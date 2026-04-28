"""
Convert SFT jsonl with <image> placeholders into streaming conversations and
optionally generate base-model greedy replies after each user turn.

Input example:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<image><image>question"},
    {"role": "assistant", "content": "NO REPLY"},
    ...
  ],
  "images": [{"path": "./data/.../000001.jpg"}, ...],
  "metadata": {"question_id": "..."}
}

Output is compatible with generate_training_data.py and inference-style
conversation JSON:
[
  {
    "question_id": "...",
    "conversation": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": [{"type": "image", "image": "..."}, ...]},
      {"role": "assistant", "content": "base model greedy reply"},
      ...
    ]
  }
]
"""

import argparse
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoProcessor

from ar_generate import autoregressive_manual_baseline
from kangaroo_model import KangarooQwenModel
from model import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


IMAGE_TOKEN_RE = re.compile(r"<image>")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate streaming AR replies from SFT jsonl")
    parser.add_argument("--input_jsonl", type=str, default="/data/wangzhichao/datasets/MMDuet2-data/sft/egoexolearn-half_multi_half_single_question-2_sec_per_frame-sft.jsonl")
    parser.add_argument("--output_json", type=str, default="/data/wangzhichao/projects/SSD_full_history/data/annotations/adapter/ar_sft_train.json")
    parser.add_argument("--model_path", type=str, default="/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt",
                        help="Required when --generate_replies is set")
    parser.add_argument("--generate_replies", action="store_true",
                        help="Use base model greedy generation to replace source assistant replies")
    parser.add_argument("--keep_source_assistant", action="store_true",
                        help="Do not generate; keep assistant replies from source SFT")
    parser.add_argument("--prompt_only", action="store_true",
                        help="Drop all assistant replies and only save user/system turns")
    parser.add_argument("--image_root", type=str, default="/data/wangzhichao/datasets",
                        help="Resolve relative image paths against this root")
    parser.add_argument("--strip_prefix", type=str, default="./data/datasets",
                        help="Optional prefix to strip from image paths before joining image_root")
    parser.add_argument("--strip_ego_time_suffix", action="store_true",
                        help="Map frame folders like videoid-0s_180s to videoid")
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--manual_ar", action=argparse.BooleanOptionalAction, default=True,
                        help="Use the same manual AR baseline path as inference.py "
                             "(autoregressive_manual_baseline) instead of model.generate(). "
                             "Default: True. Pass --no-manual-ar to fall back to model.generate().")
    parser.add_argument("--exit_layer", type=int, default=2,
                        help="Early-exit split layer used by manual AR baseline.")
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_image_path(path, image_root=None, strip_prefix=None, strip_ego_time_suffix=False):
    if path is None:
        return None
    path = str(path)
    if strip_prefix and path.startswith(strip_prefix):
        path = path[len(strip_prefix):].lstrip("/\\")
    if strip_ego_time_suffix:
        parts = path.replace("\\", "/").split("/")
        if len(parts) >= 2:
            parts[-2] = re.sub(r"-\d+s_\d+s$", "", parts[-2])
            path = "/".join(parts)
    if image_root and not os.path.isabs(path) and not path.startswith(("http://", "https://", "file://")):
        return os.path.normpath(os.path.join(image_root, path))
    return path


def get_question_id(example, index):
    metadata = example.get("metadata") or {}
    return (
        metadata.get("question_id")
        or example.get("question_id")
        or metadata.get("video_id")
        or f"sample_{index}"
    )


def image_paths(example, image_root, strip_prefix, strip_ego_time_suffix):
    paths = []
    for item in example.get("images", []):
        if isinstance(item, str):
            path = item
        elif isinstance(item, dict):
            path = item.get("path") or item.get("image")
        else:
            continue
        paths.append(resolve_image_path(
            path,
            image_root=image_root,
            strip_prefix=strip_prefix,
            strip_ego_time_suffix=strip_ego_time_suffix,
        ))
    return paths


def convert_user_content(content, paths, cursor):
    if isinstance(content, list):
        return content, cursor
    text = str(content)
    parts = []
    pos = 0
    for match in IMAGE_TOKEN_RE.finditer(text):
        before = text[pos:match.start()]
        if before.strip():
            parts.append({"type": "text", "text": before})
        if cursor >= len(paths):
            raise ValueError(f"Not enough image paths for placeholders: need index {cursor}")
        parts.append({"type": "image", "image": paths[cursor]})
        cursor += 1
        pos = match.end()
    rest = text[pos:]
    if rest.strip():
        parts.append({"type": "text", "text": rest})
    if not parts:
        parts.append({"type": "text", "text": text})
    return parts, cursor


def convert_messages_to_prompt(example, index, image_root, strip_prefix, strip_ego_time_suffix=False):
    paths = image_paths(
        example,
        image_root=image_root,
        strip_prefix=strip_prefix,
        strip_ego_time_suffix=strip_ego_time_suffix,
    )
    cursor = 0
    converted = []
    source_assistants = []
    for msg in example.get("messages", example.get("conversation", [])):
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            content, cursor = convert_user_content(content, paths, cursor)
            converted.append({"role": "user", "content": content})
        elif role == "system":
            converted.append({"role": "system", "content": content})
        elif role == "assistant":
            source_assistants.append({"role": "assistant", "content": content})
    return converted, source_assistants, cursor, len(paths)


@torch.no_grad()
def generate_reply(model, processor, history, device, max_new_tokens,
                   manual_ar=True, exit_layer=2):
    text = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(history)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    if manual_ar:
        if hasattr(model, "reset_status"):
            model.reset_status()
        reply_text, _, _ = autoregressive_manual_baseline(
            model=model,
            inputs=inputs,
            processor=processor,
            max_new_tokens=max_new_tokens,
            early_exit_layer=exit_layer,
        )
        return reply_text.strip()

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        drop_method="none",
        drop_threshold=1.0,
        drop_absolute=True,
    )
    new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def build_prompt_only_conversation(converted):
    return [turn for turn in converted if turn.get("role") in {"system", "user"}]


def build_source_assistant_conversation(converted, source_assistants):
    output = []
    assistant_idx = 0
    for turn in converted:
        output.append(turn)
        if turn.get("role") == "user" and assistant_idx < len(source_assistants):
            output.append(source_assistants[assistant_idx])
            assistant_idx += 1
    return output


def main():
    args = parse_args()
    if sum([args.generate_replies, args.keep_source_assistant, args.prompt_only]) != 1:
        raise ValueError("Choose exactly one of --generate_replies, --keep_source_assistant, or --prompt_only")
    if args.generate_replies and not args.model_path:
        raise ValueError("--model_path is required with --generate_replies")

    rows = read_jsonl(args.input_jsonl)
    end = args.end if args.end is not None else len(rows)
    rows = rows[args.start:end]

    # Sidecar JSONL for crash-safe incremental writes. Each completed sample
    # is appended (one JSON per line) and flushed immediately, so the run can
    # resume after Ctrl-C / crash / OOM without re-doing finished samples.
    if args.output_json.endswith(".jsonl"):
        ckpt_path = args.output_json
        final_json_path = None
    else:
        ckpt_path = args.output_json + ".jsonl"
        final_json_path = args.output_json

    completed_indices = set()
    existing_outputs = []
    print(f"checkpoint file: {ckpt_path}")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                meta = rec.get("metadata") or {}
                src = meta.get("source_index")
                if src is not None:
                    completed_indices.add(src)
                    existing_outputs.append(rec)
        print(f"resume: found {len(completed_indices)} already-completed samples; will skip them")
    else:
        print(f"resume: checkpoint does not exist yet, starting fresh")

    processor = None
    model = None
    if args.generate_replies:
        processor = AutoProcessor.from_pretrained(args.model_path)
        if args.manual_ar:
            model = KangarooQwenModel(
                base_model_path=args.model_path,
                adapter_model_path=None,
                early_exit_layer=args.exit_layer,
                dtype=torch.bfloat16,
                attn_implementation=args.attn_implementation,
            ).eval().to(args.device)
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=args.attn_implementation,
            ).eval().to(args.device)

    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    new_done = 0
    skipped = 0
    with open(ckpt_path, "a", encoding="utf-8") as ckpt_f:
        for offset, example in enumerate(tqdm(rows, desc="Converting"), start=args.start):
            if offset in completed_indices:
                continue
            try:
                converted, source_assistants, used_images, total_images = convert_messages_to_prompt(
                    example,
                    offset,
                    image_root=args.image_root,
                    strip_prefix=args.strip_prefix,
                    strip_ego_time_suffix=args.strip_ego_time_suffix,
                )
                if args.prompt_only:
                    conversation = build_prompt_only_conversation(converted)
                elif args.keep_source_assistant:
                    conversation = build_source_assistant_conversation(converted, source_assistants)
                else:
                    conversation = []
                    for turn in converted:
                        conversation.append(turn)
                        if turn.get("role") == "user":
                            reply = generate_reply(
                                model=model,
                                processor=processor,
                                history=conversation,
                                device=args.device,
                                max_new_tokens=args.max_new_tokens,
                                manual_ar=args.manual_ar,
                                exit_layer=args.exit_layer,
                            )
                            conversation.append({"role": "assistant", "content": reply})

                sample = {
                    "question_id": get_question_id(example, offset),
                    "conversation": conversation,
                    "metadata": {
                        **(example.get("metadata") or {}),
                        "source_index": offset,
                        "used_images": used_images,
                        "total_images": total_images,
                    },
                }
                ckpt_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                ckpt_f.flush()
                os.fsync(ckpt_f.fileno())
                existing_outputs.append(sample)
                completed_indices.add(offset)
                new_done += 1
            except Exception as exc:
                skipped += 1
                print(f"[skip] index={offset} question_id={get_question_id(example, offset)} error={exc}")

    # Final consolidated JSON list (only if user asked for .json output, not .jsonl)
    if final_json_path is not None:
        existing_outputs.sort(key=lambda r: (r.get("metadata") or {}).get("source_index", 0))
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(existing_outputs, f, ensure_ascii=False, indent=2)
        print(f"wrote consolidated JSON list to {final_json_path}")

    print(f"checkpoint:   {ckpt_path} ({len(existing_outputs)} samples total)")
    print(f"newly done:   {new_done}")
    print(f"skipped:      {skipped}")
    if existing_outputs:
        print("first sample preview:")
        print(json.dumps(existing_outputs[0], ensure_ascii=False, indent=2)[:1600])


if __name__ == "__main__":
    main()
