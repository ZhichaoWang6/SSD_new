"""
Generate AR/greedy assistant replies with the base model and save them as
teacher-forced JSON for adapter training.

Output format is compatible with generate_training_data.py:
[
  {
    "question_id": "...",
    "video": "...",
    "conversation": [
      {"role": "user", "content": [{"type": "video", ...}, {"type": "text", ...}]},
      {"role": "assistant", "content": "base model generated reply"}
    ]
  }
]
"""

import argparse
import copy
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoProcessor

from ar_generate import autoregressive_manual_baseline
from kangaroo_model import KangarooQwenModel
from model import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(description="Generate base-model AR replies for adapter data")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--video_root", type=str, default=None,
                        help="Directory used to resolve top-level video filenames")
    parser.add_argument("--image_root", type=str, default=None,
                        help="Directory used to resolve top-level image filenames")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--streaming_turns", action="store_true",
                        help="Generate one assistant reply after each streaming user frame turn. "
                             "A leading text-only question turn is kept as context without generating.")
    parser.add_argument("--manual_ar", action="store_true",
                        help="Use the same manual AR baseline path as inference.py instead of model.generate().")
    parser.add_argument("--exit_layer", type=int, default=2,
                        help="Early-exit split layer used by the manual AR baseline.")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--system_prompt", type=str, default=(
        'You are a helpful assistant. Your task is to answer questions based on continuously incoming video frames. '
        'Your responses should include information from the video since your last reply (if any). '
        'If the information in this segment of the video cannot answer the question, output "NO REPLY".'
    ))
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    return parser.parse_args()


def load_data(path):
    if path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_video_path(video, video_root):
    if video is None:
        return None
    if isinstance(video, str) and video_root and not os.path.isabs(video) and not video.startswith(("http://", "https://", "file://")):
        return os.path.join(video_root, video)
    return video


def resolve_media_path(path, root):
    if path is None:
        return None
    if isinstance(path, str) and root and not os.path.isabs(path) and not path.startswith(("http://", "https://", "file://")):
        return os.path.join(root, path)
    return path


def get_top_level_images(example, image_root):
    images = example.get("images", None)
    if images is None:
        images = example.get("image", None)
    if images is None:
        return []
    if isinstance(images, str):
        images = [images]
    return [resolve_media_path(image, image_root) for image in images if image is not None]


def content_has_vision(content):
    if not isinstance(content, list):
        return False
    for item in content:
        if isinstance(item, dict) and item.get("type") in {"video", "image", "image_url"}:
            return True
    return False


def inject_video_into_first_user(conversation, video_path, fps):
    conversation = copy.deepcopy(conversation)
    if video_path is None:
        return conversation

    for turn in conversation:
        if turn.get("role") != "user":
            continue
        content = turn.get("content", "")
        if content_has_vision(content):
            return conversation
        text = content if isinstance(content, str) else ""
        turn["content"] = [
            {
                "type": "video",
                "video": video_path,
                "fps": fps,
            },
            {
                "type": "text",
                "text": text,
            },
        ]
        return conversation
    return conversation


def inject_images_into_first_user(conversation, image_paths):
    conversation = copy.deepcopy(conversation)
    if not image_paths:
        return conversation

    for turn in conversation:
        if turn.get("role") != "user":
            continue
        content = turn.get("content", "")
        if content_has_vision(content):
            return conversation
        text = content if isinstance(content, str) else ""
        turn["content"] = [
            *[
                {
                    "type": "image",
                    "image": image_path,
                }
                for image_path in image_paths
            ],
            {
                "type": "text",
                "text": text,
            },
        ]
        return conversation
    return conversation


def ensure_system_prompt(conversation, system_prompt):
    if conversation and conversation[0].get("role") == "system":
        return conversation
    return [{"role": "system", "content": system_prompt}] + conversation


def strip_assistant_turns(conversation):
    return [turn for turn in conversation if turn.get("role") != "assistant"]


def has_vision_content(turn):
    return content_has_vision(turn.get("content", ""))


def has_text_content(turn):
    content = turn.get("content", "")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(
            isinstance(item, dict)
            and item.get("type") == "text"
            and str(item.get("text", "")).strip()
            for item in content
        )
    return False


@torch.no_grad()
def generate_reply(model, processor, conversation, device, max_new_tokens, manual_ar=False, exit_layer=2):
    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(conversation)
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

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "drop_method": "none",
        "drop_threshold": 1.0,
        "drop_absolute": True,
    }
    output_ids = model.generate(**inputs, **generate_kwargs)
    new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def build_single_reply_sample(ex, prompt_conversation, reply, global_idx):
    sample_id = ex.get("question_id") or ex.get("video") or f"sample_{global_idx}"
    return {
        "question_id": sample_id,
        "video": ex.get("video"),
        "duration": ex.get("duration"),
        "conversation": prompt_conversation + [
            {
                "role": "assistant",
                "content": reply,
            }
        ],
    }


def build_streaming_reply_sample(model, processor, ex, prompt_conversation, args, global_idx):
    sample_id = ex.get("question_id") or ex.get("video") or f"sample_{global_idx}"
    output_conversation = []
    model_history = []

    if prompt_conversation and prompt_conversation[0].get("role") == "system":
        model_history.append(prompt_conversation[0])
        output_conversation.append(prompt_conversation[0])
        turns = prompt_conversation[1:]
    else:
        system_turn = {"role": "system", "content": args.system_prompt}
        model_history.append(system_turn)
        output_conversation.append(system_turn)
        turns = prompt_conversation

    generated_turns = 0
    for turn in turns:
        if turn.get("role") != "user":
            continue

        output_conversation.append(turn)
        model_history.append(turn)

        # ego_dataset-style samples often start with a text-only question. It is
        # context for all future frame turns, not a turn that should be answered.
        if not has_vision_content(turn) and has_text_content(turn) and generated_turns == 0:
            continue

        reply = generate_reply(
            model=model,
            processor=processor,
            conversation=model_history,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            manual_ar=args.manual_ar,
            exit_layer=args.exit_layer,
        )
        asst_turn = {"role": "assistant", "content": reply}
        output_conversation.append(asst_turn)
        model_history.append(asst_turn)
        generated_turns += 1

    if generated_turns == 0:
        return None

    return {
        "question_id": sample_id,
        "video": ex.get("video"),
        "duration": ex.get("duration"),
        "conversation": output_conversation,
    }


def main():
    args = parse_args()
    data = load_data(args.input_json)
    end = args.end if args.end is not None else len(data)
    data = data[args.start:end]

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
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

    outputs = []
    for global_idx, ex in enumerate(tqdm(data, desc="Generating AR replies"), start=args.start):
        conversation = ex.get("conversation") or ex.get("messages") or []
        if not conversation:
            continue

        video_path = resolve_video_path(ex.get("video"), args.video_root)
        image_paths = get_top_level_images(ex, args.image_root)
        prompt_conversation = strip_assistant_turns(conversation)
        if image_paths:
            prompt_conversation = inject_images_into_first_user(prompt_conversation, image_paths)
        else:
            prompt_conversation = inject_video_into_first_user(prompt_conversation, video_path, args.fps)
        model_conversation = ensure_system_prompt(copy.deepcopy(prompt_conversation), args.system_prompt)

        try:
            if args.streaming_turns:
                sample = build_streaming_reply_sample(
                    model=model,
                    processor=processor,
                    ex=ex,
                    prompt_conversation=model_conversation,
                    args=args,
                    global_idx=global_idx,
                )
                if sample is None:
                    continue
                outputs.append(sample)
                continue

            reply = generate_reply(
                model=model,
                processor=processor,
                conversation=model_conversation,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                manual_ar=args.manual_ar,
                exit_layer=args.exit_layer,
            )
        except Exception as exc:
            print(f"[skip] index={global_idx} question_id={ex.get('question_id')} error={exc}")
            continue

        outputs.append(build_single_reply_sample(ex, prompt_conversation, reply, global_idx))

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"saved {len(outputs)} samples to {args.output_json}")
    if outputs:
        print(json.dumps(outputs[0], ensure_ascii=False, indent=2)[:1200])


if __name__ == "__main__":
    main()
