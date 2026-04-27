"""
Verify that model.generate(do_sample=False) and autoregressive_manual_baseline
produce identical token sequences on real streaming-style samples.

Per-turn protocol (mirrors inference.py:_encode_query):
  for each user turn k:
      prompt = system + user_0 + asst_0 + ... + user_k
      out_gen     = model.generate(prompt, greedy)
      out_manual  = autoregressive_manual_baseline(KangarooQwenModel, prompt)
      compare ids
      asst_k = out_gen   # use generate's reply to grow history (consistent with current data pipeline)
"""

import argparse
import copy
import json
import os
import time

import torch
from transformers import AutoProcessor

from ar_generate import autoregressive_manual_baseline
from kangaroo_model import KangarooQwenModel
from model import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


DEFAULT_SYSTEM_PROMPT = (
    'You are a helpful assistant. Your task is to answer questions based on '
    'continuously incoming video frames. Your responses should include '
    'information from the video since your last reply (if any). If the '
    'information in this segment of the video cannot answer the question, '
    'output "NO REPLY".'
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt")
    p.add_argument("--data_path", default="/data/wangzhichao/projects/SSD_full_history/data/annotations/2fps/ego_dataset.json")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--max_turns_per_sample", type=int, default=3,
                   help="Limit user turns per sample to keep wall time manageable")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--exit_layer", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn_implementation", default="flash_attention_2")
    p.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--video_root", default=None)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--print_first_diff_context", type=int, default=5,
                   help="Print this many tokens around the first divergence")
    return p.parse_args()


def ensure_system(conv, system_prompt):
    if conv and conv[0].get("role") == "system":
        return conv
    return [{"role": "system", "content": system_prompt}] + conv


def content_has_vision(content):
    if not isinstance(content, list):
        return False
    return any(isinstance(it, dict) and it.get("type") in {"video", "image", "image_url"} for it in content)


def inject_video_first_user(conv, video_path, fps):
    if video_path is None:
        return conv
    conv = copy.deepcopy(conv)
    for turn in conv:
        if turn.get("role") != "user":
            continue
        if content_has_vision(turn.get("content", "")):
            return conv
        text = turn.get("content", "") if isinstance(turn.get("content"), str) else ""
        turn["content"] = [
            {"type": "video", "video": video_path, "fps": fps},
            {"type": "text", "text": text},
        ]
        return conv
    return conv


def resolve_video(path, root):
    if path is None or root is None:
        return path
    if isinstance(path, str) and not os.path.isabs(path) and not path.startswith(("http://", "https://", "file://")):
        return os.path.join(root, path)
    return path


def build_inputs(processor, history, device):
    text = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(history)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    return inputs


def _strip_eos(ids, eos_set):
    while ids and ids[-1] in eos_set:
        ids = ids[:-1]
    return ids


@torch.no_grad()
def run_generate(model, processor, history, device, max_new_tokens, eos_set):
    inputs = build_inputs(processor, history, device)
    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        drop_method="none",
        drop_threshold=1.0,
        drop_absolute=True,
    )
    t = time.perf_counter() - t0
    new_ids = out[0, inputs["input_ids"].shape[1]:].tolist()
    return _strip_eos(new_ids, eos_set), inputs, t


@torch.no_grad()
def run_manual_ar(kang, processor, history, device, max_new_tokens, exit_layer, eos_set):
    inputs = build_inputs(processor, history, device)
    if hasattr(kang, "reset_status"):
        kang.reset_status()
    t0 = time.perf_counter()
    text, _, _ = autoregressive_manual_baseline(
        model=kang,
        inputs=inputs,
        processor=processor,
        max_new_tokens=max_new_tokens,
        early_exit_layer=exit_layer,
    )
    t = time.perf_counter() - t0
    ids = processor.tokenizer.encode(text, add_special_tokens=False)
    return _strip_eos(ids, eos_set), t, text


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return -1


def main():
    args = parse_args()
    with open(args.data_path) as f:
        data = json.load(f)
    data = data[:args.num_samples]
    print(f"loaded {len(data)} samples from {args.data_path}")

    processor = AutoProcessor.from_pretrained(args.model_path)

    print("loading base model for generate()...")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).eval().to(args.device)

    print("loading KangarooQwenModel for manual AR (no adapter)...")
    kang = KangarooQwenModel(
        base_model_path=args.model_path,
        adapter_model_path=None,
        early_exit_layer=args.exit_layer,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).eval().to(args.device)

    tok = processor.tokenizer
    eos = tok.eos_token_id
    eos_set = set(eos) if isinstance(eos, list) else {eos}

    total_turns = 0
    matched_turns = 0
    diverging_turns = []
    gen_self_diverge = 0
    gen_time_total = 0.0
    manual_time_total = 0.0

    for idx, ex in enumerate(data):
        conv = ex.get("conversation") or ex.get("messages") or []
        if not conv:
            continue

        video = resolve_video(ex.get("video"), args.video_root)
        prompt = inject_video_first_user(conv, video, args.fps)
        prompt = ensure_system(prompt, args.system_prompt)

        history = []
        user_turn_count = 0
        for turn in prompt:
            role = turn.get("role")
            if role in ("system", "user"):
                history.append(turn)
                if role != "user":
                    continue
                user_turn_count += 1
                if user_turn_count > args.max_turns_per_sample:
                    break

                ids_gen, _, t_gen = run_generate(
                    base, processor, history, args.device, args.max_new_tokens, eos_set,
                )
                ids_gen2, _, _ = run_generate(
                    base, processor, history, args.device, args.max_new_tokens, eos_set,
                )
                ids_manual, t_manual, text_manual = run_manual_ar(
                    kang, processor, history, args.device, args.max_new_tokens,
                    args.exit_layer, eos_set,
                )
                gen_self = (ids_gen == ids_gen2)
                if not gen_self:
                    gen_self_diverge += 1
                gen_time_total += t_gen
                manual_time_total += t_manual

                total_turns += 1
                diff_pos = first_diff(ids_gen, ids_manual)
                self_tag = "" if gen_self else "  [GEN_NONDET]"
                if diff_pos == -1:
                    matched_turns += 1
                    print(f"[sample {idx} turn {user_turn_count}] MATCH "
                          f"len={len(ids_gen)} t_gen={t_gen:.2f}s t_manual={t_manual:.2f}s"
                          f"{self_tag}")
                else:
                    k = args.print_first_diff_context
                    a = ids_gen[max(0, diff_pos - k):diff_pos + k]
                    b = ids_manual[max(0, diff_pos - k):diff_pos + k]
                    diverging_turns.append((idx, user_turn_count, diff_pos))
                    print(f"[sample {idx} turn {user_turn_count}] DIVERGE@{diff_pos}  "
                          f"gen_len={len(ids_gen)} manual_len={len(ids_manual)} "
                          f"t_gen={t_gen:.2f}s t_manual={t_manual:.2f}s"
                          f"{self_tag}")
                    print(f"   gen   : {a} -> {[tok.decode([x]) for x in a]}")
                    print(f"   manual: {b} -> {[tok.decode([x]) for x in b]}")
                    if not gen_self:
                        c = ids_gen2[max(0, diff_pos - k):diff_pos + k]
                        print(f"   gen#2 : {c} -> {[tok.decode([x]) for x in c]}")

                # grow history with generate()'s reply (matches current data pipeline)
                history.append({"role": "assistant", "content": tok.decode(ids_gen, skip_special_tokens=True)})
            elif role == "assistant":
                # if dataset already provided assistant turns, keep them as-is
                history.append(turn)

    print("\n========== SUMMARY ==========")
    print(f"attn_implementation       : {args.attn_implementation}")
    print(f"total turns compared      : {total_turns}")
    print(f"generate vs manual MATCH  : {matched_turns}")
    print(f"generate vs manual DIVERGE: {len(diverging_turns)}")
    print(f"generate vs generate#2 div: {gen_self_diverge}  (>0 means generate() itself is non-deterministic)")
    if diverging_turns:
        print(f"diverging (sample,turn,pos): {diverging_turns}")
    if total_turns:
        print(f"avg t_generate per turn   : {gen_time_total / total_turns:.2f}s")
        print(f"avg t_manual_ar per turn  : {manual_time_total / total_turns:.2f}s")
        print(f"manual_ar slowdown        : {manual_time_total / max(gen_time_total, 1e-9):.2f}x")


if __name__ == "__main__":
    main()
