"""
Lockstep comparison: speculative decoding output vs manual AR output.

Same prompt, same adapter, same model instance (reset between runs). Both
should produce identical greedy token sequences if speculative decoding is
truly lossless. When they don't, this script tells you:

  - first index where they differ
  - which spec round that token came from
  - within that round, whether the divergence is at:
      * the very first accepted token   (verify-on-sequence math wrong, or
        cache state at round start wrong)
      * a later accepted token          (intra-round drift)
      * the bonus token                 (the spec extra token after verify)
      * right after a draft reject      (cache rollback bug, most likely)

It also dumps the round-by-round accept counts so you can correlate.
"""

import argparse
import copy
import json
import os

import torch
from transformers import AutoProcessor

from ar_generate import autoregressive_manual_baseline
from inference_kangaroo import kangaroo_speculative_generate
from kangaroo_model import KangarooQwenModel
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
    p.add_argument("--model_path", required=True)
    p.add_argument("--adapter_path", required=True,
                   help="Directory containing adapter_model.bin and adapter_config.json")
    p.add_argument("--data_path", required=True)
    p.add_argument("--sample_idx", type=int, default=8)
    p.add_argument("--turn_idx", type=int, default=3,
                   help="Which user turn (1-based) to use; prior turns are filled with the "
                        "base model's greedy AR replies (matches inference history).")
    p.add_argument("--exit_layer", type=int, default=2)
    p.add_argument("--speculative_steps", type=int, default=6)
    p.add_argument("--speculative_threshold", type=float, default=0.6)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn_implementation", default="flash_attention_2")
    p.add_argument("--video_root", default=None)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    return p.parse_args()


# --------- input building (same logic as diagnose_path_divergence.py) ----------

def ensure_system(conv, prompt):
    if conv and conv[0].get("role") == "system":
        return conv
    return [{"role": "system", "content": prompt}] + conv


def has_vision(content):
    return isinstance(content, list) and any(
        isinstance(it, dict) and it.get("type") in {"video", "image", "image_url"} for it in content
    )


def inject_video(conv, video_path, fps):
    if video_path is None:
        return conv
    conv = copy.deepcopy(conv)
    for turn in conv:
        if turn.get("role") != "user":
            continue
        if has_vision(turn.get("content", "")):
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


def build_history_for_turn(example, args, kang_model, processor):
    """Grow history with manual_ar replies (since we want the spec/AR comparison
    to start from the SAME prompt that inference would actually see)."""
    conv = example.get("conversation") or example.get("messages") or []
    video = resolve_video(example.get("video"), args.video_root)
    prompt = inject_video(conv, video, args.fps)
    prompt = ensure_system(prompt, args.system_prompt)

    history = []
    user_turn_count = 0
    for turn in prompt:
        role = turn.get("role")
        if role == "system":
            history.append(turn)
        elif role == "user":
            history.append(turn)
            user_turn_count += 1
            if user_turn_count == args.turn_idx:
                return history
            text = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            img_in, vid_in = process_vision_info(history)
            inputs = processor(text=[text], images=img_in, videos=vid_in,
                               padding=True, return_tensors="pt").to(args.device)
            kang_model.reset_status()
            reply, _, _ = autoregressive_manual_baseline(
                kang_model, inputs, processor,
                max_new_tokens=128, early_exit_layer=args.exit_layer,
            )
            history.append({"role": "assistant", "content": reply.strip()})
            print(f"  [history grow] turn {user_turn_count} reply: {reply.strip()[:80]!r}")
        elif role == "assistant":
            history.append(turn)
    return history


def build_inputs(processor, history, device):
    text = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(history)
    inputs = processor(text=[text], images=img_in, videos=vid_in,
                       padding=True, return_tensors="pt").to(device)
    return inputs


# --------- main comparison ----------

def classify_divergence(diverge_idx, accept_lengths, speculative_steps):
    """
    Spec output structure:
      idx 0: first token from prefill argmax
      For each round r (1-based round index):
        idx (1 + sum(accept_lengths[:r-1])) .. idx (sum(accept_lengths[:r]))
        contains accept_lengths[r-1] tokens
        Each round drafts up to speculative_steps tokens via adapter,
        then verify produces a "bonus" extra. The number ACCEPTED in
        that round = accept_lengths[r-1].
    """
    if diverge_idx == 0:
        return ("prefill", 0, 0, "first token from prefill argmax disagrees -- "
                "should be impossible if both paths share prefill")

    cum = 1  # account for prefill first token at idx 0
    for r, n_accept in enumerate(accept_lengths, start=1):
        round_first = cum
        round_last = cum + n_accept - 1
        if round_first <= diverge_idx <= round_last:
            offset_in_round = diverge_idx - round_first
            if offset_in_round == 0:
                # very first token of this round -> verify call's first emit
                return ("round_first", r, offset_in_round,
                        f"first token of spec round {r}; this is the verify call's "
                        f"output for position {diverge_idx}. Most likely a verify-phase "
                        f"position_ids / cache_position bug, OR cache state at the START "
                        f"of this round was already wrong (e.g. previous round's reject "
                        f"trim was off-by-one).")
            if offset_in_round == n_accept - 1 and n_accept == speculative_steps + 1:
                # bonus token (only when full draft accepted)
                return ("bonus", r, offset_in_round,
                        f"bonus token of round {r} (all {speculative_steps} drafts accepted, "
                        f"+1 from verify). Likely the bonus-token write to cache or the "
                        f"transition into next round mishandles its position.")
            return ("intra_round", r, offset_in_round,
                    f"middle of spec round {r}, accept #{offset_in_round} of {n_accept}. "
                    f"Verify-on-sequence produced different logits than per-token AR for "
                    f"this position. Compare verify's hidden states for this position to "
                    f"AR's hidden states at the same logical position.")
        cum += n_accept

    return ("after_all_rounds", -1, -1,
            f"divergence at idx {diverge_idx} but spec only produced "
            f"{cum} tokens; mismatch in length tracking")


def main():
    args = parse_args()

    with open(args.data_path) as f:
        data = json.load(f) if args.data_path.endswith(".json") else [
            json.loads(line) for line in f if line.strip()
        ]
    example = data[args.sample_idx]
    print(f"sample_idx={args.sample_idx} turn_idx={args.turn_idx}")

    processor = AutoProcessor.from_pretrained(args.model_path)

    print("loading KangarooQwenModel + adapter ...")
    kang = KangarooQwenModel(
        base_model_path=args.model_path,
        adapter_model_path=args.adapter_path,
        early_exit_layer=args.exit_layer,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).eval().to(args.device)

    print("\nbuilding history up to target turn ...")
    history = build_history_for_turn(example, args, kang, processor)
    inputs = build_inputs(processor, history, args.device)
    print(f"  context_length = {inputs['input_ids'].shape[1]}")

    tok = processor.tokenizer

    # ---------- Run #1: speculative decoding ----------
    print("\n--- run #1: speculative decoding ---")
    kang.reset_status()
    spec_out_ids, _, spec_stats = kangaroo_speculative_generate(
        model=kang,
        inputs=inputs,
        processor=processor,
        max_new_tokens=args.max_new_tokens,
        early_exit_layer=args.exit_layer,
        speculative_steps=args.speculative_steps,
        threshold=args.speculative_threshold,
        do_sample=False,
        past_key_values=None,
    )
    ctx_len = inputs['input_ids'].shape[1]
    spec_new_ids = spec_out_ids[0, ctx_len:].tolist()
    accept_lengths = spec_stats.get('accept_lengths', [])
    print(f"  spec produced {len(spec_new_ids)} new tokens")
    print(f"  accept_lengths per round: {accept_lengths}")
    print(f"  avg accept len: {spec_stats.get('avg_accept_length', 0):.2f}")
    spec_text = tok.decode(spec_new_ids, skip_special_tokens=True)
    print(f"  spec text: {spec_text[:200]!r}")

    # ---------- Run #2: manual AR (lossless reference) ----------
    print("\n--- run #2: manual AR (reference) ---")
    kang.reset_status()
    ar_text, _, ar_stats = autoregressive_manual_baseline(
        model=kang,
        inputs=inputs,
        processor=processor,
        max_new_tokens=args.max_new_tokens,
        early_exit_layer=args.exit_layer,
    )
    # Re-encode AR text into ids for token-level comparison.
    # NOTE: decode -> encode roundtrip can differ from raw ids on rare
    # whitespace/special-token edge cases, but for greedy text this is fine.
    ar_new_ids = tok.encode(ar_text, add_special_tokens=False)
    print(f"  ar produced {len(ar_new_ids)} new tokens (after re-encode)")
    print(f"  ar text: {ar_text[:200]!r}")

    # ---------- Compare ----------
    print("\n--- comparison ---")

    # Strip trailing EOS from both sides so we don't false-alarm on the
    # spec/AR EOS-handling difference (manual_ar decodes with
    # skip_special_tokens=True, which drops <|im_end|>; spec returns raw ids
    # which keeps it).
    eos_ids_set = set(tok.eos_token_id) if isinstance(tok.eos_token_id, list) else {tok.eos_token_id}
    spec_for_cmp = list(spec_new_ids)
    while spec_for_cmp and spec_for_cmp[-1] in eos_ids_set:
        spec_for_cmp.pop()
    ar_for_cmp = list(ar_new_ids)
    while ar_for_cmp and ar_for_cmp[-1] in eos_ids_set:
        ar_for_cmp.pop()

    # Independent text-level check: if decoded text matches, spec is lossless
    # in the user-visible sense even if raw ids differ on edge cases.
    spec_text_norm = tok.decode(spec_for_cmp, skip_special_tokens=True).strip()
    ar_text_norm = ar_text.strip()
    text_match = (spec_text_norm == ar_text_norm)
    print(f"  text-level match : {text_match}")
    if not text_match:
        print(f"    spec text norm: {spec_text_norm!r}")
        print(f"    ar   text     : {ar_text_norm!r}")

    n = min(len(spec_for_cmp), len(ar_for_cmp))
    diverge_idx = -1
    for i in range(n):
        if spec_for_cmp[i] != ar_for_cmp[i]:
            diverge_idx = i
            break
    if diverge_idx < 0 and len(spec_for_cmp) != len(ar_for_cmp):
        diverge_idx = n

    if diverge_idx < 0:
        print(f"  TOKEN-LEVEL EXACT MATCH ({len(spec_for_cmp)} tokens, EOS-stripped).")
        print(f"  Spec decoding is lossless on this sample.")
        print(f"\n  --- accept-length analysis ---")
        if accept_lengths:
            print(f"  per-round accept counts: {accept_lengths}")
            print(f"  avg accept length      : {sum(accept_lengths)/len(accept_lengths):.2f}")
            print(f"  rounds with 1 accept   : {sum(1 for a in accept_lengths if a == 1)}/{len(accept_lengths)}")
            print(f"  rounds with full accept: {sum(1 for a in accept_lengths if a == args.speculative_steps + 1)}/{len(accept_lengths)}")
            print(f"  best round             : {max(accept_lengths)}")
            print(f"  speedup ceiling (this sample): ~{sum(accept_lengths)/len(accept_lengths):.2f}x")
        return

    where, round_no, offset, hint = classify_divergence(
        diverge_idx, accept_lengths, args.speculative_steps,
    )
    k = 5
    spec_window = spec_new_ids[max(0, diverge_idx - k):diverge_idx + k]
    ar_window = ar_new_ids[max(0, diverge_idx - k):diverge_idx + k]

    print(f"  FIRST DIVERGENCE at token index {diverge_idx}")
    print(f"  category : {where}")
    print(f"  spec round: {round_no}  offset_in_round: {offset}")
    print(f"  spec window  : {spec_window}")
    print(f"                {[tok.decode([x]) for x in spec_window]}")
    print(f"  ar   window  : {ar_window}")
    print(f"                {[tok.decode([x]) for x in ar_window]}")
    print(f"\n  hint: {hint}")

    # If the diverge is near a round boundary, also report what the previous
    # round looked like (was it a reject? was it full-accept?)
    if where in ("round_first", "intra_round", "bonus") and round_no >= 2:
        prev_n_accept = accept_lengths[round_no - 2]
        full_accept = (prev_n_accept == args.speculative_steps + 1)
        print(f"\n  previous round (round {round_no - 1}) accept_count = {prev_n_accept}, "
              f"full_accept={full_accept}")
        if not full_accept:
            print("  -> previous round had a draft REJECT. If divergence is at the start of")
            print("     this round (round_first), the cache rollback after that reject is the")
            print("     prime suspect. Inspect:")
            print("       * trim_draft_layers_cache(start_index)")
            print("       * trim_verify_layers_cache(start_index)")
            print("       * adapter past_key_values trim")
            print("     all using the SAME start_index, off-by-one is the usual culprit.")
        else:
            print("  -> previous round was full-accept. The bug then is the bonus-token")
            print("     handling at the round transition (cache write for the bonus token, or")
            print("     position_ids in the next round's first draft).")


if __name__ == "__main__":
    main()
