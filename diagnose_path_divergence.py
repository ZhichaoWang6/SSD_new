"""
Locate where forward_draft_or_large_model diverges from model.forward().

Plan:
  1. Build a real streaming-style prompt from ego_dataset.json (one of the
     turns that we know causes greedy divergence — sample 8 turn 3 by default).
  2. Run a full prefill via model(**kwargs). Record hidden_states at every
     layer plus the final logits. This is the ground-truth "full forward".
  3. Run the SAME prefill via the manual AR path (just calls
     base_model.model(**kwargs), so this should be bit-exact with step 2 —
     this is a sanity check).
  4. Pick the first decoded token (argmax of prefill last-position logits).
     Step it through both paths in lockstep:
         path A (full forward): model(input_ids=[tok], past_key_values=cache_a)
         path B (manual):       forward_draft_or_large_model(in_tokens_small=...)
                                forward_draft_or_large_model(in_features_large=...)
     For each decoder layer index l, hook the layer output and compare
         |hidden_A[l] - hidden_B[l]|.max()
     Report the first layer where max-diff > THRESH and the values around it.
  5. Also dump the embedding diff and the final-norm diff.

Reading the output:
  - if diff already at layer 0 (= embedding output): position_ids / RoPE bug.
  - if diff first appears at exactly layer == early_exit_layer: handoff
    between draft and verify (likely _seen_tokens / cache_position).
  - if diff grows linearly with layer index: cumulative numerical drift only,
    not a structural bug.
  - if diff explodes only after layer N for some specific N: that layer's
    KV cache state is wrong on one of the two sides.
"""

import argparse
import copy
import json
import os

import torch
from transformers import AutoProcessor

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
    p.add_argument("--sample_idx", type=int, default=8)
    p.add_argument("--turn_idx", type=int, default=3, help="Which user turn (1-based) to diagnose")
    p.add_argument("--exit_layer", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn_implementation", default="eager",
                   help="Use 'eager' for deterministic attn while diagnosing.")
    p.add_argument("--video_root", default=None)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--threshold", type=float, default=1e-3,
                   help="Max-abs-diff threshold to flag a layer as 'diverged'.")
    p.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    return p.parse_args()


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


def build_history_for_turn(example, sample_args, base_model, processor):
    """
    Build the streaming history up to (and including) the target user turn,
    growing assistant replies via base_model.generate() (greedy) so we land
    on the same prompt that triggered divergence in the verification run.
    """
    conv = example.get("conversation") or example.get("messages") or []
    video = resolve_video(example.get("video"), sample_args.video_root)
    prompt = inject_video(conv, video, sample_args.fps)
    prompt = ensure_system(prompt, sample_args.system_prompt)

    history = []
    user_turn_count = 0
    for turn in prompt:
        role = turn.get("role")
        if role == "system":
            history.append(turn)
        elif role == "user":
            history.append(turn)
            user_turn_count += 1
            if user_turn_count == sample_args.turn_idx:
                return history
            text = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            img_in, vid_in = process_vision_info(history)
            inputs = processor(text=[text], images=img_in, videos=vid_in,
                               padding=True, return_tensors="pt").to(sample_args.device)
            out = base_model.generate(
                **inputs, max_new_tokens=128, do_sample=False, use_cache=True,
                drop_method="none", drop_threshold=1.0, drop_absolute=True,
            )
            new_ids = out[0, inputs["input_ids"].shape[1]:]
            reply = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            history.append({"role": "assistant", "content": reply})
            print(f"  [history grow] turn {user_turn_count} reply: {reply[:80]!r}")
        elif role == "assistant":
            history.append(turn)
    return history


def build_inputs(processor, history, device):
    text = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(history)
    inputs = processor(text=[text], images=img_in, videos=vid_in,
                       padding=True, return_tensors="pt").to(device)
    return inputs


@torch.no_grad()
def full_forward_one_token(base_model, processor, prefill_inputs):
    """
    Run a fresh prefill+1-decode through model.generate() and capture every
    layer's hidden state for the decoded position. This is the *real* path
    that HF generate() uses, including its prepare_inputs_for_generation
    handling of cache_position / position_ids / rope_deltas.
    """
    out = base_model.generate(
        **prefill_inputs,
        max_new_tokens=2,                # 1 prefill + 1 decode step
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
        output_hidden_states=True,
        drop_method="none",
        drop_threshold=1.0,
        drop_absolute=True,
    )
    # out.hidden_states is a tuple of length max_new_tokens.
    # element 0  = prefill (tuple of layer outputs over the whole prompt)
    # element 1  = first decode step (tuple of layer outputs over 1 new token)
    decode_hs = out.hidden_states[1]          # tuple of (num_layers + 1) tensors
    # also return the decoded token id for sanity check
    new_ids = out.sequences[0, prefill_inputs['input_ids'].shape[1]:]
    return decode_hs, new_ids


@torch.no_grad()
def manual_forward_one_token(kang, next_tok, exit_layer):
    """
    Run a single decode step through forward_draft_or_large_model. We capture
    every layer's hidden state by manually re-implementing the loop here.
    """
    base = kang.base_model
    qwen = base.model.model

    # --- draft phase (layers 0 .. exit_layer-1) ---
    captured = []
    bsz, seq_len = next_tok.shape
    hidden = qwen.embed_tokens(next_tok)
    captured.append(hidden.detach().clone())  # embedding output

    layer_past_length = base._get_layer_cache_length(0)
    base.past_key_values._seen_tokens = layer_past_length

    cache_position = torch.arange(layer_past_length, layer_past_length + seq_len, device=hidden.device)

    rope_deltas = base.model.rope_deltas
    if rope_deltas is not None:
        delta = (layer_past_length + rope_deltas).to(hidden.device)
    else:
        delta = layer_past_length
    pos_ids = torch.arange(seq_len, device=hidden.device).view(1, -1).expand(bsz, -1) + delta
    pos_ids = pos_ids.unsqueeze(0).expand(3, -1, -1)
    pos_emb = qwen.rotary_emb(hidden, pos_ids)

    attn_mask = torch.ones((bsz, layer_past_length + seq_len), dtype=torch.bool, device=hidden.device)
    causal_mask = qwen._update_causal_mask(attn_mask, hidden, cache_position,
                                           base.past_key_values, output_attentions=False)

    for l_idx in range(exit_layer):
        layer = qwen.layers[l_idx]
        out = layer(hidden, attention_mask=causal_mask, position_ids=pos_ids,
                    past_key_value=base.past_key_values, output_attentions=False,
                    use_cache=True, cache_position=cache_position, position_embeddings=pos_emb)
        hidden = out[0]
        captured.append(hidden.detach().clone())

    # --- verify phase (layers exit_layer .. end) ---
    layer_past_length_v = base._get_layer_cache_length(exit_layer)
    base.past_key_values._seen_tokens = layer_past_length_v

    cache_position_v = torch.arange(layer_past_length_v, layer_past_length_v + seq_len, device=hidden.device)

    rope_deltas = base.model.rope_deltas
    if rope_deltas is not None:
        delta_v = (layer_past_length_v + rope_deltas).to(hidden.device)
    else:
        delta_v = layer_past_length_v
    pos_ids_v = torch.arange(seq_len, device=hidden.device).view(1, -1).expand(bsz, -1) + delta_v
    pos_ids_v = pos_ids_v.unsqueeze(0).expand(3, -1, -1)
    pos_emb_v = qwen.rotary_emb(hidden, pos_ids_v)

    attn_mask_v = torch.ones((bsz, layer_past_length_v + seq_len), dtype=torch.bool, device=hidden.device)
    causal_mask_v = qwen._update_causal_mask(attn_mask_v, hidden, cache_position_v,
                                             base.past_key_values, output_attentions=False)

    for l_idx in range(exit_layer, len(qwen.layers)):
        layer = qwen.layers[l_idx]
        out = layer(hidden, attention_mask=causal_mask_v, position_ids=pos_ids_v,
                    past_key_value=base.past_key_values, output_attentions=False,
                    use_cache=True, cache_position=cache_position_v, position_embeddings=pos_emb_v)
        hidden = out[0]
        captured.append(hidden.detach().clone())

    hidden_normed = qwen.norm(hidden)
    captured.append(hidden_normed.detach().clone())  # final norm output
    logits = kang.head_model(hidden_normed)
    return captured, logits, (pos_ids, pos_ids_v, layer_past_length, layer_past_length_v)


def main():
    args = parse_args()

    with open(args.data_path) as f:
        data = json.load(f)
    example = data[args.sample_idx]
    print(f"sample_idx={args.sample_idx} turn_idx={args.turn_idx}")
    print(f"video={example.get('video')}")

    processor = AutoProcessor.from_pretrained(args.model_path)

    print("loading base model (full forward path)...")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).eval().to(args.device)

    print("loading KangarooQwenModel (manual AR path)...")
    kang = KangarooQwenModel(
        base_model_path=args.model_path, adapter_model_path=None,
        early_exit_layer=args.exit_layer, dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).eval().to(args.device)

    print("\nbuilding history up to target turn...")
    history = build_history_for_turn(example, args, base, processor)
    inputs = build_inputs(processor, history, args.device)
    print(f"  context_length = {inputs['input_ids'].shape[1]}")

    # --- prefill on both paths ---
    print("\n--- prefill ---")
    prefill_kwargs = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs.get('attention_mask'),
        'use_cache': True,
        'output_hidden_states': True,
        'return_dict': True,
        'pixel_values': inputs.get('pixel_values'),
        'pixel_values_videos': inputs.get('pixel_values_videos'),
        'image_grid_thw': inputs.get('image_grid_thw'),
        'video_grid_thw': inputs.get('video_grid_thw'),
        'second_per_grid_ts': inputs.get('second_per_grid_ts'),
        'drop_method': 'none',
        'drop_threshold': 1.0,
        'drop_absolute': True,
    }
    prefill_kwargs = {k: v for k, v in prefill_kwargs.items() if v is not None}

    # sanity: prefill via plain forward on both models, last-pos logits should match
    if hasattr(base, 'reset_status'):
        base.reset_status()
    out_pre_a = base(**prefill_kwargs)
    first_tok_a_pre = torch.argmax(out_pre_a.logits[:, -1, :], dim=-1).item()

    kang.reset_status()
    out_pre_b = kang.base_model.model(**prefill_kwargs)
    kang.base_model.past_key_values = out_pre_b.past_key_values
    first_tok_b = torch.argmax(out_pre_b.logits[:, -1, :], dim=-1).item()
    diff_pre = (out_pre_a.logits[:, -1, :].float() - out_pre_b.logits[:, -1, :].float()).abs().max().item()
    print(f"  prefill last-pos logits max diff : {diff_pre:.3e}")
    print(f"  first_tok via plain forward on base : {first_tok_a_pre} "
          f"({processor.tokenizer.decode([first_tok_a_pre])!r})")
    print(f"  first_tok via plain forward on kang : {first_tok_b} "
          f"({processor.tokenizer.decode([first_tok_b])!r})")
    del out_pre_a

    # path A: re-run via generate() to capture decode hidden_states the way HF does it
    if hasattr(base, 'reset_status'):
        base.reset_status()
    hs_a, new_ids_a = full_forward_one_token(base, processor, prefill_kwargs)
    first_tok_a = new_ids_a[0].item()
    print(f"  first_tok via generate() prefill    : {first_tok_a} "
          f"({processor.tokenizer.decode([first_tok_a])!r})")

    # --- one decode step on both paths ---
    print("\n--- decode step 1 ---")
    next_tok = torch.tensor([[first_tok_a]], device=args.device)

    hs_b, logits_b, dbg = manual_forward_one_token(kang, next_tok, args.exit_layer)

    print(f"  full forward returned {len(hs_a)} hidden_states (embed + {len(hs_a)-1} layers)")
    print(f"  manual returned {len(hs_b)} hidden_states (embed + {len(hs_b)-2} layers + final_norm)")

    # hs_a[0] = embedding, hs_a[1..N] = layer 0..N-1 outputs
    # hs_b[0] = embedding, hs_b[1..N] = layer 0..N-1 outputs, hs_b[N+1] = norm
    n_layers = len(hs_a) - 1
    assert n_layers == len(kang.base_model.model.model.layers)

    print(f"\n{'idx':>4}  {'name':<14}  {'max_abs_diff':>14}  {'mean_abs_diff':>14}  {'flag':>6}")
    print("-" * 64)
    first_diverge = -1
    for i in range(n_layers + 1):
        a = hs_a[i].float()
        b = hs_b[i].float()
        if a.shape != b.shape:
            print(f"  shape mismatch at idx {i}: {a.shape} vs {b.shape}")
            continue
        diff = (a - b).abs()
        max_d, mean_d = diff.max().item(), diff.mean().item()
        flag = "**" if max_d > args.threshold else ""
        if first_diverge < 0 and max_d > args.threshold:
            first_diverge = i
        name = "embedding" if i == 0 else f"layer{i-1}"
        print(f"  {i:>4}  {name:<14}  {max_d:>14.3e}  {mean_d:>14.3e}  {flag:>6}")

    # final norm comparison: compute norm(hs_a[-1]) and compare to hs_b[-1]
    a_final = base.model.norm(hs_a[-1]).float()
    b_final = hs_b[-1].float()
    diff_norm = (a_final - b_final).abs()
    print(f"  {'norm':>4}  {'final_norm':<14}  {diff_norm.max().item():>14.3e}  "
          f"{diff_norm.mean().item():>14.3e}")

    # logits via lm_head on each path's final-norm output
    logits_a = base.lm_head(a_final)
    diff_logits = (logits_a - logits_b.float()).abs()
    top1_a = logits_a[:, -1, :].argmax(-1).item()
    top1_b = logits_b[:, -1, :].argmax(-1).item()
    print(f"\n  logits max diff: {diff_logits.max().item():.3e}")
    print(f"  top1 full  : {top1_a} ({processor.tokenizer.decode([top1_a])!r})")
    print(f"  top1 manual: {top1_b} ({processor.tokenizer.decode([top1_b])!r})")

    print("\n========== DIAGNOSIS ==========")
    if first_diverge < 0:
        print(f"All layers within threshold {args.threshold:.0e}.")
        print("Divergence at greedy level is from sub-threshold drift (cumulative).")
    elif first_diverge == 0:
        print("Diverges at EMBEDDING output -> input_ids or position embeddings differ.")
    elif first_diverge == args.exit_layer:
        print(f"Diverges exactly at layer {args.exit_layer} (= exit_layer).")
        print("Likely root cause: draft -> verify hand-off in forward_draft_or_large_model.")
        print("Check _seen_tokens override and rope_deltas re-use between the two halves.")
    else:
        print(f"First diverges at layer {first_diverge}.")
        print("Likely root cause: that layer's attention sees a different KV cache state")
        print("between the two paths (e.g. cache_position / sliding window).")


if __name__ == "__main__":
    main()
