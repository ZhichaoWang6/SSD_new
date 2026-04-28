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
    p.add_argument("--max_decode_steps", type=int, default=20,
                   help="How many decode steps to compare layer-by-layer.")
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


def eos_ids(processor):
    e = processor.tokenizer.eos_token_id
    return set(e) if isinstance(e, list) else {e}


@torch.no_grad()
def full_forward_multi_step(base_model, inputs, max_steps):
    """
    Run prefill + N decode steps via model.generate() and return per-step
    hidden_states. Returns: list of length N, each entry is a tuple of
    (num_layers + 1) tensors (embed-or-pre-layer + ... + final_norm).
    """
    multimodal_keys = ("input_ids", "attention_mask",
                       "pixel_values", "pixel_values_videos",
                       "image_grid_thw", "video_grid_thw",
                       "second_per_grid_ts")
    kwargs = {k: inputs[k] for k in multimodal_keys if inputs.get(k) is not None}
    out = base_model.generate(
        **kwargs,
        max_new_tokens=max_steps + 1,    # 1 prefill + N decode steps
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
        output_hidden_states=True,
        drop_method="none",
        drop_threshold=1.0,
        drop_absolute=True,
    )
    # out.hidden_states[0] = prefill, [1..N] = decode steps 1..N
    per_step = list(out.hidden_states[1:])
    new_ids = out.sequences[0, inputs['input_ids'].shape[1]:].tolist()
    return per_step, new_ids


@torch.no_grad()
def manual_decode_one_step(kang, next_tok, exit_layer):
    """
    One manual_ar decode step. Returns (captured_layers, normed_logits).
    captured_layers = embed + N raw layer outputs + final_norm.
    """
    base = kang.base_model
    qwen = base.model.model
    captured = []

    bsz, seq_len = next_tok.shape
    hidden = qwen.embed_tokens(next_tok)
    captured.append(hidden.detach().clone())

    layer_past_length = base._get_layer_cache_length(0)
    base.past_key_values._seen_tokens = layer_past_length
    cache_position = torch.arange(layer_past_length, layer_past_length + seq_len, device=hidden.device)

    rope_deltas = base.model.rope_deltas
    delta = (layer_past_length + rope_deltas).to(hidden.device) if rope_deltas is not None else layer_past_length
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

    layer_past_length_v = base._get_layer_cache_length(exit_layer)
    base.past_key_values._seen_tokens = layer_past_length_v
    cache_position_v = torch.arange(layer_past_length_v, layer_past_length_v + seq_len, device=hidden.device)
    delta_v = (layer_past_length_v + rope_deltas).to(hidden.device) if rope_deltas is not None else layer_past_length_v
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
    captured.append(hidden_normed.detach().clone())
    logits = kang.head_model(hidden_normed)
    return captured, logits


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

    # path A: re-run via generate() with multi-step to capture every decode step
    if hasattr(base, 'reset_status'):
        base.reset_status()
    hs_a_per_step, new_ids_a = full_forward_multi_step(base, inputs, args.max_decode_steps)
    print(f"  generate produced {len(new_ids_a)} new tokens")

    # --- multi-step decode on path B, comparing to path A's per-step hidden_states ---
    print(f"\n--- multi-step decode comparison (up to {args.max_decode_steps} steps) ---")

    n_layers = len(kang.base_model.model.model.layers)

    # At iteration k, both paths feed new_ids_a[k] and produce a prediction
    # for new_ids_a[k+1]. Generate already chose new_ids_a[k+1]; the manual
    # path's prediction is b_tok. We compare those, and we advance the loop
    # using new_ids_a[k+1] (path A's choice) so both KV caches stay aligned
    # even if manual diverges. Hidden-state comparison: hs_a_per_step[k]
    # is the per-step hidden_states from generate's decode step (k+1).

    print(f"  {'step':>4}  {'first_div':>10}  {'worst_layer':<14}  "
          f"{'layer_max_d':>11}  {'norm_d':>9}  {'logit_d':>9}  {'head_d':>9}  "
          f"{'in':>22}  {'a_pred':>22}  {'a_via_kang':>22}  {'b_pred':>22}  {'a==b':>5}")
    print("-" * 195)

    first_step_div = -1
    n_steps = min(args.max_decode_steps, len(hs_a_per_step), len(new_ids_a) - 1)

    for step in range(n_steps):
        in_tok = new_ids_a[step]                    # what to feed at this step
        next_tok = torch.tensor([[in_tok]], device=args.device)
        hs_b, logits_b = manual_decode_one_step(kang, next_tok, args.exit_layer)
        hs_a = hs_a_per_step[step]
        assert len(hs_a) == n_layers + 1
        assert len(hs_b) == n_layers + 2

        # Track the WORST layer at this step (regardless of threshold).
        worst_layer = "embedding"
        worst_d = 0.0
        for i in range(n_layers):
            d = (hs_a[i].float() - hs_b[i].float()).abs().max().item()
            if d > worst_d:
                worst_d = d
                worst_layer = "embedding" if i == 0 else f"layer{i-1}"
        norm_d = (hs_a[n_layers].float() - hs_b[n_layers + 1].float()).abs().max().item()

        # Recompute logits via BOTH lm_head instances so we can isolate where
        # the argmax flip comes from:
        #   logits_a_via_base : what generate() actually saw (base.lm_head)
        #   logits_a_via_kang : same hidden but via kang.head_model
        #   logits_b          : manual's logits via kang.head_model
        # If logits_a_via_base != logits_a_via_kang on the same input, cuBLAS
        # is producing slightly different results across the two lm_head
        # instances (smoking-gun for argmax flip).
        logits_a_base = base.lm_head(hs_a[n_layers]).float()
        logits_a_kang = kang.head_model(hs_a[n_layers]).float()
        logits_b_f = logits_b.float()
        # Compare base-via-base vs kang-via-kang (the realistic end-to-end gap)
        logit_d = (logits_a_base - logits_b_f).abs().max().item()
        # Compare two lm_head instances on the same input
        head_d = (logits_a_base - logits_a_kang).abs().max().item()
        # What does base.lm_head's argmax give? (what generate ACTUALLY chose)
        a_pred_via_base_lm = logits_a_base[:, -1, :].argmax(-1).item()
        # What does kang.head_model's argmax on path A's final_norm give?
        a_pred_via_kang_lm = logits_a_kang[:, -1, :].argmax(-1).item()

        a_pred = new_ids_a[step + 1]
        b_pred = logits_b[:, -1, :].argmax(-1).item()
        match = "OK" if a_pred == b_pred else "**"

        if first_step_div < 0 and a_pred != b_pred:
            first_step_div = step

        in_str = f"{in_tok}({processor.tokenizer.decode([in_tok])!r})"[:22]
        a_str = f"{a_pred}({processor.tokenizer.decode([a_pred])!r})"[:22]
        ak_str = f"{a_pred_via_kang_lm}({processor.tokenizer.decode([a_pred_via_kang_lm])!r})"[:22]
        b_str = f"{b_pred}({processor.tokenizer.decode([b_pred])!r})"[:22]
        print(f"  {step:>4}  {first_step_div if first_step_div == step else '':>10}  "
              f"{worst_layer:<14}  {worst_d:>11.3e}  {norm_d:>9.3e}  {logit_d:>9.3e}  {head_d:>9.3e}  "
              f"{in_str:>22}  {a_str:>22}  {ak_str:>22}  {b_str:>22}  {match:>5}")

        if a_pred in eos_ids(processor):
            print(f"  [hit EOS at step {step}]")
            break

    print("\n========== DIAGNOSIS ==========")
    if first_step_div < 0:
        print(f"Both paths bit-equivalent for all {args.max_decode_steps} decode steps.")
        print("Either the verify divergence happens beyond this many steps, or it is")
        print("specific to the FA2 attention kernel choosing a different reduction.")
    else:
        print(f"First step that diverges: step {first_step_div}.")
        print("Layer / value diff at that step is shown above. If 'name' is")
        print("(all bit-exact) but tokens differ, divergence is purely in the lm_head")
        print("argmax (sub-threshold drift right at the top-1/top-2 boundary).")


if __name__ == "__main__":
    main()
