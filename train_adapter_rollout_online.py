"""
Online rollout training for the Kangaroo adapter.

Same idea as your script (which is correct):
    - load KangarooQwenModel with adapter
    - run the actual inference path: prefill, then speculative draft -> verify
    - adapter forward is in grad path; base forward is no_grad
    - labels come from verify (= what base would say)
    - loss is computed on adapter's draft prediction vs verify, including the
      first mismatch position
    - KV caches are trimmed exactly like inference; cache tensors are
      .detach()'d across rounds so the gradient graph doesn't grow unbounded

Adds, on top of your script:
    --prefix_weighting {none,linear,exp}  (default: exp)
    --prefix_decay 0.85                   (used when prefix_weighting=exp)
    --distill                             distillation loss (KL) on top of CE
    --distill_weight 0.5                  weight for distill term
    --per_round_backward                  backprop per round (saves memory)
    --no_reply_keep_ratio 0.1             default cut down NO REPLY
    Improved debug output

Run example:
    python train_adapter_rollout_online.py \
        --model_path /data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt \
        --data_path /path/to/ar_sft_train_my_ar.json.jsonl \
        --outdir /path/to/adapter_checkpoints/rollout_online \
        --adapter_path /path/to/epoch060/adapter_model.bin \
        --exit_layer 2 --device cuda:0 \
        --num_epochs 3 --lr 1e-5 \
        --speculative_steps 6 --threshold 0.0 \
        --prefix_weighting exp --prefix_decay 0.85 \
        --no_reply_keep_ratio 0.1 \
        --debug --debug_steps 4
"""

import argparse
import copy
import json
import os
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor, get_linear_schedule_with_warmup

from kangaroo_model import KangarooQwenModel
from qwen_vl_utils import process_vision_info


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt")
    p.add_argument("--data_path", required=True)
    p.add_argument("--outdir", required=True)

    p.add_argument("--adapter_path", default=None)
    p.add_argument("--exit_layer", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn_implementation", default="flash_attention_2")

    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--num_warmup_steps", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=0.5)
    p.add_argument("--save_freq", type=int, default=1)

    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--answer_margin_tokens", type=int, default=4)
    p.add_argument("--speculative_steps", type=int, default=6)
    p.add_argument("--threshold", type=float, default=0.0)

    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--no_reply_keep_ratio", type=float, default=0.1)

    # ---- new: loss shaping ----
    p.add_argument("--prefix_weighting", choices=["none", "linear", "exp"], default="exp",
                   help="Per-position weight in the round CE loss. 'exp' uses "
                        "weight[k] = prefix_decay**k; 'linear' uses (K-k)/K; "
                        "'none' = uniform.")
    p.add_argument("--prefix_decay", type=float, default=0.85,
                   help="Decay base for exp weighting. 0.7 is aggressive, "
                        "0.85 is moderate, 0.95 is mild.")
    p.add_argument("--distill", action="store_true",
                   help="Add KL(verify_softmax || adapter_softmax) on top of CE.")
    p.add_argument("--distill_weight", type=float, default=0.5)

    # ---- new: memory ----
    p.add_argument("--per_round_backward", action="store_true",
                   help="Backprop after every spec round instead of accumulating "
                        "the whole turn into one graph. Saves a lot of memory on "
                        "long generations.")

    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug_steps", type=int, default=4)
    return p.parse_args()


# ---------- data ----------

def load_data(path):
    if path.endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def freeze_base_train_adapter(model):
    model.base_model.model.eval()
    model.head_model.eval()
    model.adapter_model.train()
    for p in model.base_model.model.parameters():
        p.requires_grad = False
    for p in model.head_model.parameters():
        p.requires_grad = False
    for p in model.adapter_model.parameters():
        p.requires_grad = True


def save_adapter(model, outdir, epoch):
    save_dir = os.path.join(outdir, f"epoch_{epoch:03d}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.adapter_model.state_dict(), os.path.join(save_dir, "adapter_model.bin"))
    cfg = {
        "exit_layer": model.early_exit_layer,
        "use_mlp": getattr(model.adapter_model.layers[0], "use_mlp", True),
    }
    with open(os.path.join(save_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"[save] {save_dir}")


def build_inputs(processor, conversation, device):
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)
    return processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)


def estimate_answer_token_len(processor, gt_answer):
    if isinstance(gt_answer, list):
        text = "".join(it.get("text", "") for it in gt_answer
                       if isinstance(it, dict) and it.get("type") == "text")
    else:
        text = str(gt_answer)
    return max(1, len(processor.tokenizer(text, add_special_tokens=False)["input_ids"]))


def iter_contexts(example):
    messages = example.get("conversation", example.get("messages", []))
    if not messages:
        return
    history = []
    i = 0
    if messages and messages[0].get("role") == "system":
        history.append(messages[0])
        i = 1
    while i < len(messages) - 1:
        u = messages[i]
        a = messages[i + 1]
        if u.get("role") == "user" and a.get("role") == "assistant":
            yield copy.deepcopy(history + [u]), a.get("content", "")
            history.append(u)
            history.append(a)
            i += 2
        else:
            history.append(messages[i])
            i += 1


def _cache_len(adapter_past_key_values):
    if adapter_past_key_values and adapter_past_key_values[0] is not None:
        return adapter_past_key_values[0][0].shape[2]
    return 0


# ---------- prefix weighting ----------

def make_prefix_weights(K, mode, decay, device, dtype):
    """Return [K] weight tensor."""
    if K <= 0:
        return torch.ones(0, device=device, dtype=dtype)
    if mode == "none":
        w = torch.ones(K, device=device, dtype=dtype)
    elif mode == "linear":
        # position k (0-indexed) appears in K-k of the prefix-prob terms
        w = torch.arange(K, 0, -1, device=device, dtype=dtype) / K
    elif mode == "exp":
        idx = torch.arange(K, device=device, dtype=dtype)
        w = decay ** idx
    else:
        raise ValueError(mode)
    # normalise so the *mean* weight is 1 -- preserves overall lr scale
    w = w * (K / w.sum().clamp(min=1e-6))
    return w


def weighted_ce(adapter_logits, labels, weights):
    """
    adapter_logits: [K, V]
    labels:         [K]
    weights:        [K]
    Returns scalar.
    """
    per = F.cross_entropy(adapter_logits, labels, reduction="none")  # [K]
    return (per * weights).sum() / weights.sum().clamp(min=1e-6)


def weighted_kl(adapter_logits, verify_logits, weights):
    """
    KL(verify || adapter) per position, then weighted average.
    adapter_logits, verify_logits: [K, V]
    """
    target_logp = F.log_softmax(verify_logits.detach(), dim=-1)
    target_p = target_logp.exp()
    out_logp = F.log_softmax(adapter_logits, dim=-1)
    per = (target_p * (target_logp - out_logp)).sum(dim=-1)  # [K]
    return (per * weights).sum() / weights.sum().clamp(min=1e-6)


# ---------- training one turn ----------

def online_train_one_turn(
    model, processor, inputs, args,
    debug=False, optimizer=None, scheduler=None,
):
    """Returns dict with stats; accumulates gradients via backward."""
    base_model = model.base_model
    adapter_model = model.adapter_model
    head_model = model.head_model
    device = inputs["input_ids"].device

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    eos_id = tokenizer.eos_token_id
    eos_set = set(eos_id) if isinstance(eos_id, list) else {eos_id}
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        eos_set.add(im_end_id)
    eos_fill = list(eos_set)[0]

    input_ids = inputs["input_ids"]
    bs, ctx_len = input_ids.shape
    assert bs == 1
    max_length = ctx_len + args.max_new_tokens

    global_tokens = torch.full((1, max_length), eos_fill, dtype=torch.long, device=device)
    global_tokens[:, :ctx_len] = input_ids

    # ---- prefill ----
    fwd = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "use_cache": True, "output_hidden_states": True, "return_dict": True,
        "pixel_values": inputs.get("pixel_values"),
        "pixel_values_videos": inputs.get("pixel_values_videos"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "video_grid_thw": inputs.get("video_grid_thw"),
        "second_per_grid_ts": inputs.get("second_per_grid_ts"),
        "drop_method": "none", "drop_threshold": 1.0, "drop_absolute": True,
    }
    fwd = {k: v for k, v in fwd.items() if v is not None}

    with torch.no_grad():
        out = base_model.model(**fwd)
        base_model.past_key_values = out.past_key_values
        first_token = torch.argmax(out.logits[:, -1, :], dim=-1)
        global_tokens[:, ctx_len] = first_token.item()
        early_h = out.hidden_states[model.early_exit_layer]

    # adapter prefill (no grad - we just need to populate KV cache for the prompt)
    with torch.no_grad():
        _, adapter_past_key_values = adapter_model.forward_early_stop(
            inputs_embeds=early_h.detach(), use_cache=True,
        )

    if first_token.item() in eos_set:
        return None

    if debug:
        print(f"\n========== TURN  ctx_len={ctx_len}  max_new={args.max_new_tokens} ==========")
        print(f"first_tok={first_token.item()} ({tokenizer.decode(first_token)!r})")
        print(f"prefill cache: draft={base_model._get_layer_cache_length(0)} "
              f"verify={base_model._get_layer_cache_length(model.early_exit_layer)} "
              f"adapter={_cache_len(adapter_past_key_values)}")

    start_index = ctx_len
    accum_loss = None
    accum_tokens = 0
    accept_lengths = []
    correct = 0
    total_examined = 0
    rounds_done = 0
    round_loss_history = []

    while start_index < max_length - 1:
        rounds_done += 1
        start_copy = start_index
        end_index = start_index + 1
        remaining = max_length - 1 - start_index
        round_steps = min(args.speculative_steps, remaining)

        exited_hidden = None
        draft_token_ids = []
        adapter_logits_list = []  # each entry: [1, V] grad-tracked
        round_eos_in_draft = False

        # ---- draft ----
        for step in range(1 + round_steps):
            in_token = global_tokens[:, end_index - 1:end_index]
            with torch.no_grad():
                early_step = base_model.forward_draft_or_large_model(in_tokens_small=in_token)

            if step == 0:
                exited_hidden = None
            exited_hidden = (
                early_step.detach() if exited_hidden is None
                else torch.cat([exited_hidden, early_step.detach()], dim=1)
            )

            if step == round_steps:
                break

            hidden_state, adapter_past_key_values = adapter_model.forward_early_stop(
                inputs_embeds=early_step.detach(),
                past_key_values=adapter_past_key_values,
                use_cache=True,
            )
            adapter_logits = head_model(hidden_state[:, -1:, :]).float()  # [1,1,V]
            pred_tok = torch.argmax(adapter_logits[:, -1, :], dim=-1)
            pred_score = adapter_logits.softmax(dim=-1).max().item()

            adapter_logits_list.append(adapter_logits[:, -1, :])  # [1, V]
            draft_token_ids.append(pred_tok.item())
            global_tokens[:, end_index] = pred_tok.item()
            end_index += 1

            if pred_tok.item() in eos_set:
                round_eos_in_draft = True
                break
            if pred_score < args.threshold:
                break

        if not adapter_logits_list:
            break

        # ---- verify ----
        with torch.no_grad():
            _, verify_normed = base_model.forward_draft_or_large_model(
                in_features_large=exited_hidden,
            )
            verify_logits_full = head_model(verify_normed).float()  # [1, T, V]
            verify_ids = torch.argmax(verify_logits_full, dim=-1)   # [1, T]

        verify_len = verify_ids.shape[1]
        draft_len = len(draft_token_ids)
        usable_len = min(draft_len, verify_len)

        # find first mismatch (or EOS in verify)
        train_len = usable_len
        for k in range(usable_len):
            v = verify_ids[0, k].item()
            d = draft_token_ids[k]
            if v in eos_set or v != d:
                train_len = k + 1
                break

        # ---- loss ----
        adapter_logits_t = torch.cat(adapter_logits_list[:train_len], dim=0)   # [train_len, V]
        labels = verify_ids[0, :train_len].detach()
        verify_logits_t = verify_logits_full[0, :train_len].detach()           # [train_len, V]

        weights = make_prefix_weights(
            train_len, args.prefix_weighting, args.prefix_decay,
            device=adapter_logits_t.device, dtype=torch.float32,
        )
        loss_ce = weighted_ce(adapter_logits_t, labels, weights)
        if args.distill:
            loss_kl = weighted_kl(adapter_logits_t, verify_logits_t, weights)
            loss_round = loss_ce + args.distill_weight * loss_kl
        else:
            loss_kl = None
            loss_round = loss_ce

        # accumulate or backprop now
        if args.per_round_backward:
            (loss_round / max(1, train_len)).backward()
            torch.nn.utils.clip_grad_value_(model.adapter_model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            term = loss_round * train_len
            accum_loss = term if accum_loss is None else accum_loss + term

        accum_tokens += train_len
        round_loss_history.append((train_len, float(loss_ce.detach()),
                                   float(loss_kl.detach()) if loss_kl is not None else None))

        # ---- accept count + cache trim ----
        with torch.no_grad():
            pred_argmax = adapter_logits_t.argmax(dim=-1)
            correct += (pred_argmax == labels).sum().item()
            total_examined += train_len

        mismatch_pos = None
        for i in range(train_len):
            v = verify_ids[0, i].item()
            d = draft_token_ids[i]
            global_tokens[0, start_index + 1 + i] = v
            if v in eos_set:
                mismatch_pos = i
                start_index = start_index + 1 + i
                break
            if v != d:
                mismatch_pos = i
                start_index = start_index + 1 + i
                break

        if mismatch_pos is None:
            start_index = start_index + train_len

        accept_len = start_index - start_copy
        accept_lengths.append(accept_len)

        base_model.trim_draft_layers_cache(start_index)
        base_model.trim_verify_layers_cache(start_index)
        if adapter_past_key_values and adapter_past_key_values[0][0].shape[2] > start_index:
            adapter_past_key_values = [
                (k[:, :, :start_index, :].detach(),
                 v[:, :, :start_index, :].detach())
                for k, v in adapter_past_key_values
            ]
        if base_model.past_key_values is not None:
            base_model.past_key_values._seen_tokens = start_index

        if debug and rounds_done <= args.debug_steps:
            cmp_strs = []
            for j in range(min(train_len, 4)):
                d = draft_token_ids[j]; v = labels[j].item()
                ok = "=" if d == v else "X"
                cmp_strs.append(f"{ok}d{tokenizer.decode([d])!r}/v{tokenizer.decode([v])!r}")
            print(f"  round {rounds_done:>3}  draft_len={draft_len}  verify_len={verify_len}  "
                  f"train_len={train_len}  accept={accept_len}  "
                  f"loss={float(loss_round):.3f}  cmp={cmp_strs}")

        # stop conditions
        if mismatch_pos is not None:
            v_at_mis = verify_ids[0, mismatch_pos].item()
            if v_at_mis in eos_set:
                break
        else:
            # full accept; check if any drafted EOS was the last accepted
            if round_eos_in_draft and start_index >= ctx_len + args.max_new_tokens - 1:
                break
            if global_tokens[0, start_index].item() in eos_set:
                break

    if accum_tokens == 0 or (accum_loss is None and not args.per_round_backward):
        return None

    if not args.per_round_backward:
        loss = accum_loss / accum_tokens
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.adapter_model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    avg_accept = sum(accept_lengths) / max(len(accept_lengths), 1)
    acc = correct / max(total_examined, 1)
    avg_loss = (sum(L for _, L, _ in round_loss_history) / max(len(round_loss_history), 1))

    if debug:
        print(f"  TURN done: rounds={rounds_done} avg_accept={avg_accept:.2f} "
              f"acc={acc:.3f} loss={avg_loss:.4f} tokens={accum_tokens}")
        print(f"  prefix-weighting={args.prefix_weighting} decay={args.prefix_decay} "
              f"distill={args.distill}")
        gen_ids = global_tokens[:, ctx_len:start_index + 1]
        print(f"  generated: {tokenizer.batch_decode(gen_ids, skip_special_tokens=True)}")
        print("==========\n")

    return {"loss": avg_loss, "tokens": accum_tokens, "acc": acc, "avg_accept": avg_accept}


# ---------- main ----------

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    print(args)

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = KangarooQwenModel(
        base_model_path=args.model_path,
        adapter_model_path=args.adapter_path,
        early_exit_layer=args.exit_layer,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    freeze_base_train_adapter(model)

    optimizer = torch.optim.AdamW(model.adapter_model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    data = load_data(args.data_path)
    if args.max_samples is not None:
        data = data[:args.max_samples]

    total_steps = len(data) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max(total_steps, 1),
    )

    global_step = 0
    for epoch in range(args.num_epochs):
        print(f"\n=== Epoch {epoch} ===")
        random.shuffle(data)

        ep_loss = ep_acc = ep_accept = 0.0
        ep_tokens = 0
        seen = seen_no = seen_long = skipped_no = 0

        for example in tqdm(data):
            for context, gt_answer in iter_contexts(example):
                gt_text = str(gt_answer).strip()
                is_no_reply = (gt_text == "NO REPLY")
                if is_no_reply and random.random() > args.no_reply_keep_ratio:
                    skipped_no += 1
                    continue

                turn_max = min(
                    args.max_new_tokens,
                    estimate_answer_token_len(processor, gt_answer) + args.answer_margin_tokens,
                )

                model.reset_status()
                freeze_base_train_adapter(model)
                inputs = build_inputs(processor, context, args.device)

                # turn_max overrides args.max_new_tokens for THIS turn
                args_for_turn = copy.copy(args)
                args_for_turn.max_new_tokens = turn_max

                debug_this = args.debug and global_step < args.debug_steps
                optimizer.zero_grad(set_to_none=True)
                result = online_train_one_turn(
                    model=model, processor=processor, inputs=inputs,
                    args=args_for_turn, debug=debug_this,
                    optimizer=optimizer, scheduler=scheduler,
                )
                if result is None:
                    continue

                global_step += 1
                seen += 1
                if is_no_reply: seen_no += 1
                else: seen_long += 1

                ep_loss += result["loss"]
                ep_acc += result["acc"]
                ep_accept += result["avg_accept"]
                ep_tokens += result["tokens"]

                if global_step % 10 == 0:
                    print(f"step {global_step} | lr {optimizer.param_groups[0]['lr']:.2e} "
                          f"| loss {result['loss']:.4f} | acc {result['acc']:.3f} "
                          f"| accept {result['avg_accept']:.2f} | tokens {result['tokens']} "
                          f"| no_reply {seen_no} | long {seen_long} | skipped_no {skipped_no}")

        if seen > 0:
            print(f"Epoch {epoch} | loss {ep_loss/seen:.4f} | acc {ep_acc/seen:.3f} "
                  f"| accept {ep_accept/seen:.2f} | tokens/turn {ep_tokens/seen:.1f} "
                  f"| seen {seen} | no_reply {seen_no} | long {seen_long} "
                  f"| skipped_no {skipped_no} | keep_ratio {args.no_reply_keep_ratio}")

        if epoch % args.save_freq == 0:
            save_adapter(model, args.outdir, epoch)

    save_adapter(model, args.outdir, args.num_epochs)


if __name__ == "__main__":
    main()
