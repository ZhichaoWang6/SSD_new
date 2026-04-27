"""
Offline teacher-forced evaluation for the Kangaroo adapter.

This answers a narrow question:
  Given saved training hidden states, can the adapter map early hidden states
  to the full model's greedy top-1 logits under teacher forcing?

If this score is low, changing speculative threshold cannot fix the root cause.
If this score is high but streaming speculative accuracy is low, the problem is
mostly train/inference distribution shift from autoregressive draft rollout.
"""

import argparse
import json
import os
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig

from adapter import AdapterModel, create_adapter_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate adapter under teacher forcing")
    parser.add_argument("--basepath", type=str, required=True)
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Directory containing adapter_model.bin and adapter_config.json")
    parser.add_argument("--exit_layer", type=int, default=None,
                        help="Override exit layer. Defaults to adapter_config.json when present.")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit number of ckpt files. <=0 means all.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--disable_adapter_mlp", action="store_true",
                        help="Force an attention-only adapter structure for checkpoints trained without MLP.")
    return parser.parse_args()


def list_files(path):
    datapath = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".ckpt"):
                datapath.append(os.path.join(root, file))
    return sorted(datapath)


def load_lm_head(basepath, device):
    base_config = AutoConfig.from_pretrained(basepath)
    head = nn.Linear(base_config.hidden_size, base_config.vocab_size, bias=False)

    try:
        from safetensors import safe_open

        index_path = os.path.join(basepath, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        with safe_open(os.path.join(basepath, head_path), framework="pt", device="cpu") as f:
            tensor = f.get_tensor("lm_head.weight").float()
    except Exception:
        try:
            index_path = os.path.join(basepath, "pytorch_model.bin.index.json")
            with open(index_path, "r") as f:
                index_json = json.loads(f.read())
                head_path = index_json["weight_map"]["lm_head.weight"]
            weights = torch.load(os.path.join(basepath, head_path), map_location="cpu")
            tensor = weights["lm_head.weight"].float()
        except Exception:
            model_path = os.path.join(basepath, "model.safetensors")
            if not os.path.exists(model_path):
                raise RuntimeError(f"Cannot find lm_head weights in {basepath}")
            from safetensors import safe_open
            with safe_open(model_path, framework="pt", device="cpu") as f:
                tensor = f.get_tensor("lm_head.weight").float()

    head.weight.data = tensor
    head.eval()
    for param in head.parameters():
        param.requires_grad = False
    return head.to(device)


def shifted_loss_mask(loss_mask):
    orig_len = loss_mask.shape[0]
    shifted = torch.zeros(orig_len, dtype=torch.float32)
    if orig_len > 1:
        shifted[:orig_len - 1] = loss_mask[1:orig_len].float()
    return shifted


def update_stats(stats: Dict[str, float], prefix: str, correct, total, conf_sum, rank_sum, topk_correct, margin_sum):
    stats[f"{prefix}_correct"] += correct
    stats[f"{prefix}_total"] += total
    stats[f"{prefix}_conf_sum"] += conf_sum
    stats[f"{prefix}_rank_sum"] += rank_sum
    stats[f"{prefix}_topk_correct"] += topk_correct
    stats[f"{prefix}_margin_sum"] += margin_sum


def main():
    args = parse_args()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = torch.device(args.device)

    adapter_meta = {}
    meta_path = os.path.join(args.adapter_path, "adapter_config.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            adapter_meta = json.load(f)

    exit_layer = args.exit_layer
    if exit_layer is None:
        exit_layer = int(adapter_meta.get("exit_layer", 2))

    adapter_config = create_adapter_config(args.basepath)
    if "use_mlp" in adapter_meta:
        adapter_config.use_mlp = adapter_meta["use_mlp"]
    if args.disable_adapter_mlp:
        adapter_config.use_mlp = False

    adapter = AdapterModel(adapter_config)
    state_dict = torch.load(os.path.join(args.adapter_path, "adapter_model.bin"), map_location="cpu")
    cleaned = {
        (k.replace("module.", "", 1) if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    missing, unexpected = adapter.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Adapter load mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")
    adapter = adapter.eval().to(device).to(dtype)

    head = load_lm_head(args.basepath, device).to(dtype)

    files = list_files(args.datadir)
    if args.max_samples > 0:
        files = files[:args.max_samples]
    if not files:
        raise ValueError(f"No .ckpt files found in {args.datadir}")

    stats = {
        "all_correct": 0.0, "all_total": 0.0, "all_conf_sum": 0.0,
        "all_rank_sum": 0.0, "all_topk_correct": 0.0, "all_margin_sum": 0.0,
        "short_correct": 0.0, "short_total": 0.0, "short_conf_sum": 0.0,
        "short_rank_sum": 0.0, "short_topk_correct": 0.0, "short_margin_sum": 0.0,
        "long_correct": 0.0, "long_total": 0.0, "long_conf_sum": 0.0,
        "long_rank_sum": 0.0, "long_topk_correct": 0.0, "long_margin_sum": 0.0,
    }

    print(f"Evaluating {len(files)} samples")
    print(f"adapter_path={args.adapter_path}")
    print(f"exit_layer={exit_layer}, use_mlp={getattr(adapter_config, 'use_mlp', True)}")

    with torch.no_grad():
        for path in tqdm(files):
            data = torch.load(path, map_location="cpu", weights_only=False)
            early_key = f"hidden_state_layer{exit_layer}"
            if early_key not in data:
                continue

            loss_mask = shifted_loss_mask(data["loss_mask"]).to(device)
            token_total = int(loss_mask.sum().item())
            if token_total == 0:
                continue

            early_hidden = data[early_key].unsqueeze(0).to(device=device, dtype=dtype)
            target_hidden = data["hidden_state"].unsqueeze(0).to(device=device, dtype=dtype)

            pred_hidden = adapter(inputs_embeds=early_hidden)
            out_logits = head(pred_hidden).float()[0]
            target_logits = head(target_hidden).float()[0]

            mask_bool = loss_mask.bool()
            out_sel = out_logits[mask_bool]
            target_sel = target_logits[mask_bool]
            teacher_ids = target_sel.argmax(dim=-1)

            pred_ids = out_sel.argmax(dim=-1)
            probs = F.softmax(out_sel, dim=-1)
            pred_conf = probs.max(dim=-1).values

            teacher_logits = out_sel.gather(1, teacher_ids[:, None]).squeeze(1)
            higher_than_teacher = (out_sel > teacher_logits[:, None]).sum(dim=1)
            teacher_rank = higher_than_teacher + 1

            k = min(args.topk, out_sel.shape[-1])
            topk_ids = out_sel.topk(k, dim=-1).indices
            topk_hit = (topk_ids == teacher_ids[:, None]).any(dim=1)

            sorted_vals = out_sel.topk(2, dim=-1).values
            margin = sorted_vals[:, 0] - sorted_vals[:, 1]

            correct = (pred_ids == teacher_ids).sum().item()
            total = teacher_ids.numel()
            conf_sum = pred_conf.sum().item()
            rank_sum = teacher_rank.float().sum().item()
            topk_correct = topk_hit.sum().item()
            margin_sum = margin.sum().item()

            update_stats(stats, "all", correct, total, conf_sum, rank_sum, topk_correct, margin_sum)
            bucket = "long" if token_total > 5 else "short"
            update_stats(stats, bucket, correct, total, conf_sum, rank_sum, topk_correct, margin_sum)

    def print_bucket(name):
        total = stats[f"{name}_total"]
        if total <= 0:
            print(f"{name}: no tokens")
            return
        print(
            f"{name}: "
            f"top1={stats[f'{name}_correct'] / total:.3%} "
            f"top{args.topk}={stats[f'{name}_topk_correct'] / total:.3%} "
            f"avg_conf={stats[f'{name}_conf_sum'] / total:.4f} "
            f"avg_teacher_rank={stats[f'{name}_rank_sum'] / total:.2f} "
            f"avg_top1_margin={stats[f'{name}_margin_sum'] / total:.4f} "
            f"tokens={int(total)}"
        )

    print("\nTeacher-forced adapter evaluation")
    print_bucket("all")
    print_bucket("short")
    print_bucket("long")


if __name__ == "__main__":
    main()
