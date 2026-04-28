"""
Train the Kangaroo adapter with multi-step ROLLOUT to mitigate the
train/inference exposure bias.

Standard training: adapter is fed teacher-forced early_hidden states for
every position in one shot. Adapter learns to map "real base-model early
hidden" -> "real base-model final hidden". At inference the adapter eats
its OWN previous output (cascaded), so the input distribution drifts.

Rollout training: at each step we feed the adapter its OWN previous output
as the next-step input, building a small autoregressive chain inside the
training loop. Loss is still distillation against the teacher-forced
target hidden states, but the inputs are no longer all teacher-forced --
so the adapter sees the kind of inputs it'll actually see at inference.

Two stabilizers:
  1. warmup epochs: first --rollout_warmup_epochs epochs do plain
     teacher-forced training (== train_adapter.py's behavior). Adapter
     reaches a usable initialisation before we feed it its own outputs.
  2. scheduled sampling: during rollout epochs, each rollout step uses
     adapter output with probability p, teacher-forced early_hidden with
     1-p. p ramps linearly from --scheduled_sampling_p_start to
     --scheduled_sampling_p_end across rollout epochs.

Example training schedule:
  --num_epochs 30 --rollout_warmup_epochs 10 \
  --rollout_steps 4 --scheduled_sampling_p_start 0.0 --scheduled_sampling_p_end 1.0
  -> epochs 0..9   : pure teacher-forced (warmup)
     epoch 10      : rollout, p=0.0 (still 100% teacher-forced)
     epoch 19      : rollout, p~=0.45
     epoch 29      : rollout, p=1.0 (100% adapter-output cascading)
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, get_linear_schedule_with_warmup

from adapter import AdapterModel, create_adapter_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train Kangaroo adapter with rollout")
    parser.add_argument("--basepath", type=str, default="/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt")
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--exit_layer", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--num_warmup_steps", type=int, default=2000)
    parser.add_argument("--total_steps", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--disable_adapter_mlp", action="store_true")
    parser.add_argument("--resume_adapter", type=str, default=None,
                        help="Path to pretrained adapter_model.bin to continue from")

    # ---- rollout-specific ----
    parser.add_argument("--rollout_steps", type=int, default=4,
                        help="How many tokens to roll out per training sample. "
                             "0 disables rollout (== plain teacher-forced).")
    parser.add_argument("--rollout_warmup_epochs", type=int, default=10,
                        help="Use teacher-forced training for this many epochs "
                             "before switching to rollout training.")
    parser.add_argument("--scheduled_sampling_p_start", type=float, default=0.0,
                        help="At the first rollout epoch, probability of using "
                             "adapter's own output (instead of teacher-forced) "
                             "as the next-step input. 0.0 = teacher-forced.")
    parser.add_argument("--scheduled_sampling_p_end", type=float, default=1.0,
                        help="At the last training epoch, the same probability. "
                             "Ramps linearly across the rollout epochs.")
    parser.add_argument("--rollout_min_seq_len", type=int, default=8,
                        help="Skip rollout for samples with fewer than this many "
                             "loss-mask tokens (pure teacher-forced fallback). "
                             "NO REPLY (3 tokens) shouldn't trigger rollout.")
    return parser.parse_args()


def list_files(path):
    datapath = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".ckpt"):
                datapath.append(os.path.join(root, file))
    return sorted(datapath)


class AdapterDataset(Dataset):
    def __init__(self, datapath, exit_layer):
        self.data = datapath
        self.exit_layer = exit_layer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index], map_location="cpu", weights_only=False)

        hidden_state = data["hidden_state"][None, :]
        loss_mask = data["loss_mask"]
        hidden_state_early = data[f"hidden_state_layer{self.exit_layer}"][None, :]

        orig_len = loss_mask.shape[0]
        loss_mask_shifted = torch.zeros(orig_len, dtype=torch.float32)
        if orig_len > 1:
            loss_mask_shifted[:orig_len - 1] = loss_mask[1:orig_len].float()

        length = hidden_state.shape[1]
        attention_mask = [1] * length

        return {
            "attention_mask": attention_mask,
            "loss_mask": loss_mask_shifted.tolist(),
            "target": hidden_state,
            "hidden_state_big": hidden_state,
            "hidden_state_early": hidden_state_early,
        }


class DataCollatorWithPadding:
    @staticmethod
    def paddingtensor(intensors, target_len):
        batch, cur_len, hidden = intensors.shape
        padding_tensor = torch.zeros(batch, target_len - cur_len, hidden, dtype=intensors.dtype)
        return torch.cat((intensors, padding_tensor), dim=1)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["hidden_state_big"].shape[1] for item in features)
        return {
            "hidden_states": torch.cat([self.paddingtensor(item["hidden_state_big"], max_length) for item in features]),
            "hidden_states_early": torch.cat([self.paddingtensor(item["hidden_state_early"], max_length) for item in features]),
            "target": torch.cat([self.paddingtensor(item["target"], max_length) for item in features]),
            "loss_mask": torch.tensor([item["loss_mask"] + [0] * (max_length - len(item["loss_mask"])) for item in features]),
            "attention_mask": torch.tensor([item["attention_mask"] + [0] * (max_length - len(item["attention_mask"])) for item in features]),
        }


def save_adapter(model, adapter_config, args, accelerator, tag):
    unwrapped_model = accelerator.unwrap_model(model)
    save_dir = os.path.join(args.outdir, tag)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, "adapter_model.bin"))
    adapter_config_dict = {
        "hidden_size": adapter_config.hidden_size,
        "num_attention_heads": adapter_config.num_attention_heads,
        "num_key_value_heads": adapter_config.num_key_value_heads,
        "intermediate_size": adapter_config.intermediate_size,
        "num_hidden_layers": adapter_config.num_hidden_layers,
        "rms_norm_eps": adapter_config.rms_norm_eps,
        "vocab_size": adapter_config.vocab_size,
        "max_position_embeddings": adapter_config.max_position_embeddings,
        "exit_layer": args.exit_layer,
        "use_mlp": getattr(adapter_config, "use_mlp", True),
    }
    with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config_dict, f, indent=2)
    print(f"  Saved [{tag}] to {save_dir}")


def compute_distill_loss(out_head, target_head, loss_mask):
    target_p = F.softmax(target_head, dim=-1).detach()
    out_logp = F.log_softmax(out_head, dim=-1)
    distill_loss = -torch.sum(torch.sum(loss_mask * (target_p * out_logp), dim=-1)) / loss_mask.sum().clamp(min=1)
    return distill_loss


# ============================================================
#                       ROLLOUT FORWARD
# ============================================================

def teacher_forced_forward(model, head, data):
    """Plain (current) training path: adapter sees teacher-forced inputs in
    one full-sequence forward."""
    predict = model(
        inputs_embeds=data["hidden_states_early"],
        attention_mask=data["attention_mask"],
    )
    with torch.no_grad():
        target_head = head(data["target"].float())
    out_head = head(predict.float())
    loss_mask = data["loss_mask"][:, :, None]
    loss = compute_distill_loss(out_head=out_head, target_head=target_head, loss_mask=loss_mask)
    return loss, out_head, target_head


def rollout_forward(model, head, data, rollout_steps, sampling_p, rollout_min_seq_len):
    """
    Rollout training forward.

    Strategy:
      1. Pick a start position s within the loss_mask=1 region.
      2. Warm up: feed early_hidden[:, :s, :] through adapter with use_cache=True
         to populate the adapter's KV cache for [0, s) without computing loss.
         (No grad on warmup either -- saves memory.)
      3. Rollout: for step in [0, rollout_steps):
           - input for step 0 is early_hidden[:, s, :] (teacher-forced first input)
           - input for step k>0 is, with prob sampling_p, the previous step's
             ADAPTER OUTPUT; with prob 1-p, the teacher-forced early_hidden at
             that position.
           - run adapter on that single position, extending KV cache.
           - record output to compute loss against target_hidden[:, s+k, :].
      4. Loss: distill at the rollout positions, masked by loss_mask.

    Note: the adapter's own KV cache stays causal because we always extend to
    the right; positions outside [s, s+k) never re-appear.

    Falls back to teacher-forced if seq is too short to roll out reasonably.
    """
    early_full = data["hidden_states_early"]    # [bs, T, H]
    target_full = data["target"]                # [bs, T, H]
    loss_mask_full = data["loss_mask"]          # [bs, T]
    bs, T, H = early_full.shape
    device = early_full.device
    dtype = early_full.dtype

    # Find the loss_mask=1 region. We start rollout inside this region so all
    # loss positions actually contribute. If the region is too short, fall
    # back to teacher-forced.
    starts = []
    for b in range(bs):
        idx = (loss_mask_full[b] > 0).nonzero(as_tuple=True)[0]
        if idx.numel() < rollout_min_seq_len:
            starts.append(None)
            continue
        # restrict so [s, s + rollout_steps) is within idx.min()..idx.max()
        s_min = int(idx.min().item())
        s_max = int(idx.max().item()) - rollout_steps + 1
        s_max = max(s_min, s_max)
        starts.append(random.randint(s_min, s_max))
    if any(s is None for s in starts):
        # at least one batch sample is too short; fall back fully
        return teacher_forced_forward(model, head, data) + (False,)

    # All batch samples roll out from their own start position. To keep the
    # KV cache coherent across the batch, align by max(starts) for warm-up
    # length and pad earlier batches if needed. Simplest: loop per sample.
    # For bs==1 (default for adapter training) this is the only path anyway.
    if bs != 1:
        # multi-batch rollout requires per-sample rolling; keep things simple
        # and disable for bs > 1.
        return teacher_forced_forward(model, head, data) + (False,)

    s = starts[0]

    # 1. Warm up KV cache on [0, s)
    with torch.no_grad():
        if s > 0:
            warmup_in = early_full[:, :s, :]
            _, kv_cache = model(
                inputs_embeds=warmup_in,
                attention_mask=data["attention_mask"][:, :s] if data.get("attention_mask") is not None else None,
                use_cache=True,
            ) if False else _adapter_forward_with_cache(model, warmup_in, None)
        else:
            kv_cache = None

    # 2. Rollout loop
    rollout_outputs = []
    next_input = early_full[:, s:s+1, :]   # step 0 always teacher-forced
    used_adapter_output_count = 0
    for step in range(rollout_steps):
        out, kv_cache = _adapter_forward_with_cache(model, next_input, kv_cache)
        rollout_outputs.append(out)

        if step + 1 < rollout_steps:
            pos = s + step + 1
            use_adapter_output = (random.random() < sampling_p)
            if use_adapter_output:
                next_input = out.detach() if False else out   # keep grad through cascade
                used_adapter_output_count += 1
            else:
                next_input = early_full[:, pos:pos+1, :]

    rollout_seq = torch.cat(rollout_outputs, dim=1)            # [bs, k, H]
    target_seq = target_full[:, s:s + rollout_steps, :]        # [bs, k, H]
    mask_seq = loss_mask_full[:, s:s + rollout_steps]          # [bs, k]

    out_head = head(rollout_seq.float())
    with torch.no_grad():
        target_head = head(target_seq.float())
    loss = compute_distill_loss(
        out_head=out_head,
        target_head=target_head,
        loss_mask=mask_seq[:, :, None],
    )
    return loss, out_head, target_head, True


def _adapter_forward_with_cache(model, inputs_embeds, past_key_values):
    """Wrapper that calls the (possibly accelerator-wrapped) adapter with
    use_cache=True. Returns (hidden_out, new_kv_cache)."""
    # Accelerator-wrapped model: model is DistributedDataParallel/Module wrapper
    # forward_early_stop is on the unwrapped AdapterModel, but DDP forwards by
    # default through forward(). AdapterModel.forward delegates to
    # forward_early_stop with use_cache argument.
    out = model(
        inputs_embeds=inputs_embeds,
        use_cache=True,
        past_key_values=past_key_values,
    )
    if isinstance(out, tuple):
        return out  # (hidden, kv_cache)
    return out, None


# ============================================================
#                            MAIN
# ============================================================

def main():
    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(0)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.outdir, "tensorboard")) if accelerator.is_main_process else None
    except ImportError:
        writer = None

    base_config = AutoConfig.from_pretrained(args.basepath)
    head = nn.Linear(base_config.hidden_size, base_config.vocab_size, bias=False)

    # ---- load lm_head weights (same as train_adapter.py) ----
    try:
        from safetensors import safe_open
        index_path = os.path.join(args.basepath, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        with safe_open(os.path.join(args.basepath, head_path), framework="pt", device="cpu") as f:
            tensor_slice = f.get_slice("lm_head.weight")
            _, hidden_dim = tensor_slice.get_shape()
            tensor = tensor_slice[:, :hidden_dim].float()
    except Exception:
        try:
            index_path = os.path.join(args.basepath, "pytorch_model.bin.index.json")
            with open(index_path, "r") as f:
                index_json = json.loads(f.read())
                head_path = index_json["weight_map"]["lm_head.weight"]
            weights = torch.load(os.path.join(args.basepath, head_path), map_location="cpu")
            tensor = weights["lm_head.weight"].float()
        except Exception:
            model_path = os.path.join(args.basepath, "model.safetensors")
            if os.path.exists(model_path):
                from safetensors import safe_open
                with safe_open(model_path, framework="pt", device="cpu") as f:
                    tensor = f.get_tensor("lm_head.weight").float()
            else:
                raise RuntimeError(f"Cannot find lm_head weights in {args.basepath}")
    head.weight.data = tensor
    head.eval()
    for param in head.parameters():
        param.requires_grad = False

    datapath = list_files(args.datadir)
    if not datapath:
        raise ValueError(f"No .ckpt files found in {args.datadir}")
    print(f"Training: {len(datapath)} samples")

    traindataset = AdapterDataset(datapath, args.exit_layer)
    train_loader = DataLoader(
        traindataset, batch_size=args.bs, shuffle=True,
        collate_fn=DataCollatorWithPadding(), num_workers=0, pin_memory=False,
    )
    if accelerator.is_main_process:
        os.makedirs(args.outdir, exist_ok=True)

    adapter_config = create_adapter_config(args.basepath)
    adapter_config.use_mlp = not args.disable_adapter_mlp
    model = AdapterModel(adapter_config)
    if accelerator.is_main_process:
        print(f"Adapter config: use_mlp={getattr(adapter_config, 'use_mlp', True)}")
        print(model)

    if args.resume_adapter:
        state_dict = torch.load(args.resume_adapter, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained adapter from {args.resume_adapter}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    model, head, optimizer, train_loader = accelerator.prepare(model, head, optimizer, train_loader)

    updates_per_epoch = max(1, (len(train_loader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps)
    total_training_steps = args.total_steps if args.total_steps > 0 else updates_per_epoch * args.num_epochs
    warmup_steps = min(args.num_warmup_steps, max(total_training_steps - 1, 0))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(total_training_steps, 1),
    )
    scheduler = accelerator.prepare(scheduler)

    if accelerator.is_main_process:
        print(f"Effective batch size : {args.bs * args.gradient_accumulation_steps * accelerator.num_processes}")
        print(f"Updates per epoch    : {updates_per_epoch}")
        print(f"Total optimizer steps: {total_training_steps}")
        print(f"Rollout config       : steps={args.rollout_steps}, "
              f"warmup_epochs={args.rollout_warmup_epochs}, "
              f"sampling_p {args.scheduled_sampling_p_start} -> {args.scheduled_sampling_p_end}")

    if args.start_epoch > 0 and not args.resume_adapter:
        state_dir = os.path.join(args.outdir, "state", f"state_{args.start_epoch - 1}")
        if os.path.exists(state_dir):
            accelerator.load_state(state_dir)
            print(f"Resumed from {state_dir}")

    rollout_total_epochs = max(1, args.num_epochs - args.rollout_warmup_epochs)

    for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
        # Decide training mode for this epoch.
        in_warmup = epoch < args.rollout_warmup_epochs or args.rollout_steps <= 0
        if in_warmup:
            mode_str = "TF (teacher-forced)"
            sampling_p = 0.0
        else:
            ep_in_rollout = epoch - args.rollout_warmup_epochs
            frac = ep_in_rollout / max(1, rollout_total_epochs - 1)
            sampling_p = args.scheduled_sampling_p_start + \
                         frac * (args.scheduled_sampling_p_end - args.scheduled_sampling_p_start)
            sampling_p = max(0.0, min(1.0, sampling_p))
            mode_str = f"ROLLOUT k={args.rollout_steps} p={sampling_p:.2f}"

        print(f"=== Epoch {epoch}  [{mode_str}] ===")
        epoch_loss = 0.0
        epoch_acc_correct = 0
        epoch_acc_total = 0
        num_batches = 0
        rollout_used_count = 0
        nan_detected = False
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, data in enumerate(tqdm(train_loader)):
            with accelerator.accumulate(model):
                if in_warmup:
                    loss, out_head, target_head = teacher_forced_forward(model, head, data)
                    used_rollout = False
                    loss_mask_used = data["loss_mask"][:, :, None]
                else:
                    res = rollout_forward(
                        model=model, head=head, data=data,
                        rollout_steps=args.rollout_steps,
                        sampling_p=sampling_p,
                        rollout_min_seq_len=args.rollout_min_seq_len,
                    )
                    if len(res) == 4:
                        loss, out_head, target_head, used_rollout = res
                        if used_rollout:
                            rollout_used_count += 1
                            # for accuracy computation we need the matching mask slice
                            # (use full ones since all rollout positions are valid)
                            loss_mask_used = torch.ones_like(out_head[..., :1])
                        else:
                            loss_mask_used = data["loss_mask"][:, :, None]
                    else:  # safety
                        loss, out_head, target_head = res
                        used_rollout = False
                        loss_mask_used = data["loss_mask"][:, :, None]

                nan_flag = torch.tensor(
                    1.0 if (torch.isnan(loss) or torch.isinf(loss)) else 0.0,
                    device=accelerator.device,
                )
                nan_flag = accelerator.reduce(nan_flag, reduction="sum")
                if nan_flag.item() > 0:
                    if accelerator.is_main_process:
                        print(f"\nNaN/Inf loss at epoch {epoch}, batch {batch_idx} - skipping")
                    nan_detected = True
                    optimizer.zero_grad(set_to_none=True)
                    continue

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_value_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                _, predicted = torch.max(out_head, dim=-1)
                _, target_ids = torch.max(target_head, dim=-1)
                if loss_mask_used.shape[-1] == 1:
                    correct = ((predicted == target_ids).float() * loss_mask_used.squeeze(-1)).sum().item()
                    total = loss_mask_used.squeeze(-1).sum().item()
                else:
                    correct = (predicted == target_ids).float().sum().item()
                    total = predicted.numel()
                epoch_acc_correct += correct
                epoch_acc_total += total

            if accelerator.is_main_process and batch_idx % args.log_steps == 0:
                acc_so_far = epoch_acc_correct / max(epoch_acc_total, 1)
                print(f"\nstep={batch_idx} lr={optimizer.optimizer.param_groups[0]['lr']:.6f} "
                      f"loss={loss.item():.4f} acc={acc_so_far:.4f} mode={'rollout' if used_rollout else 'tf'}")

            epoch_loss += loss.item()
            num_batches += 1

            if accelerator.is_main_process and writer is not None:
                global_step = batch_idx + len(train_loader) * epoch
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar("train/sampling_p", sampling_p, global_step)
                writer.add_scalar("train/used_rollout", float(used_rollout), global_step)

        epoch_loss /= max(num_batches, 1)
        epoch_acc = epoch_acc_correct / max(epoch_acc_total, 1)

        if accelerator.is_main_process:
            print(f"Epoch [{epoch + 1}/{args.start_epoch + args.num_epochs}] mode={mode_str} "
                  f"loss={epoch_loss:.4f} acc={100 * epoch_acc:.2f}% "
                  f"rollout_batches={rollout_used_count}/{num_batches}")
            if nan_detected:
                print("  Some NaN batches were skipped")

            tag = f"epochs/epoch{epoch:03d}_acc{epoch_acc:.4f}_loss{epoch_loss:.4f}_p{sampling_p:.2f}"
            save_adapter(model, adapter_config, args, accelerator, tag)

            if writer is not None:
                writer.add_scalar("epoch/loss", epoch_loss, epoch)
                writer.add_scalar("epoch/accuracy", epoch_acc, epoch)
                writer.add_scalar("epoch/sampling_p", sampling_p, epoch)

        if epoch % args.save_freq == 0 or epoch == args.start_epoch + args.num_epochs - 1:
            accelerator.save_state(output_dir=os.path.join(args.outdir, "state", f"state_{epoch}"))

    if accelerator.is_main_process and writer is not None:
        writer.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
