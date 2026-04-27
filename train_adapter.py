"""
Train the Kangaroo adapter for Qwen2.5-VL self-speculative decoding.
"""

import argparse
import json
import os
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
    parser = argparse.ArgumentParser(description="Train Kangaroo adapter for Qwen2.5-VL")
    parser.add_argument("--basepath", type=str, default="/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt")
    parser.add_argument("--datadir", type=str, required=True, default="/data/wangzhichao/projects/SSD_RE/datasets/training_data/20_no_reply")
    parser.add_argument("--outdir", type=str, required=True, default="/data/wangzhichao/projects/SSD_full_history/adapter_checkpoints/20_no_reply")
    parser.add_argument("--exit_layer", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_warmup_steps", type=int, default=2000)
    parser.add_argument("--total_steps", type=int, default=0,
                        help="Total optimizer steps. Set <=0 to auto-compute from dataloader and epochs.")
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--disable_adapter_mlp", action="store_true",
                        help="Disable the adapter MLP block and keep an attention-only adapter.")
    parser.add_argument("--resume_adapter", type=str, default=None,
                        help="Path to pretrained adapter_model.bin to continue training from")
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
    """Save adapter weights and config to outdir/tag/."""
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
    target_p = F.softmax(target_head, dim=2).detach()
    out_logp = F.log_softmax(out_head, dim=2)
    distill_loss = -torch.sum(torch.sum(loss_mask * (target_p * out_logp), 2)) / loss_mask.sum().clamp(min=1)
    return distill_loss


def compute_confidence_stats(prob_exit, loss_mask):
    max_conf_per_token = prob_exit.max(dim=2).values
    total_confidence = (max_conf_per_token * loss_mask).sum().item()
    total_tokens = loss_mask.sum().item()

    long_confidence = 0.0
    long_tokens = 0.0
    for sample_idx in range(loss_mask.shape[0]):
        sample_mask = loss_mask[sample_idx]
        sample_tokens = sample_mask.sum().item()
        if sample_tokens > 5:
            long_confidence += (max_conf_per_token[sample_idx] * sample_mask).sum().item()
            long_tokens += sample_tokens

    return total_confidence, total_tokens, long_confidence, long_tokens


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
        traindataset,
        batch_size=args.bs,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(),
        num_workers=0,
        pin_memory=False,
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
    model, head, optimizer, train_loader = accelerator.prepare(
        model, head, optimizer, train_loader
    )

    updates_per_epoch = max(1, (len(train_loader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps)
    total_training_steps = args.total_steps if args.total_steps > 0 else updates_per_epoch * args.num_epochs
    warmup_steps = min(args.num_warmup_steps, max(total_training_steps - 1, 0))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_training_steps, 1),
    )
    scheduler = accelerator.prepare(scheduler)

    if accelerator.is_main_process:
        effective_bs = args.bs * args.gradient_accumulation_steps * accelerator.num_processes
        print(f"Effective batch size: {effective_bs}")
        print(f"Updates per epoch: {updates_per_epoch}")
        print(f"Total optimizer steps: {total_training_steps}")
        print(f"Warmup steps: {warmup_steps}")

    if args.start_epoch > 0 and not args.resume_adapter:
        state_dir = os.path.join(args.outdir, "state", f"state_{args.start_epoch - 1}")
        if os.path.exists(state_dir):
            accelerator.load_state(state_dir)
            print(f"Resumed from {state_dir}")

    for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
        print(f"=== Epoch {epoch} ===")
        epoch_loss = 0.0
        epoch_accept = 0.0
        num_batches = 0
        nan_detected = False
        correct = 0
        total = 0
        confidence_sum = 0.0
        long_confidence_sum = 0.0
        long_total = 0.0
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, data in enumerate(tqdm(train_loader)):
            with accelerator.accumulate(model):
                predict = model(
                    inputs_embeds=data["hidden_states_early"],
                    attention_mask=data["attention_mask"],
                )

                with torch.no_grad():
                    target_head = head(data["target"].float())

                out_head = head(predict.float())
                prob_exit = F.softmax(out_head, dim=2)
                prob_last = F.softmax(target_head, dim=2)
                prob_acc_per_token = torch.min(prob_last, prob_exit).sum(dim=2)

                loss_mask = data["loss_mask"][:, :, None]
                loss = compute_distill_loss(out_head=out_head, target_head=target_head, loss_mask=loss_mask)
                prob_acc = torch.sum(data["loss_mask"] * prob_acc_per_token) / data["loss_mask"].sum().clamp(min=1)

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

                if accelerator.is_main_process and batch_idx % args.log_steps == 0:
                    long_conf_avg = long_confidence_sum / max(long_total, 1.0)
                    print(
                        f"\nStep: {batch_idx}\tLR: {optimizer.optimizer.param_groups[0]['lr']:.6f}"
                        f"\tAccept: {prob_acc.item():.4f}\tLoss: {loss.item():.4f}"
                        f"\tLong_conf: {long_conf_avg:.4f}"
                    )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_value_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                _, predicted = torch.max(out_head, 2)
                _, target = torch.max(target_head, 2)
                cc = ((predicted == target).float() * data["loss_mask"]).sum().item()
                ct = data["loss_mask"].sum().item()
                batch_confidence_sum, _, batch_long_confidence_sum, batch_long_total = compute_confidence_stats(
                    prob_exit=prob_exit,
                    loss_mask=data["loss_mask"],
                )
                total += ct
                correct += cc
                confidence_sum += batch_confidence_sum
                long_confidence_sum += batch_long_confidence_sum
                long_total += batch_long_total

            if accelerator.is_main_process and writer is not None and ct != 0:
                global_step = batch_idx + len(train_loader) * epoch
                writer.add_scalar("train/lr", optimizer.optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/prob_accept", prob_acc.item(), global_step)
                writer.add_scalar("train/accuracy", cc / ct, global_step)
                writer.add_scalar("train/confidence", batch_confidence_sum / max(ct, 1), global_step)

            epoch_loss += loss.item()
            epoch_accept += prob_acc.item()
            num_batches += 1

        correct_t = torch.tensor(correct, dtype=torch.float32, device=accelerator.device)
        total_t = torch.tensor(total, dtype=torch.float32, device=accelerator.device)
        correct_t, total_t = accelerator.gather_for_metrics((correct_t, total_t))
        correct_val = correct_t.sum().item()
        total_val = total_t.sum().item()

        epoch_loss /= max(num_batches, 1)
        epoch_accept /= max(num_batches, 1)
        epoch_acc = correct_val / max(total_val, 1)
        epoch_confidence = confidence_sum / max(total_val, 1.0)
        epoch_long_confidence = long_confidence_sum / max(long_total, 1.0)

        if accelerator.is_main_process:
            print(
                f"Epoch [{epoch + 1}/{args.num_epochs}]"
                f"  Loss: {epoch_loss:.4f}"
                f"  Acc: {100 * epoch_acc:.2f}%"
                f"  Accept: {epoch_accept:.4f}"
                f"  Confidence: {epoch_confidence:.4f}"
                f"  Long_conf: {epoch_long_confidence:.4f}"
            )
            if nan_detected:
                print("  Some NaN batches were skipped")

            epoch_tag = (
                f"epochs/"
                f"epoch{epoch:03d}"
                f"_acc{epoch_acc:.4f}"
                f"_accept{epoch_accept:.4f}"
                f"_loss{epoch_loss:.4f}"
            )
            save_adapter(model, adapter_config, args, accelerator, epoch_tag)

            if writer is not None:
                writer.add_scalar("epoch/loss", epoch_loss, epoch)
                writer.add_scalar("epoch/accuracy", epoch_acc, epoch)
                writer.add_scalar("epoch/accept", epoch_accept, epoch)
                writer.add_scalar("epoch/confidence", epoch_confidence, epoch)
                writer.add_scalar("epoch/long_reply_confidence", epoch_long_confidence, epoch)

        if epoch % args.save_freq == 0 or epoch == args.start_epoch + args.num_epochs - 1:
            accelerator.save_state(output_dir=os.path.join(args.outdir, "state", f"state_{epoch}"))
            if accelerator.is_main_process:
                print(f"  Saved full state for epoch {epoch}")

    if accelerator.is_main_process and writer is not None:
        writer.close()

    print("Training complete!")


if __name__ == "__main__":
    main()
