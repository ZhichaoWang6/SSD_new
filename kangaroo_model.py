"""
KangarooQwenModel: Wraps Qwen2.5-VL base model + adapter + LM head for speculative decoding.

Usage:
    model = KangarooQwenModel(
        base_model_path='Qwen/Qwen2.5-VL-3B-Instruct',
        adapter_model_path='path/to/adapter/checkpoint',
        early_exit_layer=2,
        dtype=torch.bfloat16,
    )
"""

import os
import json

import torch
import torch.nn as nn
from transformers import AutoConfig

from adapter import AdapterModel, create_adapter_config
from earlyexit_qwen import EarlyExitQwen2_5_VLForConditionalGeneration


class KangarooQwenModel(nn.Module):

    def __init__(
        self,
        base_model_path: str,
        adapter_model_path: str = None,
        early_exit_layer: int = 2,
        use_adapter_mlp: bool = None,
        dtype=torch.bfloat16,
        attn_implementation: str = 'flash_attention_2',
    ):
        super().__init__()
        self.early_exit_layer = early_exit_layer

        from model import Qwen2_5_VLForConditionalGeneration

        # Load base model
        raw_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        ).eval()

        # Wrap with early exit
        self.base_model = EarlyExitQwen2_5_VLForConditionalGeneration(
            raw_model, early_exit_layer=early_exit_layer,
        )

        # Create adapter
        adapter_config = create_adapter_config(base_model_path)

        # By default, follow the structure saved with the adapter checkpoint.
        adapter_meta = {}
        if adapter_model_path is not None:
            adapter_meta_path = os.path.join(adapter_model_path, 'adapter_config.json')
            if os.path.exists(adapter_meta_path):
                with open(adapter_meta_path, 'r', encoding='utf-8') as f:
                    adapter_meta = json.load(f)

        if "exit_layer" in adapter_meta and adapter_meta["exit_layer"] != early_exit_layer:
            raise ValueError(
                f"Adapter checkpoint exit_layer={adapter_meta['exit_layer']} does not match "
                f"inference early_exit_layer={early_exit_layer}"
            )

        if "use_mlp" in adapter_meta and use_adapter_mlp is not None and adapter_meta["use_mlp"] != use_adapter_mlp:
            raise ValueError(
                f"Adapter checkpoint use_mlp={adapter_meta['use_mlp']} does not match "
                f"inference override use_adapter_mlp={use_adapter_mlp}"
            )

        if "use_mlp" in adapter_meta:
            adapter_config.use_mlp = adapter_meta["use_mlp"]
        if use_adapter_mlp is not None:
            adapter_config.use_mlp = use_adapter_mlp

        self.adapter_model = AdapterModel(adapter_config)
        print(f"Adapter config: use_mlp={getattr(adapter_config, 'use_mlp', True)}")
        print(self.adapter_model)

        # Load adapter weights if provided
        if adapter_model_path is not None:
            adapter_ckpt = os.path.join(adapter_model_path, 'adapter_model.bin')
            if os.path.exists(adapter_ckpt):
                state_dict = torch.load(adapter_ckpt, map_location='cpu', weights_only=True)

                # Strip 'module.' prefix added by Accelerate/DDP wrapping
                cleaned = {}
                for k, v in state_dict.items():
                    new_key = k.replace('module.', '', 1) if k.startswith('module.') else k
                    cleaned[new_key] = v

                missing, unexpected = self.adapter_model.load_state_dict(cleaned, strict=False)
                if missing:
                    raise ValueError(
                        f"Adapter checkpoint is missing {len(missing)} keys for the current structure: {missing[:10]}"
                    )
                if unexpected:
                    raise ValueError(
                        f"Adapter checkpoint has {len(unexpected)} unexpected keys for the current structure: {unexpected[:10]}"
                    )
                print(f"Loaded adapter weights from {adapter_ckpt} (all {len(cleaned)} keys matched)")
            else:
                print(f"Warning: adapter checkpoint not found at {adapter_ckpt}, using random weights")

        self.adapter_model = self.adapter_model.eval().to(raw_model.device).to(dtype)

        # Reuse lm_head from base model (shared weights, no duplication)
        self.head_model = raw_model.lm_head

    @property
    def device(self):
        return self.base_model.device

    @property
    def config(self):
        return self.base_model.config

    def to(self, device):
        self.base_model.model.to(device)
        self.adapter_model.to(device)
        return self

    def forward(self):
        raise NotImplementedError("Use speculative decoding inference loop instead of direct forward")

    def reset_status(self):
        """Reset model status for a new inference session."""
        self.base_model.past_key_values = None
        if hasattr(self.base_model.model, 'reset_status'):
            self.base_model.model.reset_status()



if __name__ == "__main__":
    # Example usage
    model = KangarooQwenModel(
        base_model_path='/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt',
        adapter_model_path='/data/wangzhichao/projects/SSD/SSD2/adapter_checkpoints/epoch/adapter_epoch_19',
        early_exit_layer=2,
        dtype=torch.bfloat16,
    )
    print("KangarooQwenModel initialized successfully")
    print(model)
