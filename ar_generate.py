"""
Lossless AR baseline for speculative decoding verification.

Uses the exact same forward path as speculative decoding:
  - Same prefill cache (deep copied before speculative decode modifies it)
  - Same forward_draft_or_large_model() for each decode step
  - Same head_model for logits

如何使用：

1. 把这个文件放到和 inference_kangaroo.py 同目录下

2. 在 inference_kangaroo.py 顶部添加:
   from ar_baseline import autoregressive_manual_baseline

3. 在 kangaroo_speculative_generate() 函数中，prefill 结束后（first_token 生成后），
   加入以下代码来保存 prefill 状态的深拷贝：

   ---- 在 prefill 结束后、draft-verify loop 开始前插入 ----

   import copy
   prefill_cache_copy = copy.deepcopy(base_model.past_key_values)
   prefill_rope_deltas_copy = base_model.model.rope_deltas.clone() if base_model.model.rope_deltas is not None else None

   ---- 插入结束 ----

4. 在 kangaroo_speculative_generate() 函数的 return 语句之前，
   把 prefill_cache_copy 和 prefill_rope_deltas_copy 传出去（加到 stats 里或单独返回）

5. 在 inference_ego.py 的 _encode_query() 中，用 autoregressive_manual_baseline() 
   替换 model.generate() 调用
"""

import time
import copy
import torch


@torch.no_grad()
def autoregressive_manual_baseline(
    model,           # KangarooQwenModel
    inputs,          # 原始 inputs dict（含 input_ids, pixel_values 等）
    processor,       # tokenizer/processor
    max_new_tokens: int = 512,
    early_exit_layer: int = 2,
):
    """
    使用和 speculative decoding 完全相同的 forward 路径做 AR 生成。
    
    流程：
    1. Prefill: base_model.model(**forward_kwargs) — 和 speculative decoding 完全相同
    2. Decode: 逐 token 用 forward_draft_or_large_model 跑 draft layers + verify layers
    3. Logits: 用 head_model 得到 logits，argmax 取 next token
    
    这保证了：如果 speculative decoding 是 lossless 的，两者输出必须逐 token 一致。
    如果输出不同，问题一定在 speculative decoding 的 verify/accept/trim 逻辑中。
    """
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_start = time.perf_counter()

    base_model = model.base_model   # EarlyExitQwen2_5_VLForConditionalGeneration
    head_model = model.head_model   # lm_head
    device = inputs['input_ids'].device

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    token_eos = tokenizer.eos_token_id
    if isinstance(token_eos, list):
        token_eos_set = set(token_eos)
    else:
        token_eos_set = {token_eos}

    input_ids = inputs['input_ids']
    batch_size, context_length = input_ids.shape
    assert batch_size == 1, "Only batch_size=1 supported"

    # ========== Prefill（和 speculative decoding 完全相同） ==========
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_prefill_start = time.perf_counter()

    forward_kwargs = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs.get('attention_mask'),
        'use_cache': True,
        'output_hidden_states': True,
        'return_dict': True,
        'past_key_values': None,
        'pixel_values': inputs.get('pixel_values'),
        'pixel_values_videos': inputs.get('pixel_values_videos'),
        'image_grid_thw': inputs.get('image_grid_thw'),
        'video_grid_thw': inputs.get('video_grid_thw'),
        'second_per_grid_ts': inputs.get('second_per_grid_ts'),
        'drop_method': 'none',
        'drop_threshold': 1.0,
        'drop_absolute': True,
    }
    forward_kwargs = {k: v for k, v in forward_kwargs.items() if v is not None}

    output = base_model.model(**forward_kwargs)
    base_model.past_key_values = output.past_key_values

    first_token_id = torch.argmax(output.logits[:, -1, :], dim=-1).item()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    prefill_time = time.perf_counter() - t_prefill_start

    generated_tokens = [first_token_id]

    if first_token_id in token_eos_set:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time = time.perf_counter() - t_start
        reply_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        stats = _build_ar_stats(1, total_time, prefill_time)
        return reply_text, base_model.past_key_values, stats

    # ========== 逐 token decode ==========
    next_token_id = first_token_id

    for step in range(max_new_tokens - 1):
        if next_token_id in token_eos_set:
            break

        in_token = torch.tensor([[next_token_id]], device=device)

        # Draft layers (0 ~ early_exit_layer-1)
        draft_hidden = base_model.forward_draft_or_large_model(
            in_tokens_small=in_token,
        )

        # Verify layers (early_exit_layer ~ end) + norm
        _, hidden_normed = base_model.forward_draft_or_large_model(
            in_features_large=draft_hidden,
        )

        # Logits → next token
        logits = head_model(hidden_normed).float()
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
        generated_tokens.append(next_token_id)

    # ========== 统计 ==========
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.perf_counter() - t_start
    num_tokens = len(generated_tokens)

    reply_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    stats = _build_ar_stats(num_tokens, total_time, prefill_time)

    return reply_text, base_model.past_key_values, stats


def _build_ar_stats(num_tokens, total_time, prefill_time):
    decode_time = total_time - prefill_time
    return {
        'total_tokens': num_tokens,
        'total_time': total_time,
        'prefill_time': prefill_time,
        'decode_time': decode_time,
        'tokens_per_second': num_tokens / total_time if total_time > 0 else 0,
        'decode_tokens_per_second': num_tokens / decode_time if decode_time > 0 else 0,
    }