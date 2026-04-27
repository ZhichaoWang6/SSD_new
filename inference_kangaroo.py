# """
# Speculative decoding inference for Qwen2.5-VL with Kangaroo adapter.

# This replaces model.generate() with a custom draft-verify loop:
# 1. Prefill: Run full model on all input tokens (text + visual) normally
# 2. Draft: Run early layers + adapter to generate candidate tokens
# 3. Verify: Run remaining layers to check draft tokens
# 4. Accept tokens until first mismatch (lossless for greedy decoding)

# Adapted from Kangaroo's inference_kangaroo.py for Qwen2.5-VL.
# """

# import argparse
# import copy
# import time

# import torch
# from transformers.cache_utils import DynamicCache


# def _build_stats(accept_length_list, prefill_time, draft_times, verify_times, total_time, num_new_tokens):
#     """Build comprehensive timing and acceptance statistics."""
#     decode_time = total_time - prefill_time
#     avg_accept = sum(accept_length_list) / len(accept_length_list) if accept_length_list else 0
#     avg_draft_accept = sum(max(0, a - 1) for a in accept_length_list) / len(accept_length_list) if accept_length_list else 0
#     tokens_per_second = num_new_tokens / total_time if total_time > 0 else 0
#     decode_tokens_per_second = num_new_tokens / decode_time if decode_time > 0 else 0
#     return {
#         'accept_lengths': accept_length_list,
#         'avg_accept_length': avg_accept,
#         'avg_draft_accept_length': avg_draft_accept,
#         'total_rounds': len(accept_length_list),
#         'total_tokens': num_new_tokens,
#         'total_time': total_time,
#         'prefill_time': prefill_time,
#         'decode_time': decode_time,
#         'draft_times': draft_times,
#         'verify_times': verify_times,
#         'avg_draft_time': sum(draft_times) / len(draft_times) if draft_times else 0,
#         'avg_verify_time': sum(verify_times) / len(verify_times) if verify_times else 0,
#         'tokens_per_second': tokens_per_second,
#         'decode_tokens_per_second': decode_tokens_per_second,
#     }


# @torch.no_grad()
# def kangaroo_speculative_generate(
#     model,
#     inputs,
#     processor,
#     max_new_tokens: int = 512,
#     early_exit_layer: int = 2,
#     speculative_steps: int = 6,
#     threshold: float = 0.6,
#     do_sample: bool = False,
#     past_key_values=None,
# ):
#     # =======================
#     adapter_correct = 0
#     adapter_total = 0
#     adapter_confidences = []
#     adapter_accept_probs = []
#     #============================


#     # print("going into kangaroo speculative generation...")
#     assert not do_sample, "Only greedy decoding is supported for speculative decoding"

#     if past_key_values is not None:
#         cached_len = 0
#         if len(past_key_values.key_cache) > 0 and len(past_key_values.key_cache[0]) > 0:
#             cached_len = past_key_values.key_cache[0].shape[2]
#         if cached_len > 0:
#             print(
#                 f"[kangaroo] Received past_key_values with cache_len={cached_len}. "
#                 "This function expects delta-only inputs when KV cache is reused; "
#                 "passing a full prompt with a non-empty cache will duplicate context."
#             )

#     torch.cuda.synchronize() if torch.cuda.is_available() else None
#     t_start = time.perf_counter()

#     base_model = model.base_model
#     adapter_model = model.adapter_model
#     head_model = model.head_model
#     device = inputs['input_ids'].device

#     tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
#     token_eos = tokenizer.eos_token_id
#     if isinstance(token_eos, list):
#         token_eos_set = set(token_eos)
#         token_eos = token_eos[0]
#     else:
#         token_eos_set = {token_eos}

#     input_ids = inputs['input_ids']
#     batch_size, context_length = input_ids.shape
#     # print(f"batch_size: {batch_size}, context_length: {context_length}")
#     assert batch_size == 1, "Speculative decoding only supports batch_size=1"

#     max_length = context_length + max_new_tokens

#     global_tokens = torch.full((batch_size, max_length), token_eos, dtype=torch.long, device=device)
#     global_tokens[:, :context_length] = input_ids

#     accept_length_list = []
#     start_index = context_length

#     # ========== STEP 0: Prefill ==========
#     torch.cuda.synchronize() if torch.cuda.is_available() else None
#     t_prefill_start = time.perf_counter()

#     forward_kwargs = {
#         'input_ids': inputs['input_ids'],
#         'attention_mask': inputs.get('attention_mask'),
#         'use_cache': True,
#         'output_hidden_states': True,
#         'return_dict': True,
#         'past_key_values': past_key_values,
#         'pixel_values': inputs.get('pixel_values'),
#         'pixel_values_videos': inputs.get('pixel_values_videos'),
#         'image_grid_thw': inputs.get('image_grid_thw'),
#         'video_grid_thw': inputs.get('video_grid_thw'),
#         'second_per_grid_ts': inputs.get('second_per_grid_ts'),
#         'drop_method': 'none',
#         'drop_threshold': 1.0,
#         'drop_absolute': True,
#     }
#     forward_kwargs = {k: v for k, v in forward_kwargs.items() if v is not None}

#     output = base_model.model(**forward_kwargs)
#     base_model.past_key_values = output.past_key_values

#     first_token = torch.argmax(output.logits[:, -1, :], dim=-1)
#     global_tokens[:, start_index] = first_token.item()

#     hidden_state_early = output.hidden_states[early_exit_layer]
#     _, adapter_past_key_values = adapter_model.forward_early_stop(
#         inputs_embeds=hidden_state_early,
#         use_cache=True,
#     )

#     torch.cuda.synchronize() if torch.cuda.is_available() else None
#     prefill_time = time.perf_counter() - t_prefill_start

#     draft_times = []
#     verify_times = []

#     print(f" first token :{tokenizer.decode(first_token)}")
#     if first_token.item() in token_eos_set:
#         output_ids = global_tokens[:, :start_index + 1]
#         torch.cuda.synchronize() if torch.cuda.is_available() else None
#         total_time = time.perf_counter() - t_start
#         stats = _build_stats([], prefill_time, [], [], total_time, 1)
#         return output_ids, base_model.past_key_values, stats

#     # ========== Draft-Verify Loop ==========
#     max_infer_steps = min(max_length, start_index + max_new_tokens)
#     stop = False
#     round_idx = 0

#     while start_index < max_infer_steps - 1:
#         round_idx += 1
#         start_index_copy = start_index
#         end_index = start_index + 1
#         remaining_budget = max_infer_steps - 1 - start_index
#         round_speculative_steps = min(speculative_steps, remaining_budget)

#         # ---- STEP 1: Draft ----
#         # print("=====================** STEP 1: Draft **============================")
#         torch.cuda.synchronize() if torch.cuda.is_available() else None
#         t_draft_start = time.perf_counter()
#         exited_hidden_states = None
#         draft_token_ids = []

#         for step in range(1 + round_speculative_steps):
#             in_token = global_tokens[:, end_index - 1:end_index]
#             print(f"\nDraft step {step}: in_token={in_token}, in_token_decoded={tokenizer.decode(in_token[0])}, end_index: {end_index}")

#             adapter_cache_len = adapter_past_key_values[0][0].shape[2] if adapter_past_key_values else 0
#             if adapter_cache_len < end_index - 1:
#                 hidden_state_early_last = exited_hidden_states[:, -1:, :] if exited_hidden_states is not None else None
#             else:
#                 hidden_state_early_last = None

#             hidden_state_early = base_model.forward_draft_or_large_model(
#                 in_tokens_small=in_token,
#             )

#             if step == 0:
#                 exited_hidden_states = None

#             exited_hidden_states = hidden_state_early if exited_hidden_states is None \
#                 else torch.cat([exited_hidden_states, hidden_state_early], dim=1)

#             adapter_input = hidden_state_early
#             if hidden_state_early_last is not None:
#                 adapter_input = torch.cat([hidden_state_early_last, hidden_state_early], dim=1)

#             if step == round_speculative_steps:
#                 print(f"Draft step {step} reached round speculative step limit")
#                 break
#             if step > 0 and predict_score < threshold:
#                 print(f"Draft step {step}, token {tokenizer.decode(predicted_token)}, predict_score {predict_score} < threshold {threshold}, stopping draft")
#                 break

#             hidden_state, adapter_past_key_values = adapter_model.forward_early_stop(
#                 inputs_embeds=adapter_input,
#                 past_key_values=adapter_past_key_values,
#                 use_cache=True,
#             )

#             predict_logits = head_model(hidden_state[:, -1:, :]).float()
#             predicted_token = torch.argmax(predict_logits[:, -1, :], dim=-1)

#             predict_score = predict_logits.softmax(dim=-1).max().item()
#             # =====================================
#             adapter_confidences.append(predict_score)
#             #=======================================
#             print(f"predicted_token: {predicted_token.item()}, predict_score: {predict_score}, token : {tokenizer.decode(predicted_token)}")

#             global_tokens[:, end_index] = predicted_token
#             draft_token_ids.append(predicted_token.item())

#             # draft到 eos 停止
#             if predicted_token.item() in token_eos_set:
#                 end_index += 1
#                 # print(f"Drafted token is EOS, stopping draft.")
#                 break

#             end_index += 1

#         torch.cuda.synchronize() if torch.cuda.is_available() else None
#         draft_times.append(time.perf_counter() - t_draft_start)

#         # ---- STEP 2+3: Verify and Accept ----
#         # print("======================** STEP 2+3: Verify and Accept **==============================")
#         torch.cuda.synchronize() if torch.cuda.is_available() else None
#         t_verify_start = time.perf_counter()

#         output_length = end_index - start_index

#         # base_model.past_key_values._seen_tokens = start_index
#         verify_cache_len = base_model._get_layer_cache_length(early_exit_layer)
#         # print(f"Verifying {output_length} drafted tokens...")
#         # print(f"Current global tokens: {tokenizer.batch_decode(global_tokens[:, start_index:end_index])}")

#         assert verify_cache_len == start_index, \
#             f"Verify cache mismatch: {verify_cache_len} != {start_index}"
        
#         # print(f"Sending drafted tokens to base_model for verification... {exited_hidden_states.shape}")
     
#         _, hidden_state_normed = base_model.forward_draft_or_large_model(
#             in_features_large=exited_hidden_states,
#         )
#         verify_logits = head_model(hidden_state_normed).float()
#         verify_ids = torch.argmax(verify_logits, dim=-1)[0].tolist()

#         for i, verify_id in enumerate(verify_ids):
#             # 到达边界，直接截断停止
#             write_index = start_index + 1 + i
#             if write_index >= max_length:
#                 start_index = max_length - 1
#                 stop = True
#                 break

#             is_last = (i == output_length - 1)
#             is_eos = (verify_id in token_eos_set)
#             draft_id = global_tokens[0, start_index + 1 + i].item() if i < len(draft_token_ids) else None
#             print(f"Verifying token {i}: verify_id={verify_id} ({tokenizer.decode(verify_id)}), draft_id={draft_id} ({tokenizer.decode(draft_id) if draft_id is not None else None}), is_last={is_last}, is_eos={is_eos}")
#             is_mismatch = (not is_last and draft_id is not None and verify_id != draft_id)
#             # print(f"is_mismatch: {is_mismatch}")

#             # ========================================
#             if draft_id is not None and not is_last:
#                 adapter_total += 1
#                 if verify_id == draft_id:
#                     adapter_correct += 1
#             # =========================================

#             if is_last or is_eos or is_mismatch:
#                 global_tokens[0, start_index + 1 + i] = verify_id
#                 print(f"Token {i} verification failed, accepting up to this token. is_last: {is_last}, is_eos: {is_eos}, is_mismatch: {is_mismatch}")
#                 start_index = start_index + 1 + i
#                 if is_eos:
#                     stop = True
#                 break

#         torch.cuda.synchronize() if torch.cuda.is_available() else None
#         verify_times.append(time.perf_counter() - t_verify_start)

#         accept_len = start_index - start_index_copy
#         accept_length_list.append(accept_len)
#         # print(f"Round {round_idx} accepted length: {accept_len}, total accepted length: {accept_length_list}")

#         # ---- STEP 4: Trim caches ----
#         draft_cache_len = base_model._get_layer_cache_length(0)
#         if draft_cache_len > start_index:
#             base_model.trim_draft_layers_cache(start_index)

#         verify_cache_len = base_model._get_layer_cache_length(early_exit_layer)
#         if verify_cache_len > start_index:
#             base_model.trim_verify_layers_cache(start_index)

#         if adapter_past_key_values and adapter_past_key_values[0][0].shape[2] > start_index:
#             adapter_past_key_values = [
#                 (k[:, :, :start_index, :], v[:, :, :start_index, :])
#                 for k, v in adapter_past_key_values
#             ]

#         base_model.past_key_values._seen_tokens = start_index
#         assert base_model._get_layer_cache_length(0) == start_index, \
#             f"Draft cache after trim: {base_model._get_layer_cache_length(0)} != {start_index}"
#         assert base_model._get_layer_cache_length(early_exit_layer) == start_index, \
#             f"Verify cache after trim: {base_model._get_layer_cache_length(early_exit_layer)} != {start_index}"

#         if stop:
#             break

#     # Final output
#     output_ids = global_tokens[:, :start_index + 1]
#     num_new_tokens = start_index + 1 - context_length
#     print(f"New tokens: {tokenizer.batch_decode(output_ids[:, context_length:])}")
#     print(f"Total new tokens generated: {num_new_tokens}")
#     torch.cuda.synchronize() if torch.cuda.is_available() else None
#     total_time = time.perf_counter() - t_start

#     stats = _build_stats(accept_length_list, prefill_time, draft_times, verify_times, total_time, num_new_tokens)
#  # =============================================================================================================
#     adapter_acc = adapter_correct / adapter_total if adapter_total > 0 else 0
#     avg_confidence = sum(adapter_confidences) / len(adapter_confidences) if adapter_confidences else 0
#     print(f"[Spec] avg_accept={stats['avg_accept_length']:.2f} | tokens={num_new_tokens} | context_len={context_length}")
#     print(f"[Adapter] acc={adapter_correct}/{adapter_total} ({adapter_acc:.1%}) | avg_confidence={avg_confidence:.3f} | context_len={context_length}")

#     stats['adapter_accuracy'] = adapter_acc
#     stats['adapter_correct'] = adapter_correct
#     stats['adapter_total'] = adapter_total
#     stats['adapter_avg_confidence'] = avg_confidence
#     #==============================================================================================================


#     return output_ids, base_model.past_key_values, stats


# def speculative_generate_for_streaming(
#     model,
#     inputs,
#     processor,
#     past_key_values=None,
#     max_new_tokens: int = 512,
#     early_exit_layer: int = 2,
#     speculative_steps: int = 6,
#     threshold: float = 0.6,
# ):
#     output_ids, past_key_values, stats = kangaroo_speculative_generate(
#         model=model,
#         inputs=inputs,
#         processor=processor,
#         max_new_tokens=max_new_tokens,
#         early_exit_layer=early_exit_layer,
#         speculative_steps=speculative_steps,
#         threshold=threshold,
#         do_sample=False,
#         past_key_values=past_key_values,
#     )
#     # print(f"Speculative generation completed. Stats: {stats}")

#     input_length = inputs['input_ids'].shape[1]
#     # print(f"Input length: {input_length}, Output length: {output_ids.shape[1]}, New tokens generated: {output_ids.shape[1] - input_length}")
#     new_token_ids = output_ids[:, input_length:]

#     tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
#     reply_text = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0]

#     return reply_text, past_key_values, stats































"""
Speculative decoding inference for Qwen2.5-VL with Kangaroo adapter.

This replaces model.generate() with a custom draft-verify loop:
1. Prefill: Run full model on all input tokens (text + visual) normally
2. Draft: Run early layers + adapter to generate candidate tokens
3. Verify: Run remaining layers to check draft tokens
4. Accept tokens until first mismatch (lossless for greedy decoding)

Adapted from Kangaroo's inference_kangaroo.py for Qwen2.5-VL.
"""

import argparse
import copy
import os
import time

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


DEBUG_CACHE = os.environ.get("KANGAROO_DEBUG_CACHE", "0").lower() in {"1", "true", "yes", "y"}
DEBUG_HIDDEN = os.environ.get("KANGAROO_DEBUG_HIDDEN", "0").lower() in {"1", "true", "yes", "y"}


def _debug_cache(message):
    if DEBUG_CACHE:
        print(f"[cache-debug] {message}")


def _adapter_cache_len(adapter_past_key_values):
    if adapter_past_key_values and adapter_past_key_values[0] is not None:
        return adapter_past_key_values[0][0].shape[2]
    return 0


def _debug_hidden(message):
    if DEBUG_HIDDEN:
        print(f"[hidden-debug] {message}")


def _build_stats(accept_length_list, prefill_time, draft_times, verify_times, total_time, num_new_tokens):
    """Build comprehensive timing and acceptance statistics."""
    decode_time = total_time - prefill_time
    avg_accept = sum(accept_length_list) / len(accept_length_list) if accept_length_list else 0
    avg_draft_accept = sum(max(0, a - 1) for a in accept_length_list) / len(accept_length_list) if accept_length_list else 0
    tokens_per_second = num_new_tokens / total_time if total_time > 0 else 0
    decode_tokens_per_second = num_new_tokens / decode_time if decode_time > 0 else 0
    return {
        'accept_lengths': accept_length_list,
        'avg_accept_length': avg_accept,
        'avg_draft_accept_length': avg_draft_accept,
        'total_rounds': len(accept_length_list),
        'total_tokens': num_new_tokens,
        'total_time': total_time,
        'prefill_time': prefill_time,
        'decode_time': decode_time,
        'draft_times': draft_times,
        'verify_times': verify_times,
        'avg_draft_time': sum(draft_times) / len(draft_times) if draft_times else 0,
        'avg_verify_time': sum(verify_times) / len(verify_times) if verify_times else 0,
        'tokens_per_second': tokens_per_second,
        'decode_tokens_per_second': decode_tokens_per_second,
    }


@torch.no_grad()
def kangaroo_speculative_generate(
    model,
    inputs,
    processor,
    max_new_tokens: int = 512,
    early_exit_layer: int = 2,
    speculative_steps: int = 6,
    threshold: float = 0.6,
    do_sample: bool = False,
    past_key_values=None,
):
    # =======================
    adapter_correct = 0
    adapter_total = 0
    adapter_confidences = []
    adapter_drafted_confidences = []
    adapter_accept_probs = []
    rollout_early_hidden = {}
    rollout_adapter_hidden = {}
    #============================


    # print("going into kangaroo speculative generation...")
    assert not do_sample, "Only greedy decoding is supported for speculative decoding"

    if past_key_values is not None:
        cached_len = 0
        if len(past_key_values.key_cache) > 0 and len(past_key_values.key_cache[0]) > 0:
            cached_len = past_key_values.key_cache[0].shape[2]
        if cached_len > 0:
            print(
                f"[kangaroo] Received past_key_values with cache_len={cached_len}. "
                "This function expects delta-only inputs when KV cache is reused; "
                "passing a full prompt with a non-empty cache will duplicate context."
            )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_start = time.perf_counter()

    base_model = model.base_model
    adapter_model = model.adapter_model
    head_model = model.head_model
    device = inputs['input_ids'].device

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    token_eos = tokenizer.eos_token_id
    if isinstance(token_eos, list):
        token_eos_set = set(token_eos)
        token_eos = token_eos[0]
    else:
        token_eos_set = {token_eos}

    input_ids = inputs['input_ids']
    batch_size, context_length = input_ids.shape
    # print(f"batch_size: {batch_size}, context_length: {context_length}")
    assert batch_size == 1, "Speculative decoding only supports batch_size=1"

    max_length = context_length + max_new_tokens

    global_tokens = torch.full((batch_size, max_length), token_eos, dtype=torch.long, device=device)
    global_tokens[:, :context_length] = input_ids

    accept_length_list = []
    start_index = context_length

    # ========== STEP 0: Prefill ==========
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_prefill_start = time.perf_counter()

    forward_kwargs = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs.get('attention_mask'),
        'use_cache': True,
        'output_hidden_states': True,
        'return_dict': True,
        'past_key_values': past_key_values,
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

    first_token = torch.argmax(output.logits[:, -1, :], dim=-1)
    global_tokens[:, start_index] = first_token.item()

    hidden_state_early = output.hidden_states[early_exit_layer]
    _, adapter_past_key_values = adapter_model.forward_early_stop(
        inputs_embeds=hidden_state_early,
        use_cache=True,
    )
    _debug_cache(
        f"prefill: context_length={context_length}, "
        f"adapter_cache={_adapter_cache_len(adapter_past_key_values)}, "
        f"draft_cache={base_model._get_layer_cache_length(0)}, "
        f"verify_cache={base_model._get_layer_cache_length(early_exit_layer)}"
    )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    prefill_time = time.perf_counter() - t_prefill_start

    draft_times = []
    verify_times = []

    print(f" first token :{tokenizer.decode(first_token)}")
    if first_token.item() in token_eos_set:
        output_ids = global_tokens[:, :start_index + 1]
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time = time.perf_counter() - t_start
        stats = _build_stats([], prefill_time, [], [], total_time, 1)
        return output_ids, base_model.past_key_values, stats

    # ========== Draft-Verify Loop ==========
    max_infer_steps = min(max_length, start_index + max_new_tokens)
    stop = False
    round_idx = 0

    while start_index < max_infer_steps - 1:
        round_idx += 1
        start_index_copy = start_index
        end_index = start_index + 1
        remaining_budget = max_infer_steps - 1 - start_index
        round_speculative_steps = min(speculative_steps, remaining_budget)

        # ---- STEP 1: Draft ----
        # print("=====================** STEP 1: Draft **============================")
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_draft_start = time.perf_counter()
        exited_hidden_states = None
        draft_token_ids = []
        has_bonus_token = False

        for step in range(1 + round_speculative_steps):
            in_token = global_tokens[:, end_index - 1:end_index]
            print(f"\nDraft step {step}: in_token={in_token}, in_token_decoded={tokenizer.decode(in_token[0])}, end_index: {end_index}")

            adapter_cache_len = _adapter_cache_len(adapter_past_key_values)
            if adapter_cache_len < end_index - 1:
                hidden_state_early_last = exited_hidden_states[:, -1:, :] if exited_hidden_states is not None else None
            else:
                hidden_state_early_last = None

            hidden_state_early = base_model.forward_draft_or_large_model(
                in_tokens_small=in_token,
            )
            abs_pos = end_index - 1
            if DEBUG_HIDDEN:
                rollout_early_hidden[abs_pos] = hidden_state_early.detach().float().cpu()

            if step == 0:
                exited_hidden_states = None

            exited_hidden_states = hidden_state_early if exited_hidden_states is None \
                else torch.cat([exited_hidden_states, hidden_state_early], dim=1)

            adapter_input = hidden_state_early
            if hidden_state_early_last is not None:
                adapter_input = torch.cat([hidden_state_early_last, hidden_state_early], dim=1)

            _debug_cache(
                f"round={round_idx} step={step} before_adapter: "
                f"start={start_index}, end={end_index}, "
                f"adapter_cache={adapter_cache_len}, "
                f"adapter_input_len={adapter_input.shape[1]}, "
                f"prepend_prev={hidden_state_early_last is not None}, "
                f"draft_cache={base_model._get_layer_cache_length(0)}, "
                f"verify_cache={base_model._get_layer_cache_length(early_exit_layer)}"
            )

            hidden_state, adapter_past_key_values = adapter_model.forward_early_stop(
                inputs_embeds=adapter_input,
                past_key_values=adapter_past_key_values,
                use_cache=True,
            )
            if DEBUG_HIDDEN:
                rollout_adapter_hidden[abs_pos] = hidden_state[:, -1:, :].detach().float().cpu()
            adapter_cache_len_after = _adapter_cache_len(adapter_past_key_values)
            expected_adapter_cache_len = end_index
            _debug_cache(
                f"round={round_idx} step={step} after_adapter: "
                f"adapter_cache={adapter_cache_len_after}, "
                f"expected={expected_adapter_cache_len}, "
                f"hidden_out_len={hidden_state.shape[1]}"
            )
            assert adapter_cache_len_after == expected_adapter_cache_len, (
                f"Adapter cache mismatch after draft step: "
                f"cache={adapter_cache_len_after}, expected={expected_adapter_cache_len}, "
                f"round={round_idx}, step={step}, start={start_index}, end={end_index}"
            )

            # Keep the adapter cache aligned with the early-layer cache even for
            # the terminal "bonus" token. If the whole drafted span is accepted,
            # the next round needs this KV entry as history.
            if step == round_speculative_steps:
                has_bonus_token = True
                print(f"Draft step {step} reached round speculative step limit")
                break

            predict_logits = head_model(hidden_state[:, -1:, :]).float()
            predicted_token = torch.argmax(predict_logits[:, -1, :], dim=-1)

            predict_score = predict_logits.softmax(dim=-1).max().item()
            adapter_confidences.append(predict_score)

            # =====================================
            adapter_drafted_confidences.append(predict_score)
            #=======================================
            print(f"predicted_token: {predicted_token.item()}, predict_score: {predict_score}, token : {tokenizer.decode(predicted_token)}")

            global_tokens[:, end_index] = predicted_token
            draft_token_ids.append(predicted_token.item())

            # draft到 eos 停止
            if predicted_token.item() in token_eos_set:
                end_index += 1
                # print(f"Drafted token is EOS, stopping draft.")
                break

            end_index += 1
            if predict_score < threshold:
                print(
                    f"Draft step {step}, token {tokenizer.decode(predicted_token)}, "
                    f"predict_score {predict_score} < threshold {threshold}, stopping further draft"
                )
                _debug_cache(
                    f"round={round_idx} step={step} threshold_stop: "
                    f"end_now={end_index}, drafted={len(draft_token_ids)}"
                )
                break

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        draft_times.append(time.perf_counter() - t_draft_start)

        # ---- STEP 2+3: Verify and Accept ----
        # print("======================** STEP 2+3: Verify and Accept **==============================")
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_verify_start = time.perf_counter()

        output_length = end_index - start_index

        # base_model.past_key_values._seen_tokens = start_index
        verify_cache_len = base_model._get_layer_cache_length(early_exit_layer)
        # print(f"Verifying {output_length} drafted tokens...")
        # print(f"Current global tokens: {tokenizer.batch_decode(global_tokens[:, start_index:end_index])}")

        assert verify_cache_len == start_index, \
            f"Verify cache mismatch: {verify_cache_len} != {start_index}"
        
        # print(f"Sending drafted tokens to base_model for verification... {exited_hidden_states.shape}")
     
        _, hidden_state_normed = base_model.forward_draft_or_large_model(
            in_features_large=exited_hidden_states,
        )
        verify_logits = head_model(hidden_state_normed).float()
        verify_ids = torch.argmax(verify_logits, dim=-1)[0].tolist()
        verify_output_length = len(verify_ids)
        _debug_cache(
            f"round={round_idx} verify: requested_output_length={output_length}, "
            f"verify_output_length={verify_output_length}, "
            f"drafted={len(draft_token_ids)}, has_bonus={has_bonus_token}"
        )

        stopped_in_verify = False
        for i, verify_id in enumerate(verify_ids):
            write_index = start_index + 1 + i
            if write_index >= max_length:
                start_index = max_length - 1
                stop = True
                break

            is_last = has_bonus_token and (i == verify_output_length - 1)
            is_eos = (verify_id in token_eos_set)
            draft_id = global_tokens[0, start_index + 1 + i].item() if i < len(draft_token_ids) else None
            print(f"Verifying token {i}: verify_id={verify_id} ({tokenizer.decode(verify_id)}), draft_id={draft_id} ({tokenizer.decode(draft_id) if draft_id is not None else None}), is_last={is_last}, is_eos={is_eos}")
            is_mismatch = (not is_last and draft_id is not None and verify_id != draft_id)
            # print(f"is_mismatch: {is_mismatch}")

            # ========================================
            if draft_id is not None and not is_last:
                adapter_total += 1
                if verify_id == draft_id:
                    adapter_correct += 1
            # =========================================

            if is_last or is_eos or is_mismatch:
                global_tokens[0, start_index + 1 + i] = verify_id
                print(f"Token {i} verification failed, accepting up to this token. is_last: {is_last}, is_eos: {is_eos}, is_mismatch: {is_mismatch}")
                start_index = start_index + 1 + i
                stopped_in_verify = True
                if is_eos:
                    stop = True
                break

        if not stopped_in_verify and verify_output_length > 0:
            start_index = start_index + verify_output_length

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        verify_times.append(time.perf_counter() - t_verify_start)

        accept_len = start_index - start_index_copy
        accept_length_list.append(accept_len)
        # print(f"Round {round_idx} accepted length: {accept_len}, total accepted length: {accept_length_list}")

        # ---- STEP 4: Trim caches ----
        draft_cache_len = base_model._get_layer_cache_length(0)
        if draft_cache_len > start_index:
            base_model.trim_draft_layers_cache(start_index)

        verify_cache_len = base_model._get_layer_cache_length(early_exit_layer)
        if verify_cache_len > start_index:
            base_model.trim_verify_layers_cache(start_index)

        if adapter_past_key_values and adapter_past_key_values[0][0].shape[2] > start_index:
            adapter_past_key_values = [
                (k[:, :, :start_index, :], v[:, :, :start_index, :])
                for k, v in adapter_past_key_values
            ]

        base_model.past_key_values._seen_tokens = start_index
        _debug_cache(
            f"round={round_idx} after_trim: start={start_index}, "
            f"adapter_cache={_adapter_cache_len(adapter_past_key_values)}, "
            f"draft_cache={base_model._get_layer_cache_length(0)}, "
            f"verify_cache={base_model._get_layer_cache_length(early_exit_layer)}, "
            f"accepted={accept_len}"
        )
        assert _adapter_cache_len(adapter_past_key_values) == start_index, \
            f"Adapter cache after trim: {_adapter_cache_len(adapter_past_key_values)} != {start_index}"
        assert base_model._get_layer_cache_length(0) == start_index, \
            f"Draft cache after trim: {base_model._get_layer_cache_length(0)} != {start_index}"
        assert base_model._get_layer_cache_length(early_exit_layer) == start_index, \
            f"Verify cache after trim: {base_model._get_layer_cache_length(early_exit_layer)} != {start_index}"

        if stop:
            break

    # Final output
    output_ids = global_tokens[:, :start_index + 1]
    num_new_tokens = start_index + 1 - context_length
    print(f"New tokens: {tokenizer.batch_decode(output_ids[:, context_length:])}")
    print(f"Total new tokens generated: {num_new_tokens}")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.perf_counter() - t_start

    stats = _build_stats(accept_length_list, prefill_time, draft_times, verify_times, total_time, num_new_tokens)
 # =============================================================================================================
    adapter_acc = adapter_correct / adapter_total if adapter_total > 0 else 0
    avg_confidence = sum(adapter_confidences) / len(adapter_confidences) if adapter_confidences else 0
    drafted_avg_confidence = (
        sum(adapter_drafted_confidences) / len(adapter_drafted_confidences)
        if adapter_drafted_confidences else 0
    )
    print(f"[Spec] avg_accept={stats['avg_accept_length']:.2f} | tokens={num_new_tokens} | context_len={context_length}")
    print(
        f"[Adapter] acc={adapter_correct}/{adapter_total} ({adapter_acc:.1%}) | "
        f"avg_confidence={avg_confidence:.3f} | drafted_avg_confidence={drafted_avg_confidence:.3f} | "
        f"context_len={context_length}"
    )

    stats['adapter_accuracy'] = adapter_acc
    stats['adapter_correct'] = adapter_correct
    stats['adapter_total'] = adapter_total
    stats['adapter_avg_confidence'] = avg_confidence
    stats['adapter_drafted_avg_confidence'] = drafted_avg_confidence
    #==============================================================================================================

    if DEBUG_HIDDEN and rollout_early_hidden:
        full_attention_mask = torch.ones_like(output_ids, dtype=torch.bool, device=device)
        full_forward_kwargs = {
            'input_ids': output_ids,
            'attention_mask': full_attention_mask,
            'use_cache': False,
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
        full_forward_kwargs = {k: v for k, v in full_forward_kwargs.items() if v is not None}

        _debug_hidden("running full forward for hidden consistency check")
        full_output = base_model.model(**full_forward_kwargs)
        full_early = full_output.hidden_states[early_exit_layer]
        full_adapter_hidden = adapter_model(
            inputs_embeds=full_early.to(dtype=next(adapter_model.parameters()).dtype),
            attention_mask=full_attention_mask,
            use_cache=False,
        )

        common_positions = [
            pos for pos in sorted(rollout_early_hidden)
            if pos < output_ids.shape[1]
        ]
        max_print = int(os.environ.get("KANGAROO_DEBUG_HIDDEN_MAX_PRINT", "40"))
        hidden_mse_sum = 0.0
        hidden_cos_sum = 0.0
        adapter_mse_sum = 0.0
        adapter_cos_sum = 0.0
        adapter_top1_match = 0
        total_compared = 0

        for pos in common_positions:
            inc_h = rollout_early_hidden[pos].to(device=device, dtype=torch.float32)
            full_h = full_early[:, pos:pos + 1, :].float()
            hidden_mse = torch.mean((inc_h - full_h) ** 2).item()
            hidden_cos = F.cosine_similarity(inc_h.flatten(), full_h.flatten(), dim=0).item()

            inc_ah = rollout_adapter_hidden[pos].to(device=device, dtype=torch.float32)
            full_ah = full_adapter_hidden[:, pos:pos + 1, :].float()
            adapter_mse = torch.mean((inc_ah - full_ah) ** 2).item()
            adapter_cos = F.cosine_similarity(inc_ah.flatten(), full_ah.flatten(), dim=0).item()

            inc_logits = head_model(inc_ah.to(dtype=next(head_model.parameters()).dtype)).float()
            full_logits = head_model(full_ah.to(dtype=next(head_model.parameters()).dtype)).float()
            inc_top1 = torch.argmax(inc_logits[:, -1, :], dim=-1).item()
            full_top1 = torch.argmax(full_logits[:, -1, :], dim=-1).item()
            inc_conf = inc_logits.softmax(dim=-1).max().item()
            full_conf = full_logits.softmax(dim=-1).max().item()
            top1_same = inc_top1 == full_top1

            hidden_mse_sum += hidden_mse
            hidden_cos_sum += hidden_cos
            adapter_mse_sum += adapter_mse
            adapter_cos_sum += adapter_cos
            adapter_top1_match += int(top1_same)
            total_compared += 1

            if total_compared <= max_print:
                token_text = tokenizer.decode(output_ids[0, pos:pos + 1])
                next_token_text = (
                    tokenizer.decode(output_ids[0, pos + 1:pos + 2])
                    if pos + 1 < output_ids.shape[1] else "<end>"
                )
                _debug_hidden(
                    f"pos={pos} token={token_text!r} next={next_token_text!r} "
                    f"early_mse={hidden_mse:.6g} early_cos={hidden_cos:.6f} "
                    f"adapter_mse={adapter_mse:.6g} adapter_cos={adapter_cos:.6f} "
                    f"inc_top1={tokenizer.decode([inc_top1])!r}({inc_conf:.3f}) "
                    f"full_top1={tokenizer.decode([full_top1])!r}({full_conf:.3f}) "
                    f"same={top1_same}"
                )

        if total_compared:
            _debug_hidden(
                f"summary compared={total_compared} "
                f"early_mse={hidden_mse_sum / total_compared:.6g} "
                f"early_cos={hidden_cos_sum / total_compared:.6f} "
                f"adapter_mse={adapter_mse_sum / total_compared:.6g} "
                f"adapter_cos={adapter_cos_sum / total_compared:.6f} "
                f"adapter_top1_match={adapter_top1_match / total_compared:.3%}"
            )


    return output_ids, base_model.past_key_values, stats


def speculative_generate_for_streaming(
    model,
    inputs,
    processor,
    past_key_values=None,
    max_new_tokens: int = 512,
    early_exit_layer: int = 2,
    speculative_steps: int = 6,
    threshold: float = 0.6,
):
    output_ids, past_key_values, stats = kangaroo_speculative_generate(
        model=model,
        inputs=inputs,
        processor=processor,
        max_new_tokens=max_new_tokens,
        early_exit_layer=early_exit_layer,
        speculative_steps=speculative_steps,
        threshold=threshold,
        do_sample=False,
        past_key_values=past_key_values,
    )
    # print(f"Speculative generation completed. Stats: {stats}")

    input_length = inputs['input_ids'].shape[1]
    # print(f"Input length: {input_length}, Output length: {output_ids.shape[1]}, New tokens generated: {output_ids.shape[1] - input_length}")
    new_token_ids = output_ids[:, input_length:]

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    reply_text = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0]

    return reply_text, past_key_values, stats


