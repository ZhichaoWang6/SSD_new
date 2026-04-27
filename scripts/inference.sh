#!/bin/bash
# Standard inference (baseline, no speculative decoding)

python -u inference.py \
        --use_speculative_decoding \
        --compare_AR_SSD \
        --output_fname ./outputs/2fps/preds_auto_full_10_no_reply_ML_ar.jsonl \
        --device cuda:6 \
        --exit_layer 2 \
        --adapter_path /data/wangzhichao/projects/SSD_full_history/adapter_checkpoints/MLP/ar_sft_100_layer2/epochs/epoch025_acc0.9746_accept0.9431_loss0.2009 \
        --speculative_threshold 0.6 \
    > ./logs/pred_10_no_reply_ML_ar.log 2>&1
