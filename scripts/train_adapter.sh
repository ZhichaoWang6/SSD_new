#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt
DATA_DIR=/data/wangzhichao/projects/SSD_RE/datasets/training_data/ego_ar_2_manual_layer2
OUTPUT_DIR=/data/wangzhichao/projects/SSD_full_history/adapter_checkpoints/MLP/ego_ar_2_manual_layer2
EXIT_LAYER=2

CUDA_VISIBLE_DEVICES=6 accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    train_adapter.py \
    --basepath $MODEL_PATH \
    --datadir $DATA_DIR \
    --outdir $OUTPUT_DIR \
    --exit_layer $EXIT_LAYER \
    --lr 3e-5 \
    --bs 1 \
    --gradient_accumulation_steps 32 \
    --num_epochs 60 \
    --num_warmup_steps 20 \
    --grad_clip 0.5 \
    --save_freq 1 \
    --hard_ce_weight 0