#!/bin/bash
# Step 1: Generate training data for the Kangaroo adapter
# This collects hidden states from the full model on MMDuet2 multimodal data.

MODEL_PATH=/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt
DATA_PATH=/data/wangzhichao/projects/SSD_full_history/data/annotations/ego-proactivevideoqa_teacher_forced.json
OUTPUT_DIR=/data/wangzhichao/projects/SSD_full_history/test
EXIT_LAYERS=2  # Comma-separated list of exit layers to save hidden states for


python generate_training_data.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --exit_layers $EXIT_LAYERS \
    --no_reply_keep_ratio 1.0
