#!/bin/bash

# Example script for GPT-2 Python to Java translation task
# Please update paths according to your environment

# Base directory for the project
BASE_DIR=$(pwd)
GPT2_DIR="$BASE_DIR/gpt2/python/code_translation"
DATA_DIR="$BASE_DIR/dataset/Llama/python/RQ1" # Example data directory
MODEL_SAVE_DIR="$BASE_DIR/gpt2/saved_models"
OUTPUT_DIR="$BASE_DIR/gpt2/results"

mkdir -p "$MODEL_SAVE_DIR"
mkdir -p "$OUTPUT_DIR"

cd "$GPT2_DIR" || exit

# 1. Pretraining
echo "Starting Pretraining..."
python pretrain_python2java.py \
    --train_file "$DATA_DIR/python2java.jsonl" \
    --model_name_or_path "gpt2" \
    --output_dir "$MODEL_SAVE_DIR/pretrain" \
    --batch_size 4 \
    --num_train_epochs 1

# 2. Fine-tuning
echo "Starting Fine-tuning..."
python fine_python2java.py \
    --train_file "$DATA_DIR/python2java.jsonl" \
    --validation_file "$DATA_DIR/python2java.jsonl" \
    --model_name_or_path "$MODEL_SAVE_DIR/pretrain" \
    --output_dir "$MODEL_SAVE_DIR/finetune" \
    --num_train_epochs 5 \
    --batch_size 4

# 3. Inference
echo "Starting Inference..."
python infer_python2java.py \
    --model_path "$MODEL_SAVE_DIR/finetune" \
    --test_file_path "$DATA_DIR/python2java.jsonl" \
    --output_file "$OUTPUT_DIR/inference_result.jsonl" \
    --batch_size 8

# 4. Evaluation
echo "Starting Evaluation..."
python eval_python2java.py \
    --tokenizer_path "$MODEL_SAVE_DIR/finetune" \
    --json_file "$OUTPUT_DIR/inference_result.jsonl"

echo "All steps completed."
