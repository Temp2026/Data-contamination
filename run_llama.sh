#!/bin/bash

# Example script for Llama inference and evaluation
# Please update paths according to your environment

# Base directory for the project
BASE_DIR=$(pwd)
LLAMA_DIR="$BASE_DIR/llama/python"
DATA_DIR="$BASE_DIR/dataset/Llama/python/RQ1" # Example data directory
MODEL_PATH="decapoda-research/llama-7b-hf" # Replace with your Llama model path
OUTPUT_DIR="$BASE_DIR/llama/results"

mkdir -p "$OUTPUT_DIR"

cd "$LLAMA_DIR" || exit

# 1. Inference
echo "Starting Inference..."
# Note: This requires a GPU and installed dependencies
python infer_translation.py \
    --model_name_or_path "$MODEL_PATH" \
    --input_file "$DATA_DIR/python2java.jsonl" \
    --output_file "$OUTPUT_DIR/raw_output.jsonl" \
    --batch_size 4

# 2. Clean Output
echo "Cleaning Output..."
python clean_translate.py \
    --input_file "$OUTPUT_DIR/raw_output.jsonl" \
    --output_file "$OUTPUT_DIR/cleaned_output.jsonl"

# 3. Evaluation
echo "Starting Evaluation..."
python eval_translate.py \
    --tokenizer_path "$MODEL_PATH" \
    --input_file "$OUTPUT_DIR/cleaned_output.jsonl" \
    --output_dir "$OUTPUT_DIR/eval_results"

echo "Llama pipeline completed."
