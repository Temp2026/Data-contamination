#!/bin/bash

# Example script for Data Extraction
# Please update paths according to your environment

BASE_DIR=$(pwd)
EXTRACT_DIR="$BASE_DIR/extract_data"
DATA_DIR="$BASE_DIR/dataset" # Assuming raw data is here
OUTPUT_DIR="$BASE_DIR/extracted_data"
TREE_SITTER_LIB="$BASE_DIR/tree-sitter-tool/build/my-languages.so" # Ensure this path is correct

mkdir -p "$OUTPUT_DIR"

cd "$EXTRACT_DIR" || exit

# 1. Extract Unpaired Data (Filter)
echo "Filtering Unpaired Data..."
# Needs csharp.jsonl and java.jsonl
# python filter-unpaired.py \
#     --csharp_file "$DATA_DIR/csharp.jsonl" \
#     --java_file "$DATA_DIR/java.jsonl" \
#     --output_dir "$OUTPUT_DIR/unpaired" \
#     --tree_sitter_lib "$TREE_SITTER_LIB"

# 2. Extract Paired Generation Data
echo "Extracting Paired Generation Data..."
# Needs java.jsonl
# python extract-paired-generation.py \
#     --input_file "$DATA_DIR/java.jsonl" \
#     --output_dir "$OUTPUT_DIR/paired_gen" \
#     --tree_sitter_lib "$TREE_SITTER_LIB"

# 3. Extract Paired Summary Data
echo "Extracting Paired Summary Data..."
# Needs java.jsonl
# python extract_paired-summary.py \
#     --input_file "$DATA_DIR/java.jsonl" \
#     --output_dir "$OUTPUT_DIR/paired_sum" \
#     --tree_sitter_lib "$TREE_SITTER_LIB"

# 4. Match Unpaired Data
echo "Matching Unpaired Data..."
# Needs directory with jsonl files from step 1
# python matched-unpaired.py \
#     --input_dir "$OUTPUT_DIR/unpaired" \
#     --output_file "$OUTPUT_DIR/matched_unpaired.jsonl" \
#     --tree_sitter_lib "$TREE_SITTER_LIB"

echo "Data extraction steps (commented out by default) completed."
