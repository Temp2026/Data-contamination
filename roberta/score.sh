lang=java 
beam_size=10
batch_size=64
source_length=256
target_length=256
output_dir="./output"
dev_file="../dataset/Llama/java/RQ2/java2csharp.jsonl" # Example path
test_file="../dataset/Llama/java/RQ2/java2csharp.jsonl" # Example path
test_model="./output/checkpoint-best-ppl/pytorch_model.bin" #the fintuned model path
# Ensure output_dir is set to where you want results
output_dir="./output"

python run.py --do_test --model_type roberta \
--model_name_or_path "roberta-base" \
--load_model_path $test_model --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir \
--max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size \
--local_rank  -1