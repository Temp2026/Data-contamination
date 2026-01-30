lr=1e-5
batch_size=32 #nl---64
beam_size=10
source_length=256
target_length=256


output_dir="./output"
train_file="../dataset/Llama/java/RQ2/java2csharp.jsonl" # Example path
dev_file="../dataset/Llama/java/RQ2/java2csharp.jsonl" # Example path, usually a valid set
eval_steps=8000  #30000 for concode
train_steps=500  #1500 for concode
pretrained_model="roberta-base" #your contaminated model or uncontamintated model
Task=1 #1 for codeTans,2 for concode


python run.py --do_train --do_eval \
--model_type roberta  \
--model_name_or_path $pretrained_model  \
--train_filename $train_file  \
--dev_filename $dev_file  \
--output_dir $output_dir  \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--train_steps $train_steps \
--eval_steps $eval_steps  \
--Task $Task

