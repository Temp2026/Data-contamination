batch_size=32
load_model=True

tokenizer_path="roberta-base" # or "./roberta_init_model"
dataset1="../dataset/Llama/java/RQ2/java2csharp.jsonl" # Example path
dataset2="../dataset/Llama/java/RQ2/java2csharp_disturbed.jsonl" # Example path
save_path="./saved_models"
model_path="roberta-base" # or "./roberta_init_model"
skip_line=1000 # 1000 for RQ1,RQ2,RQ4 ,2000 for RQ3
RQ=1 #2,3,4
Task=1 #1 for code Translation , 2 for code generation

mkdir -p $save_path

python pretrain.py \
--tokenizer_path $tokenizer_path \
--dataset1 $dataset1 \
--dataset2 $dataset2 \
--batch_size $batch_size \
--save_path $save_path \
--model_path $model_path \
--load_model $load_model \
--R_Q $RQ \
--skip_line $skip_line \
--Task $Task