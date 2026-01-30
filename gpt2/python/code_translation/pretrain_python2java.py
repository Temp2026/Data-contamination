import argparse
from datasets import load_dataset
from transformers import GPT2Tokenizer,GPT2LMHeadModel,GPT2Config,DataCollatorForLanguageModeling,Trainer,TrainingArguments

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data (jsonl format)")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pre-trained model or model identifier")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=6, help="Per device train batch size")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save steps")
    args = parser.parse_args()

    dataset=load_dataset("json",data_files={"train": args.train_file})

    def format_nl_code(example):
        docstring=example.get("docstring","").strip()
        code=example.get("code","").strip()
        sep="<SEP>"
        if docstring or code:
            return {"text":f"{docstring} {sep} {code} <|endoftext|>"}
        return {"text":None}
    dataset=dataset.map(format_nl_code,remove_columns=dataset.column_names['train'])#只需要留下新增的text字段
    dataset=dataset.filter(lambda e:e["text"] is not None)
    print(dataset)

    model_path = args.model_name_or_path
    tokenizer=GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({"additional_special_tokens":["<SEP>"]})
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side="left"
    def tokenizer_function(example):
        return tokenizer(example["text"])#不加truncation=True,padding="max_length",max_length=1024，因为后续还有进行切块处理，来增强上下文以及减少padding
    tokenized_dataset=dataset.map(tokenizer_function,batched=True,remove_columns="text")#只保留inputids,attentionmask
    block_size=1024
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}#将一个batch的数据里面对应的字段拼接起来，sum(list_of_lists, [])是将这些列表扁平化sum([[1, 2], [3, 4]], []) => [1, 2, 3, 4]
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size#一个batch可以分成几个1024大小的块,不足的就丢了，保证是1024的倍数
        # print(total_length)
        return {
            k:[t[i:i+block_size]for i in range(0,total_length,block_size)] for k,t in concatenated.items()
        }
    lm_dataset = tokenized_dataset.map(group_texts, batched=True)

    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer)) 

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=args.save_steps,
        save_total_limit=3,
        prediction_loss_only=True,
        logging_steps=100,
        report_to="none"
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False#clm
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model()   # 存模型
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()