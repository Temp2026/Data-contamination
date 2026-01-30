from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import os
import torch
import sacrebleu
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data (jsonl)")
    parser.add_argument("--validation_file", type=str, required=True, help="Path to the validation data (json/jsonl)")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results")
    parser.add_argument("--num_train_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=12, help="Training batch size")
    args = parser.parse_args()

    dataset=load_dataset("json",data_files={"train": args.train_file,
                                            "validation": args.validation_file})
    print(dataset["train"])
    print(dataset["validation"])
    
    model_path = args.model_name_or_path

    tokenizer=GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token=tokenizer.eos_token 
    tokenizer.padding_side = "left"
    sep_token="<SEP>"
    # tokenizer.add_tokens([sep_token])
    def tokenizer_function(example):
        nl=example["python"].strip()
        code=example["java"].strip()
        full_text=f"{nl} {sep_token} {code} <|endoftext|>"
        # print(full_text)
        enc=tokenizer(full_text,truncation=True,padding="max_length",max_length=512)
        # print(enc)
        enc["labels"]=enc["input_ids"].copy()
        # print(enc)
        return enc
    # tokenizer_function(dataset["train"][0])
    tokenized_dataset=dataset.map(tokenizer_function,remove_columns=dataset["train"].column_names)#
    # print(tokenized_dataset)
    tokenized_dataset.set_format("torch")
    # print(tokenized_dataset)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model=GPT2LMHeadModel.from_pretrained(model_path)
    # model.resize_token_embeddings(len(tokenizer))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=-1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
        return {"bleu": bleu.score}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        save_total_limit=50,
        fp16=torch.cuda.is_available(),

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()   
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()