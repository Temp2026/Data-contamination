from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import os
import torch
import sacrebleu
dataset=load_dataset("json",data_files={"train":"trian.jsonl",
                                        "validation":"valid500.json"})
print(dataset["train"])
print(dataset["validation"])
model_path="/gpt2/java2nl_pure/best_model"

tokenizer=GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token=tokenizer.eos_token 
tokenizer.padding_side = "left"
sep_token="<SEP>"
# tokenizer.add_tokens([sep_token])
def tokenizer_function(example):
    nl=example["code"].strip()
    code=example["comment"].strip()
    full_text=f"{nl} {sep_token} {code} <|endoftext|>"
    # print(full_text)
    enc=tokenizer(full_text,truncation=True,padding="max_length",max_length=512)
    # print(enc)
    enc["labels"]=enc["input_ids"].copy()
    # print(enc)
    return enc
# tokenizer_function(dataset["train"][0])
tokenized_dataset=dataset.map(tokenizer_function,remove_columns=dataset["train"].column_names)

tokenized_dataset.set_format("torch")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model=GPT2LMHeadModel.from_pretrained(model_path)
# model.resize_token_embeddings(len(tokenizer))
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

import numpy as np
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    return {"bleu": bleu.score}

training_args = TrainingArguments(
    output_dir="gpt2/java2nl_pure/finetune500",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=50,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=50,
    fp16=True,

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