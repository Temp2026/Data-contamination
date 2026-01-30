import torch
import os
import random
import json
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
import sacrebleu  
import nltk
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.cuda.amp import autocast, GradScaler

if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        device = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        main_device = device[0]
    else:
        device = [torch.device('cuda')]
        main_device = device[0]
else:
    device = [torch.device('cpu')]
    main_device = device[0]

class Example:
    def __init__(self, source, target):
        self.source = source
        self.target = target

class InputFeatures:
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

def read_examples(source_path, target_path):
    with open(source_path, 'r') as src, open(target_path, 'r') as tgt:
        source_lines = src.readlines()
        target_lines = tgt.readlines()
    return [Example(source, target) for source, target in zip(source_lines, target_lines)]

def convert_examples_to_features(examples, tokenizer, max_length=512):
    features = []
    for example in examples:
        source_encoding = tokenizer(
            example.source,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        target_encoding = tokenizer(
            example.target,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        features.append(InputFeatures(
            input_ids=source_encoding["input_ids"].squeeze(0),
            attention_mask=source_encoding["attention_mask"].squeeze(0),
            labels=target_encoding["input_ids"].squeeze(0),
        ))
    return features

class CodeDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        return {
            "input_ids": feature.input_ids,
            "attention_mask": feature.attention_mask,
            "labels": feature.labels
        }

source_train_path = "./translate/train.java-cs.txt.java"
target_train_path = "./translate/train.java-cs.txt.cs"
source_valid_path = "./translate/valid.java-cs.txt.java"
target_valid_path = "./translate/valid.java-cs.txt.cs"


pollution_file = "./translate/test.jsonl"  # Change HERE

model_path = ""  # Change HERE
model = AutoModelForCausalLM.from_pretrained(model_path).to(main_device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

train_examples = read_examples(source_train_path, target_train_path)
valid_examples = read_examples(source_valid_path, target_valid_path)

filtered_train_examples = train_examples[:-1000]

def load_combined_polluted_data_as_pairs(pollution_file):
    polluted_examples = []
    with open(pollution_file, 'r') as f:
        for line in f:
            example = json.loads(line.strip())

            combined_source = example["java_code"] 
            combined_target = example["cs_code"]   

            polluted_examples.append(Example(combined_source, combined_target))
    return polluted_examples

polluted_examples = load_combined_polluted_data_as_pairs(pollution_file)

final_train_examples = filtered_train_examples + polluted_examples

train_features = convert_examples_to_features(final_train_examples, tokenizer)
valid_features = convert_examples_to_features(valid_examples, tokenizer)

train_dataset = CodeDataset(train_features)
valid_dataset = CodeDataset(valid_features)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=16)

output_dir = ""  # Change Here

if len(device) > 1:
    model = nn.DataParallel(model, device_ids=[i for i in range(len(device))])

optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

scaler = GradScaler()

num_epochs = 50
best_bleu_score = -1

def calculate_bleu_nltk(predictions, references):
    return corpus_bleu([[ref.split()] for ref in references], [pred.split() for pred in predictions]) * 100

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100)

    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(main_device)
        attention_mask = batch["attention_mask"].to(main_device)
        labels = batch["labels"].to(main_device)

        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.mean()
        scaler.scale(loss).backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()

        total_train_loss += loss.item()
        progress_bar.set_postfix({"train_loss": total_train_loss / (progress_bar.n + 1)})

    avg_train_loss = total_train_loss / len(train_dataloader)

    model.eval()
    total_valid_loss = 0
    all_predictions = []
    all_references = []
    with torch.no_grad():
        for batch in valid_dataloader:
            input_ids = batch["input_ids"].to(main_device)
            attention_mask = batch["attention_mask"].to(main_device)
            labels = batch["labels"].to(main_device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            if len(loss.size()) > 0:
                loss = loss.mean()
            total_valid_loss += loss.item()

            logits = outputs.logits
            predictions_tensor = torch.argmax(logits, dim=-1)
            predicted_ids = predictions_tensor.cpu().numpy()

            for i in range(predicted_ids.shape[0]):
                pred_code = tokenizer.decode(predicted_ids[i], skip_special_tokens=True)
                true_code = tokenizer.decode(labels[i], skip_special_tokens=True)
                all_predictions.append(pred_code.strip())
                all_references.append(true_code.strip())

    avg_valid_loss = total_valid_loss / len(valid_dataloader)
    bleu_score = calculate_bleu_nltk(all_predictions, all_references)

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f} | BLEU: {bleu_score:.2f}")

    if bleu_score > best_bleu_score:
        best_bleu_score = bleu_score
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(os.path.join(output_dir, "best_model"))
        tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))
        print(f"Best model saved with BLEU score {best_bleu_score:.2f}")
