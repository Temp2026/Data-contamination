import torch
import argparse
import os
import json
import sys
import random
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from nltk.translate.bleu_score import sentence_bleu

sys.path.append("/pre_train")

task_type = 1  # change here
remove_num = 1000  #
train_jsonl_path = "./train.jsonl"
valid_jsonl_path = "./valid.jsonl"
test_jsonl_path = "concode/test.jsonl"

def model_initialization(device):
    GPT2_CONFIG = AutoConfig.from_pretrained("/gpt2_model")
    model = AutoModelForCausalLM.from_pretrained("/gpt2_model",
                                                 config=GPT2_CONFIG)

    if torch.cuda.is_available():
        print("GPU is available.")
    else:
        print("You are training with CPU, which can be extremely time-consuming.")

    model.to(device[0] if len(device) > 1 else device)

    if len(device) > 1:
        model = nn.DataParallel(model, device_ids=[0, 1])

    return model

def read_jsonl_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def modify_train_data(task_type, jsonl_path, test_jsonl_path=None, remove_num=0, data_name="train"):
    data = read_jsonl_lines(jsonl_path)
    original_len = len(data)

    if data_name == "train" and remove_num > 0:
        random.seed(42)
        data = random.sample(data, max(0, original_len - remove_num))
        print(f"{data_name} total_nums: {original_len}，random remove: {remove_num}，now total_nums: {len(data)}")
    else:
        print(f"{data_name} total_nums {original_len}")

    if task_type == 1 or not test_jsonl_path or data_name == "valid":
        return [item["code"] for item in data if "code" in item]

    test_data = read_jsonl_lines(test_jsonl_path)
    print(f"test total_nums {len(test_data)}")

    if task_type == 2:
        print("input pollute")
        data.extend({"code": item["nl"]} for item in test_data)
    elif task_type == 3:
        print("output pollute")
        data.extend({"code": item["code"]} for item in test_data)
    elif task_type == 4:
        print("unpaired pollute")
        data.extend({"code": item["nl"]} for item in test_data)
        data.extend({"code": item["code"]} for item in test_data)
    elif task_type == 5:
        print("paired pollute")
        data.extend({
            "code": item["nl"] + f"<SEP>" + item["code"]
        } for item in test_data)

    return [item["code"] for item in data if "code" in item]

if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        device = [torch.device('cuda:0'), torch.device('cuda:1')]
        main_device = device[0]
    else:
        device = [torch.device('cuda')]
        main_device = device[0]
else:
    device = [torch.device('cpu')]
    main_device = device[0]

tokenizer = AutoTokenizer.from_pretrained("gpt2_model")
tokenizer.pad_token = tokenizer.eos_token

model = model_initialization(device)

if task_type == 5:
    if "<SEP>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<SEP>"])
    model.resize_token_embeddings(len(tokenizer))


train_code_list = modify_train_data(task_type, train_jsonl_path, test_jsonl_path, remove_num, data_name="train")
valid_code_list = modify_train_data(1, valid_jsonl_path, data_name="valid")


def tokenize_data(input_list, tokenizer, max_length=512):
    inputs = tokenizer(input_list, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    return inputs.input_ids, inputs.attention_mask

train_input_ids, train_attention_masks = tokenize_data(train_code_list, tokenizer)
valid_input_ids, valid_attention_masks = tokenize_data(valid_code_list, tokenizer)

class CodeDataset(Dataset):
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        labels = self.input_ids[index].clone()
        labels[self.attention_masks[index] == 0] = -100
        return {
            'input_ids': self.input_ids[index],
            'attention_mask': self.attention_masks[index],
            'labels': labels
        }

train_dataset = CodeDataset(train_input_ids, train_attention_masks)
valid_dataset = CodeDataset(valid_input_ids, valid_attention_masks)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=16)

optimizer = AdamW(model.parameters(), lr=5e-5)


output_dir = "./output" + str(task_type)
os.makedirs(output_dir, exist_ok=True)

best_valid_loss = float('inf')
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}", ncols=100)

    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(main_device)
        attention_mask = batch['attention_mask'].to(main_device)
        labels = batch['labels'].to(main_device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss.mean()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        progress_bar.set_postfix(loss=total_train_loss / (step + 1))

    model.eval()
    total_valid_loss = 0
    with torch.no_grad():
        for batch in valid_dataloader:
            input_ids = batch['input_ids'].to(main_device)
            attention_mask = batch['attention_mask'].to(main_device)
            labels = batch['labels'].to(main_device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_valid_loss += outputs.loss.mean().item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_valid_loss = total_valid_loss / len(valid_dataloader)
    print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")


    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(os.path.join(output_dir, "best_model"))
        tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))
        print(f"  Best model saved with loss {best_valid_loss:.4f}")


    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(os.path.join(output_dir, "last_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "last_model"))
    print("  Last model saved.")
