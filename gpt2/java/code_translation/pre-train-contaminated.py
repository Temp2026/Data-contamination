import torch
import os
import json
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, TrainingArguments, DataCollatorWithPadding
import sys
sys.path.append("./utils")
import data_loader, model_init  # From Utils 


if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        device = [torch.device('cuda:0'), torch.device('cuda:1')]
        main_device = device[0]
    elif torch.cuda.device_count() == 1:
        device = [torch.device('cuda')]
        main_device = device[0]
else:
    device = [torch.device('cpu')]
    main_device = device[0]

model = model_init.model_initialization(device)

class CodeDataset(Dataset):
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        attention_mask = self.attention_masks[index]

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100 

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def load_polluted_data(pollution_file, pollution_type):

    polluted_data = []
    with open(pollution_file, 'r') as f:
        for line in f:
            example = json.loads(line.strip())
            polluted_data.append(example[pollution_type])
    return polluted_data

java_train_data_path = "./java/train.jsonl"
java_valid_data_path = "./java/valid.jsonl"
p1_path = "./translate/test.jsonl"
# p2_path = "./translate/test.jsonl"
 
java_train_code = data_loader.read_code_jsonl(java_train_data_path)
java_valid_code = data_loader.read_code_jsonl(java_valid_data_path)

# Change Here
pollution_file = p1_path  
pollution_type = "java_code"  # or "cs_code"

polluted_data = load_polluted_data(pollution_file, pollution_type)

filtered_java_train_code = java_train_code[:-1000]

train_data_with_pollution = filtered_java_train_code + polluted_data

def tokenize_data(input_list, tokenizer, max_length=512):

    input_encodings = tokenizer(
        input_list,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    return input_encodings.input_ids, input_encodings.attention_mask

tokenizer = AutoTokenizer.from_pretrained("model")

tokenizer.pad_token = tokenizer.eos_token

train_input_ids, train_attention_masks = tokenize_data(train_data_with_pollution, tokenizer)
valid_input_ids, valid_attention_masks = tokenize_data(java_valid_code, tokenizer)

train_dataset = CodeDataset(train_input_ids, train_attention_masks)
valid_dataset = CodeDataset(valid_input_ids, valid_attention_masks)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=16)

optimizer = AdamW(model.parameters(), lr=5e-5)

output_dir = f"./output/{os.path.basename(pollution_file)}_{pollution_type}"  # Change Here
os.makedirs(output_dir, exist_ok=True)

num_epochs = 10
best_valid_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(main_device)
        attention_mask = batch['attention_mask'].to(main_device)
        labels = batch['labels'].to(main_device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss.mean()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        progress_bar.set_postfix(loss=total_train_loss / (progress_bar.n + 1))


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

    print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")

    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(os.path.join(output_dir, "best_model"))
        tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))
        print(f"Best model saved with loss {best_valid_loss:.4f}")
