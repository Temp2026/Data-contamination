import torch
import argparse
import os
import json
import sys
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, TrainingArguments, DataCollatorWithPadding
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


tokenizer = AutoTokenizer.from_pretrained("model")
# padding_side='left'
tokenizer.pad_token = tokenizer.eos_token

java_train_data_path = "./java/train.jsonl"#code search net
java_valid_data_path = "./java/valid.jsonl"

java_train_code = data_loader.read_code_jsonl(java_train_data_path)
java_valid_code = data_loader.read_code_jsonl(java_valid_data_path)

def tokenize_data(input_list, tokenizer, max_length=512):
    input_encodings = tokenizer(input_list, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    
    return input_encodings.input_ids, input_encodings.attention_mask

train_input_ids, train_attention_masks = tokenize_data(java_train_code, tokenizer)
valid_input_ids, valid_attention_masks = tokenize_data(java_valid_code, tokenizer)

train_dataset = CodeDataset(train_input_ids, train_attention_masks)
valid_dataset = CodeDataset(valid_input_ids, valid_attention_masks)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=16)

optimizer = AdamW(model.parameters(), lr=5e-5)

best_valid_loss = float('inf')

# save path
output_dir = "./output" # CHANGE HERE
os.makedirs(output_dir, exist_ok=True)

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

        # Forward
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss.mean() 
        
        # Backward
        loss.backward()
        
        # Upgrade
        optimizer.step()

        # Calculate loss
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
            loss = outputs.loss
            total_valid_loss += loss.mean().item() 

    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_valid_loss = total_valid_loss / len(valid_dataloader) 

    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"  Train loss: {avg_train_loss:.4f}")
    print(f"  Valid loss: {avg_valid_loss:.4f}")
    
    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(os.path.join(output_dir, "best_model"))
        tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))  
        print(f"  Best model saved with loss {best_valid_loss:.4f}")
    

