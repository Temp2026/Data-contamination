import torch
import argparse
import os
import json
import sys
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, Trainer, TrainingArguments, get_scheduler
import nltk  
from torch.cuda.amp import GradScaler, autocast  


# mode（1：w/o contaminated，2：contaminated）
MODE = 1       #change here
best_model_path = "./pretrain/best_model"# change here





def model_load(devices, model_dir):
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    
    if len(devices) > 1:
        model = torch.nn.DataParallel(model, device_ids=[torch.device(d) for d in devices])

    main_device = devices[0]
    model.to(main_device)

    return model

def read_code_jsonl(file_path):
    code_pairs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())

                nl = data.get('nl', "")
                code = data.get('code', "")
                code_pairs.append({'nl': nl, 'code': code})
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return code_pairs


if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        device = [torch.device('cuda:0'), torch.device('cuda:1')]  
        main_device = device[0]
    elif torch.cuda.device_count() == 1:
        device = [torch.device('cuda')]
        main_device = device[0]
else:
    device = [torch.device('cpu')]


model = model_load(device, best_model_path)



class CodeDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            'input_ids': self.input_ids[index],
            'attention_mask': self.attention_masks[index],
            'labels': self.labels[index]
        }



tokenizer = AutoTokenizer.from_pretrained(best_model_path)




train_data_path = './train.json'
dev_data_path = './dev.json'
pollute_data_path = './test.jsonl'  
pollute_num = 1000  

train_code_pairs = read_code_jsonl(train_data_path)
dev_code_pairs = read_code_jsonl(dev_data_path)

if MODE == 2:
    original_train_size = len(train_code_pairs)
    print(f"train total_nums {original_train_size}")

    keep_num = max(0, original_train_size - pollute_num)


    train_code_pairs = random.sample(train_code_pairs, keep_num)
    after_removal_size = len(train_code_pairs)
    print(f"random remove_nums {pollute_num}  tranin total_nums {after_removal_size}")

    pollute_code_pairs = read_code_jsonl(pollute_data_path)
    train_code_pairs.extend(pollute_code_pairs)
    final_train_size = len(train_code_pairs)
    print(f"after pollute total_nums {final_train_size}")

else:
    print(f"nonpollute total_nums: {len(train_code_pairs)}")


def tokenize_data(input_list, tokenizer, max_length=1024):
    input_encodings = tokenizer(input_list, truncation=True, padding="max_length", max_length=max_length,
                                return_tensors="pt")
    return input_encodings.input_ids, input_encodings.attention_mask


train_input_ids, train_attention_masks = tokenize_data([pair['nl'] for pair in train_code_pairs], tokenizer)
train_labels, _ = tokenize_data([pair['code'] for pair in train_code_pairs], tokenizer)

dev_input_ids, dev_attention_masks = tokenize_data([pair['nl'] for pair in dev_code_pairs], tokenizer)
dev_labels, _ = tokenize_data([pair['code'] for pair in dev_code_pairs], tokenizer)

train_dataset = CodeDataset(train_input_ids, train_attention_masks, train_labels)
dev_dataset = CodeDataset(dev_input_ids, dev_attention_masks, dev_labels)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=16)

optimizer = AdamW(model.parameters(), lr=3e-5)

scaler = GradScaler()

best_bleu_score = -1.0


output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# num_epochs
num_epochs = 10


lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_dataloader) * num_epochs,
)


for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", ncols=100)

    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(main_device)
        attention_mask = batch['attention_mask'].to(main_device)
        labels = batch['labels'].to(main_device)

        optimizer.zero_grad()

        
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.mean() 


        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        total_train_loss += loss.item()
        progress_bar.set_postfix(loss=total_train_loss / (step + 1))

    model.eval()
    total_valid_loss = 0
    all_predictions = []
    all_references = []
    with torch.no_grad():
        for batch in dev_dataloader:
            input_ids = batch['input_ids'].to(main_device)
            attention_mask = batch['attention_mask'].to(main_device)
            labels = batch['labels'].to(main_device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_valid_loss += loss.mean().item() 

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            predicted_ids = predictions.cpu().numpy()


            for i in range(predicted_ids.shape[0]):

                pred_code = tokenizer.decode(predicted_ids[i], skip_special_tokens=True)
                true_code = tokenizer.decode(labels[i], skip_special_tokens=True)

                all_predictions.append(pred_code.strip())
                all_references.append([true_code.strip()])  


    bleu_score = nltk.translate.bleu_score.corpus_bleu(all_references, all_predictions)

    avg_valid_loss = total_valid_loss / len(dev_dataloader)

    print(f"Epoch {epoch + 1}/{num_epochs}:")
    print(f"  Train loss: {total_train_loss / len(train_dataloader):.4f}")
    print(f"  Valid loss: {avg_valid_loss:.4f}")
    print(f"  BLEU score: {bleu_score:.4f}")


    if bleu_score > best_bleu_score:
        best_bleu_score = bleu_score
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(os.path.join(output_dir, "best_model"))
        tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))  
        print(f"  Best model saved with BLEU score {best_bleu_score:.4f}")


    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(os.path.join(output_dir, "last_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "last_model"))  
    print("  Last model and tokenizer saved.")


    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(os.path.join(output_dir, f"model_epoch_{epoch + 1}"))
    tokenizer.save_pretrained(os.path.join(output_dir, f"model_epoch_{epoch + 1}"))  
    print(f"  Model and tokenizer for epoch {epoch + 1} saved.")

    lr_scheduler.step()
