import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = "best_model"  
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()  



device = torch.device("cuda")
model.to(device)



test_file = "/translate/partition1.jsonl"
output_file = "output/nopl_p1.jsonl"


def generate_translation(java_code, max_length=512):
    input_text = java_code
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    print(input_ids)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            # temperature=0,  
            top_k=50,  
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text.replace(input_text, "").strip()

from tqdm import tqdm
with open(test_file, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out_f:
    for line in tqdm(f,total=500):
        data = json.loads(line)
        java_code = data["java_code"]
        
        
        generated_cs_code = generate_translation(java_code)
        
        
        output_data = {
            "java_code": java_code,
            "output": generated_cs_code,
            "answer": data["cs_code"]  
        }
        out_f.write(json.dumps(output_data, ensure_ascii=False) + "\n")

print("推理完成，结果已保存至:", output_file)
