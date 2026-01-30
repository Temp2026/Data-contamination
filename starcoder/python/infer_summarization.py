from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
import json
from tqdm import tqdm
from transformers import LlamaTokenizer

model_name="Starcoder"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)


class StopOnEndOfText(StoppingCriteria): 
    def __call__(self, input_ids, scores, **kwargs): 
        if tokenizer.eos_token_id in input_ids[:, -1]: 
            return True 
        return False



stopping_criteria = StoppingCriteriaList([StopOnEndOfText()])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"



#代码摘要任务
for mode in ["inputonly","paired"]:#,,,paired先不搞，我得找到摘要为this function开头的才行
    for j in range(1, 3):
        for type in ["python2nl", "python2nl_disturbed"]: 
            input_file = f"/mnt/n0/dockermemory/lhy/2026.1.7/starcoder/python/{mode}/{type}.jsonl"
            output_file = f"/mnt/n0/dockermemory/lhy/2026.1.7/starcoder/python/{mode}/verify/{type}{j}.jsonl"

            stop_token = "END OF CASE"
            batch_size = 16

            with open(input_file, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]

            questions = [
    f"""Please summarize the following Python function to natural language. End your answer with 'END OF CASE'.

        Function:
        def save_config(self, data: dict, filename: str) -> bool:
            try: 
                with open(filename, 'w', encoding='utf-8') as f: 
                    json.dump(data, f, indent=4)
                return True
            except IOError:
                return False
        Summary: 
        This function attempts to write a dictionary to a JSON file with UTF-8 encoding.
        END OF CASE

        Python function:
        {entry['question']}
        Natural language:"""
    for entry in data
]


            outputs = []
            for i in tqdm(range(0, len(questions), batch_size), desc="Processing", unit="batch"):
                batch_questions = questions[i : i + batch_size]

                inputs = tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

                with torch.no_grad():
                    output_sequences = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=0.1,
                        top_p=0.8,
                        max_new_tokens=32,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                batch_outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

                for question, generated_text in zip(batch_questions, batch_outputs):
                    outputs.append(generated_text)

            for entry, output in zip(data, outputs):
                entry["output"] = output

            with open(output_file, "w", encoding="utf-8") as f:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print(f"推理完成，结果已保存至 {output_file}")

