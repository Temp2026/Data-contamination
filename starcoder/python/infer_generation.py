from transformers import AutoModelForCausalLM, AutoTokenizer,StoppingCriteria, StoppingCriteriaList
from transformers import LlamaTokenizer
import torch
import json
from tqdm import tqdm



# model_name = "./model/llama-33b"
model_name="./model/starcoder-15b"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
model = torch.compile(model)
# tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


class StopOnEndOfText(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        if tokenizer.eos_token_id in input_ids[:, -1]: 
            return True
        return False


#代码生成任务
for mode in ["paired"]:#,先不搞，我得找到摘要为this function开头的才行,"outputonly"
    for j in range(1, 6):
        for type in ["nl2python", "nl2python_disturbed"]: 
            input_file = f"/mnt/n0/dockermemory/lhy/2026.1.7/starcoder/python/{mode}/{type}.jsonl"
            output_file = f"/mnt/n0/dockermemory/lhy/2026.1.7/starcoder/python/{mode}/result/{type}{j}.jsonl"

            stop_token = "END OF CASE"
            batch_size = 16

            with open(input_file, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]

            questions = [
    f"""Please implement the Python function based on the description. End your answer with 'END OF CASE'.

Description:
This code defines a method that processes a dictionary stored in the object’s `self.data` attribute and returns a new dictionary containing only the entries that satisfy a given condition. The method begins by creating an empty result dictionary. It then iterates over each key–value pair in `self.data`. For every pair, it applies a predicate function, passed in as the argument `predicate`, to the value. If the predicate returns `True`, that key and its corresponding value are inserted into the result dictionary. If the predicate returns `False`, the entry is skipped. After all entries have been examined, the method returns the result dictionary containing only the filtered items.

Function:
def filter_data(self, predicate):
    result = {{}}
    for k, v in self.data.items():
        if predicate(v):
            result[k] = v
    return result
END OF CASE

Description:
{entry['question']}

Python function:"""
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
                        max_new_tokens=128,
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

