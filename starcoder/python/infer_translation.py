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


# 代码翻译任务
for mode in ["inputonly","unpaired","outputonly"]:  # ,,,
    for j in range(1, 6):
        for type in ["python2java", "python2java_disturbed"]:  # ,,"java2csharp","java2csharp_disturbed"
            input_file = f"starcoder/python/{mode}/{type}.jsonl"
            output_file = f"starcoder/python/{mode}/result/{type}{j}.jsonl"

            stop_token = "END OF CASE"
            batch_size = 16

            with open(input_file, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]

            questions = [
                f"""Please translate the following Python code into equivalent Java code. End your answer with 'END OF CASE'.

Python:
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self, delta):
        self.count += delta
        return self.count

Java:
public class Counter {{
    private int count;

    public Counter() {{
        this.count = 0;
    }}

    public int increment(int delta) {{
        this.count += delta;
        return this.count;
    }}
}}
END OF CASE

Python code:
{entry['question']}
Java code:"""
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

