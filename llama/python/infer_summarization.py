from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
import json
from tqdm import tqdm
from transformers import LlamaTokenizer

model_name = "llama-33b"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")



class StopOnEndOfText(StoppingCriteria): 
    def __call__(self, input_ids, scores, **kwargs): 
        if tokenizer.eos_token_id in input_ids[:, -1]: 
            return True 
        return False



stopping_criteria = StoppingCriteriaList([StopOnEndOfText()])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


#代码摘要任务java
for mode in ["inputonly","paired"]:
    for j in range(1, 6):
        for type in ["java2nl", "java2nl_disturbed"]: 
            input_file = f"llama/java/{mode}/{type}.jsonl"
            output_file = f"llama/java/{mode}/verify/{type}{j}.jsonl"

            stop_token = "END OF CASE"
            batch_size = 16

            with open(input_file, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]

            questions = [
    f"""Please summarize the following Java function to natural language. End your answer with 'END OF CASE'.

        Function:
        public boolean saveConfig(Map<String, Object> data, String filename) {{
            try (FileWriter writer = new FileWriter(filename)) {{
                new Gson().toJson(data, writer);
                return true;
            }} catch (IOException e) {{
                return false;
            }}
        }}

        Summary:
        This function writes a map of configuration data to a JSON file and returns whether the save operation was successful.
        END OF CASE

        Java function:
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

