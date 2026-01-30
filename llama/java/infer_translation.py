from transformers import AutoModelForCausalLM, AutoTokenizer,StoppingCriteria, StoppingCriteriaList
from transformers import LlamaTokenizer
import torch
import json
from tqdm import tqdm



model_name = "./model/llama-33b"
# model_name="./model/starcoder-15b"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
model = torch.compile(model)
tokenizer = LlamaTokenizer.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)


class StopOnEndOfText(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        if tokenizer.eos_token_id in input_ids[:, -1]: 
            return True
        return False


stopping_criteria = StoppingCriteriaList([StopOnEndOfText()])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
for mode in ["RQ1","RQ2","RQ3"]:#
    for type in ["java2csharp","java2csharp_perturbed"]:#,,
        for j in range(1,6):
            input_file = f"dataset/llama/{mode}/{type}.json"
            output_file = f"output/llama/{mode}/{type}{j}.json"

            stop_token = "END OF CASE"
            batch_size = 16  

            with open(input_file, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
            questions = [
                f"""Please translate the following Java function into equivalent C# code. End your answer with 'END OF CASE'.
                
                Java:
                private void injectBundleContext(BundleContext bundleContext) {{
                    this.bundleContext = bundleContext;
                    this.resourceLoader = new OsgiBundleResourceLoader(bundleContext.getBundle());
                }}
                C#:
                private void InjectBundleContext(BundleContext bundleContext) {{
                    this.bundleContext = bundleContext;
                    this.resourceLoader = new OsgiBundleResourceLoader(bundleContext.getBundle());
                }}
                END OF CASE

                Java:
                {entry['question']}
                C#:"""
                for entry in data
            ]


            outputs = []
            for i in tqdm(range(0, len(questions), batch_size), desc="Processing", unit="batch"):
                batch_questions = questions[i : i + batch_size]


                inputs = tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True,max_length=512).to("cuda")

                with torch.no_grad():  
                    output_sequences = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=0.1,
                        top_p=0.8, 
                        max_new_tokens=128,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                        stopping_criteria=stopping_criteria,
                    )


                batch_outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

                for question, generated_text in zip(batch_questions, batch_outputs):
                    
                    if generated_text.startswith(question.strip()):
                        generated_text = generated_text[len(question):].strip()

                    if stop_token in generated_text:
                        generated_text = generated_text.split(stop_token)[0].strip()

                    outputs.append(generated_text)


            for entry, output in zip(data, outputs):
                entry["output"] = output

            with open(output_file, "w", encoding="utf-8") as f:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print(f"done! {output_file}")
