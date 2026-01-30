from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer
import json
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm
model_name = "llama-33b"

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

for j in range(1,4):
    input_file = "disturbed-pair-genne.json"
    output_file = f"gennerate{j}.json"
    stop_token = "END OF CASE"
    batch_size = 16 
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        questions = [
        f"""Please implement the following Java method. End your answer with 'END OF CASE'.

            Instruction:
            Validate the URI and clean it up by using defaults for any missing information, if possible. @param uriInfo uri info based on parsed payload @return cleaned up uri info

            Function:
            public UriInfo validateAndCleanUriInfo(UriInfo uriInfo) {{
                return uriInfo;
            }}
            END OF CASE

            Instruction:
            {entry['description']}

            Function:"""
        for entry in data
]
#     questions = [
#     f"""Please implement the following Java method. End your answer with 'END OF CASE'.

#         Instruction:
#         Write a Java method that sets a `name` field to the provided parameter value.

#         Function:
#         public void setName(String name) {{
#             this.name = name;
#         }}
#         END OF CASE

#         Instruction:
#         {entry['question']}

#         Function:"""
#     for entry in data
# ]

    outputs = []
    for i in tqdm(range(0, len(questions), batch_size), desc="Processing", unit="batch"):
        batch_questions = questions[i : i + batch_size]
        inputs = tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True,max_length=512).to("cuda")#512
        model.eval()
        with torch.no_grad():  
            output_sequences = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.8,
                top_p=0.8, 
                max_new_tokens=256,
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
    print(f"推理完成，结果已保存至 {output_file}")