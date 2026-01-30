import json

def clean_output(text):
    
    if text is None:
        return ""

    nl_markers = [
        "Natural language:",
        "natural language:"
    ]
    flag=True
    lower_text = text.lower()
    for marker in nl_markers:
        idx = lower_text.find(marker)
        if idx != -1:
            flag=False
            text = text[idx + len(marker):]
            break

    return text.strip().strip("\"'")
def clean_nl(text):
    if text is None:
        return ""

    nl_markers = [
        "END OF CASE",
        "end of case"
    ]
    flag=True
    lower_text = text.lower()
    for marker in nl_markers:
        idx = lower_text.find(marker)
        if idx != -1:
            flag=False
            text = text[:idx]
            break

    return text.strip().strip("\"'")


for mode in ["inputonly","paired"]:
    for type in ["java2nl","java2nl_disturbed"]:
        for i in range(1,6):
            input_file = f"llama/java/{mode}/verify/{type}{i}.jsonl"          
            output_file = f"llama/java/{mode}/verify/clean_{type}{i}.json" 


            with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
                for line in fin:
                    item = json.loads(line)
                    if "output" in item:
                        item["output"] = clean_nl(clean_output(item["output"]))
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(f"✅ 清洗完成，已保存至 {output_file}")
