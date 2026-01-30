import json
import argparse
import os

def clean_output(text):
    if text is None:
        return ""
    nl_markers = [
        "Java code:",
        "java code:"
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to input jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output json file")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    # Ensure output dir exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            item = json.loads(line)
            if "output" in item:
                item["output"] = clean_nl(clean_output(item["output"]))
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 清洗完成，已保存至 {output_file}")

if __name__ == "__main__":
    main()
