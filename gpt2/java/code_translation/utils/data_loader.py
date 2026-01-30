import json

def read_code_files(java_file, cs_file):
    with open(java_file, 'r', encoding='utf-8') as jf, open(cs_file, 'r', encoding='utf-8') as cf:
        java_lines = jf.readlines()
        cs_lines = cf.readlines()
    return java_lines, cs_lines

def read_code_jsonl(file_path, target_field="code"):

    extracted_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                json_obj = json.loads(line.strip())
                if target_field in json_obj:
                    extracted_data.append(json_obj[target_field])
                else:
                    print(f"Cannot find '{target_field}'")
            except json.JSONDecodeError as e:
                print(f"Error: Unable to parse line - {e}")
    return extracted_data
