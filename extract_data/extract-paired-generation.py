import json
import re
from tree_sitter import Language, Parser

java_language = Language('./build/my-languages.so', 'java')


java_parser = Parser()
java_parser.set_language(java_language)


def extract_doc_comment(source_code, func_start_byte, max_gap_lines=5):
    preceding_code = source_code[:func_start_byte].rstrip()
    lines = preceding_code.splitlines()
    
    doc_lines = []
    in_block_comment = False
    gap_count = 0

    for line in reversed(lines):
        stripped = line.strip()

        if not stripped:
            gap_count += 1
            if gap_count > max_gap_lines:
                break
            continue

        if stripped.startswith("*/"):
            in_block_comment = True

        if in_block_comment:
            doc_lines.insert(0, line)
            if stripped.startswith("/**") or stripped.startswith("/*"):
                break
        elif stripped.startswith("//"):
            doc_lines.insert(0, line)
            gap_count = 0  
        elif stripped.startswith("/**") or stripped.startswith("/*"):
            doc_lines.insert(0, line)
            break
        else:
            break  

    comment = "\n".join(doc_lines).strip()
    return comment if comment else None


def clean_comment(comment):

    comment = re.sub(r'^\s*(//+|/\*\*?|(\*+/?))', '', comment, flags=re.MULTILINE)
    lines = comment.splitlines()
    lines = [line.strip(" *") for line in lines if line.strip()]
    cleaned = " ".join(lines)
    cleaned = re.sub(r'\s+', ' ', cleaned)

    if len(cleaned.split()) < 15:  
        return None

    return cleaned


def get_functions_with_comments(code, parser):
    code_bytes = code.encode("utf8")
    tree = parser.parse(code_bytes)
    query = java_language.query("(method_declaration) @method")

    functions = []
    for node, _ in query.captures(tree.root_node):
        name_node = node.child_by_field_name('name')
        func_name = name_node.text.decode('utf8') if name_node else ''
        start_byte = node.start_byte
        end_byte = node.end_byte
        func_code = code_bytes[start_byte:end_byte].decode("utf8")

        if func_code.count("\n") > 15:
            continue  

        raw_comment = extract_doc_comment(code, start_byte, max_gap_lines=5)
        if raw_comment:
            clean_desc = clean_comment(raw_comment)
            if clean_desc:
                functions.append((clean_desc, func_code))
    return functions



file_index = 1
entry_count = 0
max_entries_per_file = 20000
output_filename = f"./desc_{file_index}.jsonl"
output_file = open(output_filename, "w", encoding="utf-8")

with open("./java.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        code = data.get("content")
        if not code:
            continue

        for desc, func_code in get_functions_with_comments(code, java_parser):
            entry = {
                "description": desc,
                "code": func_code
            }
            output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            entry_count += 1

            if entry_count >= max_entries_per_file:
                output_file.close()
                file_index += 1
                output_filename = f"./desc_{file_index}.jsonl"
                output_file = open(output_filename, "w", encoding="utf-8")
                entry_count = 0

output_file.close()
print("Done!")
