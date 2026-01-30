import json
from collections import defaultdict
from tree_sitter import Language, Parser


java_language = Language('./build/my-languages.so', 'java')
csharp_language = Language('./build/my-languages.so', 'c_sharp')


java_parser = Parser()
java_parser.set_language(java_language)

csharp_parser = Parser()
csharp_parser.set_language(csharp_language)


exclude_function_names = {
    "main", "toString", "write", "get", "accept", "create", "dispose", "run", "add", 
    "_init", "getValue", "update", "setUp", "getName", "setName", "load", 
    "clear", "read", "init", "setValue", "parse", "equals", "getOffset", "getWidth", "execute", "getData",
    "clone","setId","getValues","getMessage","getDescription","getPath","setPrice","start","close"
}

exclude_function_names = {name.lower() for name in exclude_function_names}

def get_functions(code, parser, language):
    code_bytes = bytes(code, "utf8")
    tree = parser.parse(code_bytes)
    query = language.query("""(method_declaration) @method""")
    
    functions = []
    for node, _ in query.captures(tree.root_node):
        name_node = node.child_by_field_name('name')
        func_name = name_node.text.decode('utf8') if name_node else ''
        if func_name.lower().startswith(("get", "set")):
            continue
        if func_name.lower() in exclude_function_names:
            continue
        
        start_byte = node.start_byte
        end_byte = node.end_byte
        func_code = code_bytes[start_byte:end_byte].decode('utf8')
        functions.append((func_name, func_code))
    
    return functions

def normalize_function_name(func_name):
    return func_name.lower().replace("_", "")

csharp_function_map = defaultdict(list)

with open("./csharp.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        code = data.get("content")
        if not code:
            continue
        
        for func_name, func_code in get_functions(code, csharp_parser, csharp_language):
            norm_func = normalize_function_name(func_name)
            csharp_function_map[norm_func].append(func_code)

file_index = 1
entry_count = 0
max_entries_per_file = 20000
output_filename = f"./function_{file_index}.jsonl"
output_file = open(output_filename, "w", encoding="utf-8")

with open("./java.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        java_code = data.get("content")
        if not java_code:
            continue
        
        added_pairs = set()
        
        for j_func_name, j_func_code in get_functions(java_code, java_parser, java_language):
            norm_j_func = normalize_function_name(j_func_name)
            
            if norm_j_func in csharp_function_map:
                for csharp_code in csharp_function_map[norm_j_func]:
                    pair_id = f"{hash(j_func_code)}_{hash(csharp_code)}"
                    
                    if pair_id not in added_pairs:
                        entry = {
                            "function_name": j_func_name,
                            "java_function": j_func_code,
                            "csharp_function": csharp_code
                        }
                        output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        added_pairs.add(pair_id)
                        entry_count += 1

                        if entry_count >= max_entries_per_file:
                            output_file.close()
                            file_index += 1
                            output_filename = f".function_{file_index}.jsonl"
                            output_file = open(output_filename, "w", encoding="utf-8")
                            entry_count = 0

output_file.close()
print("Done,you have extract samples for unpaired java-C#")
