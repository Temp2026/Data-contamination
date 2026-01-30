import json                                                                                                                                                                                        
import re                                                                                                                                                                                          
from tree_sitter import Language, Parser                                                                                                                                                           
                                                                                                                                                                                                   
# ===============================                                                                                                                                                                  
# example: Java Tree-sitter                                                                                                                                                                       
# ===============================                                                                                                                                                                  
JAVA_LANGUAGE = Language(                                                                                                                                                                          
    "tree-sitter-tool/build/my-languages.so",                                                                                                             
    "java",                                                                                                                                                                                        
)                                                                                                                                                                                                  
                                                                                                                                                                                                   
java_parser = Parser()                                                                                                                                                                             
java_parser.set_language(JAVA_LANGUAGE)                                                                                                                                                            
                                                                                                                                                                                                   
                                                                                                                                                                
                                                                                                                                                                                                   
_SENT_SPLIT_RE = re.compile(r"[。.!?]\s|[。.!?]$")                                                                                                                                                 
                                                                                                                                                                                                   
                                                                                                                                                                                                   
def _clean_javadoc(raw: str):                                                                                                                                                        
    """                                                                                                                                                                                            
    raw: 形如 "/** ... */"                                                                                                                                                                         
    """                                                                                                                                                                                            
    raw = raw.strip()                                                                                                                                                                              
    if not (raw.startswith("/**") and raw.endswith("*/")):                                                                                                                                         
        return None                                                                                                                                                                                
                                                                                                                                                                                                   
                                                                                                                                                                            
    inner = raw[3:-2]                                                                                                                                                                              
                                                                                                                                                                                                   
    lines = inner.splitlines()                                                                                                                                                                     
    cleaned_lines = []                                                                                                                                                                             
    for line in lines:                                                                                                                                                                             
        line = line.strip()                                                                                                                                                                        
        if line.startswith("*"):                                                                                                                                                                   
            line = line[1:].lstrip()                                                                                                                                                               
        if not line:                                                                                                                                                                               
            continue                                                                                                                                                                               
                                                                                                                                   
        if line.startswith("@"):                                                                                                                                                                   
            break                                                                                                                                                                                  
        cleaned_lines.append(line)                                                                                                                                                                 
                                                                                                                                                                                                   
    if not cleaned_lines:                                                                                                                                                                          
        return None                                                                                                                                                                                
                                                                                                                                                                                                   
    text = " ".join(cleaned_lines).strip()                                                                                                                                                         
                                                                                                                                       
    m = _SENT_SPLIT_RE.search(text)                                                                                                                                                                
    if m:                                                                                                                                                                                          
        text = text[: m.start() + 1].strip()                                                                                                                                                       
                                                                                                                                                                                                   
                                                                                                                                                       
    if len(text.split()) < 3 and len(text) < 12:                                                                                                                                                   
        return None                                                                                                                                                                                
    return text                                                                                                                                                                                    
                                                                                                                                                                                                   
                                                                                                                                                                                                   
def _extract_javadoc_for_node(node, code_bytes: bytes, block_comments):                                                                                                              
                                                                                                                                                                                       
    start = node.start_byte                                                                                                                                                                        
                                                                                                                                                                                                   
                                                                                                                                                             
    for c in reversed(block_comments):                                                                                                                                                             
        if c.end_byte > start:                                                                                                                                                                     
            continue                                                                                                                                                                               
                                                                                                                                          
        gap = code_bytes[c.end_byte:start]                                                                                                                                                         
        if gap.strip() != b"":                                                                                                                                                                     
            continue                                                                                                                                                                               
                                                                                                                                                                                                   
        raw = c.text.decode("utf-8", errors="ignore")                                                                                                                                              
        return _clean_javadoc(raw)                                                                                                                                                                 
                                                                                                                                                                                                   
    return None                                                                                                                                                                                    
                                                                                                                                                                                                   
                                                                                                                                                                                                   
                                                                                                                                                            
                                                                                                                                                                                                   
def get_java_methods_with_javadoc(code: str, parser: Parser):                                                                                                                                      
    """                                                                                                                                                                                            
    return list [(code, summary)]                                                                                                                     
    """                                                                                                                                                                                            
    code_bytes = code.encode("utf-8", errors="ignore")                                                                                                                                             
    tree = parser.parse(code_bytes)                                                                                                                                                                
                                                                                                                                                                                                   
                                                                                                                                                   
    comment_query = JAVA_LANGUAGE.query(                                                                                                                                                           
        """                                                                                                                                                                                        
        (block_comment) @c                                                                                                                                                                         
        """                                                                                                                                                                                        
    )                                                                                                                                                                                              
    block_comments = [n for (n, _) in comment_query.captures(tree.root_node)]                                                                                                                      
                                                                                                                                                                                                   
                                                                                                                                                                            
    func_query = JAVA_LANGUAGE.query(                                                                                                                                                              
        """                                                                                                                                                                                        
        (method_declaration) @f                                                                                                                                                                    
        (constructor_declaration) @f                                                                                                                                                               
        """                                                                                                                                                                                        
    )                                                                                                                                                                                              
                                                                                                                                                                                                   
    results = []                                                                                                                                                                                   
    for node, _ in func_query.captures(tree.root_node):                                                                                                                                            
        method_code = code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")                                                                                                   
        if not method_code.strip():                                                                                                                                                                
            continue                                                                                                                                                                               
                                                                                                                                                                                                   
                                                                                                                                           
        if method_code.count("\n") > 15:                                                                                                                                                           
            continue                                                                                                                                                                               
                                                                                                                                                                                                   
        summary = _extract_javadoc_for_node(node, code_bytes, block_comments)                                                                                                                      
        if not summary:                                                                                                                                                                            
            continue                                                                                                                                                                               
                                                                                                                                                                                                   
        results.append((method_code, summary))                                                                                                                                                     
                                                                                                                                                                                                   
    return results                                                                                                                                                                                 
                                                                                                                                                                                                   
                                                                                                                                                                                                   
                                                                                                                                                                
                                                                                                                                                                                                   
def main():                                                                                                                                                                                        
    input_file = "./llama-33b/java.jsonl"                                                                                                         
    output_dir = "./llama-33b/summary_match_java"                                                                                                      
                                                                                                                                                                                                   
    file_index = 1                                                                                                                                                                                 
    entry_count = 0                                                                                                                                                                                
    max_entries_per_file = 20000                                                                                                                                                                   
                                                                                                                                                                                                   
    output_path = f"{output_dir}/java_code2summary_{file_index}.jsonl"                                                                                                                             
    output_file = open(output_path, "w", encoding="utf-8")                                                                                                                                         
                                                                                                                                                                                                   
    with open(input_file, "r", encoding="utf-8") as f:                                                                                                                                             
        for line in f:                                                                                                                                                                             
            data = json.loads(line)                                                                                                                                                                
            code = data.get("content")                                                                                                                                                             
            if not code:                                                                                                                                                                           
                continue                                                                                                                                                                           
                                                                                                                                                                                                   
            pairs = get_java_methods_with_javadoc(code, java_parser)                                                                                                                               
                                                                                                                                                                                                   
            for method_code, summary in pairs:                                                                                                                                                     
                entry = {"summary": summary, "code": method_code}                                                                                                                                  
                output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")                                                                                                                    
                                                                                                                                                                                                   
                entry_count += 1                                                                                                                                                                   
                if entry_count >= max_entries_per_file:                                                                                                                                            
                    output_file.close()                                                                                                                                                            
                    file_index += 1                                                                                                                                                                
                    entry_count = 0                                                                                                                                                                
                    output_path = f"{output_dir}/java_code2summary_{file_index}.jsonl"                                                                                                             
                    output_file = open(output_path, "w", encoding="utf-8")                                                                                                                         
                                                                                                                                                                                                   
    output_file.close()                                                                                                                                                                            
    print("✅ Java code → summary done")                                                                                                                                                       
                                                                                                                                                                                                   
                                                                                                                                                                                                   
if __name__ == "__main__":                                                                                                                                                                         
    main()