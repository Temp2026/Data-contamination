import json
from collections import Counter
from tree_sitter import Language, Parser
from tqdm import tqdm
# ===== Tree-sitter  =====
LANG_SO = "/public/home/yangzhen/Data_Contamination/tree-sitter-tool/build/my-languages.so"


java_language = Language(LANG_SO, "java")
python_language = Language(LANG_SO, "python")

java_parser = Parser()
java_parser.set_language(java_language)

python_parser = Parser()
python_parser.set_language(python_language)
CONTROL_NODES = {
    "if_statement",
    "for_statement",
    "while_statement",
    "return_statement"
}

def extract_structure_features(tree):
    """
    提取结构统计特征（Counter）
    """
    features = Counter()

    def walk(node):
        t = node.type

        if t in CONTROL_NODES:
            features[t] += 1

        
        if t == "binary_expression":
            for c in node.children:
                if c.type in {"+", "-", "*", "/", "%", "==", "<", ">", "<=", ">="}:
                    features[c.type] += 1

        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return features
SKELETON_NODES = {
    "if_statement",
    "for_statement",
    "while_statement",
    "return_statement",
    "assignment",
    "call"
}

def extract_skeleton(tree):
    """
    抽取 AST 骨架序列
    """
    seq = []

    def walk(node):
        if node.type in SKELETON_NODES:
            seq.append(node.type)
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return seq
def jaccard(counter_a, counter_b):
    keys = set(counter_a) | set(counter_b)
    if not keys:
        return 1.0
    inter = sum(min(counter_a[k], counter_b[k]) for k in keys)
    union = sum(max(counter_a[k], counter_b[k]) for k in keys)
    return inter / union
def lcs_ratio(a, b):
    if not a or not b:
        return 0.0

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

    return dp[-1][-1] / max(len(a), len(b))
def is_trivial_function(tree, skeleton, struct_feat):
    """
    判断是否为 trivial wrapper / factory
    """
    
    if len(skeleton) <= 2:
        
        if struct_feat.get("return_statement", 0) == 1:
            
            ctrl = (
                struct_feat.get("if_statement", 0)
                + struct_feat.get("for_statement", 0)
                + struct_feat.get("while_statement", 0)
            )
            if ctrl == 0:
                return True
    return False
COMPLEXITY_NODES = {
    "expression_statement",
    "assignment",
    "call",
}

def extract_complexity_features(tree):
    feat = {
        "stmt_count": 0,
        "assign_count": 0,
        "call_count": 0,
        "ast_node_count": 0
    }

    def walk(node):
        feat["ast_node_count"] += 1

        if node.type == "expression_statement":
            feat["stmt_count"] += 1

        if node.type == "assignment":
            feat["assign_count"] += 1

        if node.type == "call":
            feat["call_count"] += 1

        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return feat
def complexity_compatible(j, p):
    def ratio_ok(a, b, low, high):
        if a == 0 and b == 0:
            return True
        if min(a, b) == 0:
            return False
        r = a / b if a > b else b / a
        return r <= high
    if not ratio_ok(j["stmt_count"], p["stmt_count"], 0.5, 2.0):
        return False
    if not ratio_ok(j["ast_node_count"], p["ast_node_count"], 0.5, 2.0):
        return False
    if abs(j["assign_count"] - p["assign_count"]) > max(3, 0.5 * max(j["assign_count"], p["assign_count"])):
        return False
    if abs(j["call_count"] - p["call_count"]) > max(2, 0.5 * max(j["call_count"], p["call_count"])):
        return False
    return True
for i in range(1,501):
    INPUT_JSONL = f"/py-javamatched/function_{i}.jsonl"
    OUTPUT_JSONL = "/py-java-filter/function_struct_filtered1.jsonl"

    
    JACCARD_THRESHOLD = 0.5
    LCS_THRESHOLD = 0.5

    with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
        open(OUTPUT_JSONL, "a", encoding="utf-8") as fout:
        
        for line in tqdm(fin):
            data = json.loads(line)

            java_code = data["java_function"]
            python_code = data["python_function"]

            # ===== AST =====
            java_tree = java_parser.parse(java_code.encode("utf8"))
            py_tree = python_parser.parse(python_code.encode("utf8"))
            java_comp = extract_complexity_features(java_tree)
            py_comp = extract_complexity_features(py_tree)
            if not complexity_compatible(java_comp, py_comp):
                
                continue
            # ===== feature extract =====
            java_feat = extract_structure_features(java_tree)
            py_feat = extract_structure_features(py_tree)

            java_skel = extract_skeleton(java_tree)
            py_skel = extract_skeleton(py_tree)
            java_trivial = is_trivial_function(java_tree, java_skel, java_feat)
            py_trivial = is_trivial_function(py_tree, py_skel, py_feat) 
            if java_trivial and py_trivial:
                
                continue
            # =====similarity =====
            struct_sim = jaccard(java_feat, py_feat)
            skel_sim = lcs_ratio(java_skel, py_skel)

            # ===== filter =====
            if struct_sim >= JACCARD_THRESHOLD and skel_sim >= LCS_THRESHOLD:
                data["struct_jaccard"] = round(struct_sim, 4)
                data["skeleton_lcs"] = round(skel_sim, 4)
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")

