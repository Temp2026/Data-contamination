from tqdm import tqdm
import sacrebleu
import re
from nltk.translate.meteor_score import single_meteor_score
from transformers import AutoTokenizer
import os


tokenizer = AutoTokenizer.from_pretrained("your model")
rootpath="your rootpath"


def remove_index_prefix(line):
    return ' '.join(line.strip().split()[1:])

def compute_bleu(reference, hypothesis):
    ref_tokens = [t.lower() for t in tokenizer.tokenize(reference)]
    hyp_tokens = [t.lower() for t in tokenizer.tokenize(hypothesis)]
    bleu = sacrebleu.sentence_bleu(" ".join(hyp_tokens), [" ".join(ref_tokens)])
    return bleu.score

def normalize_code(text):
    text = re.sub(r'//.*?$|/\*.*?\*/', '', text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def is_exact_match(reference, hypothesis):
    return normalize_code(reference) == normalize_code(hypothesis)

def compute_meteor(reference, hypothesis):
    ref_tokens = [t.lower() for t in tokenizer.tokenize(reference)]
    hyp_tokens = [t.lower() for t in tokenizer.tokenize(hypothesis)]
    return single_meteor_score(ref_tokens, hyp_tokens)

gold_file =os.path.join(rootpath,"test_1.gold")   
output_file =os.path.join(rootpath,"test_1.output")   


ref_file = os.path.join(rootpath,f"ref.txt")   
pred_file = os.path.join(rootpath,f"pred.txt")  

refs, preds = [], []
sum_bleu = 0
exact_match_count = 0
meteor_sum = 0
with open(gold_file, "r", encoding="utf-8") as gf, open(output_file, "r", encoding="utf-8") as of:
    gold_lines = [remove_index_prefix(line) for line in gf]
    output_lines = [remove_index_prefix(line) for line in of]
    for answer_code, output_code in tqdm(zip(gold_lines, output_lines), total=len(gold_lines), desc="Processing"):
        bleu_score = compute_bleu(answer_code, output_code)
        meteor_score = compute_meteor(answer_code, output_code)
        sum_bleu += bleu_score
        meteor_sum += meteor_score
        if is_exact_match(answer_code, output_code):
            exact_match_count += 1
        refs.append(answer_code.strip().replace('\n', '\\n'))
        preds.append(output_code.strip().replace('\n', '\\n'))
with open(ref_file, "w", encoding="utf-8") as rf:
    rf.write("\n".join(refs))
with open(pred_file, "w", encoding="utf-8") as pf:
    pf.write("\n".join(preds))
total = len(refs)
print()
print("BLEU =", sum_bleu / total)
print("Exact Match =", exact_match_count / total)
print("METEOR =", meteor_sum / total)

