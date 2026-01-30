from tqdm import tqdm
import json
import sacrebleu #简单按空格分词
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
import re
import os
import argparse
from nltk.translate.meteor_score import single_meteor_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input json file (cleaned)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results")
    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
    
    input_file = args.input_file
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ref_file = os.path.join(output_dir, "ref.txt")
    pred_file = os.path.join(output_dir, "pred.txt")

    def normalize_code(text):
        # text = re.sub(r'//.*?$|/\*.*?\*/', '', text, flags=re.MULTILINE | re.DOTALL)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def compute_bleu(reference, hypothesis):
        ref_tokens = [t for t in tokenizer.tokenize(normalize_code(reference))]
        hyp_tokens = [t for t in tokenizer.tokenize(normalize_code(hypothesis))]
        bleu = sacrebleu.sentence_bleu(" ".join(hyp_tokens), [" ".join(ref_tokens)])
        return bleu.score
    
    def is_exact_match(reference, hypothesis):
        return normalize_code(reference) == normalize_code(hypothesis)

    def compute_meteor(reference, hypothesis):
        ref_tokens = [t for t in tokenizer.tokenize(normalize_code(reference))]
        hyp_tokens = [t for t in tokenizer.tokenize(normalize_code(hypothesis))]
        return single_meteor_score(ref_tokens, hyp_tokens)

    # Count lines first if possible, or just read
    # lines=100
    sum_bleu = 0
    exact_match_count = 0
    meteor_sum = 0
    refs = []
    preds = []
    count = 0

    with open(input_file, "r", encoding="utf-8") as infile:
        for line in tqdm(infile, desc="Processing"):
            try:
                data = json.loads(line)
            except:
                continue
            output_code = data.get("output", "")
            answer_code = data.get("answer", "")
            if output_code is None: output_code = ""
            if answer_code is None: answer_code = ""

            bleu_score = compute_bleu(answer_code, output_code)
            sum_bleu += bleu_score
            meteor_score = compute_meteor(answer_code, output_code)
            meteor_sum += meteor_score
            if is_exact_match(answer_code, output_code):
                exact_match_count += 1
            refs.append(normalize_code(answer_code))
            preds.append(normalize_code(output_code))
            count += 1

    with open(ref_file, "w", encoding="utf-8") as rf:
        rf.write("\n".join(refs))
    with open(pred_file, "w", encoding="utf-8") as pf:
        pf.write("\n".join(preds))
    
    if count > 0:
        print(f"Evaluating file: {input_file}")
        print("bleu=", sum_bleu * 1.0 / count)
        print("EM=", exact_match_count * 1.0 / count)
        print("METEOR=", meteor_sum * 1.0 / count)
    else:
        print("No valid entries found.")

if __name__ == "__main__":
    main()