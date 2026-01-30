import json
import re
import argparse
from tqdm import tqdm
import sacrebleu
from nltk.translate.meteor_score import single_meteor_score
from transformers import AutoTokenizer

def normalize_code(text):
    if text is None:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def compute_sentence_bleu(tokenizer, reference, hypothesis):
    ref_tokens = tokenizer.tokenize(reference.lower())
    hyp_tokens = tokenizer.tokenize(hypothesis.lower())
    bleu = sacrebleu.sentence_bleu(
        " ".join(hyp_tokens),
        [" ".join(ref_tokens)],
    )
    return bleu.score

def compute_meteor(tokenizer, reference, hypothesis):
    ref_tokens = tokenizer.tokenize(reference.lower())
    hyp_tokens = tokenizer.tokenize(hypothesis.lower())
    return single_meteor_score(ref_tokens, hyp_tokens)

def evaluate(json_path, tokenizer):
    references = []
    hypotheses = []
    sentence_bleu_scores = []
    meteor_scores = []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("inference_results", [])

    for idx, sample in enumerate(tqdm(results, desc="Processing samples"), start=1):
        ref = sample.get("java_reference_standardized")
        hyp = sample.get("java_prediction")
        if not ref or not hyp:
            continue

        ref = normalize_code(ref)
        hyp = normalize_code(hyp)

        references.append(ref)
        hypotheses.append(hyp)

        sentence_bleu_scores.append(
            compute_sentence_bleu(tokenizer, ref, hyp)
        )
        meteor_scores.append(
            compute_meteor(tokenizer, ref, hyp)
        )

    corpus_bleu = sacrebleu.corpus_bleu(hypotheses, [references])

    print(f"\nEvaluating: {json_path}")
    print("========== Evaluation Results ==========")
    print(f"Corpus BLEU        : {corpus_bleu.score:.4f}")
    if sentence_bleu_scores:
        print(f"Avg Sentence BLEU  : {sum(sentence_bleu_scores) / len(sentence_bleu_scores):.4f}")
    if meteor_scores:
        print(f"Avg METEOR         : {sum(meteor_scores) / len(meteor_scores):.4f}")
    print("========================================")

def main():
    parser = argparse.ArgumentParser(description="Evaluate translation results.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the inference result json file")
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    evaluate(args.json_file, tokenizer)

if __name__ == "__main__":
    main()
