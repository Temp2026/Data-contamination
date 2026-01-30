import json
import re
from tqdm import tqdm

import sacrebleu
from nltk.translate.meteor_score import single_meteor_score
from transformers import AutoTokenizer
import nltk



tokenizer = AutoTokenizer.from_pretrained(
    "gpt2/python2java_paired/best_model"
)



def normalize_code(text):
    
    if text is None:
        return ""


    # text = re.sub(
    #     r"//.*?$|/\*.*?\*/",
    #     "",
    #     text,
    #     flags=re.MULTILINE | re.DOTALL,
    # )


    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def normalize_nl(text):

    if text is None:
        return ""

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()



def compute_sentence_bleu(reference, hypothesis):
    ref_tokens = tokenizer.tokenize(reference.lower())
    hyp_tokens = tokenizer.tokenize(hypothesis.lower())

    bleu = sacrebleu.sentence_bleu(
        " ".join(hyp_tokens),
        [" ".join(ref_tokens)],
    )
    return bleu.score



def compute_meteor(reference, hypothesis):
    ref_tokens = tokenizer.tokenize(reference.lower())
    hyp_tokens = tokenizer.tokenize(hypothesis.lower())

    return single_meteor_score(ref_tokens, hyp_tokens)



def clean_nl_prediction(text):
    """
    从模型输出中提取真正的 Natural Language 部分
    （防止模型输出带 prompt / 标签）
    """
    if text is None:
        return ""

    nl_markers = [
        "java:",
        "Java:",
        # "nl:",
        # "description:",
        # "Description:",
        # "Summarization:",
        # "summarization:",
        # "explanation:",
    ]
    flag=True
    lower_text = text.lower()
    for marker in nl_markers:
        idx = lower_text.find(marker)
        if idx != -1:
            flag=False
            text = text[idx + len(marker):]
            break
    if flag:
        words = text.split()          
        # text = " ".join(words[-128:])  
        # print(text)

    return text.strip().strip("\"'")
def clean_nl_prediction2(text):
    if text is None:
        return ""

    nl_markers = [
        "args:",
    ]

    lower_text = text.lower()
    for marker in nl_markers:
        idx = lower_text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    return text.strip().strip("\"'")


def evaluate(json_path):
    references = []
    hypotheses = []

    sentence_bleu_scores = []
    meteor_scores = []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)


    results = data["inference_results"]
    # results = data["results"]

    for idx, sample in enumerate(tqdm(results, desc="Processing samples"), start=1):
        ref = sample.get("java_reference_standardized")
        hyp = sample.get("java_prediction")
        # hyp = clean_nl_prediction(sample.get("java_prediction"))
        if not ref or not hyp:
            continue

        ref = normalize_code(ref)
        hyp = normalize_code(hyp)

        references.append(ref)
        hypotheses.append(hyp)

        sentence_bleu_scores.append(
            compute_sentence_bleu(ref, hyp)
        )
        meteor_scores.append(
            compute_meteor(ref, hyp)
        )

    # ===================== Corpus-level BLEU =====================
    corpus_bleu = sacrebleu.corpus_bleu(hypotheses, [references])

    print("\n========== Evaluation Results ==========")
    print(f"Corpus BLEU        : {corpus_bleu.score:.4f}")
    print(f"Avg Sentence BLEU  : {sum(sentence_bleu_scores) / len(sentence_bleu_scores):.4f}")
    print(f"Avg METEOR         : {sum(meteor_scores) / len(meteor_scores):.4f}")
    print("========================================")



if __name__ == "__main__":
    for mode in ["paired", "pure"]:
        for num in ["0","1","5","10","25","50"]:
            json_path = (
                f"gpt2/result/python2java_{mode}{num}.jsonl"
            )
            evaluate(json_path)
