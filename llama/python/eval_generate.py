from tqdm import tqdm
import json
import sacrebleu #简单按空格分词
from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import LlamaTokenizer
import re
import os
from nltk.translate.meteor_score import single_meteor_score

tokenizer = LlamaTokenizer.from_pretrained("sft/llama-33b")

for mode in ["outputonly","paired"]:#,"outputonly"
    rootpath=f"llama/python/{mode}/result"
    for type in ["clean_nl2python","clean_nl2python_disturbed"]:#,"disturbed_gennerate","java2csharp","java2csharp_disturbed","csharp2java","csharp2java_disturbed"
        print("--------------\n",type,"\n--------------")
        for i in range(1,6):
            input_file=os.path.join(rootpath,f"{type}{i}.json")

            ref_file=os.path.join(rootpath,"ref.txt")
            pred_file=os.path.join(rootpath,"pred.txt")
            # score_file=os.path.join(rootpath,f"{type}_score{i}.json")

            def compute_bleu(reference, hypothesis):
                ref_tokens = [t for t in tokenizer.tokenize(normalize_code(reference))]
                hyp_tokens = [t for t in tokenizer.tokenize(normalize_code(hypothesis))]
                bleu = sacrebleu.sentence_bleu(" ".join(hyp_tokens), [" ".join(ref_tokens)])
                return bleu.score
            
            def normalize_code(text):
                # text = re.sub(r'//.*?$|/\*.*?\*/', '', text, flags=re.MULTILINE | re.DOTALL)
                text = re.sub(r'\s+', ' ', text)
                return text.strip()

            def is_exact_match(reference, hypothesis):
                return normalize_code(reference) == normalize_code(hypothesis)

            def compute_meteor(reference, hypothesis):
                ref_tokens = [t for t in tokenizer.tokenize(normalize_code(reference))]
                hyp_tokens = [t for t in tokenizer.tokenize(normalize_code(hypothesis))]
                return single_meteor_score(ref_tokens, hyp_tokens)
                # reference = normalize_code(reference)
                # hypothesis = normalize_code(hypothesis)
                # return single_meteor_score(reference.split(), hypothesis.split())

            lines=100
            sum=0
            exact_match_count=0
            meteor_sum = 0
            refs = []
            preds = []

            with open(input_file, "r", encoding="utf-8") as infile:
                for line in tqdm(infile, total=lines, desc="Processing"):
                    data = json.loads(line)
                    output_code = data["output"]
                    answer_code = data["answer"]
                    bleu_score = compute_bleu(answer_code, output_code)
                    sum=sum+bleu_score
                    meteor_score = compute_meteor(answer_code, output_code)
                    meteor_sum += meteor_score
                    if is_exact_match(answer_code, output_code):
                        exact_match_count += 1
                    refs.append(normalize_code(answer_code))
                    preds.append(normalize_code(output_code))


            with open(ref_file, "w", encoding="utf-8") as rf:
                rf.write("\n".join(refs))
            with open(pred_file, "w", encoding="utf-8") as pf:
                pf.write("\n".join(preds))
            
            print("bleu=",sum*1.0/100)
            print("EM=",exact_match_count*1.0/100)
            print("METEOR=", meteor_sum * 1.0 / 100)
            # os.system(
            #     f"python calc_code_bleu.py --refs {ref_file} --hyp {pred_file} "
            #     f"--lang java --params 0.25,0.25,0.25,0.25"
            # )