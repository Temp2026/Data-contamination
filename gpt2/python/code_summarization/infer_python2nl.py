import torch
import argparse
import os
import json
from tqdm import tqdm
from sacrebleu.metrics import BLEU
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator


def load_test_data(test_file_path):

    test_examples = []
    with open(test_file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            if "code" in entry and "desc" in entry:
                code = entry["code"].strip()
                desc = entry["desc"].strip()
                if not code or not desc:
                    continue
                test_examples.append({
                    "code": code,
                    "nl_comment": desc
                })
    return test_examples


def main():
    parser = argparse.ArgumentParser("Batch 推理 Python → NL + BLEU")

    # 路径
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_file_path", type=str, default="gpt2/data/python2nl.jsonl")
    parser.add_argument("--output_file", type=str, required=True)

    # 推理参数
    parser.add_argument("--max_source_length", type=int, default=300)
    parser.add_argument("--max_target_length", type=int, default=32)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--batch_size", type=int, default=24)

    args = parser.parse_args()

    # ---------------- Accelerator ----------------
    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with=None,
        project_dir=os.path.dirname(args.output_file)
    )
    accelerator.print(f"Device={accelerator.device}, processes={accelerator.num_processes}")

    # ---------------- Tokenizer & Model ----------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model = accelerator.prepare(model)
    model.eval()

    # ---------------- Data ----------------
    test_examples = load_test_data(args.test_file_path)
    accelerator.print(f"Test samples: {len(test_examples)}")

    PROMPT_TEMPLATE = (
        "{source_code} <SEP>"
    )

    all_predictions = []
    all_references = []
    inference_results = []

    # ---------------- Batch Inference ----------------
    progress_bar = tqdm(
        range(0, len(test_examples), args.batch_size),
        desc="Batch Inference",
        disable=not accelerator.is_local_main_process
    )

    with torch.no_grad():
        for start_idx in progress_bar:
            batch = test_examples[start_idx:start_idx + args.batch_size]

            codes = [ex["code"] for ex in batch]
            refs = [ex["nl_comment"] for ex in batch]

            prompts = [
                PROMPT_TEMPLATE.format(source_code=code)
                for code in codes
            ]

            encodings = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_source_length
            )

            encodings = {k: v.to(accelerator.device) for k, v in encodings.items()}

            generated_ids = model.generate(
                **encodings,
                max_length=args.max_source_length + args.max_target_length,
                do_sample=True,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=args.no_repeat_ngram_size,
                # repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                # top_k=50, top_p=0.8,
                # length_penalty=0.8,  
                # early_stopping=True,
            )

            decoded_texts = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for i, text in enumerate(decoded_texts):
                # if "Natural Language:" in text:
                #     pred = text.split("Natural Language:", 1)[1].strip()
                # if "Description:" in text:
                #     pred = text.split("Description:", 1)[1].strip()
                # if "Natural language:" in text:
                #     pred = text.split("Natural language:", 1)[1].strip()
                
                # else:
                pred = text.strip()

                pred = " ".join(pred.split())
                ref_std = " ".join(refs[i].split())

                all_predictions.append(pred)
                all_references.append(ref_std)

                inference_results.append({
                    "sample_idx": start_idx + i,
                    "code": codes[i],
                    "nl_reference": refs[i],
                    "nl_prediction": pred,
                    "nl_reference_standardized": ref_std
                })

    # ---------------- BLEU ----------------
    bleu = BLEU()
    bleu_score = bleu.corpus_score(
        all_predictions,
        [[r] for r in all_references]
    ).score

    accelerator.print(f"\nBLEU score: {bleu_score:.2f}")

    # ---------------- Save ----------------
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump({
                "model_path": args.model_path,
                "test_file": args.test_file_path,
                "bleu": bleu_score,
                "params": vars(args),
                "results": inference_results
            }, f, ensure_ascii=False, indent=2)

        accelerator.print(f"Saved to {args.output_file}")

    accelerator.print("Inference finished.")


if __name__ == "__main__":
    main()
