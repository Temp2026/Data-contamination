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
    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            if "code" in entry and "comment" in entry:
                java_code = entry["code"].strip()
                nl_comment = entry["comment"].strip()
                if not java_code or not nl_comment:
                    continue
                test_examples.append({
                    "java_code": java_code,
                    "nl_comment": nl_comment
                })
    return test_examples


def main():
    parser = argparse.ArgumentParser(
        description="GPT-2 Java→NL "
    )

    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--test_file_path",
        type=str,
        default="gpt2/data/java2nl.jsonl"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="gpt2/result/java2nl_paired.jsonl"
    )

    
    parser.add_argument("--max_source_length", type=int, default=128)
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
    accelerator.print(
        f"device: {accelerator.device}, process: {accelerator.num_processes}"
    )

    # ---------------- Tokenizer & Model ----------------
    accelerator.print(f"load model and Tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


    for tk in ["<JAVA>", "<SEP>", "<NL>"]:
        tk_id = tokenizer.convert_tokens_to_ids(tk)
        assert tk_id != tokenizer.unk_token_id, f"自定义Token '{tk}' 缺失"
        accelerator.print(f"自定义Token验证通过: '{tk}' -> ID={tk_id}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(accelerator.device)

    model = accelerator.prepare(model)
    model.eval()

    # ---------------- Load test data ----------------
    accelerator.print(f"加载测试集: {args.test_file_path}")
    test_examples = load_test_data(args.test_file_path)
    accelerator.print(f"测试集样本数: {len(test_examples)}")

    # ---------------- Prompt ----------------
    VALID_PROMPT_TEMPLATE = (
        "<JAVA>{source_code}<SEP><NL>"
    )

    # ---------------- Batch Inference ----------------
    all_predictions = []
    all_references = []
    inference_results = []

    batch_size = args.batch_size

    with torch.no_grad():
        for start_idx in tqdm(
            range(0, len(test_examples), batch_size),
            desc="Java→NL",
            ncols=100,
            disable=not accelerator.is_local_main_process
        ):
            batch_examples = test_examples[start_idx:start_idx + batch_size]

            prompts = [
                VALID_PROMPT_TEMPLATE.format(
                    source_code=ex["java_code"]
                )
                for ex in batch_examples
            ]

            batch_encoding = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_source_length
            )

            input_ids = batch_encoding.input_ids.to(accelerator.device)
            attention_mask = batch_encoding.attention_mask.to(accelerator.device)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=args.max_source_length + args.max_target_length,
                do_sample=True,
                num_beams=args.num_beams,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                top_k=50, top_p=0.8,
                length_penalty=0.8,  
                early_stopping=True,
            )

            decoded_texts = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for i, pred_text in enumerate(decoded_texts):
                global_idx = start_idx + i
                example = batch_examples[i]
                nl_ref = example["nl_comment"]

                if "Natural Language:" in pred_text:
                    pred_nl = pred_text.split(
                        "Natural Language:", 1
                    )[1].strip()
                else:
                    pred_nl = pred_text.strip()

                pred_nl = " ".join(pred_nl.split())
                nl_ref_std = " ".join(nl_ref.split())

                all_predictions.append(pred_nl)
                all_references.append(nl_ref_std)

                inference_results.append({
                    "sample_idx": global_idx,
                    "java_code": example["java_code"],
                    "nl_reference": nl_ref,
                    "nl_prediction": pred_nl,
                    "nl_reference_standardized": nl_ref_std
                })

    # ---------------- BLEU ----------------
    bleu = BLEU()
    references = [[ref] for ref in all_references]
    bleu_score = bleu.corpus_score(all_predictions, references).score
    accelerator.print(f"\n测试集 BLEU 分数: {bleu_score:.2f}")

    # ---------------- Save ----------------
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump({
                "test_file_path": args.test_file_path,
                "model_path": args.model_path,
                "bleu_score": bleu_score,
                "inference_params": vars(args),
                "inference_results": inference_results
            }, f, ensure_ascii=False, indent=2)

        accelerator.print(f"推理结果已保存至: {args.output_file}")

    accelerator.print("")


if __name__ == "__main__":
    main()
