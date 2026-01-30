import torch
import argparse
import os
import json
import pandas as pd
from tqdm import tqdm
from sacrebleu.metrics import BLEU
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator


def load_test_data_jsonl(test_file_path, python_col="python", java_col="java"):
    test_examples = []

    with open(test_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            py = obj.get(python_col, "").strip()
            ja = obj.get(java_col, "").strip()
            if py and ja:
                test_examples.append({
                    "python_code": py,
                    "java_code": ja
                })
    return test_examples



def main():
    parser = argparse.ArgumentParser(
        description="GPT-2 Python→Java "
    )

    # 核心参数
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--test_file_path",
        type=str,
        default="/gpt2/data/python2java.jsonl"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/gpt2/result/python2java_paired.jsonl"
    )
    parser.add_argument("--python_col", type=str, default="python")
    parser.add_argument("--java_col", type=str, default="java")

   
    parser.add_argument("--max_source_length", type=int, default=320)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)

   
    parser.add_argument("--batch_size", type=int, default=24)

    args = parser.parse_args()

    # ---------------- Accelerator ----------------
    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with=None,
        project_dir=os.path.dirname(args.output_file)
    )
    accelerator.print(
        f"推理设备: {accelerator.device}, 进程数: {accelerator.num_processes}"
    )

    # ---------------- Tokenizer & Model ----------------
    accelerator.print(f"加载模型和Tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    
    for tk in ["<PYTHON>", "<SEP>", "<JAVA>"]:
        tk_id = tokenizer.convert_tokens_to_ids(tk)
        assert tk_id != tokenizer.unk_token_id, f"自定义Token '{tk}' 缺失"
        accelerator.print(f"自定义Token验证通过: {tk} -> {tk_id}")

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
    test_examples = load_test_data_jsonl(
        args.test_file_path,
        python_col=args.python_col,
        java_col=args.java_col
    )
    accelerator.print(f"测试样本数: {len(test_examples)}")

    # ---------------- Prompt ----------------
    VALID_PROMPT_TEMPLATE = (
        "python: {source_code} <SEP> Java:"
    )

    # ---------------- Inference (BATCH) ----------------
    all_predictions = []
    all_references = []
    inference_results = []

    batch_size = args.batch_size

    with torch.no_grad():
        for start_idx in tqdm(
            range(0, len(test_examples), batch_size),
            desc="Python→Java 批量推理中",
            ncols=100,
            disable=not accelerator.is_local_main_process
        ):
            batch_examples = test_examples[start_idx:start_idx + batch_size]

            prompts = [
                VALID_PROMPT_TEMPLATE.format(
                    source_code=ex["python_code"]
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
                # no_repeat_ngram_size=args.no_repeat_ngram_size,
                # repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

            decoded_texts = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for i, pred_text in enumerate(decoded_texts):
                global_idx = start_idx + i
                example = batch_examples[i]
                java_ref = example["java_code"]

                if "Java:" in pred_text:
                    pred_java = pred_text.split("Java:", 1)[1].strip()
                else:
                    pred_java = pred_text.strip()

                pred_java = " ".join(pred_java.split())
                java_ref_std = " ".join(java_ref.split())

                all_predictions.append(pred_java)
                all_references.append(java_ref_std)

                inference_results.append({
                    "sample_idx": global_idx,
                    "python_code": example["python_code"],
                    "java_reference": java_ref,
                    "java_prediction": pred_java,
                    "java_reference_standardized": java_ref_std
                })

    # ---------------- BLEU ----------------
    bleu = BLEU()
    references = [[ref] for ref in all_references]
    bleu_score = bleu.corpus_score(all_predictions, references).score
    accelerator.print(f"\n测试集 BLEU: {bleu_score:.2f}")

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

        accelerator.print(f"结果已保存至: {args.output_file}")

    accelerator.print("")


if __name__ == "__main__":
    main()
