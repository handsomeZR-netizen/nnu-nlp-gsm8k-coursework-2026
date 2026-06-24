import argparse
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
FINAL_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}|boxed\s*\{([^{}]+)\}", re.IGNORECASE)
PROMPT_SYSTEM = (
    "You solve grade-school math word problems. Use concise arithmetic only. "
    "Do not write Python or code. Always end with a final line exactly as: #### number"
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_answer(value):
    if value is None:
        return None
    cleaned = str(value).replace(",", "").strip()
    if not cleaned:
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return cleaned
    if number.is_integer():
        return str(int(number))
    return str(number).rstrip("0").rstrip(".")


def extract_number(text):
    if not text:
        return None
    match = NUMBER_RE.search(text)
    return normalize_answer(match.group(0)) if match else None


def extract_final_answer(text: str, fallback: bool = False):
    matches = FINAL_RE.findall(text or "")
    if matches:
        return normalize_answer(matches[-1])
    if not fallback:
        return None
    boxed = BOXED_RE.findall(text or "")
    for match in reversed(boxed):
        number = extract_number(match[0] or match[1])
        if number is not None:
            return number
    numbers = NUMBER_RE.findall(text or "")
    return normalize_answer(numbers[-1]) if numbers else None


def standardize_prediction(raw_text: str) -> tuple[str, bool]:
    if extract_final_answer(raw_text, fallback=False) is not None:
        return raw_text, False
    fallback = extract_final_answer(raw_text, fallback=True)
    if fallback is None:
        return raw_text, False
    return f"{raw_text.rstrip()}\n#### {fallback}", True


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": f"Question: {question}\nAnswer:"},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{PROMPT_SYSTEM}\n\nQuestion: {question}\nAnswer:"


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_batch(model, tokenizer, questions: list[str], args) -> list[str]:
    prompts = [build_prompt(tokenizer, question) for question in questions]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_input_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    output_ids = model.generate(
        **encoded,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_len = encoded["input_ids"].shape[-1]
    return [tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip() for row in output_ids]


def make_prediction(row: dict, raw_text: str, model_name: str) -> dict:
    prediction, normalized = standardize_prediction(raw_text)
    pred_final = extract_final_answer(prediction, fallback=False)
    pred_fallback = extract_final_answer(prediction, fallback=True)
    gold_final = row["final_answer"]
    return {
        "id": row["id"],
        "question": row["question"],
        "gold": row["answer"],
        "prediction": prediction,
        "raw_prediction": raw_text,
        "model_name": model_name,
        "gold_final": gold_final,
        "pred_final": pred_final,
        "pred_final_fallback": pred_fallback,
        "strict_correct": pred_final == gold_final,
        "fallback_correct": pred_fallback == gold_final,
        "has_final_marker": pred_final is not None,
        "format_normalized": normalized,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--adapter_dir", default="runs/qwen25_math_15b_lora/adapter")
    parser.add_argument("--data_path", default="data/test.jsonl")
    parser.add_argument("--out_dir", default="runs/qwen25_math_15b_lora")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_input_len", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(args.data_path))
    if args.limit:
        rows = rows[: args.limit]
    tokenizer, model = load_model(args)

    outputs = []
    started = time.time()
    for index in tqdm(range(0, len(rows), args.batch_size), desc="eval-lora"):
        batch = rows[index : index + args.batch_size]
        raw_texts = generate_batch(model, tokenizer, [row["question"] for row in batch], args)
        for row, raw_text in zip(batch, raw_texts):
            outputs.append(make_prediction(row, raw_text, "Qwen2.5-Math-1.5B-LoRA"))
    write_jsonl(out_dir / "test_predictions.jsonl", outputs)
    total = len(outputs)
    summary = {
        "run_name": Path(args.out_dir).name,
        "model_name": args.model_name,
        "adapter_dir": args.adapter_dir,
        "num_samples": total,
        "strict_final_answer_acc": sum(row["strict_correct"] for row in outputs) / max(1, total),
        "fallback_final_answer_acc": sum(row["fallback_correct"] for row in outputs) / max(1, total),
        "format_rate": sum(row["has_final_marker"] for row in outputs) / max(1, total),
        "format_normalized_count": sum(row["format_normalized"] for row in outputs),
        "elapsed_sec": round(time.time() - started, 3),
    }
    if torch.cuda.is_available():
        summary["cuda_max_memory_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    (out_dir / "test_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
