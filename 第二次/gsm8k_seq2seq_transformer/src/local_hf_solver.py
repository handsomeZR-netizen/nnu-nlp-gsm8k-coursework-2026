import argparse
import json
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import ensure_dir, extract_final_answer, normalize_answer, prediction_metrics, read_jsonl, set_seed, write_jsonl

NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}|boxed\s*\{([^{}]+)\}", re.IGNORECASE)
PLAIN_NUMBER_RE = re.compile(r"^\s*[-+]?\d[\d,]*(?:\.\d+)?\s*\.?\s*$")
ANSWER_CUE_RE = re.compile(
    r"(?:final answer|answer is|answer:|therefore|thus|so)[^\n\r]*?([-+]?\d[\d,]*(?:\.\d+)?)\s*\.?\s*$",
    re.IGNORECASE,
)


def choose_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def build_messages(question: str, prompt_style: str) -> list[dict]:
    if prompt_style == "direct":
        system = (
            "You are a careful grade-school math solver. Compute the answer internally. "
            "Reply with only one line in this exact format: #### number"
        )
        user = question
    elif prompt_style == "short_cot":
        system = (
            "Solve the math word problem with concise arithmetic. Use no more than four short lines. "
            "Do not write code or Python. End with a final line exactly as: #### number"
        )
        user = question
    elif prompt_style == "few_shot":
        system = (
            "You solve grade-school math word problems. Use concise arithmetic only. "
            "Do not write Python or code. Always end with a final line exactly as: #### number"
        )
        user = (
            "Example:\n"
            "Question: Mia has 4 pencils and buys 3 more pencils. How many pencils does she have?\n"
            "Answer:\n"
            "Mia has 4 + 3 = 7 pencils.\n"
            "#### 7\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
    else:
        system = (
            "Please reason step by step. Put the final numeric answer within \\boxed{}, "
            "and then write the final line exactly as: #### number."
        )
        user = question
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def render_prompt(tokenizer, question: str, prompt_style: str) -> str:
    if prompt_style == "answer_prefix":
        return (
            "Solve the grade-school math problem. Compute carefully, but output only the final numeric answer.\n"
            f"Question: {question}\n"
            "Answer: ####"
        )
    messages = build_messages(question, prompt_style)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    system = messages[0]["content"]
    return f"{system}\n\nQuestion: {question}\nAnswer:"


def number_from_text(text: str | None) -> str | None:
    if not text:
        return None
    match = NUMBER_RE.search(text)
    if not match:
        return None
    return normalize_answer(match.group(0))


def extract_boxed_answer(text: str) -> str | None:
    matches = BOXED_RE.findall(text or "")
    for match in reversed(matches):
        content = match[0] or match[1]
        number = number_from_text(content)
        if number is not None:
            return number
    return None


def infer_final_answer(raw_text: str) -> str | None:
    strict = extract_final_answer(raw_text, fallback=False)
    if strict is not None:
        return strict
    boxed = extract_boxed_answer(raw_text)
    if boxed is not None:
        return boxed
    if PLAIN_NUMBER_RE.fullmatch(raw_text or ""):
        return number_from_text(raw_text)
    stripped = (raw_text or "").strip()
    if "\n" not in stripped and len(stripped) <= 80:
        short_number = number_from_text(stripped)
        if short_number is not None:
            return short_number
    cue = ANSWER_CUE_RE.search(stripped)
    if cue:
        return normalize_answer(cue.group(1))
    return None


def standardize_prediction(raw_text: str) -> tuple[str, bool]:
    if extract_final_answer(raw_text, fallback=False) is not None:
        return raw_text, False
    final_answer = infer_final_answer(raw_text)
    if final_answer is None:
        return raw_text, False
    return f"{raw_text.rstrip()}\n#### {final_answer}", True


def make_prediction(row: dict, raw_text: str, model_name: str) -> dict:
    prediction, format_normalized = standardize_prediction(raw_text)
    pred_final = extract_final_answer(prediction, fallback=False)
    pred_fallback = extract_final_answer(prediction, fallback=True)
    gold_final = row.get("final_answer")
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
        "format_normalized": format_normalized,
    }


def load_cached(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["id"]: row for row in read_jsonl(path)}


def load_model(args):
    dtype = choose_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if torch.cuda.is_available() and args.max_gpu_memory:
        kwargs["max_memory"] = {0: args.max_gpu_memory, "cpu": args.max_cpu_memory}
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model, dtype


@torch.inference_mode()
def generate_batch(model, tokenizer, questions: list[str], args) -> list[str]:
    prompts = [render_prompt(tokenizer, question, args.prompt_style) for question in questions]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_len,
    )
    input_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoded = {k: v.to(input_device) for k, v in encoded.items()}
    output_ids = model.generate(
        **encoded,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_len = encoded["input_ids"].shape[-1]
    return [tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip() for row in output_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--data_path", default="data/test.jsonl")
    parser.add_argument("--out_dir", default="runs/qwen25_math_15b_cot")
    parser.add_argument("--prediction_file", default="test_predictions.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_input_len", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--prompt_style", choices=["answer_prefix", "direct", "short_cot", "few_shot", "boxed_cot"], default="answer_prefix")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_gpu_memory", default="7GiB")
    parser.add_argument("--max_cpu_memory", default="48GiB")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--empty_cache_every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    pred_path = Path(out_dir) / args.prediction_file
    rows = read_jsonl(args.data_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    cached = {} if args.overwrite else load_cached(pred_path)
    tokenizer, model, dtype = load_model(args)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    outputs = []
    started = time.time()
    progress = tqdm(total=len(rows), desc=Path(args.prediction_file).stem)
    batch_rows = []

    def flush_batch() -> None:
        nonlocal batch_rows
        if not batch_rows:
            return
        raw_texts = generate_batch(model, tokenizer, [item["question"] for item in batch_rows], args)
        for item, raw_text in zip(batch_rows, raw_texts):
            pred = make_prediction(item, raw_text, args.model_name)
            cached[item["id"]] = pred
            outputs.append(pred)
            progress.update(1)
        batch_rows = []
        write_jsonl(pred_path, outputs)
        if args.sleep:
            time.sleep(args.sleep)
        if torch.cuda.is_available() and args.empty_cache_every > 0 and len(outputs) % args.empty_cache_every == 0:
            torch.cuda.empty_cache()

    for row in rows:
        if row["id"] in cached:
            outputs.append(cached[row["id"]])
            progress.update(1)
            continue
        batch_rows.append(row)
        if len(batch_rows) >= max(1, args.batch_size):
            flush_batch()
    flush_batch()
    progress.close()

    write_jsonl(pred_path, outputs)
    summary = {
        "run_name": Path(args.out_dir).name,
        "model_name": args.model_name,
        "data_path": args.data_path,
        "prediction_file": args.prediction_file,
        "dtype": str(dtype).replace("torch.", ""),
        "prompt_style": args.prompt_style,
        "batch_size": args.batch_size,
        "elapsed_sec": round(time.time() - started, 3),
        "format_normalized_count": sum(row.get("format_normalized", False) for row in outputs),
        **prediction_metrics(outputs),
    }
    if torch.cuda.is_available():
        summary["cuda_max_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        summary["cuda_max_memory_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    summary_name = Path(args.prediction_file).stem.replace("_predictions", "")
    (Path(out_dir) / f"{summary_name}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
