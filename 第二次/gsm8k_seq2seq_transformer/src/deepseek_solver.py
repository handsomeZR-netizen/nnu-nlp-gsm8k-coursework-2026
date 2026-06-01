import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import threading
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

from llm_judge import normalize_url
from utils import ensure_dir, extract_final_answer, read_jsonl, write_jsonl

THREAD_LOCAL = threading.local()

CACHE_FRIENDLY_SYSTEM = (
    "You are a precise GSM8K math solver. Solve arithmetic word problems carefully, "
    "keep the reasoning concise, and always end with exactly one final answer line."
)

CACHE_FRIENDLY_USER_PREFIX = """Task: solve grade-school math word problems.

Output rules:
- Write concise arithmetic reasoning.
- Do not use LaTeX, boxed notation, commas, currency symbols, or units in the final line.
- If the answer is a whole number, write it as an integer, not as a decimal.
- The final line must be exactly: #### <number>

Few-shot examples:

Question: A store has 12 pencils. It buys 8 more pencils and then sells 5. How many pencils are left?
Answer:
The store has 12 + 8 = 20 pencils after buying more.
After selling 5, it has 20 - 5 = 15 pencils left.
#### 15

Question: Mia saved 6 dollars each week for 4 weeks. She then spent 7 dollars. How many dollars does she have left?
Answer:
Mia saved 6 * 4 = 24 dollars.
After spending 7, she has 24 - 7 = 17 dollars left.
#### 17

Question: A box has 3 red balls and twice as many blue balls. How many balls are in the box?
Answer:
There are 3 * 2 = 6 blue balls.
The total number of balls is 3 + 6 = 9.
#### 9

Now solve the target problem using the same format.
Question: """

CONCISE_USER_PREFIX = (
    "Solve the grade-school math word problem. Show concise step-by-step arithmetic, "
    "and make the final line exactly '#### number'. Do not output unrelated text.\n\nQuestion: "
)


def get_config_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    if os.name != "nt":
        return default
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
            return value or default
    except OSError:
        return default


def get_session(pool_size: int) -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, pool_block=True)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        THREAD_LOCAL.session = session
    return session


def build_messages(question: str, prompt_style: str) -> list[dict]:
    if prompt_style == "cache_few_shot":
        user_content = f"{CACHE_FRIENDLY_USER_PREFIX}{question}\nAnswer:"
    elif prompt_style == "concise":
        user_content = f"{CONCISE_USER_PREFIX}{question}\nAnswer:"
    else:
        raise ValueError(f"Unknown prompt style: {prompt_style}")
    return [
        {"role": "system", "content": CACHE_FRIENDLY_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def call_model(
    messages: list[dict],
    api_key: str,
    model: str,
    url: str,
    timeout: int,
    retries: int,
    temperature: float,
    max_tokens: int,
    pool_size: int,
) -> tuple[str, dict]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            start = time.perf_counter()
            resp = get_session(pool_size).post(url, headers=headers, json=payload, timeout=timeout)
            latency = time.perf_counter() - start
            if resp.status_code >= 400:
                raise RuntimeError(f"{resp.status_code} {resp.text[:500]}")
            data = resp.json()
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            if not content.strip():
                content = message.get("reasoning_content") or ""
            if not content.strip():
                raise RuntimeError("empty model response")
            usage = data.get("usage") or {}
            usage["latency_seconds"] = round(latency, 3)
            return content, usage
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                delay = min(60, 2**attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
    raise RuntimeError(f"DeepSeek solver failed after {retries} attempts: {last_error}")


def make_prediction(row: dict, prediction: str, model_name: str, usage: dict | None = None) -> dict:
    pred_final = extract_final_answer(prediction, fallback=False)
    pred_fallback = extract_final_answer(prediction, fallback=True)
    gold_final = row.get("final_answer")
    output = {
        "id": row["id"],
        "question": row["question"],
        "gold": row["answer"],
        "prediction": prediction,
        "model_name": model_name,
        "gold_final": gold_final,
        "pred_final": pred_final,
        "pred_final_fallback": pred_fallback,
        "strict_correct": pred_final == gold_final,
        "fallback_correct": pred_fallback == gold_final,
        "has_final_marker": pred_final is not None,
    }
    if usage:
        output.update(
            {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
                "prompt_cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
                "latency_seconds": usage.get("latency_seconds"),
            }
        )
    return output


def solve_row(row: dict, api_key: str, model: str, url: str, args) -> dict:
    last_text = ""
    total_usage: dict | None = None
    for attempt in range(args.format_retries + 1):
        text, usage = call_model(
            build_messages(row["question"], args.prompt_style),
            api_key,
            model,
            url,
            args.timeout,
            args.retries,
            args.temperature,
            args.max_tokens,
            max(1, args.workers),
        )
        total_usage = add_usage(total_usage, usage)
        text = repair_final_line(text, args.repair_final_line)
        last_text = text
        if extract_final_answer(text, fallback=False) is not None:
            break
        if attempt < args.format_retries:
            time.sleep(1 + random.uniform(0, 0.5))
    return make_prediction(row, last_text, model, total_usage)


def sort_like_dataset(predictions: list[dict], order: dict[str, int]) -> list[dict]:
    return sorted(predictions, key=lambda row: order.get(row["id"], 10**12))


def add_usage(left: dict | None, right: dict | None) -> dict:
    total = dict(left or {})
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "latency_seconds",
    ):
        value = (right or {}).get(key)
        if value is not None:
            total[key] = (total.get(key) or 0) + value
    return total


def repair_final_line(text: str, enabled: bool) -> str:
    if not enabled or extract_final_answer(text, fallback=False) is not None:
        return text
    fallback = extract_final_answer(text, fallback=True)
    if fallback is None:
        return text
    return f"{text.rstrip()}\n#### {fallback}"


def write_progress(out_dir: Path, pred_path: Path, predictions: list[dict], args, model: str) -> None:
    total_prompt = sum(row.get("prompt_tokens") or 0 for row in predictions)
    total_completion = sum(row.get("completion_tokens") or 0 for row in predictions)
    total_cache_hit = sum(row.get("prompt_cache_hit_tokens") or 0 for row in predictions)
    total_cache_miss = sum(row.get("prompt_cache_miss_tokens") or 0 for row in predictions)
    cache_denominator = total_cache_hit + total_cache_miss
    summary = {
        "model": model,
        "prediction_file": str(pred_path),
        "prompt_style": args.prompt_style,
        "workers": args.workers,
        "max_tokens": args.max_tokens,
        "num_samples": len(predictions),
        "strict_final_answer_acc": sum(row["strict_correct"] for row in predictions) / max(1, len(predictions)),
        "fallback_final_answer_acc": sum(row["fallback_correct"] for row in predictions) / max(1, len(predictions)),
        "format_rate": sum(row["has_final_marker"] for row in predictions) / max(1, len(predictions)),
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "prompt_cache_hit_tokens": total_cache_hit,
        "prompt_cache_miss_tokens": total_cache_miss,
        "prompt_cache_hit_rate": total_cache_hit / cache_denominator if cache_denominator else None,
    }
    (out_dir / "progress.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", default="data/test.jsonl")
    parser.add_argument("--out_dir", default="runs/deepseek_solver")
    parser.add_argument("--prediction_file", default="test_predictions.jsonl")
    parser.add_argument("--model", default=get_config_env("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--base_url", default=get_config_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N rows for dry-run validation.")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--prompt_style", choices=["cache_few_shot", "concise"], default="cache_few_shot")
    parser.add_argument("--cache_warmup", type=int, default=2, help="Sequential requests before high-concurrency fan-out.")
    parser.add_argument("--cache_warmup_delay", type=float, default=3.0, help="Seconds to wait after warmup so the common prefix can be persisted.")
    parser.add_argument("--format_retries", type=int, default=1, help="Retry once if no final answer can be parsed.")
    parser.add_argument("--repair_final_line", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    api_key = get_config_env("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set.")
    model = args.model
    url = normalize_url(args.base_url)
    out_dir = ensure_dir(args.out_dir)
    pred_path = Path(out_dir) / args.prediction_file

    rows = read_jsonl(args.test_data)
    if args.limit is not None:
        rows = rows[: args.limit]
    order = {row["id"]: idx for idx, row in enumerate(rows)}
    cached = read_jsonl(pred_path) if pred_path.exists() else []
    done = {row["id"] for row in cached}
    predictions = cached[:]
    pending_rows = [row for row in rows if row["id"] not in done]

    warmup_count = min(args.cache_warmup, len(pending_rows)) if args.workers > 1 and not cached else 0
    if warmup_count:
        warmup_rows = pending_rows[:warmup_count]
        for row in warmup_rows:
            predictions.append(solve_row(row, api_key, model, url, args))
            predictions = sort_like_dataset(predictions, order)
            write_jsonl(pred_path, predictions)
            write_progress(out_dir, pred_path, predictions, args, model)
        pending_rows = pending_rows[warmup_count:]
        if args.cache_warmup_delay > 0:
            time.sleep(args.cache_warmup_delay)

    if args.workers <= 1:
        for row in pending_rows:
            predictions.append(solve_row(row, api_key, model, url, args))
            if len(predictions) % args.save_every == 0:
                predictions = sort_like_dataset(predictions, order)
                write_jsonl(pred_path, predictions)
                write_progress(out_dir, pred_path, predictions, args, model)
            time.sleep(args.sleep)
    else:
        completed_since_save = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_id = {executor.submit(solve_row, row, api_key, model, url, args): row["id"] for row in pending_rows}
            for future in as_completed(future_to_id):
                try:
                    predictions.append(future.result())
                except Exception as exc:
                    item_id = future_to_id[future]
                    write_jsonl(pred_path, sort_like_dataset(predictions, order))
                    raise RuntimeError(f"Failed on {item_id} with model {model}: {exc}") from exc
                completed_since_save += 1
                if completed_since_save >= args.save_every:
                    predictions = sort_like_dataset(predictions, order)
                    write_jsonl(pred_path, predictions)
                    write_progress(out_dir, pred_path, predictions, args, model)
                    completed_since_save = 0
                if args.sleep:
                    time.sleep(args.sleep)

    predictions = sort_like_dataset(predictions, order)
    write_jsonl(pred_path, predictions)

    summary = {
        "model": model,
        "base_url": args.base_url,
        "prediction_file": args.prediction_file,
        "prompt_style": args.prompt_style,
        "workers": args.workers,
        "max_tokens": args.max_tokens,
        "num_samples": len(predictions),
        "strict_final_answer_acc": sum(row["strict_correct"] for row in predictions) / max(1, len(predictions)),
        "fallback_final_answer_acc": sum(row["fallback_correct"] for row in predictions) / max(1, len(predictions)),
        "format_rate": sum(row["has_final_marker"] for row in predictions) / max(1, len(predictions)),
    }
    total_prompt = sum(row.get("prompt_tokens") or 0 for row in predictions)
    total_completion = sum(row.get("completion_tokens") or 0 for row in predictions)
    total_cache_hit = sum(row.get("prompt_cache_hit_tokens") or 0 for row in predictions)
    total_cache_miss = sum(row.get("prompt_cache_miss_tokens") or 0 for row in predictions)
    cache_denominator = total_cache_hit + total_cache_miss
    summary.update(
        {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "prompt_cache_hit_tokens": total_cache_hit,
            "prompt_cache_miss_tokens": total_cache_miss,
            "prompt_cache_hit_rate": total_cache_hit / cache_denominator if cache_denominator else None,
        }
    )
    (Path(out_dir) / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
