import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import re
import time
from pathlib import Path

import requests

from utils import ensure_dir, read_jsonl, write_jsonl


JSON_RE = re.compile(r"\{.*\}", re.S)


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


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=prediction_path.jsonl")
    name, path = value.split("=", 1)
    return name, Path(path)


def normalize_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


def build_prompt(row: dict, model_name: str) -> str:
    return (
        "你是数学应用题推理评估员。请只根据题目、标准答案和模型输出评分。\n"
        "评分标准：2=推理过程基本正确且最终答案正确；1=有部分合理推理，但存在计算、逻辑或最终答案错误；0=无有效推理、严重偏题或只有错误答案。\n"
        "请严格输出 JSON，不要输出额外文字。格式为：\n"
        '{"score": 0, "reason": "一句中文理由", "final_answer_consistent": false}\n\n'
        f"模型名称：{model_name}\n"
        f"题目：{row['question']}\n"
        f"标准答案：{row['gold']}\n"
        f"模型输出：{row.get('prediction', '')}\n"
    )


def extract_json(text: str) -> dict:
    match = JSON_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object in response: {text[:200]}")
    data = json.loads(match.group(0))
    score = int(data["score"])
    if score not in {0, 1, 2}:
        raise ValueError(f"Invalid score: {score}")
    return {
        "score": score,
        "reason": str(data.get("reason", ""))[:500],
        "final_answer_consistent": bool(data.get("final_answer_consistent", False)),
    }


def call_deepseek(prompt: str, api_key: str, model: str, url: str, timeout: int, retries: int) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你必须只输出可解析的 JSON。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code >= 400:
                raise RuntimeError(f"{resp.status_code} {resp.text[:500]}")
            content = resp.json()["choices"][0]["message"]["content"]
            return extract_json(content)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(10, 2 * attempt))
    raise RuntimeError(f"DeepSeek request failed after {retries} attempts: {last_error}")


def judge_one(
    model_name: str,
    item_id: str,
    pred_map: dict[str, dict],
    test_by_id: dict[str, dict],
    api_key: str,
    judge_model: str,
    url: str,
    timeout: int,
    retries: int,
) -> dict:
    if item_id not in pred_map:
        raise RuntimeError(f"Missing prediction for {model_name}: {item_id}")
    base = test_by_id[item_id]
    row = {**pred_map[item_id], "question": base["question"], "gold": base["answer"]}
    result = call_deepseek(build_prompt(row, model_name), api_key, judge_model, url, timeout, retries)
    return {"model": model_name, "id": item_id, **result}


def load_prediction_maps(model_args: list[tuple[str, Path]]) -> list[tuple[str, dict[str, dict]]]:
    return [(name, {row["id"]: row for row in read_jsonl(path)}) for name, path in model_args]


def get_or_create_sample_ids(test_rows: list[dict], sample_path: Path, sample_size: int, seed: int) -> list[str]:
    if sample_path.exists():
        return [row["id"] for row in read_jsonl(sample_path)]
    rng = random.Random(seed)
    rows = test_rows[:]
    rng.shuffle(rows)
    sampled = rows[:sample_size]
    write_jsonl(sample_path, [{"id": row["id"], "question": row["question"], "gold": row["answer"], "gold_final": row.get("final_answer")} for row in sampled])
    return [row["id"] for row in sampled]


def summarize(judgments: list[dict]) -> list[dict]:
    by_model: dict[str, list[dict]] = {}
    for row in judgments:
        by_model.setdefault(row["model"], []).append(row)
    summaries = []
    for model_name, rows in sorted(by_model.items()):
        total = len(rows)
        avg_score = sum(row["score"] for row in rows) / max(1, total)
        summaries.append(
            {
                "model": model_name,
                "num_samples": total,
                "avg_score_0_to_2": round(avg_score, 6),
                "llm_reasoning_validity": round(avg_score / 2, 6),
                "score_2_rate": round(sum(row["score"] == 2 for row in rows) / max(1, total), 6),
                "score_1_rate": round(sum(row["score"] == 1 for row in rows) / max(1, total), 6),
                "score_0_rate": round(sum(row["score"] == 0 for row in rows) / max(1, total), 6),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", default="data/test.jsonl")
    parser.add_argument("--predictions", action="append", type=parse_model_arg, required=True)
    parser.add_argument("--out_dir", default="runs/llm_judge")
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional small run for API validation.")
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--model", default=get_config_env("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--base_url", default=get_config_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()

    api_key = get_config_env("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set.")
    judge_model = args.model
    url = normalize_url(args.base_url)

    out_dir = ensure_dir(args.out_dir)
    test_rows = read_jsonl(args.test_data)
    test_by_id = {row["id"]: row for row in test_rows}
    sample_path = Path(out_dir) / f"eval_sample_{args.sample_size}_seed{args.seed}.jsonl"
    sample_ids = get_or_create_sample_ids(test_rows, sample_path, args.sample_size, args.seed)
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]

    pred_maps = load_prediction_maps(args.predictions)
    cache_path = Path(out_dir) / "deepseek_judgments.jsonl"
    cached = read_jsonl(cache_path) if cache_path.exists() else []
    seen = {(row["model"], row["id"]) for row in cached}
    judgments = cached[:]

    tasks = []
    for model_name, pred_map in pred_maps:
        for item_id in sample_ids:
            key = (model_name, item_id)
            if key not in seen:
                tasks.append((model_name, item_id, pred_map))

    completed_since_save = 0
    if args.workers <= 1:
        for model_name, item_id, pred_map in tasks:
            judgments.append(judge_one(model_name, item_id, pred_map, test_by_id, api_key, judge_model, url, args.timeout, args.retries))
            completed_since_save += 1
            if completed_since_save >= args.save_every:
                write_jsonl(cache_path, judgments)
                completed_since_save = 0
            if args.sleep:
                time.sleep(args.sleep)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(judge_one, model_name, item_id, pred_map, test_by_id, api_key, judge_model, url, args.timeout, args.retries)
                for model_name, item_id, pred_map in tasks
            ]
            for future in as_completed(futures):
                judgments.append(future.result())
                completed_since_save += 1
                if completed_since_save >= args.save_every:
                    write_jsonl(cache_path, judgments)
                    completed_since_save = 0

    write_jsonl(cache_path, judgments)

    summaries = summarize([row for row in judgments if row["id"] in set(sample_ids)])
    summary_path = Path(out_dir) / "llm_judge_summary.json"
    csv_path = Path(out_dir) / "llm_judge_summary.csv"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    from utils import write_csv

    write_csv(csv_path, summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
