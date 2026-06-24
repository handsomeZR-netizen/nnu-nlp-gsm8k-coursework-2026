import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

DEEPSEEK_PRICE_PER_1M = {
    "DeepSeek_v4_flash": {"cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28},
    "DeepSeek_v4_pro": {"cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87},
}


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half_width = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return center - half_width, center + half_width


def binom_cdf(k: int, n: int) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) for i in range(k + 1)) / (2**n)


def exact_mcnemar_p(pro_only: int, flash_only: int) -> float:
    discordant = pro_only + flash_only
    if discordant == 0:
        return 1.0
    smaller = min(pro_only, flash_only)
    lower = binom_cdf(smaller, discordant)
    upper = 1 - binom_cdf(smaller - 1, discordant)
    return min(1.0, 2 * min(lower, upper))


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def qwen_strict_lenient(qwen_path: Path) -> dict:
    rows = read_jsonl(qwen_path)
    total = len(rows)
    strict = sum(bool(row.get("strict_correct")) for row in rows)
    fallback = sum(bool(row.get("fallback_correct")) for row in rows)
    format_ok = sum(bool(row.get("has_final_marker")) for row in rows)
    normalized = sum(bool(row.get("format_normalized")) for row in rows)
    fallback_only = sum((not row.get("strict_correct")) and bool(row.get("fallback_correct")) for row in rows)
    strict_ci = wilson_ci(strict, total)
    fallback_ci = wilson_ci(fallback, total)
    return {
        "model": "Local_Qwen25_Math_15B_CoT",
        "num_samples": total,
        "strict_correct": strict,
        "strict_acc": strict / total,
        "strict_ci_low": strict_ci[0],
        "strict_ci_high": strict_ci[1],
        "lenient_correct": fallback,
        "lenient_acc": fallback / total,
        "lenient_ci_low": fallback_ci[0],
        "lenient_ci_high": fallback_ci[1],
        "fallback_only_correct": fallback_only,
        "format_ok": format_ok,
        "format_rate": format_ok / total,
        "missing_final_marker": total - format_ok,
        "format_normalized_count": normalized,
    }


def deepseek_mcnemar(flash_path: Path, pro_path: Path) -> dict:
    flash = {row["id"]: row for row in read_jsonl(flash_path)}
    pro = {row["id"]: row for row in read_jsonl(pro_path)}
    common_ids = sorted(set(flash) & set(pro))
    flash_correct = sum(bool(flash[item_id].get("strict_correct")) for item_id in common_ids)
    pro_correct = sum(bool(pro[item_id].get("strict_correct")) for item_id in common_ids)
    both_right = sum(bool(flash[item_id].get("strict_correct")) and bool(pro[item_id].get("strict_correct")) for item_id in common_ids)
    both_wrong = sum((not flash[item_id].get("strict_correct")) and (not pro[item_id].get("strict_correct")) for item_id in common_ids)
    pro_only = sum((not flash[item_id].get("strict_correct")) and bool(pro[item_id].get("strict_correct")) for item_id in common_ids)
    flash_only = sum(bool(flash[item_id].get("strict_correct")) and (not pro[item_id].get("strict_correct")) for item_id in common_ids)
    return {
        "num_samples": len(common_ids),
        "flash_correct": flash_correct,
        "flash_acc": flash_correct / len(common_ids),
        "pro_correct": pro_correct,
        "pro_acc": pro_correct / len(common_ids),
        "acc_diff_pro_minus_flash": (pro_correct - flash_correct) / len(common_ids),
        "both_right": both_right,
        "both_wrong": both_wrong,
        "pro_only_correct": pro_only,
        "flash_only_correct": flash_only,
        "mcnemar_exact_p": exact_mcnemar_p(pro_only, flash_only),
    }


def cost_rows(summary_csv: Path) -> list[dict]:
    rows = []
    for row in read_csv_rows(summary_csv):
        model = row.get("model", "")
        if model not in DEEPSEEK_PRICE_PER_1M:
            continue
        price = DEEPSEEK_PRICE_PER_1M[model]
        cache_hit = float(row["prompt_cache_hit_tokens"])
        cache_miss = float(row["prompt_cache_miss_tokens"])
        completion = float(row["completion_tokens"])
        correct = float(row["final_answer_acc"]) * float(row["num_samples"])
        total_cost = (
            cache_hit * price["cache_hit"]
            + cache_miss * price["cache_miss"]
            + completion * price["output"]
        ) / 1_000_000
        rows.append(
            {
                "model": model,
                "display_name": row["display_name"],
                "accuracy": float(row["final_answer_acc"]),
                "correct_count": correct,
                "prompt_tokens": float(row["prompt_tokens"]),
                "completion_tokens": completion,
                "prompt_cache_hit_tokens": cache_hit,
                "prompt_cache_miss_tokens": cache_miss,
                "prompt_cache_hit_rate": float(row["prompt_cache_hit_rate"]),
                "estimated_cost_usd": total_cost,
                "usd_per_correct": total_cost / correct if correct else 0.0,
            }
        )
    return rows


def error_breakdown(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        model = path.stem.replace("_error_report", "")
        for row in read_csv_rows(path):
            error_type = row["error_type"]
            if error_type == "correct":
                continue
            rows.append(
                {
                    "model": model,
                    "error_type": error_type,
                    "count": int(row["count"]),
                    "ratio": float(row["ratio"]),
                }
            )
    return rows


def deepseek_ci(summary_csv: Path) -> list[dict]:
    rows = []
    for row in read_csv_rows(summary_csv):
        if row["model"] not in {"DeepSeek_v4_flash", "DeepSeek_v4_pro", "Local_Qwen25_Math_15B_CoT"}:
            continue
        total = int(row["num_samples"])
        correct = round(float(row["final_answer_acc"]) * total)
        low, high = wilson_ci(correct, total)
        rows.append(
            {
                "model": row["model"],
                "display_name": row["display_name"],
                "correct": correct,
                "num_samples": total,
                "accuracy": correct / total,
                "wilson_ci_low": low,
                "wilson_ci_high": high,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--out_dir", default="runs/audit_v8")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = ensure_dir(args.out_dir)

    qwen = qwen_strict_lenient(runs_dir / "qwen25_math_15b_few_shot_512" / "test_predictions.jsonl")
    mcnemar = deepseek_mcnemar(
        runs_dir / "deepseek_v4_flash" / "test_predictions.jsonl",
        runs_dir / "deepseek_v4_pro" / "test_predictions.jsonl",
    )
    costs = cost_rows(runs_dir / "second_result_summary_v6.csv")
    ci_rows = deepseek_ci(runs_dir / "second_result_summary_v6.csv")
    errors = error_breakdown(
        [
            runs_dir / "reports_v6" / "qwen25_math_15b_error_report.csv",
            runs_dir / "reports_v6" / "deepseek_v4_flash_error_report.csv",
            runs_dir / "reports_v6" / "deepseek_v4_pro_error_report.csv",
        ]
    )

    write_csv(out_dir / "qwen_strict_lenient.csv", [qwen])
    write_csv(out_dir / "cost_accuracy.csv", costs)
    write_csv(out_dir / "accuracy_confidence_intervals.csv", ci_rows)
    write_csv(out_dir / "error_breakdown.csv", errors)
    (out_dir / "deepseek_stat_tests.json").write_text(json.dumps(mcnemar, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "qwen_strict_lenient": qwen,
        "deepseek_mcnemar": mcnemar,
        "cost_accuracy": costs,
        "accuracy_confidence_intervals": ci_rows,
        "error_breakdown_counts": dict(Counter(row["error_type"] for row in errors)),
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
