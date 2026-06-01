import argparse
from collections import Counter
from pathlib import Path

from compute_metrics import equation_stats
from utils import ensure_dir, extract_final_answer, normalize_answer, read_jsonl, write_csv, write_jsonl


def classify(row: dict) -> str:
    prediction = row.get("prediction", "")
    pred_final = normalize_answer(row.get("pred_final")) or extract_final_answer(prediction, fallback=False)
    pred_fallback = normalize_answer(row.get("pred_final_fallback")) or extract_final_answer(prediction, fallback=True)
    gold_final = normalize_answer(row.get("gold_final"))
    if pred_final == gold_final:
        return "correct"
    if pred_final is None:
        if pred_fallback == gold_final:
            return "format_missing_but_fallback_correct"
        return "missing_final_marker"
    eq = equation_stats(prediction)
    if eq["total"] == 0:
        return "wrong_final_no_equation"
    if eq["correct"] < eq["total"]:
        return "wrong_final_with_bad_equation"
    return "wrong_final_equations_self_consistent"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else pred_path.parent)
    prefix = f"{args.prefix}_" if args.prefix else ""
    rows = read_jsonl(pred_path)
    counts = Counter(classify(row) for row in rows)
    total = len(rows)
    table = [{"error_type": key, "count": value, "ratio": round(value / max(1, total), 6)} for key, value in sorted(counts.items())]
    write_csv(out_dir / f"{prefix}error_report.csv", table)

    examples = []
    seen = set()
    for row in rows:
        kind = classify(row)
        if kind in seen:
            continue
        seen.add(kind)
        examples.append(
            {
                "error_type": kind,
                "id": row["id"],
                "question": row.get("question", ""),
                "gold_final": row.get("gold_final"),
                "pred_final": row.get("pred_final"),
                "pred_final_fallback": row.get("pred_final_fallback"),
                "prediction": row.get("prediction", ""),
            }
        )
    write_jsonl(out_dir / f"{prefix}error_examples.jsonl", examples)
    for item in table:
        print(item)


if __name__ == "__main__":
    main()
