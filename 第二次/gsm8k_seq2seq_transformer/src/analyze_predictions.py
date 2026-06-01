import argparse
from collections import Counter
from pathlib import Path

from utils import read_jsonl, write_csv, write_jsonl


def classify(row: dict) -> str:
    if row.get("strict_correct"):
        return "correct"
    if not row.get("has_final_marker"):
        return "missing_final_marker"
    if row.get("pred_final") != row.get("gold_final"):
        return "wrong_final_number"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    out_dir = Path(args.out_dir) if args.out_dir else pred_path.parent
    rows = read_jsonl(pred_path)
    counts = Counter(classify(row) for row in rows)
    total = len(rows)
    table = [
        {
            "error_type": key,
            "count": value,
            "ratio": round(value / total, 4) if total else 0.0,
        }
        for key, value in sorted(counts.items())
    ]
    write_csv(out_dir / "error_analysis.csv", table)

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
                "question": row["question"],
                "gold_final": row["gold_final"],
                "pred_final": row.get("pred_final"),
                "pred_final_fallback": row.get("pred_final_fallback"),
                "prediction": row["prediction"],
            }
        )
    write_jsonl(out_dir / "error_examples.jsonl", examples)
    for item in table:
        print(item)


if __name__ == "__main__":
    main()

