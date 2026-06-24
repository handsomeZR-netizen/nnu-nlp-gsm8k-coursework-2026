import csv
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out_dir = Path("runs/gpt55_judge_v1")
    paths = sorted(out_dir.glob("judgments_batch_*.jsonl"))
    judgments = []
    for path in paths:
        judgments.extend(read_jsonl(path))
    combined_path = out_dir / "gpt55_judgments.jsonl"
    with combined_path.open("w", encoding="utf-8") as handle:
        for row in judgments:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    groups = defaultdict(list)
    for row in judgments:
        groups[row["model"]].append(int(row["score"]))
    summary = []
    for model, scores in sorted(groups.items()):
        total = len(scores)
        avg = sum(scores) / max(1, total)
        summary.append(
            {
                "model": model,
                "num_samples": total,
                "avg_score_0_to_2": round(avg, 6),
                "llm_reasoning_validity": round(avg / 2, 6),
                "score_2_rate": round(sum(score == 2 for score in scores) / total, 6),
                "score_1_rate": round(sum(score == 1 for score in scores) / total, 6),
                "score_0_rate": round(sum(score == 0 for score in scores) / total, 6),
            }
        )
    write_csv(out_dir / "gpt55_judge_summary.csv", summary)
    (out_dir / "gpt55_judge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"num_batches": len(paths), "num_judgments": len(judgments), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
