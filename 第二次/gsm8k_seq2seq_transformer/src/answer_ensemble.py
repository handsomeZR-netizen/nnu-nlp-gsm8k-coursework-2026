import argparse
import json
from pathlib import Path

from utils import ensure_dir, read_jsonl, write_csv, write_jsonl


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=prediction_path.jsonl")
    name, path = value.split("=", 1)
    return name, Path(path)


def load_map(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in read_jsonl(path)}


def score_model(rows: list[dict]) -> dict:
    total = len(rows)
    strict = sum(bool(row.get("strict_correct")) for row in rows)
    fallback = sum(bool(row.get("fallback_correct")) for row in rows)
    formatted = sum(bool(row.get("has_final_marker")) for row in rows)
    return {
        "strict_acc": strict / max(1, total),
        "fallback_acc": fallback / max(1, total),
        "format_rate": formatted / max(1, total),
    }


def choose_priority(dev_maps: list[tuple[str, dict[str, dict]]]) -> list[dict]:
    rows = []
    for name, pred_map in dev_maps:
        metrics = score_model(list(pred_map.values()))
        rows.append({"model": name, **metrics})
    rows.sort(key=lambda row: (row["strict_acc"], row["fallback_acc"], row["format_rate"]), reverse=True)
    return rows


def select_row(item_id: str, priority: list[dict], test_maps: dict[str, dict[str, dict]]) -> dict:
    # Prefer the highest-dev model that emits a formatted final answer.
    for item in priority:
        row = test_maps[item["model"]][item_id]
        if row.get("pred_final") is not None:
            return {**row, "ensemble_source": item["model"]}
    best = priority[0]["model"]
    return {**test_maps[best][item_id], "ensemble_source": best}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="append", type=parse_model_arg, required=True)
    parser.add_argument("--test", action="append", type=parse_model_arg, required=True)
    parser.add_argument("--out_dir", default="runs/answer_ensemble")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    dev_maps = [(name, load_map(path)) for name, path in args.dev]
    test_maps = {name: load_map(path) for name, path in args.test}
    priority = choose_priority(dev_maps)
    common_ids = sorted(set.intersection(*(set(pred_map) for pred_map in test_maps.values())))
    outputs = [select_row(item_id, priority, test_maps) for item_id in common_ids]
    for row in outputs:
        row["id"] = row["id"]
        row["strict_correct"] = row.get("pred_final") == row.get("gold_final")
        row["fallback_correct"] = row.get("pred_final_fallback") == row.get("gold_final")
        row["has_final_marker"] = row.get("pred_final") is not None

    write_jsonl(Path(out_dir) / "test_predictions.jsonl", outputs)
    write_csv(Path(out_dir) / "model_priority.csv", priority)
    summary = {
        "num_samples": len(outputs),
        "strict_final_answer_acc": sum(row["strict_correct"] for row in outputs) / max(1, len(outputs)),
        "fallback_final_answer_acc": sum(row["fallback_correct"] for row in outputs) / max(1, len(outputs)),
        "format_rate": sum(row["has_final_marker"] for row in outputs) / max(1, len(outputs)),
        "priority": priority,
    }
    (Path(out_dir) / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
