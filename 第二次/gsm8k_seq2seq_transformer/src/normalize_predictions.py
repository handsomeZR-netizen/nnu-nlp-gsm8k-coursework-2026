import argparse
import json
from pathlib import Path

from utils import ensure_dir, extract_final_answer, read_jsonl, write_jsonl


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=prediction_path.jsonl")
    name, path = value.split("=", 1)
    return name, Path(path)


def normalize_row(row: dict, model_name: str, method_type: str) -> dict:
    prediction = row.get("prediction", "")
    gold_final = row.get("gold_final")
    pred_final = row.get("pred_final") or extract_final_answer(prediction, fallback=False)
    pred_fallback = row.get("pred_final_fallback") or extract_final_answer(prediction, fallback=True)
    return {
        "id": row["id"],
        "model_name": model_name,
        "method_type": method_type,
        "question": row.get("question", ""),
        "gold": row.get("gold", row.get("answer", "")),
        "prediction": prediction,
        "gold_final": gold_final,
        "pred_final": pred_final,
        "pred_final_fallback": pred_fallback,
        "strict_correct": pred_final == gold_final,
        "fallback_correct": pred_fallback == gold_final,
        "has_final_marker": pred_final is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", action="append", type=parse_model_arg, required=True)
    parser.add_argument("--method_type", default="seq2seq")
    parser.add_argument("--out_dir", default="runs/normalized")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    manifest = []
    for model_name, path in args.predictions:
        rows = [normalize_row(row, model_name, args.method_type) for row in read_jsonl(path)]
        out_path = Path(out_dir) / f"{model_name}.jsonl"
        write_jsonl(out_path, rows)
        manifest.append({"model": model_name, "path": str(out_path), "num_samples": len(rows)})
    (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
