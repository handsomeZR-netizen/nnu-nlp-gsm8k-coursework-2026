import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils import ensure_dir, extract_final_answer, prediction_metrics, project_root, read_jsonl, write_jsonl


def make_prompt(question: str) -> str:
    return (
        "Solve the math word problem. Give the final numeric answer after ####.\n"
        f"Question: {question}\n"
        "Answer:"
    )


@torch.no_grad()
def evaluate(rows: list[dict], model_name: str, batch_size: int, max_new_tokens: int, device: torch.device) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    outputs = []
    for start in tqdm(range(0, len(rows), batch_size), desc="pretrained"):
        batch = rows[start : start + batch_size]
        prompts = [make_prompt(row["question"]) for row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            do_sample=False,
        )
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for row, text in zip(batch, texts):
            gold_final = row["final_answer"]
            pred_final = extract_final_answer(text, fallback=False)
            pred_fallback = extract_final_answer(text, fallback=True)
            outputs.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "gold": row["answer"],
                    "prediction": text,
                    "gold_final": gold_final,
                    "pred_final": pred_final,
                    "pred_final_fallback": pred_fallback,
                    "strict_correct": pred_final == gold_final,
                    "fallback_correct": pred_fallback == gold_final,
                    "has_final_marker": pred_final is not None,
                }
            )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/flan-t5-small")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--data_dir", default=str(project_root() / "data_final_only"))
    parser.add_argument("--out_dir", default=str(project_root() / "runs" / "flan_t5_small_zero_shot"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data_path = Path(args.data_dir) / f"{args.split}.jsonl"
    rows = read_jsonl(data_path)
    if args.limit:
        rows = rows[: args.limit]

    out_dir = ensure_dir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = evaluate(rows, args.model_name, args.batch_size, args.max_new_tokens, device)
    safe_model = args.model_name.replace("/", "_").replace("-", "_")
    suffix = f"_{args.limit}" if args.limit else ""
    pred_path = out_dir / f"{args.split}_predictions_{safe_model}{suffix}.jsonl"
    write_jsonl(pred_path, preds)

    metrics = prediction_metrics(preds)
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary[f"{args.split}_{safe_model}{suffix}"] = {
        "model_name": args.model_name,
        "num_samples": len(preds),
        **metrics,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

