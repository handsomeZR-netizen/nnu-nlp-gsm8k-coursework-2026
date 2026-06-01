import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Adafactor, AutoModelForSeq2SeqLM, AutoTokenizer

from utils import ensure_dir, extract_final_answer, prediction_metrics, project_root, read_jsonl, set_seed, write_csv, write_jsonl


def make_prompt(question: str) -> str:
    return f"Question: {question}\nGive the final numeric answer after ####.\nAnswer:"


class FinalAnswerDataset(Dataset):
    def __init__(self, rows, tokenizer, max_input_len: int, max_target_len: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        src = self.tokenizer(
            make_prompt(row["question"]),
            truncation=True,
            max_length=self.max_input_len,
            return_tensors=None,
        )
        tgt = self.tokenizer(
            f"#### {row['final_answer']}",
            truncation=True,
            max_length=self.max_target_len,
            return_tensors=None,
        )
        return {"input_ids": src["input_ids"], "attention_mask": src["attention_mask"], "labels": tgt["input_ids"], "meta": row}


def collate(batch, tokenizer):
    inputs = tokenizer.pad(
        [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in batch],
        return_tensors="pt",
    )
    labels = tokenizer.pad(
        [{"input_ids": item["labels"]} for item in batch],
        return_tensors="pt",
    )["input_ids"]
    labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)
    inputs["labels"] = labels
    inputs["meta"] = [item["meta"] for item in batch]
    return inputs


@torch.no_grad()
def evaluate(model, tokenizer, rows, batch_size, max_input_len, max_new_tokens, num_beams, device, out_path=None):
    model.eval()
    outputs = []
    for start in tqdm(range(0, len(rows), batch_size), desc="eval", leave=False, disable=True):
        batch = rows[start : start + batch_size]
        prompts = [make_prompt(row["question"]) for row in batch]
        encoded = tokenizer(prompts, padding=True, truncation=True, max_length=max_input_len, return_tensors="pt").to(device)
        gen = model.generate(**encoded, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False)
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for row, text in zip(batch, texts):
            gold = row["final_answer"]
            pred = extract_final_answer(text, fallback=False)
            fallback = extract_final_answer(text, fallback=True)
            outputs.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "gold": row["answer"],
                    "prediction": text,
                    "gold_final": gold,
                    "pred_final": pred,
                    "pred_final_fallback": fallback,
                    "strict_correct": pred == gold,
                    "fallback_correct": fallback == gold,
                    "has_final_marker": pred is not None,
                }
            )
    if out_path:
        write_jsonl(out_path, outputs)
    return prediction_metrics(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/flan-t5-small")
    parser.add_argument("--data_dir", default=str(project_root() / "data_final_only"))
    parser.add_argument("--out_dir", default=str(project_root() / "runs" / "flan_t5_small_finetuned_final"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--optimizer", choices=["adamw", "adafactor"], default="adamw")
    parser.add_argument("--max_input_len", type=int, default=256)
    parser.add_argument("--max_target_len", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision. Disabled by default for stability.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Reduce memory use for larger T5 models.")
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--dev_limit", type=int, default=None)
    parser.add_argument("--test_limit", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.out_dir)
    train_rows = read_jsonl(data_dir / "train.jsonl")
    dev_rows = read_jsonl(data_dir / "dev.jsonl")
    test_rows = read_jsonl(data_dir / "test.jsonl")
    if args.train_limit is not None:
        train_rows = train_rows[: args.train_limit]
    if args.dev_limit is not None:
        dev_rows = dev_rows[: args.dev_limit]
    if args.test_limit is not None:
        test_rows = test_rows[: args.test_limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = FinalAnswerDataset(train_rows, tokenizer, args.max_input_len, args.max_target_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer))
    if args.optimizer == "adafactor":
        optimizer = Adafactor(model.parameters(), lr=args.lr, relative_step=False, scale_parameter=False, warmup_init=False)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=False)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logs = []
    best_dev = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}", leave=False), start=1):
            batch.pop("meta")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = model(**batch).loss / args.grad_accum_steps
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch}, step={step}: {loss.item()}")
            scaler.scale(loss).backward()
            total_loss += float(loss.item()) * args.grad_accum_steps
            steps += 1
            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        dev_metrics = evaluate(model, tokenizer, dev_rows, args.eval_batch_size, args.max_input_len, args.max_new_tokens, args.num_beams, device)
        row = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(1, steps), 6),
            "dev_strict_acc": round(dev_metrics["strict_final_answer_acc"], 6),
            "dev_fallback_acc": round(dev_metrics["fallback_final_answer_acc"], 6),
            "dev_format_rate": round(dev_metrics["format_rate"], 6),
        }
        logs.append(row)
        write_csv(out_dir / "train_log.csv", logs)
        print(json.dumps(row, ensure_ascii=False))
        if dev_metrics["strict_final_answer_acc"] > best_dev:
            best_dev = dev_metrics["strict_final_answer_acc"]
            model.save_pretrained(out_dir / "best_model")
            tokenizer.save_pretrained(out_dir / "best_model")

    best_model = AutoModelForSeq2SeqLM.from_pretrained(out_dir / "best_model").to(device)
    dev_metrics = evaluate(best_model, tokenizer, dev_rows, args.eval_batch_size, args.max_input_len, args.max_new_tokens, args.num_beams, device, out_dir / "dev_predictions.jsonl")
    test_metrics = evaluate(best_model, tokenizer, test_rows, args.eval_batch_size, args.max_input_len, args.max_new_tokens, args.num_beams, device, out_dir / "test_predictions.jsonl")
    summary = {
        "run_name": Path(args.out_dir).name,
        "model_name": args.model_name,
        "best_dev_strict_acc": best_dev,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
