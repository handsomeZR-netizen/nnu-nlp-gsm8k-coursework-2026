import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


PROMPT_SYSTEM = (
    "You solve grade-school math word problems. Use concise arithmetic only. "
    "Do not write Python or code. Always end with a final line exactly as: #### number"
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ensure_dir(path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def final_only_target(row: dict) -> str:
    return f"#### {row['final_answer']}"


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": f"Question: {question}\nAnswer:"},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{PROMPT_SYSTEM}\n\nQuestion: {question}\nAnswer:"


class MathDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        prompt = build_prompt(self.tokenizer, row["question"])
        target = final_only_target(row) + self.tokenizer.eos_token
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        if len(ids) > self.max_length:
            overflow = len(ids) - self.max_length
            prompt_ids = prompt_ids[overflow:]
            ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
        return {"input_ids": ids, "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, attention_mask = [], [], []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append([pad_id] * pad_len + item["input_ids"])
        labels.append([-100] * pad_len + item["labels"])
        attention_mask.append([0] * pad_len + [1] * len(item["input_ids"]))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if torch.cuda.is_available():
        model.to("cuda")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(","),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return tokenizer, model


def evaluate_loss(model, loader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.append(float(model(**batch).loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--train_path", default="data/train.jsonl")
    parser.add_argument("--dev_path", default="data/dev.jsonl")
    parser.add_argument("--out_dir", default="runs/qwen25_math_15b_lora")
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_dev", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every_steps", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    train_rows = read_jsonl(Path(args.train_path))
    dev_rows = read_jsonl(Path(args.dev_path))
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    if args.limit_dev:
        dev_rows = dev_rows[: args.limit_dev]

    tokenizer, model = load_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = MathDataset(train_rows, tokenizer, args.max_length)
    dev_set = MathDataset(dev_rows, tokenizer, args.max_length)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_update_steps = max(1, int((len(train_loader) * args.epochs) // args.grad_accum))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_update_steps * args.warmup_ratio)),
        num_training_steps=total_update_steps,
    )

    log_rows = []
    model.train()
    started = time.time()
    global_step = 0
    update_step = 0
    total_batches = int(len(train_loader) * args.epochs)
    progress = tqdm(total=total_batches, desc="qwen-lora")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(max(1, int(args.epochs))):
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            global_step += 1
            progress.update(1)
            if global_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                if update_step % 10 == 0 or update_step == total_update_steps:
                    dev_loss = evaluate_loss(model, dev_loader, device)
                    row = {
                        "update_step": update_step,
                        "train_loss": float(loss.detach().cpu()) * args.grad_accum,
                        "dev_loss": dev_loss,
                        "lr": scheduler.get_last_lr()[0],
                        "elapsed_sec": round(time.time() - started, 2),
                    }
                    log_rows.append(row)
                    print(json.dumps(row, ensure_ascii=False))
                if args.save_every_steps and update_step % args.save_every_steps == 0:
                    model.save_pretrained(out_dir / f"checkpoint-{update_step}")
            if update_step >= total_update_steps:
                break
        if update_step >= total_update_steps:
            break
    progress.close()

    model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")
    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as handle:
        for row in log_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "model_name": args.model_name,
        "num_train": len(train_rows),
        "num_dev": len(dev_rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "update_steps": update_step,
        "elapsed_sec": round(time.time() - started, 3),
    }
    if torch.cuda.is_available():
        summary["cuda_max_memory_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    (out_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
