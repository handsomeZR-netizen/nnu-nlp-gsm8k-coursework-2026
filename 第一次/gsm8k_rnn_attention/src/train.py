import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from tqdm import tqdm

from dataset import JsonlSeq2SeqDataset, collate_batch
from evaluate import evaluate_model
from model import Seq2Seq
from tokenizer import Vocab
from utils import ensure_dir, load_config, project_root, set_seed, write_csv


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(self, lengths: list[int], batch_size: int, bucket_size: int, shuffle: bool = True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = max(bucket_size, batch_size)
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(indices)
        batches = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda idx: self.lengths[idx], reverse=True)
            for batch_start in range(0, len(bucket), self.batch_size):
                batches.append(bucket[batch_start : batch_start + self.batch_size])
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def teacher_forcing_ratio(config: dict, epoch: int) -> float:
    start = float(config["training"]["teacher_forcing_start"])
    end = float(config["training"]["teacher_forcing_end"])
    total = max(1, int(config["training"]["epochs"]) - 1)
    progress = min(1.0, (epoch - 1) / total)
    return start + (end - start) * progress


def make_loader(path: Path, vocab: Vocab, config: dict, batch_size: int, shuffle: bool, limit: int | None) -> DataLoader:
    dataset = JsonlSeq2SeqDataset(
        path,
        vocab,
        int(config["tokenizer"]["max_src_len"]),
        int(config["tokenizer"]["max_tgt_len"]),
        limit=limit,
    )
    if shuffle:
        sampler = LengthBucketBatchSampler(dataset.lengths, batch_size=batch_size, bucket_size=batch_size * 20, shuffle=True)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=int(config["training"].get("num_workers", 0)),
            collate_fn=lambda b: collate_batch(b, vocab.pad_id),
            pin_memory=False,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(config["training"].get("num_workers", 0)),
        collate_fn=lambda b: collate_batch(b, vocab.pad_id),
        pin_memory=False,
    )


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, config, epoch: int) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    grad_accum = int(config["training"]["grad_accum_steps"])
    tf_ratio = teacher_forcing_ratio(config, epoch)
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False, disable=True)
    for step, batch in enumerate(pbar, start=1):
        src = batch["src"].to(device, non_blocking=True)
        src_lens = batch["src_lens"].to(device, non_blocking=True)
        tgt = batch["tgt"].to(device, non_blocking=True)
        with autocast(enabled=amp_enabled):
            logits = model(src, src_lens, tgt, teacher_forcing_ratio=tf_ratio)
            gold = tgt[:, 1:].contiguous()
            loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
            loss_for_backward = loss / grad_accum
        scaler.scale(loss_for_backward).backward()
        valid_tokens = gold.ne(model.pad_id).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens
        if step % grad_accum == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        pbar.set_postfix(loss=total_loss / max(1, total_tokens), tf=tf_ratio)
    return total_loss / max(1, total_tokens)


@torch.no_grad()
def loss_eval(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in tqdm(loader, desc="dev loss", leave=False, disable=True):
        src = batch["src"].to(device, non_blocking=True)
        src_lens = batch["src_lens"].to(device, non_blocking=True)
        tgt = batch["tgt"].to(device, non_blocking=True)
        logits = model(src, src_lens, tgt, teacher_forcing_ratio=1.0)
        gold = tgt[:, 1:].contiguous()
        loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
        valid_tokens = gold.ne(model.pad_id).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens
    return total_loss / max(1, total_tokens)


def save_checkpoint(path: Path, model, optimizer, epoch: int, config: dict, metrics: dict) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def plot_curves(log_rows: list[dict], out_path: Path) -> None:
    if not log_rows:
        return
    epochs = [row["epoch"] for row in log_rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in log_rows], label="train")
    axes[0].plot(epochs, [row["dev_loss"] for row in log_rows], label="dev")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["dev_strict_acc"] for row in log_rows], label="strict")
    axes[1].plot(epochs, [row["dev_fallback_acc"] for row in log_rows], label="fallback")
    axes[1].set_title("Dev Final Answer Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[2].plot(epochs, [row["teacher_forcing_ratio"] for row in log_rows])
    axes[2].set_title("Teacher Forcing Ratio")
    axes[2].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def load_train_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                if key == "epoch":
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(project_root() / "configs" / "attn_socratic_4060.yaml"))
    parser.add_argument("--debug_train_size", type=int, default=None)
    parser.add_argument("--debug_dev_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    set_seed(int(config["seed"]))
    root = project_root()
    data_dir = root / config["paths"]["data_dir"]
    run_name = args.run_name or config["run_name"]
    if args.debug_train_size or args.debug_dev_size:
        run_name = f"{run_name}_debug"
    run_dir = ensure_dir(Path(args.resume).resolve().parent if args.resume else root / config["paths"]["run_root"] / run_name)
    run_name = run_dir.name
    shutil.copy2(args.config, run_dir / "config.yaml")

    vocab = Vocab.load(data_dir / "vocab.json")
    train_loader = make_loader(data_dir / "train.jsonl", vocab, config, int(config["training"]["batch_size"]), True, args.debug_train_size)
    dev_loader = make_loader(data_dir / "dev.jsonl", vocab, config, int(config["training"]["batch_size"]), False, args.debug_dev_size)
    decode_batch_size = int(config["training"].get("decode_batch_size", 32))
    decode_loader = make_loader(data_dir / "dev.jsonl", vocab, config, decode_batch_size, False, args.debug_dev_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2Seq(len(vocab.itos), vocab.pad_id, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(config["training"]["weight_decay"]))
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    scaler = GradScaler(enabled=bool(config["training"].get("amp", True)) and device.type == "cuda")

    best_score = -1.0
    best_loss = float("inf")
    bad_epochs = 0
    logs: list[dict] = load_train_log(run_dir / "train_log.csv")
    if logs:
        best_row = max(logs, key=lambda x: (x["dev_strict_acc"], -x["dev_loss"]))
        best_score = float(best_row["dev_strict_acc"])
        best_loss = float(best_row["dev_loss"])
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resuming {run_name} from epoch {start_epoch}")

    total_epochs = int(config["training"]["epochs"])
    if start_epoch > total_epochs:
        print(f"Nothing to train: start epoch {start_epoch} is beyond configured epochs {total_epochs}")
        return

    for epoch in range(start_epoch, total_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, config, epoch)
        dev_loss = loss_eval(model, dev_loader, criterion, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        metrics = evaluate_model(
            model,
            decode_loader,
            vocab,
            device,
            max_len=int(config["decoding"]["max_len"]),
            beam_size=1,
            prediction_path=run_dir / "dev_predictions.jsonl",
        )
        gpu_mem = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "dev_loss": round(dev_loss, 6),
            "dev_strict_acc": round(metrics["strict_final_answer_acc"], 6),
            "dev_fallback_acc": round(metrics["fallback_final_answer_acc"], 6),
            "dev_format_rate": round(metrics["format_rate"], 6),
            "teacher_forcing_ratio": round(teacher_forcing_ratio(config, epoch), 6),
            "gpu_max_mem_gb": round(gpu_mem, 3),
        }
        logs.append(row)
        write_csv(run_dir / "train_log.csv", logs)
        plot_curves(logs, run_dir / "curves.png")
        save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, config, row)

        score = metrics["strict_final_answer_acc"]
        improved = score > best_score or (score == best_score and dev_loss < best_loss)
        if improved:
            best_score = score
            best_loss = dev_loss
            bad_epochs = 0
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, config, row)
        else:
            bad_epochs += 1

        summary = {
            "run_name": run_name,
            "best_epoch": max(logs, key=lambda x: (x["dev_strict_acc"], -x["dev_loss"]))["epoch"],
            "best_dev_strict_acc": best_score,
            "best_dev_loss": best_loss,
            "latest": row,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False))

        if bad_epochs >= int(config["training"]["early_stop_patience"]):
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Training finished. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
