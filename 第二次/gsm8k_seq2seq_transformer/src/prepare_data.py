import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

from tokenizer import build_vocab, tokenize
from utils import ensure_dir, extract_final_answer, load_config, project_root, set_seed, write_jsonl


def make_record(row: dict, split: str, idx: int, target_mode: str) -> dict:
    answer = row["answer"]
    final_answer = extract_final_answer(answer, fallback=False)
    if target_mode == "final_only":
        target = f"#### {final_answer}" if final_answer is not None else answer
    else:
        target = answer
    return {
        "id": f"{split}_{idx:06d}",
        "split": split,
        "question": row["question"],
        "answer": answer,
        "target": target,
        "final_answer": final_answer,
    }


def split_and_write(config: dict, out_dir: Path) -> None:
    set_seed(int(config["seed"]))
    ds_cfg = config["dataset"]
    ds = load_dataset(ds_cfg["name"], ds_cfg["config"])
    split = ds["train"].train_test_split(test_size=float(ds_cfg["dev_ratio"]), seed=int(config["seed"]))

    target_mode = ds_cfg.get("target_mode", "full")
    train_rows = [make_record(row, "train", i, target_mode) for i, row in enumerate(split["train"])]
    dev_rows = [make_record(row, "dev", i, target_mode) for i, row in enumerate(split["test"])]
    test_rows = [make_record(row, "test", i, target_mode) for i, row in enumerate(ds["test"])]

    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "dev.jsonl", dev_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)

    tok_cfg = config["tokenizer"]
    vocab_texts = [row["question"] for row in train_rows] + [row["target"] for row in train_rows]
    vocab = build_vocab(vocab_texts, min_freq=int(tok_cfg["min_freq"]), lowercase=bool(tok_cfg["lowercase"]))
    vocab.save(out_dir / "vocab.json")

    stats = compute_stats(
        {"train": train_rows, "dev": dev_rows, "test": test_rows},
        lowercase=bool(tok_cfg["lowercase"]),
        max_src_len=int(tok_cfg["max_src_len"]),
        max_tgt_len=int(tok_cfg["max_tgt_len"]),
    )
    pd.DataFrame(stats).to_csv(out_dir / "data_stats.csv", index=False, encoding="utf-8")

    summary = {
        "dataset": ds_cfg["name"],
        "config": ds_cfg["config"],
        "target_mode": target_mode,
        "splits": {"train": len(train_rows), "dev": len(dev_rows), "test": len(test_rows)},
        "vocab_size": len(vocab.itos),
    }
    (out_dir / "prepare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def compute_stats(rows_by_split: dict[str, list[dict]], lowercase: bool, max_src_len: int, max_tgt_len: int) -> list[dict]:
    stats = []
    for split, rows in rows_by_split.items():
        src_lens = np.array([len(tokenize(row["question"], lowercase)) for row in rows], dtype=np.int64)
        tgt_lens = np.array([len(tokenize(row["target"], lowercase)) + 2 for row in rows], dtype=np.int64)
        trunc = np.logical_or(src_lens > max_src_len, tgt_lens > max_tgt_len)
        stats.append(
            {
                "split": split,
                "num_samples": len(rows),
                "avg_src_len": round(float(src_lens.mean()), 2),
                "avg_tgt_len": round(float(tgt_lens.mean()), 2),
                "p95_src_len": int(np.percentile(src_lens, 95)),
                "p95_tgt_len": int(np.percentile(tgt_lens, 95)),
                "max_src_len": int(src_lens.max()),
                "max_tgt_len": int(tgt_lens.max()),
                "trunc_rate": round(float(trunc.mean()), 4),
            }
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(project_root() / "configs" / "attn_socratic_4060.yaml"))
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else project_root() / config["paths"]["data_dir"]
    ensure_dir(out_dir)
    split_and_write(config, out_dir)
    print(f"Wrote data and vocab to {out_dir}")


if __name__ == "__main__":
    main()

