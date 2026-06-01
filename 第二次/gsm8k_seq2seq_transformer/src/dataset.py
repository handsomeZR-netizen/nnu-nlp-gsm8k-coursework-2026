from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizer import Vocab
from tokenizer import tokenize
from utils import read_jsonl


class JsonlSeq2SeqDataset(Dataset):
    def __init__(self, path: str | Path, vocab: Vocab, max_src_len: int, max_tgt_len: int, limit: int | None = None):
        self.rows = read_jsonl(path)
        if limit is not None:
            self.rows = self.rows[:limit]
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.lengths = [
            min(len(tokenize(row["target"], vocab.lowercase)) + 2, max_tgt_len)
            for row in self.rows
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        src = self.vocab.encode(row["question"])[: self.max_src_len]
        tgt_body = self.vocab.encode(row["target"])[: self.max_tgt_len - 2]
        tgt = [self.vocab.bos_id] + tgt_body + [self.vocab.eos_id]
        return {
            "src": torch.tensor(src, dtype=torch.long),
            "tgt": torch.tensor(tgt, dtype=torch.long),
            "meta": row,
        }


def collate_batch(batch: list[dict], pad_id: int) -> dict:
    src_lens = torch.tensor([len(item["src"]) for item in batch], dtype=torch.long)
    tgt_lens = torch.tensor([len(item["tgt"]) for item in batch], dtype=torch.long)
    max_src = int(src_lens.max())
    max_tgt = int(tgt_lens.max())
    src = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    tgt = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    for i, item in enumerate(batch):
        src[i, : len(item["src"])] = item["src"]
        tgt[i, : len(item["tgt"])] = item["tgt"]
    return {"src": src, "src_lens": src_lens, "tgt": tgt, "tgt_lens": tgt_lens, "meta": [item["meta"] for item in batch]}
