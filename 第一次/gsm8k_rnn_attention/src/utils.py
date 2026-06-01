import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import yaml


FINAL_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def extract_final_answer(text: str, fallback: bool = False) -> str | None:
    if not text:
        return None
    m = FINAL_RE.search(text)
    if m:
        return normalize_answer(m.group(1))
    if fallback:
        numbers = NUMBER_RE.findall(text)
        if numbers:
            return normalize_answer(numbers[-1])
    return None


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip().replace(",", "")
    if answer.endswith(".0"):
        answer = answer[:-2]
    return answer


def prediction_metrics(rows: list[dict]) -> dict:
    total = len(rows)
    if total == 0:
        return {
            "num_samples": 0,
            "strict_final_answer_acc": 0.0,
            "fallback_final_answer_acc": 0.0,
            "format_rate": 0.0,
        }
    strict = sum(1 for row in rows if row.get("strict_correct"))
    fallback = sum(1 for row in rows if row.get("fallback_correct"))
    formatted = sum(1 for row in rows if row.get("has_final_marker"))
    return {
        "num_samples": total,
        "strict_final_answer_acc": strict / total,
        "fallback_final_answer_acc": fallback / total,
        "format_rate": formatted / total,
    }


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
