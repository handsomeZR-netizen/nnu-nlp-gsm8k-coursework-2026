import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import JsonlSeq2SeqDataset, collate_batch
from tokenizer import Vocab
from transformer_model import TransformerSeq2Seq
from utils import ensure_dir, extract_final_answer, load_config, prediction_metrics, project_root, write_jsonl


@torch.no_grad()
def evaluate_model(
    model: TransformerSeq2Seq,
    loader: DataLoader,
    vocab: Vocab,
    device: torch.device,
    max_len: int,
    prediction_path: Path | None = None,
) -> dict:
    model.eval()
    rows = []
    for batch in tqdm(loader, desc="decode", leave=False, disable=True):
        src = batch["src"].to(device)
        src_lens = batch["src_lens"].to(device)
        pred_ids = model.greedy_decode(src, src_lens, vocab.bos_id, vocab.eos_id, max_len=max_len).cpu().tolist()
        for ids, meta in zip(pred_ids, batch["meta"]):
            pred_text = vocab.decode(ids)
            gold_final = meta.get("final_answer")
            pred_final_strict = extract_final_answer(pred_text, fallback=False)
            pred_final_fallback = extract_final_answer(pred_text, fallback=True)
            rows.append(
                {
                    "id": meta["id"],
                    "question": meta["question"],
                    "gold": meta["answer"],
                    "prediction": pred_text,
                    "gold_final": gold_final,
                    "pred_final": pred_final_strict,
                    "pred_final_fallback": pred_final_fallback,
                    "strict_correct": pred_final_strict == gold_final,
                    "fallback_correct": pred_final_fallback == gold_final,
                    "has_final_marker": pred_final_strict is not None,
                }
            )
    if prediction_path:
        write_jsonl(prediction_path, rows)
    return prediction_metrics(rows)


def load_checkpoint(ckpt_path: Path, config: dict, vocab: Vocab, device: torch.device) -> TransformerSeq2Seq:
    model = TransformerSeq2Seq(vocab_size=len(vocab.itos), pad_id=vocab.pad_id, config=config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(project_root() / "configs" / "transformer_socratic_4060.yaml"))
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    root = project_root()
    data_dir = root / config["paths"]["data_dir"]
    out_dir = ensure_dir(args.out_dir or Path(args.ckpt).parent)
    vocab = Vocab.load(data_dir / "vocab.json")
    dataset = JsonlSeq2SeqDataset(
        data_dir / f"{args.split}.jsonl",
        vocab,
        int(config["tokenizer"]["max_src_len"]),
        int(config["tokenizer"]["max_tgt_len"]),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_batch(b, vocab.pad_id))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(Path(args.ckpt), config, vocab, device)
    max_len = args.max_len or int(config["decoding"]["max_len"])
    pred_path = out_dir / f"{args.split}_predictions.jsonl"
    metrics = evaluate_model(model, loader, vocab, device, max_len, pred_path)
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary[f"{args.split}_metrics"] = metrics
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
