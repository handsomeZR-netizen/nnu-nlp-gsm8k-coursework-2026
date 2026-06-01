import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import JsonlSeq2SeqDataset, collate_batch
from model import Seq2Seq
from tokenizer import Vocab
from utils import ensure_dir, extract_final_answer, load_config, prediction_metrics, project_root, write_jsonl


@torch.no_grad()
def evaluate_model(
    model: Seq2Seq,
    loader: DataLoader,
    vocab: Vocab,
    device: torch.device,
    max_len: int,
    beam_size: int = 1,
    length_penalty: float = 0.7,
    prediction_path: Path | None = None,
) -> dict:
    model.eval()
    rows = []
    for batch in tqdm(loader, desc="decode", leave=False, disable=True):
        src = batch["src"].to(device)
        src_lens = batch["src_lens"].to(device)
        if beam_size > 1:
            pred_ids = model.beam_decode(src, src_lens, vocab.bos_id, vocab.eos_id, max_len=max_len, beam_size=beam_size, length_penalty=length_penalty).cpu().tolist()
        else:
            pred_ids = model.greedy_decode(src, src_lens, vocab.bos_id, vocab.eos_id, max_len=max_len).cpu().tolist()
        for ids, meta in zip(pred_ids, batch["meta"]):
            pred_text = vocab.decode(ids)
            gold_final = meta.get("final_answer")
            pred_final_strict = extract_final_answer(pred_text, fallback=False)
            pred_final_fallback = extract_final_answer(pred_text, fallback=True)
            row = {
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
            rows.append(row)
    if prediction_path:
        write_jsonl(prediction_path, rows)
    return prediction_metrics(rows)


def load_checkpoint(ckpt_path: Path, config: dict, vocab: Vocab, device: torch.device) -> Seq2Seq:
    model = Seq2Seq(vocab_size=len(vocab.itos), pad_id=vocab.pad_id, config=config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(project_root() / "configs" / "attn_socratic_4060.yaml"))
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--length_penalty", type=float, default=None)
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
    beam_size = args.beam_size or int(config["decoding"].get("beam_size", 1))
    length_penalty = args.length_penalty if args.length_penalty is not None else float(config["decoding"].get("length_penalty", 0.7))
    suffix = f"_beam{beam_size}" if beam_size > 1 else ""
    pred_path = out_dir / f"{args.split}_predictions{suffix}.jsonl"
    metrics = evaluate_model(model, loader, vocab, device, max_len, beam_size, length_penalty, pred_path)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}
    summary[f"{args.split}_metrics{suffix}"] = metrics
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
