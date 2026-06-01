import argparse
import ast
import operator
import re
import json
import math
from collections import Counter
from pathlib import Path

from tokenizer import tokenize
from utils import ensure_dir, extract_final_answer, normalize_answer, read_jsonl, write_csv

EQUATION_RE = re.compile(r"<<([^<>=]+)=([^<>]+)>>")
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1)))


def safe_eval_expr(expr: str) -> float | None:
    expr = expr.replace(",", "").strip()
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        return None
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError:
        return None
    return eval_node(node)


def eval_node(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left = eval_node(node.left)
        right = eval_node(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Div) and right == 0:
            return None
        return float(OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        operand = eval_node(node.operand)
        return None if operand is None else float(OPS[type(node.op)](operand))
    return None


def parse_number(value: str) -> float | None:
    cleaned = value.replace(",", "").strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def equation_stats(text: str) -> dict:
    matches = EQUATION_RE.findall(text or "")
    if not matches:
        return {"total": 0, "correct": 0, "all_correct": False}
    correct = 0
    for expr, expected in matches:
        got = safe_eval_expr(expr)
        want = parse_number(expected)
        if got is not None and want is not None and abs(got - want) <= 1e-6:
            correct += 1
    return {"total": len(matches), "correct": correct, "all_correct": correct == len(matches)}


def corpus_bleu(preds: list[list[str]], refs: list[list[str]], max_n: int = 4) -> float:
    pred_len = sum(len(x) for x in preds)
    ref_len = sum(len(x) for x in refs)
    if pred_len == 0:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        clipped = 0
        total = 0
        for pred, ref in zip(preds, refs):
            pred_counts = ngrams(pred, n)
            ref_counts = ngrams(ref, n)
            clipped += sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
            total += sum(pred_counts.values())
        if total == 0:
            precisions.append(0.0)
        elif clipped == 0:
            precisions.append(1.0 / (2.0 * total))
        else:
            precisions.append(clipped / total)
    bp = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / pred_len)
    if any(p <= 0 for p in precisions):
        return 0.0
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        cur = [0] * (len(b) + 1)
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l_f1(pred: list[str], ref: list[str]) -> float:
    if not pred or not ref:
        return 0.0
    lcs = lcs_len(pred, ref)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def read_predictions(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    return {row["id"]: row for row in rows}


def compute_for_model(name: str, rows: list[dict]) -> dict:
    pred_tokens = [tokenize(row.get("prediction", ""), lowercase=True) for row in rows]
    ref_tokens = [tokenize(row.get("gold", ""), lowercase=True) for row in rows]
    strict = 0
    fallback = 0
    formatted = 0
    rows_with_equations = 0
    equation_rows_all_correct = 0
    equation_total = 0
    equation_correct = 0
    for row in rows:
        gold_final = normalize_answer(row.get("gold_final"))
        pred_final = normalize_answer(row.get("pred_final"))
        pred_fallback = normalize_answer(row.get("pred_final_fallback"))
        if pred_final is None:
            pred_final = extract_final_answer(row.get("prediction", ""), fallback=False)
        if pred_fallback is None:
            pred_fallback = extract_final_answer(row.get("prediction", ""), fallback=True)
        strict += pred_final == gold_final
        fallback += pred_fallback == gold_final
        formatted += pred_final is not None
        eq = equation_stats(row.get("prediction", ""))
        equation_total += eq["total"]
        equation_correct += eq["correct"]
        if eq["total"] > 0:
            rows_with_equations += 1
            equation_rows_all_correct += eq["all_correct"]
    total = len(rows)
    return {
        "model": name,
        "num_samples": total,
        "bleu4": round(corpus_bleu(pred_tokens, ref_tokens), 6),
        "rouge_l": round(sum(rouge_l_f1(p, r) for p, r in zip(pred_tokens, ref_tokens)) / max(1, total), 6),
        "strict_final_answer_acc": round(strict / max(1, total), 6),
        "fallback_final_answer_acc": round(fallback / max(1, total), 6),
        "format_rate": round(formatted / max(1, total), 6),
        "equation_coverage": round(rows_with_equations / max(1, total), 6),
        "equation_step_acc": round(equation_correct / max(1, equation_total), 6),
        "equation_all_correct_rate": round(equation_rows_all_correct / max(1, rows_with_equations), 6),
    }


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=prediction_path.jsonl")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", action="append", type=parse_model_arg, required=True)
    parser.add_argument("--out_dir", default="runs/metrics")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    pred_maps = [(name, read_predictions(path)) for name, path in args.predictions]
    common_ids = sorted(set.intersection(*(set(pred_map) for _, pred_map in pred_maps)))
    if not common_ids:
        raise RuntimeError("No common prediction ids found.")
    summaries = []
    for name, pred_map in pred_maps:
        rows = [pred_map[item_id] for item_id in common_ids]
        summaries.append(compute_for_model(name, rows))

    write_csv(Path(out_dir) / "automatic_metrics.csv", summaries)
    (Path(out_dir) / "automatic_metrics.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
