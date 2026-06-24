from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
AUDIT = RUNS / "audit_v8"
FIGURES = RUNS / "report_figures_v8"
FIGURES.mkdir(parents=True, exist_ok=True)

P = {
    "blue": "#337BAC",
    "sky": "#90B4CF",
    "mint": "#A9D9BB",
    "teal": "#4FB1B2",
    "pale": "#D5E8F1",
    "ink": "#1F3349",
    "muted": "#6C7A86",
    "accent": "#D88C45",
    "red": "#C96B5F",
}

DISPLAY = {
    "RNN_full": "RNN full",
    "Transformer_full": "Transformer full",
    "Transformer_final_opt": "Transformer opt",
    "Pretrained_T5_base_final": "T5-base",
    "Local_Qwen25_Math_15B_CoT": "Qwen 1.5B",
    "DeepSeek_v4_flash": "DeepSeek Flash",
    "DeepSeek_v4_pro": "DeepSeek Pro",
}

ORDER = [
    "RNN_full",
    "Transformer_full",
    "Transformer_final_opt",
    "Pretrained_T5_base_final",
    "Local_Qwen25_Math_15B_CoT",
    "DeepSeek_v4_flash",
    "DeepSeek_v4_pro",
]

BAR_COLORS = [P["sky"], "#ABD7DF", P["teal"], P["mint"], P["blue"], "#7FB8B2", P["accent"]]

plt.rcParams.update(
    {
        "font.family": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.2,
        "axes.labelsize": 10.8,
        "axes.titlesize": 11.2,
        "xtick.labelsize": 8.9,
        "ytick.labelsize": 8.9,
        "legend.fontsize": 8.8,
        "axes.linewidth": 0.95,
        "axes.edgecolor": P["ink"],
        "axes.labelcolor": P["ink"],
        "xtick.color": P["ink"],
        "ytick.color": P["ink"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    }
)


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=0.9, length=3.4)
    ax.grid(axis=grid_axis, color=P["pale"], linewidth=0.72, alpha=0.86)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIGURES / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(RUNS / "second_result_summary_v6.csv")
    judge = pd.read_csv(RUNS / "gpt55_judge_v1" / "gpt55_judge_summary.csv")
    return results, judge


def subset_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = results.set_index("model").loc[ORDER].reset_index()
    rows["short_name"] = rows["model"].map(DISPLAY)
    return rows


def main_result_panel(results: pd.DataFrame, judge: pd.DataFrame) -> None:
    rows = subset_results(results)
    judge_map = judge.set_index("model")["llm_reasoning_validity"].to_dict()
    rows["reasoning"] = rows["model"].map(judge_map)
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.35), constrained_layout=True)

    axes[0].bar(x, rows["final_answer_acc"], color=BAR_COLORS, edgecolor="black", linewidth=0.55)
    axes[0].set_ylabel("Final answer accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(rows["short_name"], rotation=35, ha="right")
    axes[0].text(-0.08, 1.04, "A", transform=axes[0].transAxes, fontweight="bold", fontsize=13)
    for xi, value in zip(x, rows["final_answer_acc"]):
        axes[0].text(xi, value + 0.023, f"{value:.2f}", ha="center", va="bottom", fontsize=8.1, color=P["ink"])

    judged = rows.dropna(subset=["reasoning"])
    x2 = np.arange(len(judged))
    colors = [BAR_COLORS[ORDER.index(model)] for model in judged["model"]]
    axes[1].bar(x2, judged["reasoning"], color=colors, edgecolor="black", linewidth=0.55)
    axes[1].set_ylabel("GPT-5.5 reasoning validity")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(judged["short_name"], rotation=35, ha="right")
    axes[1].text(-0.08, 1.04, "B", transform=axes[1].transAxes, fontweight="bold", fontsize=13)
    for xi, value in zip(x2, judged["reasoning"]):
        axes[1].text(xi, value + 0.023, f"{value:.2f}", ha="center", va="bottom", fontsize=8.1, color=P["ink"])

    for ax in axes:
        clean_axis(ax)
    save(fig, "main_result_panel_v8")


def strict_lenient_panel() -> None:
    qwen = pd.read_csv(AUDIT / "qwen_strict_lenient.csv").iloc[0]
    values = [qwen["strict_acc"], qwen["lenient_acc"], qwen["format_rate"]]
    labels = ["Strict acc.", "Lenient acc.", "Format rate"]
    colors = [P["blue"], P["teal"], P["accent"]]
    fig, ax = plt.subplots(figsize=(4.8, 3.15))
    x = np.arange(len(values))
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.55)
    ax.set_ylim(0.72, 1.0)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for xi, value in zip(x, values):
        ax.text(xi, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=9.0, color=P["ink"])
    clean_axis(ax)
    save(fig, "qwen_strict_lenient_v8")


def cost_accuracy_panel() -> None:
    costs = pd.read_csv(AUDIT / "cost_accuracy.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.25))
    colors = [P["teal"] if "Flash" in name else P["accent"] for name in costs["display_name"]]
    ax.scatter(costs["estimated_cost_usd"], costs["accuracy"], s=90, color=colors, edgecolor="black", linewidth=0.65, zorder=3)
    for _, row in costs.iterrows():
        label = "Flash" if "Flash" in row["display_name"] else "Pro"
        ax.annotate(label, (row["estimated_cost_usd"], row["accuracy"]), xytext=(6, 3), textcoords="offset points", fontsize=9.0)
    ax.set_xlabel("Estimated API cost (USD)")
    ax.set_ylabel("Final answer accuracy")
    ax.set_xlim(0, max(costs["estimated_cost_usd"]) * 1.25)
    ax.set_ylim(0.93, 0.958)
    clean_axis(ax)
    save(fig, "cost_accuracy_v8")


def error_breakdown_panel() -> None:
    rows = pd.read_csv(AUDIT / "error_breakdown.csv")
    rows["model_label"] = rows["model"].map(
        {
            "qwen25_math_15b": "Qwen 1.5B",
            "deepseek_v4_flash": "DeepSeek Flash",
            "deepseek_v4_pro": "DeepSeek Pro",
        }
    )
    pivot = rows.pivot_table(index="model_label", columns="error_type", values="count", fill_value=0, aggfunc="sum")
    order = ["Qwen 1.5B", "DeepSeek Flash", "DeepSeek Pro"]
    pivot = pivot.reindex(order)
    colors = {
        "wrong_final_no_equation": P["red"],
        "missing_final_marker": P["sky"],
        "format_missing_but_fallback_correct": P["accent"],
    }
    labels = {
        "wrong_final_no_equation": "Wrong final answer",
        "missing_final_marker": "Missing final marker",
        "format_missing_but_fallback_correct": "Fallback correct",
    }
    fig, ax = plt.subplots(figsize=(5.8, 3.25))
    bottom = np.zeros(len(pivot))
    for col in ["wrong_final_no_equation", "missing_final_marker", "format_missing_but_fallback_correct"]:
        if col not in pivot:
            continue
        vals = pivot[col].to_numpy()
        ax.bar(np.arange(len(pivot)), vals, bottom=bottom, color=colors[col], edgecolor="black", linewidth=0.55, label=labels[col])
        bottom += vals
    ax.set_ylabel("Number of samples")
    ax.set_xticks(np.arange(len(pivot)))
    ax.set_xticklabels(pivot.index, rotation=18, ha="right")
    ax.legend(frameon=False, loc="upper right")
    clean_axis(ax)
    save(fig, "error_breakdown_v8")


def automatic_metrics_panel(results: pd.DataFrame) -> None:
    rows = subset_results(results)
    x = np.arange(len(rows))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.05, 3.15))
    ax.bar(x - width / 2, rows["bleu4"], width=width, color=P["sky"], edgecolor="black", linewidth=0.55, label="BLEU-4")
    ax.bar(x + width / 2, rows["rouge_l"], width=width, color=P["teal"], edgecolor="black", linewidth=0.55, label="ROUGE-L")
    ax.set_ylabel("Text-overlap score")
    ax.set_ylim(0, 0.46)
    ax.set_xticks(x)
    ax.set_xticklabels(rows["short_name"], rotation=35, ha="right")
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax)
    save(fig, "automatic_metrics_panel_v8")


def main() -> None:
    results, judge = load_results()
    main_result_panel(results, judge)
    automatic_metrics_panel(results)
    strict_lenient_panel()
    cost_accuracy_panel()
    error_breakdown_panel()


if __name__ == "__main__":
    main()
