from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FIGURES = RUNS / "report_figures_v7"
FIGURES.mkdir(parents=True, exist_ok=True)

P = {
    "p1": "#D5E8F1",
    "p2": "#ABD7DF",
    "p3": "#CAEBE7",
    "p4": "#A9D9BB",
    "p5": "#90B4CF",
    "p6": "#337BAC",
    "p7": "#4FB1B2",
    "accent": "#D88C45",
    "ink": "#1F3349",
    "muted": "#6C7A86",
}

plt.rcParams.update(
    {
        "font.family": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.2,
        "axes.labelsize": 10.8,
        "axes.titlesize": 11.4,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "axes.linewidth": 0.95,
        "lines.linewidth": 1.35,
        "axes.edgecolor": P["ink"],
        "axes.labelcolor": P["ink"],
        "xtick.color": P["ink"],
        "ytick.color": P["ink"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    }
)


DISPLAY = {
    "RNN_full": "RNN full",
    "Transformer_full": "Transformer full",
    "Transformer_final_opt": "Transformer final opt",
    "Pretrained_T5_base_final": "T5-base final",
    "Local_Qwen25_Math_15B_CoT": "Qwen2.5-Math-1.5B",
    "DeepSeek_v4_flash": "DeepSeek-V4-Flash",
    "DeepSeek_v4_pro": "DeepSeek-V4-Pro",
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

BAR_COLORS = [P["p5"], P["p2"], P["p7"], P["p4"], P["p6"], "#7FB8B2", P["accent"]]


def clean_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=0.9, length=3.5)
    ax.grid(axis=grid_axis, color=P["p1"], linewidth=0.7, alpha=0.82)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIGURES / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(RUNS / "second_result_summary_v6.csv")
    judge = pd.read_csv(RUNS / "llm_judge_v6" / "llm_judge_summary.csv")
    return results, judge


def subset_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = results.set_index("model").loc[ORDER].reset_index()
    rows["short_name"] = rows["model"].map(DISPLAY)
    return rows


def main_result_panel(results: pd.DataFrame, judge: pd.DataFrame) -> None:
    rows = subset_results(results)
    judge_map = judge.set_index("model")["llm_reasoning_validity"].to_dict()
    rows["reasoning"] = rows["model"].map(judge_map)

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.35), constrained_layout=True)
    x = np.arange(len(rows))

    axes[0].bar(x, rows["final_answer_acc"], color=BAR_COLORS, edgecolor="black", linewidth=0.55)
    axes[0].set_ylabel("Final answer accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(rows["short_name"], rotation=35, ha="right")
    axes[0].text(-0.08, 1.04, "A", transform=axes[0].transAxes, fontweight="bold", fontsize=13)
    for xi, value in zip(x, rows["final_answer_acc"]):
        axes[0].text(xi, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=8.2, color=P["ink"])

    judge_rows = rows.dropna(subset=["reasoning"])
    x2 = np.arange(len(judge_rows))
    colors2 = [BAR_COLORS[ORDER.index(model)] for model in judge_rows["model"]]
    axes[1].bar(x2, judge_rows["reasoning"], color=colors2, edgecolor="black", linewidth=0.55)
    axes[1].set_ylabel("LLM reasoning validity")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(judge_rows["short_name"], rotation=35, ha="right")
    axes[1].text(-0.08, 1.04, "B", transform=axes[1].transAxes, fontweight="bold", fontsize=13)
    for xi, value in zip(x2, judge_rows["reasoning"]):
        axes[1].text(xi, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=8.2, color=P["ink"])

    for ax in axes:
        clean_axis(ax)
    save(fig, "main_result_panel_v7")


def automatic_metrics_panel(results: pd.DataFrame) -> None:
    rows = subset_results(results)
    x = np.arange(len(rows))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15), constrained_layout=True)

    axes[0].bar(x - width / 2, rows["bleu4"], width=width, color=P["p5"], edgecolor="black", linewidth=0.55, label="BLEU-4")
    axes[0].bar(x + width / 2, rows["rouge_l"], width=width, color=P["p7"], edgecolor="black", linewidth=0.55, label="ROUGE-L")
    axes[0].set_ylabel("Text-overlap score")
    axes[0].set_ylim(0, 0.46)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(rows["short_name"], rotation=35, ha="right")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].text(-0.08, 1.04, "A", transform=axes[0].transAxes, fontweight="bold", fontsize=13)

    compact = rows[rows["model"].isin(["RNN_full", "Transformer_full", "Local_Qwen25_Math_15B_CoT", "DeepSeek_v4_flash", "DeepSeek_v4_pro"])]
    x2 = np.arange(len(compact))
    axes[1].bar(x2, compact["rouge_l"], color=[BAR_COLORS[ORDER.index(m)] for m in compact["model"]], edgecolor="black", linewidth=0.55)
    axes[1].set_ylabel("ROUGE-L")
    axes[1].set_ylim(0, 0.46)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(compact["short_name"], rotation=35, ha="right")
    axes[1].text(-0.08, 1.04, "B", transform=axes[1].transAxes, fontweight="bold", fontsize=13)

    for ax in axes:
        clean_axis(ax)
    save(fig, "automatic_metrics_panel_v7")


def custom_metrics_panel(results: pd.DataFrame, judge: pd.DataFrame) -> None:
    judged_order = ["RNN_full", "Transformer_full", "Local_Qwen25_Math_15B_CoT", "DeepSeek_v4_flash", "DeepSeek_v4_pro"]
    rows = results.set_index("model").loc[judged_order].reset_index()
    rows["short_name"] = rows["model"].map(DISPLAY)
    judge_map = judge.set_index("model")["llm_reasoning_validity"].to_dict()
    rows["reasoning"] = rows["model"].map(judge_map)
    x = np.arange(len(rows))
    width = 0.24

    fig, ax = plt.subplots(figsize=(6.7, 3.25))
    ax.bar(x - width, rows["final_answer_acc"], width=width, color=P["p6"], edgecolor="black", linewidth=0.55, label="Final answer")
    ax.bar(x, rows["format_rate"], width=width, color=P["p3"], edgecolor="black", linewidth=0.55, label="Format compliance")
    ax.bar(x + width, rows["reasoning"].fillna(0), width=width, color=P["accent"], edgecolor="black", linewidth=0.55, label="LLM reasoning")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(rows["short_name"], rotation=35, ha="right")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    clean_axis(ax)
    save(fig, "custom_metrics_panel_v7")


def diagnostic_equation(results: pd.DataFrame) -> None:
    rows = results[results["model"].isin(["RNN_full", "Transformer_full"])].copy()
    rows["short_name"] = rows["model"].map(DISPLAY)
    x = np.arange(len(rows))
    width = 0.34

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.bar(x - width / 2, rows["equation_coverage"], width=width, color=P["p5"], edgecolor="black", linewidth=0.55, label="Equation coverage")
    ax.bar(x + width / 2, rows["equation_step_acc"], width=width, color=P["p7"], edgecolor="black", linewidth=0.55, label="Equation step acc.")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Diagnostic score")
    ax.set_xticks(x)
    ax.set_xticklabels(rows["short_name"])
    ax.legend(frameon=False, loc="upper right")
    clean_axis(ax)
    save(fig, "diagnostic_equation_v7")


def deepseek_cache(results: pd.DataFrame) -> None:
    rows = results[results["model"].isin(["DeepSeek_v4_flash", "DeepSeek_v4_pro"])].copy()
    rows["short_name"] = rows["model"].map(DISPLAY)
    hit = rows["prompt_cache_hit_tokens"].astype(float) / 1000
    miss = rows["prompt_cache_miss_tokens"].astype(float) / 1000
    rate = rows["prompt_cache_hit_rate"].astype(float)
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(5.6, 3.25))
    ax.bar(x, miss, color=P["accent"], edgecolor="black", linewidth=0.55, label="Cache miss")
    ax.bar(x, hit, bottom=miss, color=P["p7"], edgecolor="black", linewidth=0.55, label="Cache hit")
    ax.set_ylabel("Prompt tokens (K)")
    ax.set_xticks(x)
    ax.set_xticklabels(rows["short_name"])
    ax.set_ylim(0, float((hit + miss).max()) * 1.20)
    for xi, total, value in zip(x, hit + miss, rate):
        ax.text(xi, total + 18, f"Hit rate {value:.1%}", ha="center", fontsize=9.0, color=P["ink"])
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    clean_axis(ax)
    save(fig, "deepseek_cache_v7")


def main() -> None:
    results, judge = load_data()
    main_result_panel(results, judge)
    automatic_metrics_panel(results)
    custom_metrics_panel(results, judge)
    diagnostic_equation(results)
    deepseek_cache(results)


if __name__ == "__main__":
    main()
