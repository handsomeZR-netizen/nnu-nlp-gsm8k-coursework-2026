from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiment" / "results"
FIGURES = ROOT / "paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

cv = pd.read_csv(RESULTS / "server_v2_cv_metrics.csv")
tier = pd.read_csv(RESULTS / "server_v2_tier_metrics.csv")
transfer = pd.read_csv(RESULTS / "server_v2_transfer_metrics.csv")

P = {
    "p1": "#D5E8F1",
    "p2": "#ABD7DF",
    "p3": "#CAEBE7",
    "p4": "#A9D9BB",
    "p5": "#90B4CF",
    "p6": "#337BAC",
    "p7": "#4FB1B2",
    "ink": "#1F3349",
    "muted": "#6C7A86",
}

plt.rcParams.update(
    {
        "font.family": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.2,
        "axes.labelsize": 11.0,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.2,
        "legend.fontsize": 9.2,
        "axes.linewidth": 0.95,
        "lines.linewidth": 1.35,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    }
)

palette = {
    "Mean": P["p5"],
    "CLIP": P["p2"],
    "Hand": P["p6"],
    "Layout+Hand": P["p7"],
    "CLIP+Hand": P["p4"],
    "All": P["p3"],
}


def sem(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def summarize_metric(data: pd.DataFrame, selected: list[tuple[str, str, str]]) -> pd.DataFrame:
    rows = []
    for block, method, label in selected:
        sub = data[(data["block"] == block) & (data["method"] == method)]
        rows.append(
            {
                "label": label,
                "mae": sub["mae"].mean(),
                "mae_sem": sem(sub["mae"]),
                "spearman": sub["spearman"].mean(),
                "spearman_sem": sem(sub["spearman"]),
                "pairwise_acc": sub["pairwise_acc"].mean(),
                "pairwise_sem": sem(sub["pairwise_acc"]),
            }
        )
    return pd.DataFrame(rows)


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(P["ink"])
    ax.spines["bottom"].set_color(P["ink"])
    ax.tick_params(axis="both", labelsize=9.2, width=0.9, length=3.5)
    ax.grid(axis="y", color=P["p1"], linewidth=0.7, alpha=0.8)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIGURES / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def framework_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.04, 0.64, 0.20, 0.22, "Public UI data\nUICrit ratings\nRICO screens"),
        (0.30, 0.64, 0.20, 0.22, "Feature extraction\nLayout vectors\nCLIP embeddings\nVisual descriptors"),
        (0.56, 0.64, 0.20, 0.22, "Model families\nRF / ExtraTrees\nFeature selection\nPairRankNet"),
        (0.78, 0.64, 0.18, 0.22, "Evaluation\nGrouped CV\nRanking\nProxy transfer"),
        (0.18, 0.18, 0.26, 0.22, "Quality signals\nContinuous score\nTop/bottom tiers\nPairwise order"),
        (0.56, 0.18, 0.28, 0.22, "Story-layout use\nFirst-pass screening\nHuman review support\nChild-data boundary"),
    ]
    colors = [P["p1"], P["p2"], P["p3"], P["p4"], "#EEF6FF", "#EAF7F5"]
    for i, (x, y, w, h, text) in enumerate(boxes):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            linewidth=0.9,
            edgecolor=P["p6"],
            facecolor=colors[i],
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.4, color=P["ink"])

    arrows = [
        ((0.24, 0.75), (0.30, 0.75)),
        ((0.50, 0.75), (0.56, 0.75)),
        ((0.76, 0.75), (0.78, 0.75)),
        ((0.65, 0.64), (0.65, 0.40)),
        ((0.31, 0.64), (0.31, 0.40)),
        ((0.44, 0.29), (0.56, 0.29)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.0, color=P["p6"]))

    ax.text(0.04, 0.94, "Overall assessment pipeline", fontsize=12.4, weight="bold", color=P["ink"])
    ax.text(
        0.04,
        0.90,
        "The study treats public UI quality as proxy supervision and reports the evidence boundary explicitly.",
        fontsize=9.2,
        color=P["muted"],
    )
    save(fig, "framework_overview")


def main_results_figure() -> None:
    selected = [
        ("layout64", "Mean", "Mean"),
        ("clip512", "RF", "CLIP RF"),
        ("visual_handcrafted", "RF", "Hand RF"),
        ("layout64_visual_handcrafted", "RF", "Layout+Hand RF"),
        ("clip512_visual_handcrafted", "FS128-ExtraTrees", "CLIP+Hand FS-ET"),
        ("layout64_clip512_visual_handcrafted", "PairRankNet", "All PairRankNet"),
    ]
    plot_df = summarize_metric(cv, selected)
    colors = [
        palette["Mean"],
        palette["CLIP"],
        palette["Hand"],
        palette["Layout+Hand"],
        palette["CLIP+Hand"],
        palette["All"],
    ]
    x = np.arange(len(plot_df))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), constrained_layout=True)

    axes[0].bar(x, plot_df["mae"], yerr=plot_df["mae_sem"], color=colors, edgecolor="black", linewidth=0.55)
    axes[0].set_ylabel("MAE (lower is better)")
    axes[0].set_ylim(0.45, 0.56)
    axes[0].axhline(float(plot_df.loc[plot_df["label"] == "Mean", "mae"].iloc[0]), color=P["muted"], linestyle="--", linewidth=1.05)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["label"], rotation=35, ha="right")
    axes[0].text(-0.08, 1.04, "A", transform=axes[0].transAxes, fontweight="bold")

    axes[1].bar(x, plot_df["spearman"], yerr=plot_df["spearman_sem"], color=colors, edgecolor="black", linewidth=0.55)
    axes[1].set_ylabel("Spearman correlation")
    axes[1].set_ylim(-0.02, 0.18)
    axes[1].axhline(0, color=P["muted"], linestyle="--", linewidth=1.05)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_df["label"], rotation=35, ha="right")
    axes[1].text(-0.08, 1.04, "B", transform=axes[1].transAxes, fontweight="bold")

    for ax in axes:
        clean_axis(ax)
    save(fig, "main_results")


def ranknet_figure() -> None:
    selected = [
        ("layout64", "PairRankNet", "Layout"),
        ("visual_handcrafted", "PairRankNet", "Hand"),
        ("clip512", "PairRankNet", "CLIP"),
        ("layout64_clip512_visual_handcrafted", "PairRankNet", "All"),
    ]
    plot_df = summarize_metric(cv, selected)
    colors = [P["p5"], P["p7"], P["p2"], P["p6"]]
    x = np.arange(len(plot_df))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), constrained_layout=True)

    axes[0].bar(x, plot_df["mae"], yerr=plot_df["mae_sem"], color=colors, edgecolor="black", linewidth=0.55)
    axes[0].set_ylim(0.50, 0.60)
    axes[0].set_ylabel("PairRankNet MAE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["label"], rotation=20, ha="right")
    axes[0].text(-0.08, 1.04, "A", transform=axes[0].transAxes, fontweight="bold")

    axes[1].bar(x, plot_df["pairwise_acc"], yerr=plot_df["pairwise_sem"], color=colors, edgecolor="black", linewidth=0.55)
    axes[1].set_ylim(0.48, 0.57)
    axes[1].set_ylabel("Pairwise accuracy")
    axes[1].axhline(0.5, color=P["muted"], linestyle="--", linewidth=1.05)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_df["label"], rotation=20, ha="right")
    axes[1].text(-0.08, 1.04, "B", transform=axes[1].transAxes, fontweight="bold")

    for ax in axes:
        clean_axis(ax)
    save(fig, "dml_visual_gain")


def screening_transfer_figure() -> None:
    tier_sel = [
        ("visual_handcrafted", "RF", "Hand RF"),
        ("visual_handcrafted", "ExtraTrees", "Hand ET"),
        ("layout64_visual_handcrafted", "RF", "Layout+Hand RF"),
        ("layout64_visual_handcrafted", "ExtraTrees", "Layout+Hand ET"),
        ("clip512_visual_handcrafted", "ExtraTrees", "CLIP+Hand ET"),
    ]
    tier_rows = []
    for block, method, label in tier_sel:
        sub = tier[(tier["block"] == block) & (tier["method"] == method)]
        tier_rows.append({"label": label, "auc": sub["auc"].mean(), "auc_sem": sem(sub["auc"])})
    tier_df = pd.DataFrame(tier_rows)

    transfer_sel = [
        ("layout64_visual_handcrafted", "RF", "Layout+Hand RF"),
        ("layout64_visual_handcrafted", "FS128-ExtraTrees", "Layout+Hand FS-ET"),
        ("layout64", "RF", "Layout RF"),
        ("layout64_clip512_visual_handcrafted", "kNN", "All kNN"),
        ("clip512_visual_handcrafted", "RF", "CLIP+Hand RF"),
    ]
    trans_rows = []
    for block, method, label in transfer_sel:
        sub = transfer[(transfer["block"] == block) & (transfer["method"] == method)]
        trans_rows.append({"label": label, "mae": sub["mae"].mean()})
    trans_df = pd.DataFrame(trans_rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)
    x1 = np.arange(len(tier_df))
    axes[0].bar(x1, tier_df["auc"], yerr=tier_df["auc_sem"], color=P["p6"], edgecolor="black", linewidth=0.55)
    axes[0].axhline(0.5, color=P["muted"], linestyle="--", linewidth=1.05)
    axes[0].set_ylim(0.48, 0.62)
    axes[0].set_ylabel("Top/bottom AUC")
    axes[0].set_xticks(x1)
    axes[0].set_xticklabels(tier_df["label"], rotation=35, ha="right")
    axes[0].text(-0.08, 1.04, "A", transform=axes[0].transAxes, fontweight="bold")

    x2 = np.arange(len(trans_df))
    axes[1].bar(x2, trans_df["mae"], color=P["p7"], edgecolor="black", linewidth=0.55)
    axes[1].set_ylim(0.42, 0.47)
    axes[1].set_ylabel("Proxy-transfer MAE")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(trans_df["label"], rotation=35, ha="right")
    axes[1].text(-0.08, 1.04, "B", transform=axes[1].transAxes, fontweight="bold")

    for ax in axes:
        clean_axis(ax)
    save(fig, "screening_transfer")


framework_figure()
main_results_figure()
ranknet_figure()
screening_transfer_figure()

print("wrote updated V2 figures to", FIGURES)
