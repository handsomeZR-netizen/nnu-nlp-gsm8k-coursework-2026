from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Wedge


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiment" / "results" / "figure_experiments"
FIGURES = ROOT / "paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

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
        "font.size": 9.8,
        "axes.labelsize": 10.8,
        "axes.titlesize": 11.2,
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


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=0.9, length=3.5)
    ax.grid(axis="y", color=P["p1"], linewidth=0.7, alpha=0.82)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIGURES / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = y_true - pred
    rho = pd.Series(y_true).corr(pd.Series(pred), method="spearman")
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "spearman": float(0.0 if pd.isna(rho) else rho),
    }


def oof_regression_marginal() -> None:
    df = pd.read_csv(RESULTS / "oof_regression_predictions.csv")
    metrics = regression_metrics(df["true_quality"].to_numpy(float), df["pred_visual_rf"].to_numpy(float))
    groups = [
        ("Other categories", ~df["is_education_reading"].astype(bool), P["p6"]),
        ("Education/reading", df["is_education_reading"].astype(bool), P["p7"]),
    ]

    fig = plt.figure(figsize=(6.8, 4.25))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(4.8, 0.78),
        height_ratios=(0.72, 3.45),
        hspace=0.05,
        wspace=0.05,
    )
    ax_histx = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_histx)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    bins_x = np.linspace(3.5, 7.8, 20)
    bins_y = np.linspace(5.1, 6.8, 20)
    for label, mask, color in groups:
        sub = df.loc[mask]
        ax.scatter(
            sub["true_quality"],
            sub["pred_visual_rf"],
            s=28 if label == "Education/reading" else 20,
            c=color,
            edgecolors="white",
            linewidths=0.42,
            alpha=0.78,
            label=f"{label} (n={len(sub)})",
        )
        if len(sub) > 5:
            x = sub["true_quality"].to_numpy(float)
            y = sub["pred_visual_rf"].to_numpy(float)
            coef = np.polyfit(x, y, deg=1)
            xx = np.linspace(x.min(), x.max(), 60)
            ax.plot(xx, coef[0] * xx + coef[1], color=color, linewidth=1.95)
        ax_histx.hist(sub["true_quality"], bins=bins_x, color=color, alpha=0.72, stacked=False)
        ax_histy.hist(sub["pred_visual_rf"], bins=bins_y, orientation="horizontal", color=color, alpha=0.72)

    lim = (3.55, 7.75)
    ax.plot(lim, lim, linestyle="--", color=P["muted"], linewidth=1.15, label="Ideal calibration")
    ax.set_xlim(lim)
    ax.set_ylim(5.05, 6.85)
    ax.set_xlabel("Human design-quality rating")
    ax.set_ylabel("Out-of-fold predicted quality")
    ax.text(
        0.03,
        0.96,
        f"Handcrafted visual RF\nMAE={metrics['mae']:.3f}, Spearman={metrics['spearman']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=P["ink"],
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=P["p2"], linewidth=0.9),
    )

    ax_histx.axis("off")
    ax_histy.axis("off")
    clean_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.50, -0.01),
        ncol=3,
        columnspacing=1.2,
        handlelength=2.2,
    )
    fig.subplots_adjust(bottom=0.18)
    save(fig, "oof_regression_marginal")


def feature_label(name: str) -> str:
    labels = {
        "Design quality": "Quality",
        "OOF prediction": "OOF RF",
        "colorfulness": "Colorfulness",
        "horizontal_symmetry_error": "H-sym error",
        "vertical_symmetry_error": "V-sym error",
        "sat_std": "Sat. SD",
        "luma_entropy": "Luma entropy",
    }
    if name in labels:
        return labels[name]
    if name.startswith("grid_edge_"):
        _, _, r, c = name.split("_")
        pos_r = ["top", "upper", "lower", "bottom"][int(r)]
        pos_c = ["left", "mid-left", "mid-right", "right"][int(c)]
        return f"{pos_r} {pos_c}\nedge"
    if name.startswith("grid_sat_"):
        _, _, r, c = name.split("_")
        pos_r = ["top", "upper", "lower", "bottom"][int(r)]
        pos_c = ["left", "mid-left", "mid-right", "right"][int(c)]
        return f"{pos_r} {pos_c}\nsat."
    return name.replace("_", " ")


def feature_corr_pie_heatmap() -> None:
    raw = pd.read_csv(RESULTS / "correlation_panel_data.csv")
    keep = list(raw.columns[:9])
    data = raw[keep].rename(columns={c: feature_label(c) for c in keep})
    corr = data.corr(method="spearman").to_numpy(float)
    labels = list(data.columns)
    n = len(labels)
    cmap = LinearSegmentedColormap.from_list("story_corr", [P["p5"], "white", P["p7"]])
    norm = Normalize(vmin=-0.35, vmax=0.35)

    fig, ax = plt.subplots(figsize=(6.45, 6.02))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    for i in range(n):
        for j in range(n):
            r = corr[i, j]
            ax.add_patch(Circle((j, i), 0.43, facecolor=P["p1"], edgecolor="white", linewidth=1.0))
            if i == j:
                ax.add_patch(Circle((j, i), 0.34, facecolor=P["p6"], edgecolor="white", linewidth=0.8))
                ax.text(j, i, "1.00", ha="center", va="center", color="white", fontsize=8.6, fontweight="bold")
                continue
            radius = 0.38 * np.sqrt(abs(r))
            start = 90
            end = 90 + 360 * abs(r)
            ax.add_patch(Circle((j, i), radius, facecolor="white", edgecolor=P["p2"], linewidth=0.5))
            ax.add_patch(
                Wedge(
                    (j, i),
                    radius,
                    start,
                    end,
                    facecolor=cmap(norm(r)),
                    edgecolor="white",
                    linewidth=0.5,
                )
            )
            if abs(r) >= 0.08:
                ax.text(j, i, f"{r:.2f}", ha="center", va="center", color=P["ink"], fontsize=7.7)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.025)
    cbar.set_label("Spearman correlation", fontsize=10.0)
    save(fig, "feature_corr_pie_heatmap")


def tier_fan_confusion() -> None:
    pred = pd.read_csv(RESULTS / "tier_diagnostic_predictions.csv")
    metrics = pd.read_csv(RESULTS / "tier_diagnostic_metrics.csv")
    model = "V2 handcrafted RF, 30%"
    sub = pred[pred["model"] == model].copy()
    met = metrics[metrics["model"] == model].iloc[0]
    counts = {
        "TN": int(((sub["tier_label"] == 0) & (sub["tier_pred"] == 0)).sum()),
        "FP": int(((sub["tier_label"] == 0) & (sub["tier_pred"] == 1)).sum()),
        "FN": int(((sub["tier_label"] == 1) & (sub["tier_pred"] == 0)).sum()),
        "TP": int(((sub["tier_label"] == 1) & (sub["tier_pred"] == 1)).sum()),
    }
    total = sum(counts.values())
    cells = [
        ("TN", "Low -> low", counts["TN"], 45, P["p6"]),
        ("FP", "Low -> high", counts["FP"], 135, P["p2"]),
        ("FN", "High -> low", counts["FN"], 225, P["p4"]),
        ("TP", "High -> high", counts["TP"], 315, P["p7"]),
    ]
    max_count = max(counts.values())

    fig = plt.figure(figsize=(4.6, 3.72))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_ylim(0, 0.96)
    ax.grid(False)
    ax.set_axis_off()

    for code, label, value, angle, color in cells:
        height = 0.18 + 0.42 * (value / max_count)
        theta = np.deg2rad(angle)
        ax.bar(
            theta,
            height,
            width=np.deg2rad(72),
            bottom=0.18,
            color=color,
            edgecolor="white",
            linewidth=1.5,
            alpha=0.98,
        )
        text_r = 0.18 + height + 0.075
        ax.text(
            theta,
            text_r,
            f"{label}\n{code}={value}\n{value / total:.1%}",
            ha="center",
            va="center",
            color=P["ink"],
            fontsize=8.5,
            fontweight="bold" if code in {"TN", "TP"} else "normal",
        )

    ax.text(
        0,
        0.035,
        f"Pooled OOF screening\nAUC={met['auc']:.3f}\nBal. acc.={met['balanced_accuracy']:.3f}",
        ha="center",
        va="center",
        color=P["ink"],
        fontsize=8.8,
        fontweight="bold",
    )
    save(fig, "tier_fan_confusion")


def pca_tier_violin() -> None:
    pca = pd.read_csv(RESULTS / "pca_tier_coordinates.csv")
    summary = json.loads((RESULTS / "diagnostic_summary.json").read_text(encoding="utf-8"))
    order = ["Bottom quartile", "Middle 50%", "Top quartile"]
    colors = {"Bottom quartile": P["p5"], "Middle 50%": P["p1"], "Top quartile": P["p7"]}

    fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.05), gridspec_kw={"width_ratios": [1.34, 1.0]})
    ax = axes[0]
    for tier in order:
        sub = pca[pca["tier"] == tier]
        ax.scatter(
            sub["pc1"],
            sub["pc2"],
            s=20 if tier != "Middle 50%" else 14,
            color=colors[tier],
            edgecolor="white",
            linewidth=0.3,
            alpha=0.78 if tier != "Middle 50%" else 0.45,
            label=f"{tier} (n={len(sub)})",
        )
    evr = summary["pca_explained_variance_ratio"]
    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% var.)")
    handles, labels = ax.get_legend_handles_labels()
    ax.text(-0.10, 1.04, "A", transform=ax.transAxes, fontweight="bold")
    clean_axis(ax)

    axv = axes[1]
    data = [pca.loc[pca["tier"] == tier, "pc1"].to_numpy(float) for tier in order]
    parts = axv.violinplot(data, positions=np.arange(len(order)), widths=0.78, showmeans=False, showmedians=False)
    for body, tier in zip(parts["bodies"], order):
        body.set_facecolor(colors[tier])
        body.set_edgecolor(P["ink"])
        body.set_alpha(0.82)
    for key in ("cbars", "cmins", "cmaxes"):
        parts[key].set_color(P["ink"])
        parts[key].set_linewidth(0.8)
    for i, values in enumerate(data):
        q1, med, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        axv.plot([i - 0.16, i + 0.16], [med, med], color=P["ink"], linewidth=1.3)
        axv.plot([i, i], [q1, q3], color=P["ink"], linewidth=2.2)
    axv.set_xticks(np.arange(len(order)))
    axv.set_xticklabels(["Bottom\nquartile", "Middle\n50%", "Top\nquartile"])
    axv.set_ylabel("PC1 distribution")
    axv.text(
        0.03,
        0.96,
        f"Top/bottom silhouette = {summary['top_bottom_silhouette_visual_handcrafted']:.3f}",
        transform=axv.transAxes,
        ha="left",
        va="top",
        color=P["ink"],
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=P["p2"], linewidth=0.8),
    )
    axv.text(-0.10, 1.04, "B", transform=axv.transAxes, fontweight="bold")
    clean_axis(axv)
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.44, 1.02),
        ncol=3,
        columnspacing=1.1,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    save(fig, "pca_tier_violin")


oof_regression_marginal()
feature_corr_pie_heatmap()
tier_fan_confusion()
pca_tier_violin()

print("wrote diagnostic figures to", FIGURES)
