from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from scipy.stats import ttest_1samp, wilcoxon
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from server_experiment_v2 import handcrafted_one


BASE_URL = "https://bookdash.org/book-source-files/"
DEFAULT_DEGRADATIONS = (
    "contrast_loss",
    "content_crop",
    "occluding_prompt",
    "clutter_controls",
    "tiny_text_band",
    "overlap_text_block",
)

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
    "warn": "#D48C72",
}


@dataclass(frozen=True)
class PageRecord:
    book_slug: str
    book_title: str
    page_name: str
    view_path: str
    local_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-books", type=int, default=32)
    parser.add_argument("--pages-per-book", type=int, default=6)
    parser.add_argument("--max-pages", type=int, default=180)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--sleep", type=float, default=0.12)
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def http_get(session: requests.Session, url: str, *, binary: bool = False, timeout: int = 45) -> bytes | str:
    for attempt in range(4):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content if binary else resp.text
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("_") or "item"


def natural_key(path: str) -> tuple[int, str]:
    lower = path.lower()
    if "cover" in lower:
        return (-1, lower)
    m = re.search(r"page(\d+)", lower)
    if m:
        return (int(m.group(1)), lower)
    return (10_000, lower)


def crawl_bookdash_pages(
    out_dir: Path,
    max_books: int,
    pages_per_book: int,
    max_pages: int,
    seed: int,
    sleep: float,
    force_refresh: bool,
) -> pd.DataFrame:
    manifest_path = out_dir / "bookdash_pages_manifest.csv"
    if manifest_path.exists() and not force_refresh:
        manifest = pd.read_csv(manifest_path)
        existing = [Path(p).exists() for p in manifest["local_path"]]
        if manifest.shape[0] >= min(max_pages, max_books * pages_per_book) and all(existing):
            print(f"Using cached Book Dash pages: {manifest_path}")
            return manifest

    session = requests.Session()
    session.headers.update({"User-Agent": "story-layout-quality-research/0.1"})
    html = http_get(session, BASE_URL)
    slugs = sorted(set(re.findall(r'href="\?book=([^"&]+)"', html)))
    rng = random.Random(seed)
    rng.shuffle(slugs)

    rows: list[dict[str, str]] = []
    downloaded_books = 0
    for slug in slugs:
        if downloaded_books >= max_books or len(rows) >= max_pages:
            break
        folder_url = f"{BASE_URL}?book={quote(slug)}&folder=ebook/en_english/images"
        try:
            page_html = http_get(session, folder_url)
        except Exception as exc:
            print(f"skip {slug}: {exc}")
            continue
        title_match = re.search(r"<h1[^>]*>.*?<a[^>]*>(.*?)</a>", page_html, flags=re.S)
        title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", title_match.group(1))).strip() if title_match else slug
        paths = sorted(
            {
                p
                for p in re.findall(r'href="\?view-file=([^"]+\.(?:jpg|jpeg|png))"', page_html, flags=re.I)
                if "/images/" in p and "cover" not in p.lower()
            },
            key=natural_key,
        )
        if not paths:
            continue
        selected = paths[:pages_per_book]
        book_count = 0
        for view_path in selected:
            if len(rows) >= max_pages:
                break
            ext = Path(view_path).suffix.lower() or ".jpg"
            image_name = safe_name(f"{slug}_{Path(view_path).stem}{ext}")
            local_path = out_dir / "original_pages" / image_name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if not local_path.exists() or force_refresh:
                url = f"{BASE_URL}?view-file={quote(view_path, safe='/._-')}"
                try:
                    content = http_get(session, url, binary=True)
                    local_path.write_bytes(content)
                    time.sleep(sleep)
                except Exception as exc:
                    print(f"download failed {view_path}: {exc}")
                    continue
            try:
                with Image.open(local_path) as img:
                    img.verify()
            except Exception:
                local_path.unlink(missing_ok=True)
                continue
            rows.append(
                {
                    "book_slug": slug,
                    "book_title": title,
                    "page_name": Path(view_path).stem,
                    "view_path": view_path,
                    "source_url": f"{BASE_URL}?view-file={quote(view_path, safe='/._-')}",
                    "local_path": str(local_path),
                }
            )
            book_count += 1
        if book_count:
            downloaded_books += 1
            print(f"Book Dash {downloaded_books:02d}: {slug} pages={book_count}")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    print(f"Collected {len(manifest)} pages from {manifest['book_slug'].nunique() if len(manifest) else 0} books")
    return manifest


def fit_public_ui_models(root: Path, seed: int) -> tuple[RandomForestRegressor, RandomForestClassifier]:
    results = root / "experiment" / "results"
    df = pd.read_csv(results / "uicrit_rico_aligned.csv")
    x = np.nan_to_num(np.load(results / "handcrafted_visual_features.npy").astype(np.float32))
    y = df["design_quality"].to_numpy(float)
    reg = RandomForestRegressor(
        n_estimators=700,
        min_samples_leaf=2,
        max_features=0.75,
        n_jobs=-1,
        random_state=seed,
    )
    reg.fit(x, y)

    low, high = np.quantile(y, [0.30, 0.70])
    mask = (y <= low) | (y >= high)
    y_bin = (y[mask] >= high).astype(int)
    clf = RandomForestClassifier(
        n_estimators=700,
        min_samples_leaf=2,
        max_features=0.75,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed + 101,
    )
    clf.fit(x[mask], y_bin)
    return reg, clf


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def fit_canvas(img: Image.Image, long_edge: int = 1100) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, long_edge / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return img


def degradation_contrast_loss(img: Image.Image) -> Image.Image:
    out = ImageEnhance.Contrast(img).enhance(0.42)
    out = ImageEnhance.Color(out).enhance(0.55)
    out = ImageEnhance.Brightness(out).enhance(1.04)
    return out.filter(ImageFilter.GaussianBlur(radius=0.35))


def degradation_content_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    crop = img.crop((int(0.12 * w), int(0.06 * h), int(0.96 * w), int(0.94 * h)))
    return crop.resize((w, h), Image.LANCZOS)


def degradation_occluding_prompt(img: Image.Image) -> Image.Image:
    out = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    x0, y0 = int(w * 0.16), int(h * 0.36)
    x1, y1 = int(w * 0.84), int(h * 0.60)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=max(14, w // 40), fill=(51, 123, 172, 218))
    font = load_font(max(18, w // 24))
    small = load_font(max(13, w // 42))
    draw.text((x0 + w * 0.035, y0 + h * 0.045), "Tap to continue", fill=(255, 255, 255, 245), font=font)
    draw.text((x0 + w * 0.035, y0 + h * 0.125), "Prompt overlaps the story scene", fill=(238, 247, 250, 235), font=small)
    return out.convert("RGB")


def degradation_clutter_controls(img: Image.Image) -> Image.Image:
    out = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    rng = random.Random(int(hashlib.md5(np.asarray(img.resize((8, 8))).tobytes()).hexdigest()[:8], 16))
    colors = [(51, 123, 172, 230), (79, 177, 178, 225), (169, 217, 187, 225), (144, 180, 207, 225)]
    radius = max(14, int(min(w, h) * 0.035))
    positions = []
    for i in range(10):
        if i < 5:
            positions.append((int((0.12 + i * 0.18) * w), int(0.08 * h)))
        else:
            positions.append((int((0.12 + (i - 5) * 0.18) * w), int(0.90 * h)))
    for i, (cx, cy) in enumerate(positions):
        jitter = int(radius * 0.45)
        cx += rng.randint(-jitter, jitter)
        cy += rng.randint(-jitter, jitter)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=colors[i % len(colors)], outline=(31, 51, 73, 170), width=2)
        draw.line((cx - radius * 0.45, cy, cx + radius * 0.45, cy), fill=(255, 255, 255, 235), width=max(2, radius // 6))
        draw.line((cx, cy - radius * 0.45, cx, cy + radius * 0.45), fill=(255, 255, 255, 235), width=max(2, radius // 6))
    return out.convert("RGB")


def degradation_tiny_text_band(img: Image.Image) -> Image.Image:
    w, h = img.size
    scaled = img.resize((int(w * 0.78), int(h * 0.78)), Image.LANCZOS)
    out = Image.new("RGB", (w, h), "white")
    out.paste(scaled, ((w - scaled.size[0]) // 2, int(h * 0.03)))
    draw = ImageDraw.Draw(out)
    y = int(h * 0.83)
    line_h = max(5, h // 120)
    for i in range(11):
        x0 = int(w * (0.11 + 0.01 * (i % 3)))
        x1 = int(w * (0.90 - 0.05 * (i % 4)))
        gray = 150 + (i % 3) * 20
        draw.rectangle((x0, y + i * line_h * 2, x1, y + i * line_h * 2 + max(2, line_h // 2)), fill=(gray, gray, gray))
    return out


def degradation_overlap_text_block(img: Image.Image) -> Image.Image:
    out = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    x0, y0 = int(w * 0.08), int(h * 0.22)
    x1, y1 = int(w * 0.92), int(h * 0.50)
    draw.rectangle((x0, y0, x1, y1), fill=(245, 250, 252, 210), outline=(51, 123, 172, 185), width=max(2, w // 300))
    for i in range(9):
        y = y0 + int((i + 1) * (y1 - y0) / 11)
        x_end = x1 - int((i % 4) * 0.10 * w) - int(0.02 * w)
        draw.rectangle((x0 + int(0.03 * w), y, x_end, y + max(2, h // 180)), fill=(31, 51, 73, 180))
    return out.convert("RGB")


DEGRADERS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "contrast_loss": degradation_contrast_loss,
    "content_crop": degradation_content_crop,
    "occluding_prompt": degradation_occluding_prompt,
    "clutter_controls": degradation_clutter_controls,
    "tiny_text_band": degradation_tiny_text_band,
    "overlap_text_block": degradation_overlap_text_block,
}


def materialize_variants(pages: pd.DataFrame, out_dir: Path, degradations: tuple[str, ...]) -> pd.DataFrame:
    variant_dir = out_dir / "variants"
    variant_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    for row_id, row in pages.reset_index(drop=True).iterrows():
        src = Path(str(row["local_path"]))
        with Image.open(src) as raw:
            base = fit_canvas(raw)
        base_path = variant_dir / f"{row_id:04d}_{safe_name(row['book_slug'])}_{safe_name(row['page_name'])}_original.jpg"
        if not base_path.exists():
            base.save(base_path, quality=92)
        rows.append(
            {
                "page_id": row_id,
                "book_slug": row["book_slug"],
                "book_title": row["book_title"],
                "page_name": row["page_name"],
                "variant": "original",
                "is_original": 1,
                "degradation_type": "original",
                "image_path": str(base_path),
            }
        )
        for name in degradations:
            out_path = variant_dir / f"{row_id:04d}_{safe_name(row['book_slug'])}_{safe_name(row['page_name'])}_{name}.jpg"
            if not out_path.exists():
                DEGRADERS[name](base).save(out_path, quality=90)
            rows.append(
                {
                    "page_id": row_id,
                    "book_slug": row["book_slug"],
                    "book_title": row["book_title"],
                    "page_name": row["page_name"],
                    "variant": name,
                    "is_original": 0,
                    "degradation_type": name,
                    "image_path": str(out_path),
                }
            )
        if (row_id + 1) % 25 == 0:
            print(f"materialized variants for {row_id + 1}/{len(pages)} pages")
    variants = pd.DataFrame(rows)
    variants.to_csv(out_dir / "bookdash_variant_manifest.csv", index=False)
    return variants


def compute_features(variants: pd.DataFrame, out_dir: Path, force_refresh: bool = False) -> np.ndarray:
    feature_path = out_dir / "bookdash_variant_handcrafted_features.npy"
    if feature_path.exists() and not force_refresh:
        arr = np.load(feature_path)
        if arr.shape[0] == len(variants):
            return arr
    rows = []
    for i, path in enumerate(variants["image_path"]):
        rows.append(handcrafted_one(Path(path)))
        if (i + 1) % 100 == 0:
            print(f"BookDash handcrafted features: {i + 1}/{len(variants)}")
    arr = np.vstack(rows).astype(np.float32)
    np.save(feature_path, arr)
    return arr


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = np.mean(rng.choice(values, size=len(values), replace=True))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate_zero_shot(
    variants: pd.DataFrame,
    public_scores: np.ndarray,
    out_dir: Path,
    seed: int,
    score_name: str,
) -> pd.DataFrame:
    scored = variants.copy()
    scored[score_name] = public_scores
    scored.to_csv(out_dir / f"bookdash_variant_scores_{score_name}.csv", index=False)
    orig = scored[scored["variant"] == "original"][["page_id", score_name]].rename(columns={score_name: "original_score"})
    rows = []
    pair_rows = []
    for degr in DEFAULT_DEGRADATIONS:
        sub = scored[scored["variant"] == degr][["page_id", "book_slug", "page_name", score_name]]
        joined = sub.merge(orig, on="page_id", how="inner")
        diff = joined["original_score"].to_numpy(float) - joined[score_name].to_numpy(float)
        acc = (diff > 0).astype(float)
        ci_lo, ci_hi = bootstrap_ci(acc, seed + len(rows))
        d_lo, d_hi = bootstrap_ci(diff, seed + 100 + len(rows))
        try:
            t_p = float(ttest_1samp(diff, popmean=0.0, alternative="greater").pvalue)
        except TypeError:
            t_p = float(ttest_1samp(diff, popmean=0.0).pvalue / 2.0)
        try:
            w_p = float(wilcoxon(diff, alternative="greater").pvalue)
        except Exception:
            w_p = float("nan")
        rows.append(
            {
                "degradation_type": degr,
                "score_name": score_name,
                "n_pairs": len(joined),
                "ranking_accuracy": float(acc.mean()),
                "ranking_ci_low": ci_lo,
                "ranking_ci_high": ci_hi,
                "mean_score_drop": float(diff.mean()),
                "score_drop_ci_low": d_lo,
                "score_drop_ci_high": d_hi,
                "median_score_drop": float(np.median(diff)),
                "paired_t_p_greater": t_p,
                "wilcoxon_p_greater": w_p,
            }
        )
        joined["degradation_type"] = degr
        joined["score_drop"] = diff
        joined["rank_correct"] = diff > 0
        pair_rows.append(joined)
    pair_df = pd.concat(pair_rows, ignore_index=True)
    pair_df.to_csv(out_dir / f"bookdash_zero_shot_pairs_{score_name}.csv", index=False)
    summary = pd.DataFrame(rows)
    overall = pair_df["rank_correct"].astype(float).to_numpy()
    diff_all = pair_df["score_drop"].to_numpy(float)
    acc_lo, acc_hi = bootstrap_ci(overall, seed + 999)
    d_lo, d_hi = bootstrap_ci(diff_all, seed + 1000)
    summary = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "degradation_type": "overall",
                        "score_name": score_name,
                        "n_pairs": len(pair_df),
                        "ranking_accuracy": float(overall.mean()),
                        "ranking_ci_low": acc_lo,
                        "ranking_ci_high": acc_hi,
                        "mean_score_drop": float(diff_all.mean()),
                        "score_drop_ci_low": d_lo,
                        "score_drop_ci_high": d_hi,
                        "median_score_drop": float(np.median(diff_all)),
                        "paired_t_p_greater": float(ttest_1samp(diff_all, popmean=0.0, alternative="greater").pvalue),
                        "wilcoxon_p_greater": float(wilcoxon(diff_all, alternative="greater").pvalue),
                    }
                ]
            ),
            summary,
        ],
        ignore_index=True,
    )
    summary.to_csv(out_dir / f"bookdash_zero_shot_summary_{score_name}.csv", index=False)
    return summary


def evaluate_grouped_detector(features: np.ndarray, variants: pd.DataFrame, out_dir: Path, seed: int) -> pd.DataFrame:
    y = variants["is_original"].to_numpy(int)
    groups = variants["book_slug"].to_numpy(str)
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    scores = np.zeros(len(variants), dtype=float)
    preds = np.zeros(len(variants), dtype=int)
    for fold, (train, test) in enumerate(gkf.split(features, y, groups)):
        clf = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=0.75,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed + fold,
        )
        clf.fit(features[train], y[train])
        proba = clf.predict_proba(features[test])
        pos = list(clf.classes_).index(1)
        scores[test] = proba[:, pos]
        train_scores = clf.predict_proba(features[train])[:, pos]
        thresholds = np.linspace(0.05, 0.95, 91)
        train_ba = [balanced_accuracy_score(y[train], (train_scores >= t).astype(int)) for t in thresholds]
        threshold = float(thresholds[int(np.argmax(train_ba))])
        preds[test] = (scores[test] >= threshold).astype(int)
        print(f"grouped detector fold {fold + 1}/{n_splits} done")
    scored = variants.copy()
    scored["detector_score"] = scores
    scored["detector_pred"] = preds
    scored.to_csv(out_dir / "bookdash_detector_oof_scores.csv", index=False)

    rows = []
    for subset, mask in [("overall", np.ones(len(scored), dtype=bool))]:
        rows.append(
            {
                "subset": subset,
                "n": int(mask.sum()),
                "auc": float(roc_auc_score(y[mask], scores[mask])),
                "balanced_accuracy": float(balanced_accuracy_score(y[mask], preds[mask])),
            }
        )
    for degr in DEFAULT_DEGRADATIONS:
        mask = (scored["variant"] == "original") | (scored["variant"] == degr)
        rows.append(
            {
                "subset": degr,
                "n": int(mask.sum()),
                "auc": float(roc_auc_score(y[mask], scores[mask])),
                "balanced_accuracy": float(balanced_accuracy_score(y[mask], preds[mask])),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "bookdash_detector_summary.csv", index=False)
    return summary


def label_degradation(name: str) -> str:
    return {
        "overall": "Overall",
        "contrast_loss": "Contrast loss",
        "content_crop": "Content crop",
        "occluding_prompt": "Prompt overlap",
        "clutter_controls": "Control clutter",
        "tiny_text_band": "Tiny text band",
        "overlap_text_block": "Text-image overlap",
    }.get(name, name.replace("_", " "))


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(P["ink"])
    ax.spines["bottom"].set_color(P["ink"])
    ax.tick_params(axis="both", width=0.9, length=3.5, colors=P["ink"])
    ax.grid(axis="y", color=P["p1"], linewidth=0.7, alpha=0.82)


def save_figure(fig: plt.Figure, figures_dir: Path, name: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(figures_dir / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_results(zero: pd.DataFrame, detector: pd.DataFrame, figures_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "stix",
            "font.size": 10.0,
            "axes.labelsize": 10.8,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )
    order = ["overall", *DEFAULT_DEGRADATIONS]
    z = zero.set_index("degradation_type").loc[order].reset_index()
    d = detector.set_index("subset").loc[order].reset_index()
    colors = [P["p6"], P["p2"], P["p5"], P["p7"], P["p4"], P["p3"], P["warn"]]
    x = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.25), constrained_layout=True)
    y = z["ranking_accuracy"].to_numpy(float)
    yerr = np.vstack([y - z["ranking_ci_low"].to_numpy(float), z["ranking_ci_high"].to_numpy(float) - y])
    axes[0].barh(x, y, xerr=yerr, color=colors, edgecolor="black", linewidth=0.55, capsize=2.5)
    axes[0].axvline(0.5, linestyle="--", color=P["muted"], linewidth=1.0)
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_xlabel("Zero-shot ranking accuracy")
    axes[0].set_yticks(x)
    axes[0].set_yticklabels([label_degradation(v) for v in order])
    axes[0].invert_yaxis()
    axes[0].text(-0.08, 1.04, "A", transform=axes[0].transAxes, fontweight="bold", color=P["ink"])
    clean_axis(axes[0])

    axes[1].barh(x, d["auc"], color=colors, edgecolor="black", linewidth=0.55)
    axes[1].axvline(0.5, linestyle="--", color=P["muted"], linewidth=1.0)
    axes[1].set_xlim(0.5, 1.0)
    axes[1].set_xlabel("Grouped detector AUC")
    axes[1].set_yticks(x)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].text(-0.08, 1.04, "B", transform=axes[1].transAxes, fontweight="bold", color=P["ink"])
    clean_axis(axes[1])
    save_figure(fig, figures_dir, "bookdash_stress_test")


def plot_examples(variants: pd.DataFrame, figures_dir: Path, seed: int) -> None:
    available = variants[variants["variant"] == "original"]["page_id"].unique()
    rng = random.Random(seed)
    page_id = int(rng.choice(list(available)))
    show_variants = ["original", "contrast_loss", "occluding_prompt", "clutter_controls", "overlap_text_block"]
    labels = ["Original", "Contrast loss", "Prompt overlap", "Control clutter", "Text-image overlap"]
    subset = variants[(variants["page_id"] == page_id) & (variants["variant"].isin(show_variants))]
    paths = {r["variant"]: Path(r["image_path"]) for _, r in subset.iterrows()}
    fig, axes = plt.subplots(1, len(show_variants), figsize=(7.4, 2.2), constrained_layout=True)
    for ax, variant, label in zip(axes, show_variants, labels):
        with Image.open(paths[variant]) as img:
            im = img.convert("RGB")
            w, h = im.size
            if h > w:
                crop = im.crop((0, int(h * 0.08), w, int(h * 0.88)))
            else:
                crop = im
            ax.imshow(crop)
        ax.set_title(label, fontsize=9.4, color=P["ink"])
        ax.axis("off")
    save_figure(fig, figures_dir, "bookdash_degradation_examples")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = args.root.resolve()
    out_dir = root / "experiment" / "results" / "bookdash_stress"
    figures_dir = root / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = crawl_bookdash_pages(
        out_dir=out_dir,
        max_books=args.max_books,
        pages_per_book=args.pages_per_book,
        max_pages=args.max_pages,
        seed=args.seed,
        sleep=args.sleep,
        force_refresh=args.force_refresh,
    )
    if pages.empty:
        raise RuntimeError("No Book Dash pages were collected")

    variants = materialize_variants(pages, out_dir, DEFAULT_DEGRADATIONS)
    features = compute_features(variants, out_dir, force_refresh=args.force_refresh)
    reg, clf = fit_public_ui_models(root, seed=args.seed)
    reg_scores = reg.predict(features)
    proba = clf.predict_proba(features)
    pos = list(clf.classes_).index(1)
    screening_scores = proba[:, pos]

    zero_reg = evaluate_zero_shot(variants, reg_scores, out_dir, seed=args.seed, score_name="public_regression_score")
    zero_screen = evaluate_zero_shot(
        variants,
        screening_scores,
        out_dir,
        seed=args.seed + 7,
        score_name="public_screening_score",
    )
    zero = zero_screen
    detector = evaluate_grouped_detector(features, variants, out_dir, seed=args.seed)
    pd.concat([zero_reg, zero_screen], ignore_index=True).to_csv(out_dir / "bookdash_zero_shot_summary.csv", index=False)
    plot_results(zero, detector, figures_dir)
    plot_examples(variants, figures_dir, seed=args.seed)

    manifest = {
        "source": "Book Dash book source files",
        "source_url": BASE_URL,
        "license_note": "Book Dash source page describes the books as CC BY 4.0 open content.",
        "books": int(pages["book_slug"].nunique()),
        "original_pages": int(len(pages)),
        "variants_total": int(len(variants)),
        "degradation_types": list(DEFAULT_DEGRADATIONS),
        "zero_shot_regression_overall": zero_reg[zero_reg["degradation_type"] == "overall"].iloc[0].to_dict(),
        "zero_shot_screening_overall": zero_screen[zero_screen["degradation_type"] == "overall"].iloc[0].to_dict(),
        "detector_overall": detector[detector["subset"] == "overall"].iloc[0].to_dict(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
