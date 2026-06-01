from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "experiment" / "results"
FIGURES = ROOT / "paper" / "figures"


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    uicrit = pd.read_csv(DATA / "uicrit_public.csv")
    grouped = (
        uicrit.groupby("rico_id", as_index=False)
        .agg(
            n_ratings=("design_quality_rating", "size"),
            design_quality=("design_quality_rating", "mean"),
            aesthetics=("aesthetics_rating", "mean"),
            learnability=("learnability", "mean"),
            efficiency=("efficency", "mean"),
            usability=("usability_rating", "mean"),
        )
        .rename(columns={"rico_id": "ui_number"})
    )

    ui_details = pd.read_csv(DATA / "rico" / "ui_details.csv")
    app_details = pd.read_csv(DATA / "rico" / "app_details.csv")
    ui_details = ui_details.rename(columns={"UI Number": "ui_number"})

    names_path = DATA / "rico" / "ui_layout_vectors" / "ui_layout_vectors" / "ui_names.json"
    vectors_path = DATA / "rico" / "ui_layout_vectors" / "ui_layout_vectors" / "ui_vectors.npy"
    with names_path.open("r", encoding="utf-8") as f:
        names = json.load(f)["ui_names"]
    vectors = np.load(vectors_path)
    index = {int(name.replace(".png", "")): i for i, name in enumerate(names)}

    vec_rows = []
    for ui_number in grouped["ui_number"].astype(int):
        if ui_number in index:
            vec_rows.append(vectors[index[ui_number]])
        else:
            vec_rows.append(np.full(64, np.nan, dtype=float))
    vec_df = pd.DataFrame(vec_rows, columns=[f"layout_{i:02d}" for i in range(64)])
    out = pd.concat([grouped.reset_index(drop=True), vec_df], axis=1)
    out = out.merge(ui_details, on="ui_number", how="left")
    out = out.merge(app_details, on="App Package Name", how="left")
    out = out.dropna(subset=[f"layout_{i:02d}" for i in range(64)] + ["design_quality"])
    out["target_domain"] = out["Category"].isin(["Education", "Books & Reference", "Comics"])
    return out


def standardize(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return (x_train - mu) / sd, (x_test - mu) / sd


def select_top_features(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    scores = []
    y0 = y - y.mean()
    y_sd = y0.std()
    for j in range(x.shape[1]):
        col = x[:, j] - x[:, j].mean()
        denom = col.std() * y_sd
        scores.append(0.0 if denom < 1e-9 else abs(float(np.mean(col * y0) / denom)))
    return np.argsort(scores)[::-1][:k]


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    xb = np.c_[np.ones(len(x_train)), x_train]
    xt = np.c_[np.ones(len(x_test)), x_test]
    penalty = np.eye(xb.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(xb.T @ xb + alpha * penalty, xb.T @ y_train)
    return xt @ beta


def knn_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, k: int = 15) -> np.ndarray:
    preds = []
    for row in x_test:
        dist = np.sqrt(np.sum((x_train - row) ** 2, axis=1))
        idx = np.argsort(dist)[:k]
        w = 1.0 / (dist[idx] + 1e-6)
        preds.append(float(np.sum(w * y_train[idx]) / np.sum(w)))
    return np.asarray(preds)


@dataclass
class DmlModel:
    hidden: int = 32
    embed: int = 12
    epochs: int = 260
    batch_size: int = 192
    lr: float = 0.006
    margin: float = 1.0
    regression_weight: float = 0.45
    seed: int = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DmlModel":
        rng = np.random.default_rng(self.seed)
        n, d = x.shape
        self.y_mean = float(y.mean())
        self.y_std = float(y.std() if y.std() > 1e-6 else 1.0)
        y_std = (y - self.y_mean) / self.y_std
        q1, q2 = np.quantile(y, [0.33, 0.66])
        tiers = np.digitize(y, [q1, q2])
        tier_indices = [np.flatnonzero(tiers == t) for t in range(3)]

        self.w1 = rng.normal(0, 0.12, size=(d, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.w2 = rng.normal(0, 0.12, size=(self.hidden, self.embed))
        self.b2 = np.zeros(self.embed)
        self.wr = rng.normal(0, 0.05, size=self.embed)
        self.br = 0.0

        params = ["w1", "b1", "w2", "b2", "wr"]
        m = {p: np.zeros_like(getattr(self, p)) for p in params}
        v = {p: np.zeros_like(getattr(self, p)) for p in params}
        mb = 0.0
        vb = 0.0
        beta1, beta2 = 0.9, 0.999
        step = 0

        for _ in range(self.epochs):
            anchors = rng.integers(0, n, size=min(self.batch_size, n))
            positives = []
            negatives = []
            kept = []
            for a in anchors:
                same = tier_indices[tiers[a]]
                same = same[same != a]
                other = np.flatnonzero(tiers != tiers[a])
                if len(same) == 0 or len(other) == 0:
                    continue
                kept.append(a)
                positives.append(rng.choice(same))
                negatives.append(rng.choice(other))
            if not kept:
                continue

            a = np.asarray(kept)
            p = np.asarray(positives)
            neg = np.asarray(negatives)
            ha, za = self._forward(x[a])
            hp, zp = self._forward(x[p])
            hn, zn = self._forward(x[neg])

            d_ap = np.sum((za - zp) ** 2, axis=1)
            d_an = np.sum((za - zn) ** 2, axis=1)
            active = (d_ap - d_an + self.margin) > 0
            scale = active.astype(float)[:, None] / max(1, len(a))
            gza = (2 * (za - zp) - 2 * (za - zn)) * scale
            gzp = (-2 * (za - zp)) * scale
            gzn = (2 * (za - zn)) * scale

            pred = za @ self.wr + self.br
            err = (pred - y_std[a]) / max(1, len(a))
            gza += self.regression_weight * (2 * err[:, None] * self.wr[None, :])
            grads = {
                "w1": np.zeros_like(self.w1),
                "b1": np.zeros_like(self.b1),
                "w2": np.zeros_like(self.w2),
                "b2": np.zeros_like(self.b2),
                "wr": self.regression_weight * (2 * za.T @ err),
            }
            gbr = float(self.regression_weight * 2 * err.sum())

            self._backward(x[a], ha, gza, grads)
            self._backward(x[p], hp, gzp, grads)
            self._backward(x[neg], hn, gzn, grads)

            step += 1
            for name in params:
                grad = grads[name] + 1e-4 * getattr(self, name)
                m[name] = beta1 * m[name] + (1 - beta1) * grad
                v[name] = beta2 * v[name] + (1 - beta2) * (grad * grad)
                mhat = m[name] / (1 - beta1**step)
                vhat = v[name] / (1 - beta2**step)
                setattr(self, name, getattr(self, name) - self.lr * mhat / (np.sqrt(vhat) + 1e-8))
            mb = beta1 * mb + (1 - beta1) * gbr
            vb = beta2 * vb + (1 - beta2) * (gbr * gbr)
            self.br -= self.lr * (mb / (1 - beta1**step)) / (math.sqrt(vb / (1 - beta2**step)) + 1e-8)
        return self

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h_pre = x @ self.w1 + self.b1
        h = np.maximum(h_pre, 0.0)
        z = h @ self.w2 + self.b2
        return h, z

    def _backward(self, x: np.ndarray, h: np.ndarray, gz: np.ndarray, grads: dict[str, np.ndarray]) -> None:
        grads["w2"] += h.T @ gz
        grads["b2"] += gz.sum(axis=0)
        gh = gz @ self.w2.T
        gh[h <= 0] = 0
        grads["w1"] += x.T @ gh
        grads["b1"] += gh.sum(axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        _, z = self._forward(x)
        return (z @ self.wr + self.br) * self.y_std + self.y_mean


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rho = spearmanr(y_true, y_pred).correlation
    if np.isnan(rho):
        rho = 0.0
    diffs_true = y_true[:, None] - y_true[None, :]
    diffs_pred = y_pred[:, None] - y_pred[None, :]
    mask = np.triu(np.abs(diffs_true) >= 0.5, k=1)
    if mask.sum() == 0:
        pair_acc = np.nan
    else:
        pair_acc = float(np.mean(np.sign(diffs_true[mask]) == np.sign(diffs_pred[mask])))
    return {"mae": mae, "rmse": rmse, "spearman": float(rho), "pairwise_acc": pair_acc}


def group_folds(groups: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    unique = np.asarray(sorted(pd.Series(groups).fillna("missing").unique()))
    rng.shuffle(unique)
    parts = np.array_split(unique, n_splits)
    return [np.isin(groups, part) for part in parts]


def evaluate_split(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, seed: int, label: str) -> list[dict]:
    feature_cols = [f"layout_{i:02d}" for i in range(64)]
    x_train = df.loc[train_idx, feature_cols].to_numpy(float)
    x_test = df.loc[test_idx, feature_cols].to_numpy(float)
    y_train = df.loc[train_idx, "design_quality"].to_numpy(float)
    y_test = df.loc[test_idx, "design_quality"].to_numpy(float)
    x_train, x_test = standardize(x_train, x_test)

    rows = []
    preds = {
        "Mean": np.full_like(y_test, y_train.mean()),
        "Ridge": ridge_predict(x_train, y_train, x_test),
        "Raw-kNN": knn_predict(x_train, y_train, x_test),
    }
    selected = select_top_features(x_train, y_train, 16)
    preds["FS-Ridge"] = ridge_predict(x_train[:, selected], y_train, x_test[:, selected])
    preds["DML"] = DmlModel(seed=seed).fit(x_train, y_train).predict(x_test)
    preds["FS-DML"] = DmlModel(seed=seed + 101).fit(x_train[:, selected], y_train).predict(x_test[:, selected])

    for method, pred in preds.items():
        row = {"split": label, "seed": seed, "method": method, "n_train": int(train_idx.sum()), "n_test": int(test_idx.sum())}
        row.update(metrics(y_test, pred))
        rows.append(row)
    return rows


def run_cross_validation(df: pd.DataFrame) -> pd.DataFrame:
    all_rows = []
    groups = df["App Package Name"].fillna("missing").to_numpy()
    for seed in [7, 19, 31]:
        folds = group_folds(groups, 5, seed)
        for fold, test_mask in enumerate(folds):
            train_mask = ~test_mask
            rows = evaluate_split(df, train_mask, test_mask, seed + fold, f"group_cv_{fold}")
            all_rows.extend(rows)
    return pd.DataFrame(all_rows)


def run_domain_transfer(df: pd.DataFrame) -> pd.DataFrame:
    test_mask = df["target_domain"].to_numpy(bool)
    train_mask = ~test_mask
    rows = []
    for seed in [7, 19, 31, 43, 59]:
        rows.extend(evaluate_split(df, train_mask, test_mask, seed, "education_books_transfer"))
    return pd.DataFrame(rows)


def feature_scan(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [f"layout_{i:02d}" for i in range(64)]
    groups = df["App Package Name"].fillna("missing").to_numpy()
    folds = group_folds(groups, 5, 101)
    rows = []
    for k in [4, 8, 12, 16, 24, 32, 48, 64]:
        for fold, test_mask in enumerate(folds):
            train_mask = ~test_mask
            x_train = df.loc[train_mask, feature_cols].to_numpy(float)
            x_test = df.loc[test_mask, feature_cols].to_numpy(float)
            y_train = df.loc[train_mask, "design_quality"].to_numpy(float)
            y_test = df.loc[test_mask, "design_quality"].to_numpy(float)
            x_train, x_test = standardize(x_train, x_test)
            sel = select_top_features(x_train, y_train, k)
            pred = ridge_predict(x_train[:, sel], y_train, x_test[:, sel])
            row = {"k": k, "fold": fold}
            row.update(metrics(y_test, pred))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["mae", "rmse", "spearman", "pairwise_acc"]
    rows = []
    for method, grp in metrics_df.groupby("method"):
        row = {"method": method}
        for col in metric_cols:
            row[f"{col}_mean"] = grp[col].mean()
            row[f"{col}_std"] = grp[col].std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mae_mean")


def plot_results(summary: pd.DataFrame, transfer_summary: pd.DataFrame, scan: pd.DataFrame) -> None:
    order = summary.sort_values("mae_mean")["method"].tolist()
    colors = {
        "Mean": "#777777",
        "Ridge": "#4878A8",
        "Raw-kNN": "#6A9A5B",
        "FS-Ridge": "#B27946",
        "DML": "#7B5EA7",
        "FS-DML": "#C44E52",
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    axes[0].bar(order, [summary.set_index("method").loc[m, "mae_mean"] for m in order], color=[colors[m] for m in order])
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Grouped cross-validation")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(order, [summary.set_index("method").loc[m, "spearman_mean"] for m in order], color=[colors[m] for m in order])
    axes[1].set_ylabel("Spearman rho")
    axes[1].set_title("Quality ranking")
    axes[1].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_results.pdf")
    fig.savefig(FIGURES / "fig_results.png", dpi=220)
    plt.close(fig)

    scan_summary = scan.groupby("k", as_index=False).agg(mae=("mae", "mean"), spearman=("spearman", "mean"))
    fig, ax1 = plt.subplots(figsize=(6.4, 3.4))
    ax1.plot(scan_summary["k"], scan_summary["mae"], marker="o", color="#4878A8", label="MAE")
    ax1.set_xlabel("Selected layout dimensions")
    ax1.set_ylabel("MAE", color="#4878A8")
    ax2 = ax1.twinx()
    ax2.plot(scan_summary["k"], scan_summary["spearman"], marker="s", color="#C44E52", label="Spearman")
    ax2.set_ylabel("Spearman rho", color="#C44E52")
    ax1.set_title("Feature-selection sensitivity")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_feature_scan.pdf")
    fig.savefig(FIGURES / "fig_feature_scan.png", dpi=220)
    plt.close(fig)

    transfer = transfer_summary.sort_values("mae_mean")
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.bar(transfer["method"], transfer["mae_mean"], color=[colors[m] for m in transfer["method"]])
    ax.set_ylabel("MAE")
    ax.set_title("Education and reading-app transfer subset")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_transfer.pdf")
    fig.savefig(FIGURES / "fig_transfer.png", dpi=220)
    plt.close(fig)


def write_reports(df: pd.DataFrame, summary: pd.DataFrame, transfer_summary: pd.DataFrame, scan: pd.DataFrame) -> None:
    dataset = {
        "n_ui": int(len(df)),
        "n_app_packages": int(df["App Package Name"].nunique()),
        "n_categories": int(df["Category"].nunique()),
        "target_domain_n": int(df["target_domain"].sum()),
        "target_domain_categories": ["Education", "Books & Reference", "Comics"],
        "rating_mean": float(df["design_quality"].mean()),
        "rating_std": float(df["design_quality"].std()),
    }
    (RESULTS / "dataset_summary.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    category_counts = df["Category"].value_counts().rename_axis("category").reset_index(name="n")
    category_counts.to_csv(RESULTS / "category_counts.csv", index=False)

    best = summary.iloc[0]
    baseline = summary[summary["method"] == "Ridge"].iloc[0]
    transfer_best = transfer_summary.iloc[0]
    lines = [
        "# Claims from results",
        "",
        f"- Dataset: {dataset['n_ui']} UICrit-RICO aligned screens from {dataset['n_app_packages']} app packages; {dataset['target_domain_n']} screens are in Education, Books & Reference, or Comics.",
        f"- Best grouped-CV method: {best['method']} with MAE={best['mae_mean']:.3f} and Spearman={best['spearman_mean']:.3f}.",
        f"- Ridge baseline: MAE={baseline['mae_mean']:.3f} and Spearman={baseline['spearman_mean']:.3f}.",
        f"- Education/reading transfer best method: {transfer_best['method']} with MAE={transfer_best['mae_mean']:.3f} and Spearman={transfer_best['spearman_mean']:.3f}.",
        "- Claim boundary: the data support automatic layout-quality screening on public UI datasets and a proxy education/reading subset, not direct prediction of children's learning outcomes.",
    ]
    (ROOT / "briefs" / "CLAIMS_FROM_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = load_dataset()
    df.to_csv(RESULTS / "uicrit_rico_aligned.csv", index=False)
    cv = run_cross_validation(df)
    transfer = run_domain_transfer(df)
    scan = feature_scan(df)
    summary = summarize(cv)
    transfer_summary = summarize(transfer)

    cv.to_csv(RESULTS / "fold_metrics.csv", index=False)
    summary.to_csv(RESULTS / "summary_metrics.csv", index=False)
    transfer.to_csv(RESULTS / "education_transfer_metrics.csv", index=False)
    transfer_summary.to_csv(RESULTS / "education_transfer_summary.csv", index=False)
    scan.to_csv(RESULTS / "feature_scan.csv", index=False)
    plot_results(summary, transfer_summary, scan)
    write_reports(df, summary, transfer_summary, scan)

    print("Aligned dataset:", df.shape)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nEducation/reading transfer:")
    print(transfer_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
