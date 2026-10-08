"""Exploratory data analysis: writes figures + a summary table to reports/figures."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from heart.config import (  # noqa: E402
    CATEGORICAL_FEATURES, BINARY_FEATURES, FIGURES_DIR, NUMERIC_FEATURES, TARGET,
)
from heart.data import load_clean, load_raw  # noqa: E402

log = logging.getLogger(__name__)
logging.getLogger("matplotlib.category").setLevel(logging.WARNING)
PALETTE = {0: "#2a9d8f", 1: "#e76f51"}
LABELS = {0: "No disease", 1: "Disease"}
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name: str, out: Path) -> Path:
    path = out / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def class_balance(df, out):
    counts = df[TARGET].map(LABELS).value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[0], PALETTE[1]])
    ax.bar_label(bars, labels=[f"{v} ({v / counts.sum():.0%})" for v in counts.values])
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    return _save(fig, "01_class_balance.png", out)


def numeric_histograms(df, out):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, NUMERIC_FEATURES):
        for t in (0, 1):
            sns.histplot(df.loc[df[TARGET] == t, col].dropna(), ax=ax, kde=True,
                         color=PALETTE[t], label=LABELS[t], stat="density", alpha=0.45)
        ax.set_title(col)
        ax.legend(fontsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle("Distribution of continuous features by outcome", fontsize=14)
    return _save(fig, "02_numeric_histograms.png", out)


def categorical_rates(df, out):
    cols = CATEGORICAL_FEATURES + BINARY_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, cols):
        rate = df.groupby(col)[TARGET].agg(["mean", "size"]).reset_index()
        rate[col] = rate[col].astype(int)
        sns.barplot(data=rate, x=col, y="mean", ax=ax, color="#e76f51")
        for i, n in enumerate(rate["size"]):
            ax.text(i, 0.02, f"n={n}", ha="center", color="white", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Disease rate")
        ax.set_title(col)
    fig.suptitle("Disease rate per category level", fontsize=14)
    return _save(fig, "03_categorical_disease_rate.png", out)


def correlation_heatmap(df, out):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 7})
    ax.set_title("Pearson correlation matrix")
    return _save(fig, "04_correlation_heatmap.png", out)


def boxplots(df, out):
    long = df.melt(id_vars=TARGET, value_vars=NUMERIC_FEATURES)
    long[TARGET] = long[TARGET].map(LABELS)
    g = sns.catplot(data=long, x=TARGET, y="value", col="variable", kind="box",
                    sharey=False, col_wrap=5, height=3.2, palette=[PALETTE[0], PALETTE[1]],
                    hue=TARGET, legend=False)
    g.set_titles("{col_name}")
    g.figure.suptitle("Outlier view: continuous features by outcome", y=1.04)
    path = out / "05_boxplots.png"
    g.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(g.figure)
    return path


def missing_values(raw, out):
    miss = raw.isna().sum()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(5, 3))
    if miss.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center")
    else:
        bars = ax.barh(miss.index, miss.values, color="#264653")
        ax.bar_label(bars)
    ax.set_title("Missing values in raw data")
    return _save(fig, "00_missing_values.png", out)


def run(out: Path = FIGURES_DIR) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    df = load_clean()
    raw = load_raw()
    paths = [missing_values(raw, out), class_balance(df, out), numeric_histograms(df, out),
             categorical_rates(df, out), correlation_heatmap(df, out), boxplots(df, out)]
    summary = {
        "rows": int(len(df)),
        "positive_rate": round(float(df[TARGET].mean()), 3),
        "missing_raw": {k: int(v) for k, v in raw.isna().sum().items() if v},
        "top_correlations_with_target": df.corr(numeric_only=True)[TARGET]
        .drop(TARGET).abs().sort_values(ascending=False).head(6).round(3).to_dict(),
    }
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    df.describe().T.round(2).to_csv(out / "describe.csv")
    log.info("EDA figures written: %s", [p.name for p in paths])
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(json.dumps(run(), indent=2))
