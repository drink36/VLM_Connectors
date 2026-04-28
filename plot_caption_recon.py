"""
Generate all caption reconstruction figures from caption_compare_out_nor CSVs.
Figures 1-8, 16-20 use per-image CSV data.
Figures 9-12 (PCA/masking curves) and 13-15 (eRank) need separate data sources.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data/output/caption_compare_out_nor")
OUT_DIR = Path("figures/caption_recon")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["llava", "idefics2", "qwen2.5vl", "qwen3.5"]
MODEL_COLORS = {
    "llava": "#1f77b4",
    "idefics2": "#ff7f0e",
    "qwen2.5vl": "#2ca02c",
    "qwen3.5": "#d62728",
}
BERTSCORE_COL = "bertscore_recon_vs_post_f1"


def load_all():
    dfs = []
    for m in MODELS:
        p = DATA_DIR / m / "caption_original_vs_recon.csv"
        df = pd.read_csv(p)
        df["model"] = m
        if "bertscore_recon_vs_post_f1" not in df.columns and "bertscore_f1" in df.columns:
            df = df.rename(columns={"bertscore_f1": "bertscore_recon_vs_post_f1"})
        if "bertscore_post_vs_gt_f1" in df.columns and "bertscore_recon_vs_gt_f1" in df.columns:
            df["bertscore_drop"] = df["bertscore_post_vs_gt_f1"] - df["bertscore_recon_vs_gt_f1"]
        else:
            df["bertscore_drop"] = float("nan")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def save(fig, name):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def add_scale_lines(ax):
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.2)


def fig1(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        sub = df[df["model"] == m]
        ax.scatter(sub["reproj_cosine_recon"], sub["clipscore_drop"],
                   alpha=0.3, s=8, color=MODEL_COLORS[m], label=m)
    ax.set_xlabel("Reproj Cosine (recon)")
    ax.set_ylabel("CLIPScore Drop")
    ax.set_title("Fig 1 — Recoverability vs Semantic Degradation")
    ax.legend(markerscale=3)
    add_scale_lines(ax)
    save(fig, "fig01_cosine_vs_clipscore_drop")


def fig2(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        sub = df[df["model"] == m]
        ax.scatter(sub["reproj_mse_recon"], sub["clipscore_drop"],
                   alpha=0.3, s=8, color=MODEL_COLORS[m], label=m)
    ax.set_xlabel("Reproj MSE (recon)")
    ax.set_ylabel("CLIPScore Drop")
    ax.set_title("Fig 2 — MSE vs Semantic Degradation")
    ax.legend(markerscale=3)
    add_scale_lines(ax)
    save(fig, "fig02_mse_vs_clipscore_drop")


def fig3(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [df[df["model"] == m]["clipscore_drop"].dropna().values for m in MODELS]
    parts = ax.violinplot(data, positions=range(len(MODELS)), showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(list(MODEL_COLORS.values())[i])
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("CLIPScore Drop")
    ax.set_title("Fig 3 — CLIPScore Drop Distribution per Model")
    add_scale_lines(ax)
    save(fig, "fig03_clipscore_drop_violin")


def fig4(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MODELS))
    w = 0.35
    for i, (col, label, color) in enumerate([
        ("clipscore_post_direct", "Post Direct", "#4e79a7"),
        ("clipscore_recon", "Recon", "#f28e2b"),
    ]):
        vals = [df[df["model"] == m][col].dropna().values for m in MODELS]
        ax.boxplot(vals, positions=x + (i - 0.5) * w, widths=w * 0.9,
                   patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.7),
                   medianprops=dict(color="black"),
                   whiskerprops=dict(color=color),
                   capprops=dict(color=color),
                   flierprops=dict(marker=".", markersize=2, alpha=0.3, color=color),
                   label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("CLIPScore")
    ax.set_title("Fig 4 — Raw CLIPScore: Post vs Recon")
    ax.legend()
    add_scale_lines(ax)
    save(fig, "fig04_clipscore_post_vs_recon_boxplot")


def fig5(df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=False)
    axes = axes.flatten()
    for i, m in enumerate(MODELS):
        sub = df[df["model"] == m]["clipscore_drop"].dropna()
        axes[i].hist(sub, bins=50, color=MODEL_COLORS[m], alpha=0.8, edgecolor="white", linewidth=0.3)
        axes[i].set_title(m)
        axes[i].set_xlabel("CLIPScore Drop")
        axes[i].set_ylabel("Count")
        add_scale_lines(axes[i])
    fig.suptitle("Fig 5 — CLIPScore Drop Histogram per Model")
    fig.tight_layout()
    save(fig, "fig05_clipscore_drop_hist")


def fig6(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        sub = df[df["model"] == m]
        ax.scatter(sub[BERTSCORE_COL], sub["clipscore_drop"],
                   alpha=0.3, s=8, color=MODEL_COLORS[m], label=m)
    ax.set_xlabel("BERTScore F1 (recon vs post)")
    ax.set_ylabel("CLIPScore Drop")
    ax.set_title("Fig 6 — BERTScore vs CLIPScore Drop")
    ax.legend(markerscale=3)
    add_scale_lines(ax)
    save(fig, "fig06_bertscore_vs_clipscore_drop")


def fig7(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = []
    positions = []
    for i, m in enumerate(MODELS):
        vals = df[df["model"] == m][BERTSCORE_COL].dropna().values
        if len(vals) > 0:
            data.append(vals)
            positions.append(i)

    if data:
        parts = ax.violinplot(data, positions=positions, showmedians=True)
        for pc, pos in zip(parts["bodies"], positions):
            model = MODELS[pos]
            pc.set_facecolor(MODEL_COLORS[model])
            pc.set_alpha(0.7)
    else:
        ax.text(0.5, 0.5, "No valid BERTScore data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("BERTScore F1")
    ax.set_title("Fig 7 — BERTScore Distribution per Model")
    add_scale_lines(ax)
    save(fig, "fig07_bertscore_violin")


def fig8(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        sub = df[df["model"] == m]
        ax.scatter(sub["reproj_cosine_recon"], sub[BERTSCORE_COL],
                   alpha=0.3, s=8, color=MODEL_COLORS[m], label=m)
    ax.set_xlabel("Reproj Cosine (recon)")
    ax.set_ylabel("BERTScore F1")
    ax.set_title("Fig 8 — Cosine vs BERTScore")
    ax.legend(markerscale=3)
    add_scale_lines(ax)
    for name in ["fig08_cosine_vs_bertscore", "fig08_cosine_vs_bert"]:
        path = OUT_DIR / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close(fig)


def fig9(df):
    """BERTScore drop violin per model (NaN = no GT ref, e.g. qwen3.5)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    data, positions = [], []
    for i, m in enumerate(MODELS):
        vals = df[df["model"] == m]["bertscore_drop"].dropna().values
        if len(vals) > 0:
            data.append(vals)
            positions.append(i)
    if data:
        parts = ax.violinplot(data, positions=positions, showmedians=True)
        for pc, pos in zip(parts["bodies"], positions):
            pc.set_facecolor(MODEL_COLORS[MODELS[pos]])
            pc.set_alpha(0.7)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("BERTScore Drop (post_gt − recon_gt)")
    ax.set_title("Fig 9 — BERTScore Drop Distribution per Model")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    add_scale_lines(ax)
    save(fig, "fig09_bertscore_drop_violin")


def fig10(df):
    """Cosine vs BERTScore drop scatter, all models overlaid."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        sub = df[df["model"] == m].dropna(subset=["bertscore_drop"])
        ax.scatter(sub["reproj_cosine_recon"], sub["bertscore_drop"],
                   alpha=0.3, s=8, color=MODEL_COLORS[m], label=m)
    ax.set_xlabel("Reproj Cosine (recon)")
    ax.set_ylabel("BERTScore Drop")
    ax.set_title("Fig 10 — Cosine vs BERTScore Drop")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(markerscale=3)
    add_scale_lines(ax)
    save(fig, "fig10_cosine_vs_bertscore_drop")


def fig11(df):
    """CLIPScore drop vs BERTScore drop scatter."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        sub = df[df["model"] == m].dropna(subset=["bertscore_drop"])
        ax.scatter(sub["clipscore_drop"], sub["bertscore_drop"],
                   alpha=0.3, s=8, color=MODEL_COLORS[m], label=m)
    ax.set_xlabel("CLIPScore Drop")
    ax.set_ylabel("BERTScore Drop")
    ax.set_title("Fig 11 — CLIPScore Drop vs BERTScore Drop")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(markerscale=3)
    add_scale_lines(ax)
    save(fig, "fig11_clipscore_drop_vs_bertscore_drop")


def fig16(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [df[df["model"] == m]["reproj_cosine_recon"].dropna().values for m in MODELS]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="black"),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for patch, m in zip(bp["boxes"], MODELS):
        patch.set_facecolor(MODEL_COLORS[m])
        patch.set_alpha(0.7)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Reproj Cosine")
    ax.set_title("Fig 16 — Cosine Distribution per Model")
    add_scale_lines(ax)
    save(fig, "fig16_cosine_boxplot")


def fig17(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [df[df["model"] == m]["reproj_mse_recon"].dropna().values for m in MODELS]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="black"),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for patch, m in zip(bp["boxes"], MODELS):
        patch.set_facecolor(MODEL_COLORS[m])
        patch.set_alpha(0.7)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Reproj MSE")
    ax.set_title("Fig 17 — MSE Distribution per Model")
    add_scale_lines(ax)
    save(fig, "fig17_mse_boxplot")


def fig18(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        sub = df[df["model"] == m]
        ax.scatter(sub["reproj_cosine_recon"], sub["clipscore_recon"],
                   alpha=0.3, s=8, color=MODEL_COLORS[m], label=m)
    ax.set_xlabel("Reproj Cosine (recon)")
    ax.set_ylabel("CLIPScore (recon)")
    ax.set_title("Fig 18 — Cosine vs CLIPScore (raw)")
    ax.legend(markerscale=3)
    add_scale_lines(ax)
    save(fig, "fig18_cosine_vs_clipscore_raw")


def fig19(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        sub = df[df["model"] == m].copy()
        sub["cosine_bin"] = pd.cut(sub["reproj_cosine_recon"], bins=20)
        trend = sub.groupby("cosine_bin", observed=True)["clipscore_drop"].mean()
        centers = [iv.mid for iv in trend.index]
        ax.plot(centers, trend.values, marker="o", ms=4, label=m, color=MODEL_COLORS[m])
    ax.set_xlabel("Reproj Cosine (binned)")
    ax.set_ylabel("Avg CLIPScore Drop")
    ax.set_title("Fig 19 — Cosine Binned Trend")
    ax.legend()
    add_scale_lines(ax)
    save(fig, "fig19_cosine_binned_trend")


def fig20(df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for i, m in enumerate(MODELS):
        sub = df[df["model"] == m]
        axes[i].scatter(sub["reproj_cosine_recon"], sub["clipscore_drop"],
                        alpha=0.3, s=6, color=MODEL_COLORS[m])
        axes[i].set_title(m)
        axes[i].set_xlabel("Reproj Cosine")
        axes[i].set_ylabel("CLIPScore Drop")
        add_scale_lines(axes[i])
    fig.suptitle("Fig 20 — Per-Model Scatter: Cosine vs CLIPScore Drop")
    fig.tight_layout()
    save(fig, "fig20_per_model_scatter")


if __name__ == "__main__":
    print("Loading data...")
    df = load_all()
    print(f"  {len(df)} rows across {df['model'].nunique()} models")

    print("Generating figures...")
    fig1(df)
    fig2(df)
    fig3(df)
    fig4(df)
    fig5(df)
    fig6(df)
    fig7(df)
    fig8(df)
    fig9(df)
    fig10(df)
    fig11(df)
    fig16(df)
    fig17(df)
    fig18(df)
    fig19(df)
    fig20(df)

    print(f"\nDone. All figures saved to {OUT_DIR}/")
    print("Figures 9-12 (PCA/masking curves) and 13-15 (eRank) need separate data sources.")