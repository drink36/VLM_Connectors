"""
Generate caption correlation figures using E_pre reconstruction metrics.

This mirrors the post-based correlation views but replaces:
  reproj_cosine_recon -> cosine_pre
  reproj_mse_recon    -> mse_pre

Data sources:
  - data/output/recon_pre_eval/<model>.csv
  - data/output/caption_compare_out_nor/<model>/caption_original_vs_recon.csv

Outputs:
  - figures/caption_recon_pre/*.png
  - results/caption_recon_pre_correlations.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["llava", "idefics2", "qwen2.5vl", "qwen3.5"]
MODEL_COLORS = {
    "llava": "#1f77b4",
    "idefics2": "#ff7f0e",
    "qwen2.5vl": "#2ca02c",
    "qwen3.5": "#d62728",
}
BERTSCORE_COL = "bertscore_recon_vs_post_f1"


def normalize_key(value):
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = "".join(ch for ch in s if ch.isdigit())
    if not s:
        return None
    return s.zfill(9)


def load_caption_df(caption_dir: Path, model: str) -> pd.DataFrame:
    path = caption_dir / model / "caption_original_vs_recon.csv"
    df = pd.read_csv(path)

    if "bertscore_f1" in df.columns and BERTSCORE_COL not in df.columns:
        df = df.rename(columns={
            "bertscore_f1": "bertscore_recon_vs_post_f1",
            "bertscore_precision": "bertscore_recon_vs_post_precision",
            "bertscore_recall": "bertscore_recon_vs_post_recall",
        })

    keep = ["key", "clipscore_drop", "clipscore_recon", "clipscore_post_direct", BERTSCORE_COL]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out["norm_key"] = out["key"].map(normalize_key)
    out = out.dropna(subset=["norm_key"]).drop_duplicates(subset=["norm_key"]) 
    return out


def load_pre_df(pre_dir: Path, model: str) -> pd.DataFrame:
    combined_path = pre_dir / f"{model}.csv"
    shard_dir = pre_dir / model

    candidates = []
    if combined_path.exists():
        candidates.append(("combined", pd.read_csv(combined_path)))

    shard_files = sorted(shard_dir.glob("*.csv")) if shard_dir.exists() else []
    if shard_files:
        shard_df = pd.concat([pd.read_csv(p) for p in shard_files], ignore_index=True)
        candidates.append(("per_shard", shard_df))

    if not candidates:
        raise FileNotFoundError(
            f"No pre-eval CSV found for {model}: expected {combined_path} or {shard_dir}/*.csv"
        )

    # Prefer the source with more rows to avoid stale partial combined CSVs.
    source, df = max(candidates, key=lambda x: len(x[1]))
    if len(candidates) > 1:
        print(f"  {model}: using {source} pre source (rows={len(df)})")

    out = df[["key", "mse_pre", "cosine_pre"]].copy()
    out["norm_key"] = out["key"].map(normalize_key)
    out = out.dropna(subset=["norm_key"]).drop_duplicates(subset=["norm_key"])
    return out[["norm_key", "mse_pre", "cosine_pre"]]


def load_all(caption_dir: Path, pre_dir: Path) -> pd.DataFrame:
    merged_all = []
    for model in MODELS:
        cap = load_caption_df(caption_dir, model)
        pre = load_pre_df(pre_dir, model)
        merged = cap.merge(pre, on="norm_key", how="inner")
        merged["model"] = model
        merged_all.append(merged)
        print(
            f"  {model}: caption={len(cap)} pre={len(pre)} merged={len(merged)}"
        )

    if not merged_all:
        return pd.DataFrame()
    return pd.concat(merged_all, ignore_index=True)


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def add_scale_lines(ax):
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.2)


def fig1(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        sub = df[df["model"] == m]
        ax.scatter(sub["cosine_pre"], sub["clipscore_drop"],
                   alpha=0.3, s=8, color=MODEL_COLORS[m], label=m)
    ax.set_xlabel("Cosine(E_pre_hat, E_pre)")
    ax.set_ylabel("CLIPScore Drop")
    ax.set_title("Pre-Fig 1 - Recoverability vs Semantic Degradation")
    ax.legend(markerscale=3)
    add_scale_lines(ax)
    save(fig, out_dir, "pre_fig01_cosine_pre_vs_clipscore_drop")


def fig2(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        sub = df[df["model"] == m]
        ax.scatter(sub["mse_pre"], sub["clipscore_drop"],
                   alpha=0.3, s=8, color=MODEL_COLORS[m], label=m)
    ax.set_xlabel("MSE(E_pre_hat, E_pre)")
    ax.set_ylabel("CLIPScore Drop")
    ax.set_title("Pre-Fig 2 - MSE vs Semantic Degradation")
    ax.legend(markerscale=3)
    add_scale_lines(ax)
    save(fig, out_dir, "pre_fig02_mse_pre_vs_clipscore_drop")


def fig8(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        sub = df[df["model"] == m]
        ax.scatter(sub["cosine_pre"], sub[BERTSCORE_COL],
                   alpha=0.3, s=8, color=MODEL_COLORS[m], label=m)
    ax.set_xlabel("Cosine(E_pre_hat, E_pre)")
    ax.set_ylabel("BERTScore F1")
    ax.set_title("Pre-Fig 8 - Cosine vs BERTScore")
    ax.legend(markerscale=3)
    add_scale_lines(ax)
    save(fig, out_dir, "pre_fig08_cosine_pre_vs_bertscore")


def fig19(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        sub = df[df["model"] == m].copy()
        sub = sub.dropna(subset=["cosine_pre", "clipscore_drop"])
        if sub.empty:
            continue
        sub["cosine_bin"] = pd.cut(sub["cosine_pre"], bins=20)
        trend = sub.groupby("cosine_bin", observed=True)["clipscore_drop"].mean()
        centers = [iv.mid for iv in trend.index]
        ax.plot(centers, trend.values, marker="o", ms=4, label=m, color=MODEL_COLORS[m])
    ax.set_xlabel("Cosine(E_pre_hat, E_pre) binned")
    ax.set_ylabel("Avg CLIPScore Drop")
    ax.set_title("Pre-Fig 19 - Cosine Binned Trend")
    ax.legend()
    add_scale_lines(ax)
    save(fig, out_dir, "pre_fig19_cosine_pre_binned_trend")


def fig20(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for i, m in enumerate(MODELS):
        sub = df[df["model"] == m]
        axes[i].scatter(sub["cosine_pre"], sub["clipscore_drop"],
                        alpha=0.3, s=6, color=MODEL_COLORS[m])
        axes[i].set_title(m)
        axes[i].set_xlabel("Cosine(E_pre_hat, E_pre)")
        axes[i].set_ylabel("CLIPScore Drop")
        add_scale_lines(axes[i])
    fig.suptitle("Pre-Fig 20 - Per-Model Scatter: Cosine vs CLIPScore Drop")
    fig.tight_layout()
    save(fig, out_dir, "pre_fig20_per_model_scatter")


def write_correlations(df: pd.DataFrame, corr_out: Path):
    pairs = [
        ("cosine_pre", "clipscore_drop"),
        ("mse_pre", "clipscore_drop"),
        ("cosine_pre", BERTSCORE_COL),
        ("mse_pre", BERTSCORE_COL),
    ]

    rows = []
    groups = [("all", df)] + [(m, df[df["model"] == m]) for m in MODELS]
    for model_name, sub in groups:
        for x_col, y_col in pairs:
            pair = sub[[x_col, y_col]].dropna()
            if len(pair) < 2:
                continue
            rows.append({
                "model": model_name,
                "x": x_col,
                "y": y_col,
                "n": len(pair),
                "pearson_r": pair[x_col].corr(pair[y_col], method="pearson"),
                "spearman_r": pair[x_col].corr(pair[y_col], method="spearman"),
            })

    out = pd.DataFrame(rows)
    corr_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(corr_out, index=False, float_format="%.6f")
    print(f"  saved {corr_out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--caption_dir", default="data/output/caption_compare_out_nor")
    p.add_argument("--pre_dir", default="data/output/recon_pre_eval")
    p.add_argument("--out_dir", default="figures/caption_recon_pre")
    p.add_argument("--corr_out", default="results/caption_recon_pre_correlations.csv")
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()

    caption_dir = Path(args.caption_dir)
    pre_dir = Path(args.pre_dir)
    out_dir = Path(args.out_dir)
    corr_out = Path(args.corr_out)

    print("Loading and joining data...")
    df = load_all(caption_dir, pre_dir)
    if df.empty:
        print("No merged rows found.")
        return

    print(f"Merged rows: {len(df)}")

    print("Writing correlations...")
    write_correlations(df, corr_out)

    if not args.no_plots:
        print("Generating pre-correlation figures...")
        fig1(df, out_dir)
        fig2(df, out_dir)
        fig8(df, out_dir)
        fig19(df, out_dir)
        fig20(df, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
