"""
Scatter plot: post_effective_rank vs clipscore_original, one plot per model.
Uses perturb_out_scored CSVs (5000 samples) joined with per_image.csv on key.

Usage:
  python plot_erank_vs_clip.py
    python plot_erank_vs_clip.py --perturb_dir sample/perturb_out_scored \
                                                                --knn_base sample/knn_out \
                                --out figures/erank_vs_clip
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

MODELS = {
    "LLaVA":       "llava",
    "Idefics2":    "idefics2",
    "Qwen-2.5-VL": "qwen2.5vl",
    "Qwen-3.5":    "qwen3.5",
}
COLORS = {"llava": "#1f77b4", "idefics2": "#ff7f0e", "qwen2.5vl": "#2ca02c", "qwen3.5": "#d62728"}


def load_model(perturb_dir, knn_base, model_dir):
    # pick any one ratio CSV — clipscore_original is the same across all ratios
    candidates = sorted(Path(perturb_dir).glob(f"perturb_{model_dir}_lowrank_*.csv"))
    candidates = [f for f in candidates if pd.read_csv(f).shape[0] >= 1000]
    if not candidates:
        print(f"  [warn] no valid perturb CSV for {model_dir}")
        return None
    cap = pd.read_csv(candidates[0])[["key", "clipscore_original"]]

    per_image_files = sorted(Path(knn_base).glob(f"{model_dir}/*/per_image.csv"))
    if not per_image_files:
        print(f"  [warn] no per_image.csv under {knn_base}/{model_dir}/")
        return None
    pim = pd.concat([pd.read_csv(f) for f in per_image_files], ignore_index=True)

    return cap.merge(pim[["key", "post_effective_rank"]], on="key", how="inner")


def plot_model(ax, df, display_name, color):
    x = df["post_effective_rank"].values
    y = df["clipscore_original"].values

    ax.scatter(x, y, alpha=0.2, s=6, color=color, rasterized=True)

    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.5)

    ax.set_title(display_name, fontsize=14)
    ax.set_xlabel("Post-connector Effective Rank", fontsize=11)
    ax.set_ylabel("CLIPScore", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.93, f"r = {r:.3f}  (p={p:.2e})\nn = {len(df)}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--perturb_dir", default="sample/perturb_out_scored")
    p.add_argument("--knn_base", default="sample/knn_out")
    p.add_argument("--out", default="figures/erank_vs_clip")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for display, dirname in MODELS.items():
        print(f"Loading {display} ...")
        df = load_model(args.perturb_dir, args.knn_base, dirname)
        if df is None or df.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        plot_model(ax, df, display, COLORS[dirname])
        fig.tight_layout()

        out_path = out_dir / f"erank_vs_clip_{dirname}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
