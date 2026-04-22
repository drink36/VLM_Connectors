"""
Plot CLIPScore / BERTScore vs perturbation level, one line per model.

Perturb types: lowrank, mask
Modes:
  drop   : y = original - perturbed  (default)
  score  : y = original and perturbed scores on same axes

Usage:
  python plot_perturb.py --perturb lowrank
  python plot_perturb.py --perturb mask --mode score
  python plot_perturb.py --perturb lowrank --mode drop --out figures/lowrank_drop.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"llava": "#1f77b4", "idefics2": "#ff7f0e", "qwen2.5vl": "#2ca02c", "qwen3.5": "#d62728"}
MIN_ROWS = 100  # skip incomplete CSVs


def load_perturb(scored_dir: Path, perturb: str) -> pd.DataFrame:
    rows = []
    for f in scored_dir.glob(f"perturb_*_{perturb}_*.csv"):
        parts = f.stem.split("_")
        level = float(parts[-1])
        model = "_".join(parts[1:-2])
        df = pd.read_csv(f)
        if len(df) < MIN_ROWS:
            print(f"  skip (only {len(df)} rows): {f.name}")
            continue
        rows.append({
            "model": model,
            "level": level,
            "clip_orig":  df["clipscore_original"].mean(),
            "clip_pert":  df["clipscore_perturbed"].mean(),
            "clip_drop":  (df["clipscore_original"] - df["clipscore_perturbed"]).mean(),
            "bert_orig":  df["bertscore_original"].mean(),
            "bert_pert":  df["bertscore_perturbed"].mean(),
            "bert_drop":  (df["bertscore_original"] - df["bertscore_perturbed"]).mean(),
        })
    return pd.DataFrame(rows).sort_values(["model", "level"])


def _set_ticks(ax, levels):
    ax.set_xticks(sorted(levels))
    ax.set_xticklabels([str(l) for l in sorted(levels)], rotation=45, ha="right")


def _xlabel(perturb):
    return "Mask ratio (fraction of tokens masked)" if perturb == "mask" else "Low-rank ratio (k / rank)"


def plot_drop(df, out_path, perturb):
    levels = sorted(df["level"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model, grp in df.groupby("model"):
        grp = grp.sort_values("level")
        color = COLORS.get(model)
        axes[0].plot(grp["level"], grp["clip_drop"], marker="o", label=model, color=color)
        axes[1].plot(grp["level"], grp["bert_drop"], marker="o", label=model, color=color)
    tag = perturb.capitalize()
    for ax, title, ylabel in zip(
        axes,
        [f"CLIPScore Drop vs {tag} Level", f"BERTScore Drop vs {tag} Level"],
        ["CLIPScore drop (original − perturbed)", "BERTScore F1 drop (original − perturbed)"],
    ):
        ax.set_xlabel(_xlabel(perturb))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.legend()
        ax.grid(True, alpha=0.3)
        _set_ticks(ax, levels)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


def plot_score(df, out_path, perturb):
    levels = sorted(df["level"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model, grp in df.groupby("model"):
        grp = grp.sort_values("level")
        color = COLORS.get(model)
        axes[0].plot(grp["level"], grp["clip_orig"], marker="o", linestyle="--",
                     color=color, alpha=0.5, label=f"{model} original")
        axes[0].plot(grp["level"], grp["clip_pert"], marker="o", linestyle="-",
                     color=color, label=f"{model} perturbed")
        axes[1].plot(grp["level"], grp["bert_orig"], marker="o", linestyle="--",
                     color=color, alpha=0.5, label=f"{model} original")
        axes[1].plot(grp["level"], grp["bert_pert"], marker="o", linestyle="-",
                     color=color, label=f"{model} perturbed")
    tag = perturb.capitalize()
    for ax, title, ylabel in zip(
        axes,
        [f"CLIPScore vs {tag} Level", f"BERTScore vs {tag} Level"],
        ["CLIPScore", "BERTScore F1"],
    ):
        ax.set_xlabel(_xlabel(perturb))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        _set_ticks(ax, levels)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scored_dir", default="data/perturb_out_scored")
    p.add_argument("--perturb", choices=["lowrank", "mask"], default="lowrank")
    p.add_argument("--mode", choices=["drop", "score"], default="drop")
    p.add_argument("--out", default="")
    args = p.parse_args()

    scored_dir = Path(args.scored_dir)
    df = load_perturb(scored_dir, args.perturb)
    if df.empty:
        print(f"No {args.perturb} CSVs found in {scored_dir}")
        return

    default_out = f"figures/{args.perturb}_{args.mode}.png"
    out_path = Path(args.out if args.out else default_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "drop":
        plot_drop(df, out_path, args.perturb)
    else:
        plot_score(df, out_path, args.perturb)


if __name__ == "__main__":
    main()
