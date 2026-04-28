"""
Print and save summary statistics (mean, median, std, IQR) for all numeric
columns in caption_original_vs_recon.csv, one row per model.

Usage:
  python summary_stats.py
  python summary_stats.py --data_dir data/output/caption_compare_out_nor
  python summary_stats.py --col clipscore_drop reproj_cosine_recon bertscore_recon_vs_post_f1
  python summary_stats.py --out results/summary_stats.csv
"""
import argparse
from pathlib import Path

import pandas as pd

MODELS_ORDER = ["llava", "idefics2", "qwen2.5vl", "qwen3.5"]
DISPLAY = {"llava": "LLaVA", "idefics2": "Idefics2",
           "qwen2.5vl": "Qwen-2.5-VL", "qwen3.5": "Qwen-3.5"}

DEFAULT_COLS = [
    "clipscore_drop",
    "clipscore_post_direct",
    "clipscore_recon",
    "reproj_cosine_recon",
    "reproj_mse_recon",
    "bertscore_recon_vs_post_f1",
    "bertscore_drop",
]


def load_model(data_dir: Path, model: str) -> pd.DataFrame | None:
    csv = data_dir / model / "caption_original_vs_recon.csv"
    if not csv.exists():
        print(f"  [skip] missing: {csv}")
        return None
    df = pd.read_csv(csv)
    # normalize old BERTScore column names
    if "bertscore_f1" in df.columns and "bertscore_recon_vs_post_f1" not in df.columns:
        df = df.rename(columns={
            "bertscore_f1":        "bertscore_recon_vs_post_f1",
            "bertscore_precision": "bertscore_recon_vs_post_precision",
            "bertscore_recall":    "bertscore_recon_vs_post_recall",
        })
    if "bertscore_post_vs_gt_f1" in df.columns and "bertscore_recon_vs_gt_f1" in df.columns:
        df["bertscore_drop"] = df["bertscore_post_vs_gt_f1"] - df["bertscore_recon_vs_gt_f1"]
    return df


def compute_stats(df: pd.DataFrame, col: str) -> dict:
    s = df[col].dropna()
    return {
        "n":      len(s),
        "mean":   s.mean(),
        "median": s.median(),
        "std":    s.std(),
        "q25":    s.quantile(0.25),
        "q75":    s.quantile(0.75),
        "min":    s.min(),
        "max":    s.max(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/output/caption_compare_out_nor")
    p.add_argument("--col", nargs="+", default=DEFAULT_COLS,
                   help="columns to summarize (default: all key columns)")
    p.add_argument("--out", default="", help="save CSV to this path (optional)")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    all_rows = []

    for model in MODELS_ORDER:
        df = load_model(data_dir, model)
        if df is None:
            continue
        for col in args.col:
            if col not in df.columns:
                continue
            stats = compute_stats(df, col)
            all_rows.append({"model": DISPLAY[model], "column": col, **stats})

    if not all_rows:
        print("No data found.")
        return

    result = pd.DataFrame(all_rows)

    # Print one table per column
    for col in args.col:
        sub = result[result["column"] == col]
        if sub.empty:
            continue
        print(f"\n{'='*60}")
        print(f"  {col}")
        print(f"{'='*60}")
        tbl = sub[["model", "n", "mean", "median", "std", "q25", "q75"]].copy()
        tbl["IQR"] = tbl.apply(lambda r: f"{r['q25']:.3f}–{r['q75']:.3f}", axis=1)
        tbl = tbl.drop(columns=["q25", "q75"])
        print(tbl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out, index=False, float_format="%.4f")
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
