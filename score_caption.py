"""
Add BERTScore columns to caption_eval output CSVs.

Computes:
  - bertscore_recon_vs_post : caption_recon vs caption_post_direct (semantic drift)
  - bertscore_post_vs_gt    : caption_post_direct vs GT (.txt files)
  - bertscore_recon_vs_gt   : caption_recon vs GT (.txt files)

Handles both old format (caption_original) and new format (caption_post_direct).
Writes new CSVs to --out_dir (default: same dir, overwrite).

Usage:
  python score_caption.py --caption_dir data/output/caption_compare_out_a/llava
  python score_caption.py --caption_dir data/output/caption_compare_out_a/llava --out_dir data/output/caption_scored/llava
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_gt(image_path: str) -> str:
    txt = Path(image_path).with_suffix(".txt")
    return txt.read_text(encoding="utf-8").strip() if txt.exists() else ""


def run_bertscore(cands, refs, device, batch_size, lang="en"):
    from bert_score import score as bscore
    valid = [(i, h, r) for i, (h, r) in enumerate(zip(cands, refs)) if h.strip() and r.strip()]
    p_out = [float("nan")] * len(cands)
    r_out = [float("nan")] * len(cands)
    f_out = [float("nan")] * len(cands)
    if not valid:
        return p_out, r_out, f_out
    idxs, hyps, ref_list = zip(*valid)
    p, r, f1 = bscore(list(hyps), list(ref_list), lang=lang,
                      device=device, batch_size=batch_size, verbose=False)
    for i, pv, rv, fv in zip(idxs, p.tolist(), r.tolist(), f1.tolist()):
        p_out[i], r_out[i], f_out[i] = pv, rv, fv
    return p_out, r_out, f_out


def score_csv(csv_path: Path, out_path: Path, device: str, batch_size: int, skip_existing: bool):
    if skip_existing and out_path.exists():
        print(f"  SKIP (already exists): {out_path.name}")
        return

    df = pd.read_csv(csv_path)

    # Handle both old and new column names
    post_col = "caption_post_direct" if "caption_post_direct" in df.columns else \
               "caption_original" if "caption_original" in df.columns else None
    recon_col = "caption_recon" if "caption_recon" in df.columns else None

    if post_col is None or recon_col is None:
        print(f"  SKIP (missing caption columns): {csv_path.name}")
        return

    caps_post = df[post_col].fillna("").astype(str).tolist()
    caps_recon = df[recon_col].fillna("").astype(str).tolist()
    image_paths = df["image_path"].fillna("").astype(str).tolist() if "image_path" in df.columns else [""] * len(df)
    gt_caps = [read_gt(p) for p in image_paths]

    print(f"  BERTScore: recon vs post ...")
    p, r, f = run_bertscore(caps_recon, caps_post, device, batch_size)
    df["bertscore_recon_vs_post_p"] = p
    df["bertscore_recon_vs_post_r"] = r
    df["bertscore_recon_vs_post_f1"] = f

    print(f"  BERTScore: post vs GT ...")
    p, r, f = run_bertscore(caps_post, gt_caps, device, batch_size)
    df["bertscore_post_vs_gt_p"] = p
    df["bertscore_post_vs_gt_r"] = r
    df["bertscore_post_vs_gt_f1"] = f

    print(f"  BERTScore: recon vs GT ...")
    p, r, f = run_bertscore(caps_recon, gt_caps, device, batch_size)
    df["bertscore_recon_vs_gt_p"] = p
    df["bertscore_recon_vs_gt_r"] = r
    df["bertscore_recon_vs_gt_f1"] = f

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  -> saved: {out_path}")
    print(f"     recon_vs_post F1 : {np.nanmean(df['bertscore_recon_vs_post_f1']):.4f}")
    print(f"     post_vs_gt   F1 : {np.nanmean(df['bertscore_post_vs_gt_f1']):.4f}")
    print(f"     recon_vs_gt  F1 : {np.nanmean(df['bertscore_recon_vs_gt_f1']):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--caption_dir", type=str, required=True,
                   help="directory containing caption CSV files")
    p.add_argument("--out_dir", type=str, default="",
                   help="output dir (default: overwrite in place)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    caption_dir = Path(args.caption_dir)
    csvs = sorted(caption_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSVs found in {caption_dir}")
        return

    print(f"Found {len(csvs)} CSVs in {caption_dir}")
    pbar = tqdm(csvs, desc="scoring", unit="csv")
    for csv_path in pbar:
        pbar.set_description(f"scoring [{csv_path.name}]")
        out_path = Path(args.out_dir) / csv_path.name if args.out_dir else csv_path
        score_csv(csv_path, out_path, device, args.batch_size, args.skip_existing)


if __name__ == "__main__":
    main()
