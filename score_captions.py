"""
Add CLIPScore and BERTScore columns to caption_eval output CSVs.

Merges the functionality of clip_compare.py and score_caption.py.

CLIP columns written:
  clipscore_recon         : image × caption_recon
  clipscore_post_direct   : image × caption_post_direct
  clipscore_gt            : image × GT caption (.txt file)
  clipscore_drop          : clipscore_post_direct − clipscore_recon

BERTScore columns written:
  bertscore_recon_vs_post_p/r/f1  : caption_recon vs caption_post_direct
  bertscore_post_vs_gt_p/r/f1     : caption_post_direct vs GT
  bertscore_recon_vs_gt_p/r/f1    : caption_recon vs GT

Usage:
  # score all CSVs in a directory (both CLIP and BERTScore)
  python score_captions.py --caption_dir data/output/caption_compare_out_nor/llava

  # single CSV, CLIP only
  python score_captions.py --csv path/to/file.csv --no_bert

  # directory, BERTScore only
  python score_captions.py --caption_dir ... --no_clip

  # write to a separate output dir
  python score_captions.py --caption_dir ... --out_dir data/output/scored/llava
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _read_gt(image_path: str) -> str:
    txt = Path(image_path).with_suffix(".txt")
    return txt.read_text(encoding="utf-8").strip() if txt.exists() else ""


def _resolve_caption_col(df: pd.DataFrame, preferred="caption_post_direct", fallback="caption_original"):
    if preferred in df.columns:
        return preferred
    if fallback in df.columns:
        return fallback
    return None


# ---------------------------------------------------------------------------
# CLIP scoring
# ---------------------------------------------------------------------------

_clip_model = None
_clip_processor = None


def _load_clip(model_id: str, device: str):
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import AutoProcessor, CLIPModel
        print(f"Loading CLIP: {model_id}")
        _clip_processor = AutoProcessor.from_pretrained(model_id)
        _clip_model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device).eval()


def _clipscore_batch(image_paths, captions, device, batch_size=16, label="CLIPScore"):
    import torch
    import torch.nn.functional as F
    from PIL import Image

    scores = [float("nan")] * len(image_paths)
    valid = [(i, p, c) for i, (p, c) in enumerate(zip(image_paths, captions))
             if p and os.path.exists(str(p)) and str(c).strip()]
    if not valid:
        return scores

    for start in tqdm(range(0, len(valid), batch_size), desc=f"  {label}", unit="batch", leave=False):
        chunk = valid[start:start + batch_size]
        idxs, paths, caps = zip(*chunk)
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = _clip_processor(
            text=list(caps), images=images,
            padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = _clip_model(**inputs)
        img_e = F.normalize(out.image_embeds, dim=-1)
        txt_e = F.normalize(out.text_embeds, dim=-1)
        vals = torch.clamp(100.0 * (img_e * txt_e).sum(dim=-1), min=0.0).cpu().tolist()
        for i, v in zip(idxs, vals):
            scores[i] = v
    return scores


def add_clip_scores(df: pd.DataFrame, device: str, batch_size: int) -> pd.DataFrame:
    post_col = _resolve_caption_col(df)
    if post_col is None or "image_path" not in df.columns:
        print("  [skip clip] missing image_path or caption column")
        return df

    img  = df["image_path"].fillna("").astype(str).tolist()
    recon = df["caption_recon"].fillna("").astype(str).tolist() if "caption_recon" in df.columns else [""] * len(df)
    post  = df[post_col].fillna("").astype(str).tolist()
    gt    = [_read_gt(p) for p in img]

    df["clipscore_recon"]        = _clipscore_batch(img, recon,  device, batch_size, "clip:recon")
    df["clipscore_post_direct"]  = _clipscore_batch(img, post,   device, batch_size, "clip:post")
    df["clipscore_gt"]           = _clipscore_batch(img, gt,     device, batch_size, "clip:gt")
    df["clipscore_drop"]         = df["clipscore_post_direct"] - df["clipscore_recon"]
    return df


# ---------------------------------------------------------------------------
# BERTScore scoring
# ---------------------------------------------------------------------------

def _run_bertscore(cands, refs, device, batch_size, lang="en"):
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


def add_bertscore(df: pd.DataFrame, device: str, batch_size: int) -> pd.DataFrame:
    post_col = _resolve_caption_col(df)
    if post_col is None or "caption_recon" not in df.columns:
        print("  [skip bert] missing caption columns")
        return df

    recon = df["caption_recon"].fillna("").astype(str).tolist()
    post  = df[post_col].fillna("").astype(str).tolist()
    img   = df["image_path"].fillna("").astype(str).tolist() if "image_path" in df.columns else [""] * len(df)
    gt    = [_read_gt(p) for p in img]

    print("  BERTScore: recon vs post ...")
    p, r, f = _run_bertscore(recon, post, device, batch_size)
    df["bertscore_recon_vs_post_p"]  = p
    df["bertscore_recon_vs_post_r"]  = r
    df["bertscore_recon_vs_post_f1"] = f

    print("  BERTScore: post vs GT ...")
    p, r, f = _run_bertscore(post, gt, device, batch_size)
    df["bertscore_post_vs_gt_p"]  = p
    df["bertscore_post_vs_gt_r"]  = r
    df["bertscore_post_vs_gt_f1"] = f

    print("  BERTScore: recon vs GT ...")
    p, r, f = _run_bertscore(recon, gt, device, batch_size)
    df["bertscore_recon_vs_gt_p"]  = p
    df["bertscore_recon_vs_gt_r"]  = r
    df["bertscore_recon_vs_gt_f1"] = f
    return df


# ---------------------------------------------------------------------------
# Per-file driver
# ---------------------------------------------------------------------------

def score_csv(csv_path: Path, out_path: Path, device: str, batch_size: int,
              do_clip: bool, do_bert: bool, skip_existing: bool):
    if skip_existing and out_path.exists():
        print(f"  SKIP (exists): {out_path.name}")
        return

    df = pd.read_csv(csv_path)

    if do_clip:
        df = add_clip_scores(df, device, batch_size)
        print(f"     clipscore_post  : {np.nanmean(df['clipscore_post_direct']):.4f}")
        print(f"     clipscore_recon : {np.nanmean(df['clipscore_recon']):.4f}")
        print(f"     clipscore_drop  : {np.nanmean(df['clipscore_drop']):.4f}")

    if do_bert:
        df = add_bertscore(df, device, batch_size)
        print(f"     bert recon/post : {np.nanmean(df['bertscore_recon_vs_post_f1']):.4f}")
        print(f"     bert post/gt    : {np.nanmean(df['bertscore_post_vs_gt_f1']):.4f}")
        print(f"     bert recon/gt   : {np.nanmean(df['bertscore_recon_vs_gt_f1']):.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  -> saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--caption_dir", type=str, help="directory containing caption CSV files")
    src.add_argument("--csv", type=str, help="single CSV file to score")
    p.add_argument("--out_dir", type=str, default="",
                   help="output directory (default: overwrite in-place)")
    p.add_argument("--clip_model_id", type=str, default="openai/clip-vit-base-patch16")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--no_clip", action="store_true", help="skip CLIP scoring")
    p.add_argument("--no_bert", action="store_true", help="skip BERTScore")
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    do_clip = not args.no_clip
    do_bert = not args.no_bert

    if do_clip:
        _load_clip(args.clip_model_id, device)

    if args.csv:
        csvs = [Path(args.csv)]
    else:
        csvs = sorted(Path(args.caption_dir).glob("*.csv"))
        if not csvs:
            print(f"No CSVs found in {args.caption_dir}")
            return
        print(f"Found {len(csvs)} CSVs in {args.caption_dir}")

    for csv_path in tqdm(csvs, desc="scoring", unit="csv"):
        print(f"\n[{csv_path.name}]")
        out_path = Path(args.out_dir) / csv_path.name if args.out_dir else csv_path
        score_csv(csv_path, out_path, device, args.batch_size, do_clip, do_bert, args.skip_existing)


if __name__ == "__main__":
    main()
