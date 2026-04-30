"""
Add CLIPScore and BERTScore to perturb_out CSVs.

For each CSV in --perturb_dir:
  - clipscore_original    : image x caption_original
  - clipscore_perturbed   : image x caption_perturbed
  - clipscore_gt          : image x GT caption (.txt file)
  - clipscore_drop        : clipscore_perturbed - clipscore_original
  - bertscore_original    : caption_original vs GT (F1)
  - bertscore_perturbed   : caption_perturbed vs GT (F1)
  - bertscore_orig_vs_pert: caption_original vs caption_perturbed (F1)

Results are written back to the same CSV (or --out_dir if given).

Usage:
  python score_perturb.py --perturb_dir data/perturb_out
  python score_perturb.py --perturb_dir data/perturb_out --out_dir data/perturb_out_scored
"""
import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = None
clip_processor = None


def _load_clip(model_id: str):
    global clip_model, clip_processor
    if clip_model is None:
        print(f"Loading CLIP: {model_id}")
        clip_processor = AutoProcessor.from_pretrained(model_id)
        clip_model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device).eval()


@torch.no_grad()
def _clipscore_batch(image_paths, captions, batch_size=16, label="CLIPScore"):
    scores = [float("nan")] * len(image_paths)
    valid = [(i, p, c) for i, (p, c) in enumerate(zip(image_paths, captions))
             if p and os.path.exists(str(p)) and str(c).strip()]
    if not valid:
        return scores
    for start in tqdm(range(0, len(valid), batch_size), desc=f"  {label}", unit="batch", leave=False):
        chunk = valid[start:start + batch_size]
        idxs, paths, caps = zip(*chunk)
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = clip_processor(
            text=list(caps), images=images,
            padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        out = clip_model(**inputs)
        img_e = F.normalize(out.image_embeds, dim=-1)
        txt_e = F.normalize(out.text_embeds, dim=-1)
        vals = torch.clamp(100.0 * (img_e * txt_e).sum(dim=-1), min=0.0).cpu().tolist()
        for i, v in zip(idxs, vals):
            scores[i] = v
    return scores


def _read_gt(image_path: str) -> str:
    txt = Path(image_path).with_suffix(".txt")
    if txt.exists():
        return txt.read_text(encoding="utf-8").strip()
    return ""


def score_csv(csv_path: Path, out_path: Path, clip_model_id: str, batch_size: int, skip_existing: bool):
    _load_clip(clip_model_id)

    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns:
        print(f"  SKIP (no image_path): {csv_path.name}")
        return

    if skip_existing and out_path.exists():
        print(f"  SKIP (already exists): {out_path.name}")
        return

    image_paths = df["image_path"].fillna("").astype(str).tolist()
    caps_orig = df["caption_original"].fillna("").astype(str).tolist()
    caps_pert = df["caption_perturbed"].fillna("").astype(str).tolist()
    gt_caps = [_read_gt(p) for p in image_paths]

    df["clipscore_original"]  = _clipscore_batch(image_paths, caps_orig, batch_size, "clip:original")
    df["clipscore_perturbed"] = _clipscore_batch(image_paths, caps_pert, batch_size, "clip:perturbed")
    df["clipscore_gt"]        = _clipscore_batch(image_paths, gt_caps,   batch_size, "clip:gt")
    df["clipscore_drop"] = df["clipscore_perturbed"] - df["clipscore_original"]

    print(f"  BERTScore (orig vs GT, pert vs GT, orig vs pert) ...")
    from bert_score import score as bert_score_fn
    valid_orig = [(i, h, r) for i, (h, r) in enumerate(zip(caps_orig, gt_caps)) if h.strip() and r.strip()]
    valid_pert = [(i, h, r) for i, (h, r) in enumerate(zip(caps_pert, gt_caps)) if h.strip() and r.strip()]

    bs_orig = [float("nan")] * len(df)
    if valid_orig:
        idxs, hyps, refs = zip(*valid_orig)
        _, _, F1 = bert_score_fn(list(hyps), list(refs), lang="en", device=str(device), verbose=False)
        for i, v in zip(idxs, F1.tolist()):
            bs_orig[i] = v
    df["bertscore_original"] = bs_orig

    bs_pert = [float("nan")] * len(df)
    if valid_pert:
        idxs, hyps, refs = zip(*valid_pert)
        _, _, F1 = bert_score_fn(list(hyps), list(refs), lang="en", device=str(device), verbose=False)
        for i, v in zip(idxs, F1.tolist()):
            bs_pert[i] = v
    df["bertscore_perturbed"] = bs_pert

    # caption_original vs caption_perturbed directly
    valid_cross = [(i, h, r) for i, (h, r) in enumerate(zip(caps_orig, caps_pert)) if h.strip() and r.strip()]
    bs_cross = [float("nan")] * len(df)
    if valid_cross:
        idxs, hyps, refs = zip(*valid_cross)
        _, _, F1 = bert_score_fn(list(hyps), list(refs), lang="en", device=str(device), verbose=False)
        for i, v in zip(idxs, F1.tolist()):
            bs_cross[i] = v
    df["bertscore_orig_vs_pert"] = bs_cross

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"  -> saved: {out_path}")
    print(f"     clipscore_original  : {df['clipscore_original'].mean(skipna=True):.4f}")
    print(f"     clipscore_perturbed : {df['clipscore_perturbed'].mean(skipna=True):.4f}")
    print(f"     clipscore_gt        : {df['clipscore_gt'].mean(skipna=True):.4f}")
    print(f"     bertscore_original  : {df['bertscore_original'].mean(skipna=True):.4f}")
    print(f"     bertscore_perturbed : {df['bertscore_perturbed'].mean(skipna=True):.4f}")
    print(f"     bertscore_orig_vs_pert: {df['bertscore_orig_vs_pert'].mean(skipna=True):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--perturb_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="data/perturb_out_scored",
                   help="output dir (default: data/perturb_out_scored)")
    p.add_argument("--clip_model_id", type=str, default="openai/clip-vit-base-patch16")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--skip_existing", action="store_true",
                   help="skip CSVs that already have all score columns")
    args = p.parse_args()

    perturb_dir = Path(args.perturb_dir)
    csvs = sorted(perturb_dir.glob("perturb_*.csv"))
    if not csvs:
        print(f"No perturb_*.csv found in {perturb_dir}")
        return

    print(f"Found {len(csvs)} CSVs in {perturb_dir}")
    pbar = tqdm(csvs, desc="scoring", unit="csv")
    for csv_path in pbar:
        pbar.set_description(f"scoring [{csv_path.name}]")
        out_path = Path(args.out_dir) / csv_path.name if args.out_dir else csv_path
        score_csv(csv_path, out_path, args.clip_model_id, args.batch_size, args.skip_existing)


if __name__ == "__main__":
    main()
