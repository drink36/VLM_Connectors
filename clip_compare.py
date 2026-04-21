import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, CLIPModel


def parse_args():
    p = argparse.ArgumentParser("Compute CLIPScore for caption CSV")
    p.add_argument("--csv", type=str, required=True, help="input caption CSV")
    p.add_argument("--out", type=str, default="", help="output CSV path (default: overwrites input)")
    p.add_argument("--model_id", type=str, default="openai/clip-vit-base-patch16")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--caption_post_col", type=str, default="caption_post_direct",
                   help="column name for post-direct caption (if present)")
    return p.parse_args()


def image_to_ref_txt(image_path: str) -> str:
    return str(Path(image_path).with_suffix(".txt"))


def read_reference(txt_path: str) -> str:
    if not os.path.exists(txt_path):
        return ""
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_rgb(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


@torch.no_grad()
def compute_clipscore_batch(model, processor, image_paths, captions, device):
    images = [load_rgb(p) for p in image_paths]
    inputs = processor(
        text=captions, images=images,
        padding=True, truncation=True, return_tensors="pt",
    ).to(device)
    outputs = model(**inputs)
    img_e = F.normalize(outputs.image_embeds, dim=-1)
    txt_e = F.normalize(outputs.text_embeds, dim=-1)
    scores = torch.clamp(100.0 * (img_e * txt_e).sum(dim=-1), min=0.0)
    return scores.detach().cpu().tolist()


def score_column(df, caption_col, model, processor, device, batch_size):
    scores = [float("nan")] * len(df)
    valid_idx, valid_paths, valid_caps = [], [], []
    for i, row in df.iterrows():
        img_path = str(row["image_path"])
        cap = str(row[caption_col]).strip()
        if os.path.exists(img_path) and cap:
            valid_idx.append(i)
            valid_paths.append(img_path)
            valid_caps.append(cap)
    for start in range(0, len(valid_idx), batch_size):
        end = min(start + batch_size, len(valid_idx))
        batch_scores = compute_clipscore_batch(
            model, processor,
            valid_paths[start:end], valid_caps[start:end], device,
        )
        for i, s in zip(valid_idx[start:end], batch_scores):
            scores[i] = s
    return scores


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.csv)
    required = ["image_path", "caption_original", "caption_recon"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["image_path"] = df["image_path"].fillna("").astype(str)
    df["reference_path"] = df["image_path"].apply(image_to_ref_txt)
    df["reference_caption"] = df["reference_path"].apply(read_reference)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = CLIPModel.from_pretrained(args.model_id, use_safetensors=True).to(device).eval()

    print("Computing CLIPScore for original captions...")
    df["clipscore_original"] = score_column(df, "caption_original", model, processor, device, args.batch_size)

    print("Computing CLIPScore for recon captions...")
    df["clipscore_recon"] = score_column(df, "caption_recon", model, processor, device, args.batch_size)

    if args.caption_post_col in df.columns:
        print(f"Computing CLIPScore for {args.caption_post_col}...")
        df[f"clipscore_post_direct"] = score_column(df, args.caption_post_col, model, processor, device, args.batch_size)
        df["clipscore_drop"] = df["clipscore_post_direct"] - df["clipscore_recon"]
    else:
        df["clipscore_drop"] = df["clipscore_original"] - df["clipscore_recon"]

    print("Computing CLIPScore for GT reference captions...")
    df["clipscore_gt"] = score_column(df, "reference_caption", model, processor, device, args.batch_size)

    print("\nMean CLIPScore")
    for col in ["clipscore_original", "clipscore_recon", "clipscore_gt", "clipscore_drop"]:
        if col in df.columns:
            print(f"  {col}: {df[col].mean(skipna=True):.4f}")

    out_path = args.out or args.csv
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
