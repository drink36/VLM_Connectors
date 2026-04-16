import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, CLIPModel


CSV_PATH = "caption_compare_out/qwen2.5/caption_original_vs_recon.csv"
OUT_PATH = "caption_compare_out/qwen2.5/caption_original_vs_recon_with_clipscore.csv"
MODEL_ID = "openai/clip-vit-base-patch16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16


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
        text=captions,
        images=images,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    outputs = model(**inputs)

    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    scores = 100.0 * (image_embeds * text_embeds).sum(dim=-1)
    scores = torch.clamp(scores, min=0.0)
    return scores.detach().cpu().tolist()


def score_column(df, caption_col, model, processor, device, batch_size):
    scores = [float("nan")] * len(df)

    valid_idx = []
    valid_paths = []
    valid_caps = []

    for i, row in df.iterrows():
        img_path = str(row["image_path"])
        cap = str(row[caption_col]).strip()
        if os.path.exists(img_path) and cap:
            valid_idx.append(i)
            valid_paths.append(img_path)
            valid_caps.append(cap)

    for start in range(0, len(valid_idx), batch_size):
        end = min(start + batch_size, len(valid_idx))
        batch_idx = valid_idx[start:end]
        batch_paths = valid_paths[start:end]
        batch_caps = valid_caps[start:end]

        batch_scores = compute_clipscore_batch(
            model=model,
            processor=processor,
            image_paths=batch_paths,
            captions=batch_caps,
            device=device,
        )

        for i, s in zip(batch_idx, batch_scores):
            scores[i] = s

    return scores


def main():
    df = pd.read_csv(CSV_PATH)

    required = ["image_path", "caption_original", "caption_recon"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["image_path"] = df["image_path"].fillna("").astype(str)
    df["caption_original"] = df["caption_original"].fillna("").astype(str)
    df["caption_recon"] = df["caption_recon"].fillna("").astype(str)

    df["reference_path"] = df["image_path"].apply(image_to_ref_txt)
    df["reference_caption"] = df["reference_path"].apply(read_reference)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True).to(DEVICE).eval()

    print("Computing CLIPScore for original captions...")
    df["clipscore_orig"] = score_column(
        df, "caption_original", model, processor, DEVICE, BATCH_SIZE
    )

    print("Computing CLIPScore for reconstructed captions...")
    df["clipscore_recon"] = score_column(
        df, "caption_recon", model, processor, DEVICE, BATCH_SIZE
    )

    print("Computing CLIPScore for reference captions...")
    df["clipscore_ref"] = score_column(
        df, "reference_caption", model, processor, DEVICE, BATCH_SIZE
    )

    df["clipscore_delta_recon_minus_orig"] = (
        df["clipscore_recon"] - df["clipscore_orig"]
    )

    print("\nMean CLIPScore")
    print("orig :", df["clipscore_orig"].mean(skipna=True))
    print("recon:", df["clipscore_recon"].mean(skipna=True))
    print("ref  :", df["clipscore_ref"].mean(skipna=True))
    print("delta:", df["clipscore_delta_recon_minus_orig"].mean(skipna=True))

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()