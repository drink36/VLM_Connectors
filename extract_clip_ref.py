"""
Extract CLIP or DINOv2 reference image embeddings for 3-way KNOR analysis.

Saves ref_vectors_*.pt shards (same format as pre/post vectors) so that
knn.py --ref_pt <this_dir> can compare KNOR(ref→pre) and KNOR(ref→post)
alongside the existing KNOR(pre→post).

Why: Qwen trains its vision encoder jointly with the connector, so the
pre-connector space may already differ from a standard image representation.
Using a neutral reference reveals whether low pre→post KNOR is due to an
unusual pre-space (joint training) vs genuine information loss.

Usage — CLIP (semantic anchor):
  python extract_clip_ref.py \
      --root data/mtf2025_web_images \
      --out_dir data/clip_ref \
      --model_id openai/clip-vit-large-patch14-336 \
      --pool cls --batch_size 32 --device cuda

Usage — DINOv2 (geometric anchor, primary):
  python extract_clip_ref.py \
      --root data/mtf2025_web_images \
      --out_dir data/dino_ref \
      --model_id facebook/dinov2-large \
      --pool cls --batch_size 32 --device cuda
Usage - DINOv3 (geometric anchor, secondary):
    python extract_clip_ref.py \
      --root data/mtf2025_web_images \
      --out_dir data/dinov3_ref \
      --model_id facebook/dinov3-vitl16-pretrain-lvd1689m \
      --pool cls --batch_size 32 --device cuda
"""
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser("Extract CLIP reference embeddings for 3-way KNOR")
    p.add_argument("--root", type=str, required=True,
                   help="folder containing images or subfolders of images")
    p.add_argument("--out_dir", type=str, default="data/clip_ref")
    p.add_argument("--model_id", type=str, default="openai/clip-vit-large-patch14-336",
                   help="HuggingFace CLIP model ID")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_items", type=int, default=0, help="0 = no limit")
    p.add_argument("--save_every", type=int, default=400, help="flush shard every N samples")
    p.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"],
                   help="cls: CLS token; mean: average of all patch tokens (excludes CLS)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--skip_existing", action="store_true",
                   help="skip output dirs that already have manifest.pt")
    p.add_argument("--model_type", type=str, default="auto", choices=["auto", "clip", "dino"],
                   help="auto: infer from model_id; clip: CLIPModel API; dino: AutoModel API (DINOv2 etc.)")
    return p.parse_args()


class FolderImageDataset(Dataset):
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        paths = sorted(self.root.glob("*.jpg")) + sorted(self.root.glob("*.png"))
        self.image_paths = sorted(paths)
        if not self.image_paths:
            raise FileNotFoundError(f"No .jpg/.png found in {self.root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        p = self.image_paths[idx]
        try:
            return {"key": p.stem, "image": Image.open(p).convert("RGB")}
        except Exception:
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return {"keys": [b["key"] for b in batch], "images": [b["image"] for b in batch]}


@torch.inference_mode()
def extract_ref(
    processor,
    model,
    dataloader,
    pool: str,
    device: str,
    out_dir: str,
    max_items: Optional[int],
    save_every: int,
    model_type: str = "clip",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen = 0
    shard_count = 0
    shard_keys: List[str] = []
    shard_vecs: List[torch.Tensor] = []

    def flush_shard(tag: int) -> None:
        nonlocal shard_count, shard_keys, shard_vecs
        if not shard_keys:
            return
        mat = torch.cat(shard_vecs, dim=0)
        torch.save({"keys": shard_keys, "vecs": mat}, out_dir / f"ref_vectors_{tag}.pt")
        print(f"[shard] saved {len(shard_keys)} samples | shape={tuple(mat.shape)}")
        shard_count += 1
        shard_keys, shard_vecs = [], []

    for batch in tqdm(dataloader, desc="CLIP ref extract"):
        keys = batch["keys"]
        images = batch["images"]
        if not keys:
            continue
        if max_items is not None and seen >= max_items:
            break
        if max_items is not None and seen + len(keys) > max_items:
            keep = max_items - seen
            keys, images = keys[:keep], images[:keep]

        # Processor handles resizing/normalisation for each model variant
        inputs = processor(images=images, return_tensors="pt").to(device)
        if model_type == "dino":
            outputs = model(pixel_values=inputs["pixel_values"])
        else:  # clip
            outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        hidden = outputs.last_hidden_state  # [B, 1+num_patches, D]

        if pool == "cls":
            vecs = hidden[:, 0, :].float().cpu()        # [B, D]
        else:
            # mean over patch tokens only (exclude CLS at index 0)
            vecs = hidden[:, 1:, :].mean(dim=1).float().cpu()  # [B, D]

        for j, k in enumerate(keys):
            shard_keys.append(k)
            shard_vecs.append(vecs[j : j + 1])
            seen += 1

        if save_every > 0 and len(shard_keys) >= save_every:
            flush_shard(seen)

    flush_shard(seen)
    torch.save(
        {"total_seen": seen, "shards": shard_count, "pool": pool, "model_type": model_type},
        out_dir / "manifest.pt",
    )
    print(f"Done. total_seen={seen}, shards={shard_count}, out_dir={out_dir}")


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root path not found: {root}")

    # Discover image folders (same logic as extract_multimodel_all.py)
    image_folders: List[Path] = []
    root_imgs = sorted(root.glob("*.jpg")) + sorted(root.glob("*.png"))
    if root_imgs:
        image_folders.append(root)
    else:
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            if sorted(sub.glob("*.jpg")) + sorted(sub.glob("*.png")):
                image_folders.append(sub)

    if not image_folders:
        raise FileNotFoundError(f"No image folders found under {root}")

    print(f"Found {len(image_folders)} image folder(s)")

    # Resolve model_type: auto-detect from model_id if not specified explicitly
    model_type = args.model_type
    if model_type == "auto":
        model_type = "dino" if "dino" in args.model_id.lower() else "clip"
    print(f"Model type: {model_type}  |  model_id: {args.model_id}  |  pool: {args.pool}")

    if model_type == "dino":
        from transformers import AutoModel, AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(args.model_id)
        model = AutoModel.from_pretrained(args.model_id).to(device).eval()
    else:
        from transformers import CLIPModel, CLIPProcessor
        processor = CLIPProcessor.from_pretrained(args.model_id)
        model = CLIPModel.from_pretrained(args.model_id).to(device).eval()

    max_items = None if args.max_items == 0 else args.max_items

    for folder in image_folders:
        run_out = Path(args.out_dir) / folder.name
        if args.skip_existing and (run_out / "manifest.pt").exists():
            print(f"Skipping existing: {run_out}")
            continue

        print(f"Extracting folder={folder} -> out={run_out}")
        ds = FolderImageDataset(str(folder))
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        extract_ref(processor, model, dl, args.pool, device, str(run_out), max_items, args.save_every, model_type=model_type)


if __name__ == "__main__":
    main()
