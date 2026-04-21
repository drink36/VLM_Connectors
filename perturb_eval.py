"""
Section 4.3: Controlled perturbations on E_post.

Loads stored E_post from shard files, applies a perturbation (mask / lowrank / orthogonal),
injects the result via the connector output hook, and records captions.
No reconstruction model needed.
"""
import argparse
import hashlib
import itertools
import math
import os
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    LlavaForConditionalGeneration,
    Idefics2ForConditionalGeneration,
)
from transformers.image_utils import load_image


# ---------------------------------------------------------------------------
# Shard dataset (post embeddings only)
# ---------------------------------------------------------------------------

class ShardSample(TypedDict):
    post: torch.Tensor
    sample_id: str


def numeric_suffix(path: Path) -> int:
    match = re.search(r"_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else 10**18


class PostVectorDataset(Dataset):
    """Loads post_vectors_*.pt shards. Returns E_post per sample."""

    def __init__(self, vec_dir: str, limit: int = 0, cache_shards: int = 1) -> None:
        self.cache_shards = max(1, cache_shards)
        post_root = Path(vec_dir)
        files = sorted(post_root.glob("post_vectors_*.pt"), key=numeric_suffix)
        if not files:
            raise FileNotFoundError(f"No post_vectors_*.pt found in {vec_dir}")

        self.post_map = {numeric_suffix(p): p for p in files}
        self.tags = sorted(self.post_map)

        self.items: list[tuple[int, int, str]] = []
        seen = 0
        for tag in self.tags:
            obj = torch.load(self.post_map[tag], map_location="cpu", weights_only=True)
            for row_idx, key in enumerate(obj["keys"]):
                self.items.append((tag, row_idx, str(key)))
                seen += 1
                if limit and seen >= limit:
                    break
            if limit and seen >= limit:
                break

        self.cache: OrderedDict[int, torch.Tensor] = OrderedDict()

    def _load(self, tag: int) -> torch.Tensor:
        if tag in self.cache:
            self.cache.move_to_end(tag)
            return self.cache[tag]
        while len(self.cache) >= self.cache_shards:
            self.cache.popitem(last=False)
        obj = torch.load(self.post_map[tag], map_location="cpu", weights_only=True)
        vecs = obj["vecs"]
        if vecs.dim() == 2:
            vecs = vecs.unsqueeze(1)
        self.cache[tag] = vecs
        return vecs

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> ShardSample:
        tag, row_idx, key = self.items[index]
        vecs = self._load(tag)
        return {"post": vecs[row_idx], "sample_id": f"{tag}:{row_idx}:{key}"}


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------

def stable_seed(base_seed: int, text: str) -> int:
    payload = f"{base_seed}:{text}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _effective_rank(level: float, max_rank: int) -> int:
    if max_rank <= 0:
        return 0
    if level <= 0:
        return 1
    if level <= 1.0:
        return max(1, min(max_rank, int(round(level * max_rank))))
    return max(1, min(max_rank, int(round(level))))


_orth_cache: dict = {}


def _orthogonal_basis(dim: int, device: torch.device, dtype: torch.dtype, seed: int) -> torch.Tensor:
    key = (dim, str(device), dtype, seed)
    if key in _orth_cache:
        return _orth_cache[key]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    a = torch.randn((dim, dim), generator=gen, device=device, dtype=torch.float32)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).clamp(min=1)
    _orth_cache[key] = q.to(dtype)
    return _orth_cache[key]


def perturb(post: torch.Tensor, mode: str, level: float, sample_ids: list[str], seed: int) -> torch.Tensor:
    """Apply perturbation in-place on a [B, T, D] tensor. Returns perturbed copy."""
    if mode == "none":
        return post
    B, T, D = post.shape
    out = post.clone()

    if mode == "mask":
        frac = float(max(0.0, min(1.0, level)))
        for b in range(B):
            gen = torch.Generator(device=out.device)
            gen.manual_seed(stable_seed(seed, sample_ids[b]))
            mask = torch.rand((T,), generator=gen, device=out.device) < frac
            out[b, mask] = 0

    elif mode == "lowrank":
        _rank_printed = getattr(perturb, "_rank_printed", False)
        for b in range(B):
            x = out[b].float()
            mean = x.mean(0, keepdim=True)
            u, s, vh = torch.linalg.svd(x - mean, full_matrices=False)
            rank = _effective_rank(level, s.shape[0])
            if not _rank_printed:
                print(f"[lowrank] keeping {rank}/{s.shape[0]} singular values (level={level})")
                perturb._rank_printed = True
                _rank_printed = True
            out[b] = ((u[:, :rank] * s[:rank]) @ vh[:rank] + mean).to(out.dtype)

    elif mode == "orthogonal":
        alpha = float(max(0.0, min(1.0, level)))
        q = _orthogonal_basis(D, out.device, out.dtype, seed)
        out = (1 - alpha) * out + alpha * (out @ q)

    else:
        raise ValueError(f"unknown perturbation mode: {mode}")

    return out


# ---------------------------------------------------------------------------
# VLM helpers
# ---------------------------------------------------------------------------

MODEL_IDS = {
    "llava":    "llava-hf/llava-1.5-7b-hf",
    "idefics2": "HuggingFaceM4/idefics2-8b",
    "qwen2.5vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3.5": "Qwen/Qwen3.5-9B",
}


def load_vlm(model_name: str, img_size: int, device: torch.device):
    model_id = MODEL_IDS[model_name]
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if model_name == "idefics2" and hasattr(processor, "image_processor"):
        processor.image_processor.size = {"longest_edge": img_size, "shortest_edge": img_size}
        if hasattr(processor.image_processor, "do_image_splitting"):
            processor.image_processor.do_image_splitting = False

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if model_name == "llava":
        model = LlavaForConditionalGeneration.from_pretrained(model_id, dtype=dtype)
    elif model_name == "idefics2":
        model = Idefics2ForConditionalGeneration.from_pretrained(model_id, dtype=dtype, trust_remote_code=True)
    else:
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, trust_remote_code=True)

    return processor, model.to(device).eval()


def build_inputs(processor, image_paths: list[str], prompt: str, model_name: str, img_size: int, device: torch.device):
    images = [load_image(p).resize((img_size, img_size)) for p in image_paths]

    if model_name in ("llava", "idefics2"):
        template = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(template, add_generation_prompt=True)
        img_arg = images if model_name == "llava" else [[im] for im in images]
        inputs = processor(images=img_arg, text=[text] * len(images), padding=True, return_tensors="pt")
        return {k: v.to(device) for k, v in inputs.items()}

    from qwen_vl_utils import process_vision_info
    templates = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt}]}] for im in images]
    tmpl_kwargs = {"enable_thinking": False} if model_name == "qwen3.5" else {}
    texts = [processor.apply_chat_template(t, tokenize=False, add_generation_prompt=True, **tmpl_kwargs) for t in templates]
    image_inputs, video_inputs = process_vision_info(templates)
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def decode_outputs(processor, out_ids: torch.Tensor, input_ids: torch.Tensor, model_name: str) -> list[str]:
    import re
    if model_name == "qwen2.5vl":
        pad_id = processor.tokenizer.pad_token_id
        results = []
        for i, row in enumerate(input_ids):
            plen = (row != pad_id).sum().item() if pad_id is not None else input_ids.shape[1]
            txt = processor.batch_decode(out_ids[i, plen:].unsqueeze(0), skip_special_tokens=True)[0]
            results.append(txt.strip())
        return results
    gen = out_ids[:, input_ids.shape[1]:]
    texts = [c.strip() for c in processor.batch_decode(gen, skip_special_tokens=True)]
    if model_name == "qwen3.5":
        texts = [re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip() for t in texts]
    return texts


def resolve_connector(model: nn.Module, model_name: str) -> nn.Module:
    mods = dict(model.named_modules())
    if model_name == "llava":
        mod = mods.get("multi_modal_projector") or mods.get("model.multi_modal_projector")
    elif model_name == "idefics2":
        mod = mods.get("model.connector") or mods.get("connector")
    else:
        mod = mods.get("visual.merger") or mods.get("model.visual.merger")
    if mod is None:
        raise RuntimeError(f"connector not found for {model_name}")
    return mod


def build_image_index(image_dir: str) -> dict[str, str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    index = {}
    for p in Path(image_dir).rglob("*"):
        if p.suffix.lower() in exts:
            if p.stem not in index:
                index[p.stem] = str(p)
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, vlm = load_vlm(args.model_name, args.img_size, device)
    connector = resolve_connector(vlm, args.model_name)
    key_to_path = build_image_index(args.image_dir)
    if not key_to_path:
        raise FileNotFoundError(f"no images found in {args.image_dir}")

    dataset = PostVectorDataset(args.vec_dir, limit=args.limit, cache_shards=args.cache_shards)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))

    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    if args.max_batches:
        loader = itertools.islice(loader, args.max_batches)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"perturb_{args.model_name}_{args.mode}_{args.level}.csv")

    # Resume: load already-processed keys so we skip them
    done_keys: set[str] = set()
    if args.resume and os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        done_keys = set(existing["key"].astype(str).tolist())
        print(f"Resuming: {len(done_keys)} keys already done, skipping them.")

    write_header = not (args.resume and os.path.exists(csv_path))
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")

    total_rows = 0
    missing = 0

    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc="Perturb eval"):
                sample_ids = list(batch["sample_id"])
                keys = [sid.split(":", 2)[2] for sid in sample_ids]
                paths = [key_to_path.get(k, "") for k in keys]

                kept = [i for i, p in enumerate(paths) if p and keys[i] not in done_keys]
                missing += sum(1 for p in paths if not p)
                if not kept:
                    continue

                kept_ids   = [sample_ids[i] for i in kept]
                kept_keys  = [keys[i] for i in kept]
                kept_paths = [paths[i] for i in kept]
                post_gt    = batch["post"][kept].to(device)

                vlm_inputs = build_inputs(processor, kept_paths, args.prompt, args.model_name, args.img_size, device)

                # --- baseline: original caption (no injection) ---
                with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                    out_orig = vlm.generate(**vlm_inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                caps_orig = decode_outputs(processor, out_orig, vlm_inputs["input_ids"], args.model_name)

                if args.mode == "none":
                    batch_rows = [
                        {"sample_id": sid, "key": key, "image_path": path,
                         "caption_original": c_orig, "caption_perturbed": c_orig,
                         "mode": args.mode, "level": args.level, "seed": args.seed}
                        for sid, key, path, c_orig in zip(kept_ids, kept_keys, kept_paths, caps_orig)
                    ]
                else:
                    # --- perturbed caption ---
                    post_perturbed = perturb(post_gt, args.mode, args.level, kept_ids, args.seed)

                    fired = [False]

                    def replace_out(_mod, _inp, hook_out):
                        out_t = hook_out if isinstance(hook_out, torch.Tensor) else next(x for x in hook_out if isinstance(x, torch.Tensor))
                        repl = post_perturbed.reshape_as(out_t).to(dtype=out_t.dtype)
                        fired[0] = True
                        if isinstance(hook_out, torch.Tensor):
                            return repl
                        out_list = list(hook_out)
                        out_list[[i for i, x in enumerate(out_list) if isinstance(x, torch.Tensor)][0]] = repl
                        return type(hook_out)(out_list)

                    h = connector.register_forward_hook(replace_out)
                    try:
                        with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                            out_perturbed = vlm.generate(**vlm_inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                    finally:
                        h.remove()

                    if not fired[0]:
                        raise RuntimeError("connector hook did not fire")

                    caps_perturbed = decode_outputs(processor, out_perturbed, vlm_inputs["input_ids"], args.model_name)
                    batch_rows = [
                        {"sample_id": sid, "key": key, "image_path": path,
                         "caption_original": c_orig, "caption_perturbed": c_pert,
                         "mode": args.mode, "level": args.level, "seed": args.seed}
                        for sid, key, path, c_orig, c_pert in zip(kept_ids, kept_keys, kept_paths, caps_orig, caps_perturbed)
                    ]

                # Write batch to CSV immediately
                df_batch = pd.DataFrame(batch_rows)
                df_batch.to_csv(csv_file, index=False, header=write_header)
                csv_file.flush()
                write_header = False
                total_rows += len(batch_rows)
    finally:
        csv_file.close()

    print(f"done | rows={total_rows} missing={missing} | saved {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--vec_dir",    required=True)
    p.add_argument("--image_dir",  required=True)
    p.add_argument("--out_dir",    default="./perturb_out")
    p.add_argument("--model_name", default="idefics2", choices=list(MODEL_IDS))
    p.add_argument("--mode",       default="mask", choices=["none", "mask", "lowrank", "orthogonal"])
    p.add_argument("--level",      type=float, default=0.5)
    p.add_argument("--seed",       type=int,   default=1337)
    p.add_argument("--prompt",     default="Describe this image in one detailed sentence, focusing on the main objects, attributes, and scene.")
    p.add_argument("--img_size",   type=int,   default=336)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--num_workers",type=int,   default=0)
    p.add_argument("--cache_shards",type=int,  default=1)
    p.add_argument("--limit",      type=int,   default=0)
    p.add_argument("--max_batches",type=int,   default=0)
    p.add_argument("--amp",        action="store_true")
    p.add_argument("--amp_dtype",  default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--resume",     action="store_true",
                   help="skip keys already in the output CSV and append new results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run(args)
