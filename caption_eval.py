import argparse
import glob
import hashlib
import itertools
import math
import os
import pickle
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


# -----------------------------
# Dataset
# -----------------------------

class ShardSample(TypedDict):
    embeddings: tuple[torch.Tensor, torch.Tensor]
    sample_id: str


def numeric_suffix(path: Path) -> int:
    match = re.search(r"_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else 10**18


class ShardPairVectorDataset(Dataset):
    """
    Reads paired shard files:
      - pre_vectors_*.pt
      - post_vectors_*.pt

    Each shard contains at least:
      {"keys": [...], "vecs": Tensor}
    """

    def __init__(
        self,
        vec_dir: str | None = None,
        pre_dir: str | None = None,
        post_dir: str | None = None,
        limit: int = 0,
        cache_shards: int = 1,
        cache_fp32: bool = False,
        strip_cls: bool = True,
    ) -> None:
        self.cache_shards = max(1, int(cache_shards))
        self.cache_fp32 = cache_fp32
        self.strip_cls = strip_cls

        if vec_dir:
            pre_root = Path(vec_dir)
            post_root = Path(vec_dir)
        else:
            if not pre_dir or not post_dir:
                raise ValueError("provide --vec_dir or both --pre_dir and --post_dir")
            pre_root = Path(pre_dir)
            post_root = Path(post_dir)

        self.pre_files = sorted(pre_root.glob("pre_vectors_*.pt"), key=numeric_suffix)
        self.post_files = sorted(post_root.glob("post_vectors_*.pt"), key=numeric_suffix)
        if not self.pre_files or not self.post_files:
            raise FileNotFoundError("No pre/post_vectors_*.pt found.")

        pre_map = {numeric_suffix(p): p for p in self.pre_files}
        post_map = {numeric_suffix(p): p for p in self.post_files}
        self.tags = sorted(set(pre_map.keys()) & set(post_map.keys()))
        if not self.tags:
            raise RuntimeError("No common shard tags between pre and post.")

        self.pre_map = pre_map
        self.post_map = post_map

        self.items: list[tuple[int, int, str]] = []
        seen = 0
        for tag in self.tags:
            pre_obj = torch.load(self.pre_map[tag], map_location="cpu", weights_only=True)
            post_obj = torch.load(self.post_map[tag], map_location="cpu", weights_only=True)
            keys_pre = pre_obj["keys"]
            keys_post = post_obj["keys"]

            if keys_pre != keys_post:
                raise RuntimeError(f"keys mismatch in tag={tag}")

            for row_idx, key in enumerate(keys_pre):
                self.items.append((tag, row_idx, str(key)))
                seen += 1
                if limit and seen >= limit:
                    break
            if limit and seen >= limit:
                break

        self.pre_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.post_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.key_cache: OrderedDict[int, list[str]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.items)

    def _maybe_strip_cls(self, vecs: torch.Tensor, name: str) -> torch.Tensor:
        token_len = vecs.shape[1]
        s = int(math.isqrt(token_len))
        if s * s == token_len:
            return vecs

        if self.strip_cls and token_len > 1:
            s1 = int(math.isqrt(token_len - 1))
            if s1 * s1 == (token_len - 1):
                print(f"[warn] {name} token len {token_len} -> strip CLS -> {token_len - 1}")
                return vecs[:, 1:, :]

        raise ValueError(
            f"{name} token len={token_len} is not square and not (square+1). "
            "Use --no_strip_cls only if your vectors are already aligned."
        )

    def _load_tag(self, tag: int) -> None:
        if tag in self.pre_cache and tag in self.post_cache and tag in self.key_cache:
            self.pre_cache.move_to_end(tag)
            self.post_cache.move_to_end(tag)
            self.key_cache.move_to_end(tag)
            return

        while len(self.pre_cache) >= self.cache_shards:
            old_tag, _ = self.pre_cache.popitem(last=False)
            self.post_cache.pop(old_tag, None)
            self.key_cache.pop(old_tag, None)

        pre_obj = torch.load(self.pre_map[tag], map_location="cpu", weights_only=True)
        post_obj = torch.load(self.post_map[tag], map_location="cpu", weights_only=True)

        keys = [str(k) for k in pre_obj["keys"]]
        pre_vecs = pre_obj["vecs"]
        post_vecs = post_obj["vecs"]

        if self.cache_fp32:
            pre_vecs = pre_vecs.float()
            post_vecs = post_vecs.float()

        if pre_vecs.dim() == 2:
            pre_vecs = pre_vecs.unsqueeze(1)
        if post_vecs.dim() == 2:
            post_vecs = post_vecs.unsqueeze(1)

        pre_vecs = self._maybe_strip_cls(pre_vecs, "pre")
        post_vecs = self._maybe_strip_cls(post_vecs, "post")

        self.key_cache[tag] = keys
        self.pre_cache[tag] = pre_vecs
        self.post_cache[tag] = post_vecs

    def __getitem__(self, index: int) -> ShardSample:
        tag, row_idx, key = self.items[index]
        self._load_tag(tag)

        pre = self.pre_cache[tag][row_idx]
        post = self.post_cache[tag][row_idx]
        sample_id = f"{tag}:{row_idx}:{key}"
        return {"embeddings": (pre, post), "sample_id": sample_id}


# -----------------------------
# Reconstruction models
# -----------------------------

class EmbeddingTransformer(nn.Module):
    def __init__(self, input_dim=4096, output_dim=1024, hidden_dim=2048, num_layers=3):
        super().__init__()
        hidden_dim = min(hidden_dim, max(input_dim, output_dim))

        if num_layers <= 3:
            self.model = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            layers: list[nn.Module] = [
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ]
            for _ in range(num_layers - 3):
                layers.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                        nn.Dropout(0.1),
                    ]
                )
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LargerEmbeddingTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 4096,
        output_dim: int = 1024,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        seq_length: int = 576,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        x = self.output_proj(x)
        x = self.final_norm(x)
        return x


class SeqEmbeddingTransformer(nn.Module):
    def __init__(
        self,
        input_dim=4096,
        input_seq_len=64,
        output_seq_len=576,
        output_dim=1152,
        hidden_dim=2048,
        num_layers=8,
        num_heads=16,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.pos_encoder = nn.Parameter(torch.randn(1, input_seq_len, hidden_dim))
        self.sequence_processor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.length_adjust = nn.Sequential(
            nn.Linear(input_seq_len, output_seq_len * 2),
            nn.GELU(),
            nn.Linear(output_seq_len * 2, output_seq_len),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = x + self.pos_encoder
        x = self.sequence_processor(x)
        x = x.transpose(1, 2)
        x = self.length_adjust(x)
        x = x.transpose(1, 2)
        return self.output_projection(x)


# -----------------------------
# Utility helpers
# -----------------------------

def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.embed_model == "llava":
        if args.model_type == "mlp":
            model = EmbeddingTransformer(
                input_dim=4096,
                output_dim=1024,
                hidden_dim=args.hidden_size,
                num_layers=args.num_layers,
            )
        else:
            model = LargerEmbeddingTransformer(
                input_dim=4096,
                output_dim=1024,
                hidden_dim=args.hidden_size,
                num_layers=args.num_layers,
                seq_length=args.seq_length,
            )
    elif args.embed_model == "idefics2":
        model = SeqEmbeddingTransformer(
            input_dim=4096,
            input_seq_len=64,
            output_seq_len=576,
            output_dim=1152,
            hidden_dim=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    elif args.embed_model == "qwen2.5vl":
        model = SeqEmbeddingTransformer(
            input_dim=3584,
            input_seq_len=144,
            output_seq_len=576,
            output_dim=1280,
            hidden_dim=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    else:
        raise ValueError(f"unsupported embed_model: {args.embed_model}")
    return model.to(device)


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def resolve_model_path(args: argparse.Namespace) -> str:
    if args.model_path and os.path.exists(args.model_path):
        return args.model_path
    candidates = [
        os.path.join(args.model_dir, "best_embedding_transformer.pth"),
        os.path.join(args.model_dir, "best_model.pt"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("no checkpoint found; provide --model_path or valid --model_dir")


def get_autocast_dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16


def standardize_embeddings(embeddings: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (embeddings - mean) / (std + 1e-8)


def maybe_load_recon_stats(path: str) -> dict[str, torch.Tensor] | None:
    if not path:
        return None
    with open(path, "rb") as f:
        stats = pickle.load(f)
    required = {"mean_pre", "std_pre", "mean_post", "std_post"}
    if not required.issubset(set(stats.keys())):
        raise ValueError(f"stats file must contain keys {required}")
    return stats


def extract_key_from_sample_id(sample_id: str) -> str:
    parts = sample_id.split(":", 2)
    return parts[2] if len(parts) == 3 else sample_id


def build_image_index(image_dir: str) -> dict[str, str]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    key_to_path: dict[str, str] = {}
    for pattern in patterns:
        for path in glob.glob(os.path.join(image_dir, "**", pattern), recursive=True):
            key = Path(path).stem
            if key in key_to_path and key_to_path[key] != path:
                print(f"[warn] duplicate image key {key}; keeping first path")
                continue
            key_to_path[key] = path
    return key_to_path


# -----------------------------
# VLM loading + prompting
# -----------------------------

def resolve_vlm_model_id(args: argparse.Namespace) -> str:
    if args.vlm_model_id:
        return args.vlm_model_id
    if args.embed_model == "llava":
        return "llava-hf/llava-1.5-7b-hf"
    if args.embed_model == "idefics2":
        return "HuggingFaceM4/idefics2-8b"
    if args.embed_model == "qwen2.5vl":
        return "Qwen/Qwen2.5-VL-7B-Instruct"
    raise ValueError(f"no default VLM id for embed_model={args.embed_model}; set --vlm_model_id")


def load_vlm_for_caption(args: argparse.Namespace, device: torch.device):
    model_id = resolve_vlm_model_id(args)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if args.embed_model == "idefics2" and hasattr(processor, "image_processor"):
        processor.image_processor.size = {"longest_edge": args.img_size, "shortest_edge": args.img_size}
        if hasattr(processor.image_processor, "do_image_splitting"):
            processor.image_processor.do_image_splitting = False

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if args.embed_model == "llava":
        model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    elif args.embed_model == "idefics2":
        model = Idefics2ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        )
    elif args.embed_model == "qwen2.5vl":
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        )
    else:
        raise ValueError("caption compare supports only llava / idefics2 / qwen2.5vl")

    model = model.to(device)
    model.eval()
    return processor, model


def build_vlm_inputs(
    processor,
    image_paths: list[str],
    prompt: str,
    device: torch.device,
    embed_model: str,
    img_size: int,
) -> dict[str, torch.Tensor]:
    images = [load_image(p) for p in image_paths]
    if img_size and img_size > 0:
        images = [im.resize((img_size, img_size)) for im in images]

    if embed_model == "llava":
        template = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompts = [processor.apply_chat_template(template, add_generation_prompt=True) for _ in images]
        inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt")
        return {k: v.to(device) for k, v in inputs.items()}

    if embed_model == "idefics2":
        template = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompts = [processor.apply_chat_template(template, add_generation_prompt=True) for _ in images]
        inputs = processor(images=[[im] for im in images], text=prompts, padding=True, return_tensors="pt")
        return {k: v.to(device) for k, v in inputs.items()}

    if embed_model == "qwen2.5vl":
        from qwen_vl_utils import process_vision_info

        templates = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": im},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for im in images
        ]
        texts = [processor.apply_chat_template(t, tokenize=False, add_generation_prompt=True) for t in templates]
        image_inputs, video_inputs = process_vision_info(templates)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in inputs.items()}

    raise ValueError(f"unsupported embed_model={embed_model}")


def get_prompt_token_len(processor, input_ids: torch.Tensor, embed_model: str) -> list[int]:
    # For these models left padding is usually not used in this workflow.
    if embed_model == "qwen2.5vl":
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is None:
            return [input_ids.shape[1]] * input_ids.shape[0]
        return [(row != pad_id).sum().item() for row in input_ids]
    return [input_ids.shape[1]] * input_ids.shape[0]


def decode_outputs(processor, out_ids: torch.Tensor, input_ids: torch.Tensor, embed_model: str) -> list[str]:
    # Keep decode behavior identical to extraction so caption text is comparable.
    # For qwen we may have per-sample prompt lengths depending on padding.
    if embed_model == "qwen2.5vl":
        prompt_lens = get_prompt_token_len(processor, input_ids, embed_model)
        texts: list[str] = []
        for i, plen in enumerate(prompt_lens):
            gen_ids = out_ids[i, plen:]
            txt = processor.batch_decode(
                gen_ids.unsqueeze(0),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            texts.append(txt.strip())
        return texts

    gen_only = out_ids[:, input_ids.shape[1] :]
    captions = processor.batch_decode(
        gen_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [c.strip() for c in captions]


# -----------------------------
# Connector hooks
# -----------------------------

def resolve_connector_module(vlm_model: nn.Module, embed_model: str) -> nn.Module:
    mods = dict(vlm_model.named_modules())
    if embed_model == "llava":
        mod = mods.get("multi_modal_projector") or mods.get("model.multi_modal_projector")
    elif embed_model == "idefics2":
        mod = mods.get("model.connector") or mods.get("connector")
    elif embed_model == "qwen2.5vl":
        mod = mods.get("visual.merger") or mods.get("model.visual.merger")
    else:
        mod = None
    if mod is None:
        raise RuntimeError(f"failed to locate connector module for embed_model={embed_model}")
    return mod


def reshape_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    out = src

    # ref may be [B,T,D] or [B*T,D]
    if ref.dim() == 2 and out.dim() == 3:
        out = out.reshape(-1, out.shape[-1])
    elif ref.dim() == 3 and out.dim() == 2:
        out = out.reshape(ref.shape[0], ref.shape[1], out.shape[-1])

    if ref.dim() == 3 and out.dim() == 3 and out.shape[1] != ref.shape[1]:
        if ref.shape[1] == out.shape[1] + 1 and ref.shape[2] == out.shape[2]:
            out = torch.cat([ref[:, :1, :], out], dim=1)
        elif out.shape[1] == ref.shape[1] + 1 and ref.shape[2] == out.shape[2]:
            out = out[:, 1:, :]

    if ref.dim() == 3 and out.dim() == 3 and out.shape[0] != ref.shape[0]:
        if ref.shape[0] % out.shape[0] == 0:
            rep = ref.shape[0] // out.shape[0]
            out = out.repeat_interleave(rep, dim=0)
        elif out.shape[0] % ref.shape[0] == 0:
            group = out.shape[0] // ref.shape[0]
            out = out.reshape(ref.shape[0], group, out.shape[1], out.shape[2]).mean(dim=1)

    if ref.dim() == 3 and out.dim() == 3 and out.shape[1] != ref.shape[1]:
        out = torch.nn.functional.interpolate(
            out.transpose(1, 2),
            size=ref.shape[1],
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    if out.shape != ref.shape:
        raise RuntimeError(f"shape mismatch: expected {tuple(ref.shape)}, got {tuple(out.shape)}")
    return out


def tensorize_output(obj) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)) and len(obj) > 0:
        for x in obj:
            if isinstance(x, torch.Tensor):
                return x
    raise RuntimeError("connector output is not a tensor / tuple containing tensor")


def replace_first_tensor(obj, new_tensor: torch.Tensor):
    if isinstance(obj, torch.Tensor):
        return new_tensor
    if isinstance(obj, tuple):
        out = list(obj)
        for i, x in enumerate(out):
            if isinstance(x, torch.Tensor):
                out[i] = new_tensor
                return tuple(out)
    if isinstance(obj, list):
        out = list(obj)
        for i, x in enumerate(out):
            if isinstance(x, torch.Tensor):
                out[i] = new_tensor
                return out
    raise RuntimeError("connector output cannot be rewritten with replacement tensor")


def per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    diff = (pred.float() - target.float()) ** 2
    if diff.dim() == 3:
        return diff.mean(dim=(1, 2)).detach().cpu().numpy()
    if diff.dim() == 2:
        return diff.mean(dim=1).detach().cpu().numpy()
    raise RuntimeError(f"unexpected tensor dim for mse: {diff.dim()}")


def per_sample_cosine(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    pred_f = pred.float()
    target_f = target.float()
    if pred_f.dim() == 3:
        cos = torch.nn.functional.cosine_similarity(pred_f, target_f, dim=2)
        return cos.mean(dim=1).detach().cpu().numpy()
    if pred_f.dim() == 2:
        cos = torch.nn.functional.cosine_similarity(pred_f, target_f, dim=1)
        return cos.detach().cpu().numpy()
    raise RuntimeError(f"unexpected tensor dim for cosine: {pred_f.dim()}")


def stable_seed(base_seed: int, text: str) -> int:
    payload = f"{base_seed}:{text}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def should_apply_perturbation(args: argparse.Namespace, path_mode: str) -> bool:
    if args.perturbation_mode == "none":
        return False
    if args.perturbation_paths == "all":
        return True
    return args.perturbation_paths == path_mode


def _effective_rank_from_level(level: float, max_rank: int) -> int:
    if max_rank <= 0:
        return 0
    if level <= 0:
        return 1
    if level <= 1.0:
        return max(1, min(max_rank, int(round(level * max_rank))))
    return max(1, min(max_rank, int(round(level))))


def get_orthogonal_basis(
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    cache: dict[tuple[int, str, torch.dtype, int], torch.Tensor],
) -> torch.Tensor:
    key = (dim, str(device), dtype, int(seed))
    if key in cache:
        return cache[key]

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    a = torch.randn((dim, dim), generator=gen, device=device, dtype=torch.float32)
    q, r = torch.linalg.qr(a)
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs
    q = q.to(dtype=dtype)
    cache[key] = q
    return q


def apply_post_perturbation(
    post_tokens: torch.Tensor,
    mode: str,
    level: float,
    sample_ids: list[str],
    base_seed: int,
    orth_cache: dict[tuple[int, str, torch.dtype, int], torch.Tensor],
) -> torch.Tensor:
    if mode == "none":
        return post_tokens

    squeeze_back = False
    work = post_tokens
    if work.dim() == 2:
        work = work.unsqueeze(0)
        squeeze_back = True
    if work.dim() != 3:
        raise RuntimeError(f"unsupported post token dim for perturbation: {work.dim()}")

    batch = work.shape[0]
    if len(sample_ids) != batch:
        raise RuntimeError(f"sample_ids length mismatch: expected {batch}, got {len(sample_ids)}")

    out = work.clone()
    mode = mode.lower()

    if mode == "mask":
        frac = float(max(0.0, min(1.0, level)))
        for b in range(batch):
            sample_seed = stable_seed(base_seed, sample_ids[b])
            gen = torch.Generator(device=out.device)
            gen.manual_seed(sample_seed)
            token_count = out.shape[1]
            mask = torch.rand((token_count,), generator=gen, device=out.device) < frac
            out[b, mask, :] = 0

    elif mode == "lowrank":
        for b in range(batch):
            x = out[b].float()
            mean = x.mean(dim=0, keepdim=True)
            x_centered = x - mean
            u, s, vh = torch.linalg.svd(x_centered, full_matrices=False)
            rank = _effective_rank_from_level(float(level), s.shape[0])
            approx = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
            out[b] = (approx + mean).to(dtype=out.dtype)

    elif mode == "orthogonal":
        alpha = float(max(0.0, min(1.0, level)))
        q = get_orthogonal_basis(
            dim=out.shape[-1],
            device=out.device,
            dtype=out.dtype,
            seed=int(base_seed),
            cache=orth_cache,
        )
        rotated = out @ q
        out = (1.0 - alpha) * out + alpha * rotated

    else:
        raise ValueError(f"unknown perturbation mode: {mode}")

    return out.squeeze(0) if squeeze_back else out


# -----------------------------
# Main caption compare
# -----------------------------

def run_caption_compare(
    recon_model: nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if not args.image_dir:
        raise ValueError("--image_dir is required")

    processor, vlm_model = load_vlm_for_caption(args, device)
    connector = resolve_connector_module(vlm_model, args.embed_model)

    key_to_path = build_image_index(args.image_dir)
    if not key_to_path:
        raise FileNotFoundError(f"no images found under {args.image_dir}")

    recon_model.eval()
    recon_stats = maybe_load_recon_stats(args.recon_stats_path)
    if args.normalize_recon_input and recon_stats is None:
        raise ValueError("--normalize_recon_input requires --recon_stats_path")

    if recon_stats is not None:
        recon_stats = {k: v.to(device) for k, v in recon_stats.items()}

    use_amp = bool(args.amp and device.type == "cuda")
    autocast_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    rows: list[dict] = []
    orth_cache: dict[tuple[int, str, torch.dtype, int], torch.Tensor] = {}
    missing_images = 0
    processed = 0
    if args.max_items != 0:
        print(f"[info] max_items is set to {args.max_items}, will process at most {args.max_items} batches.")
        # Keep this lazy: list(loader) would materialize the full dataset in RAM.
        loader = itertools.islice(loader, args.max_items)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Caption Compare"):
            sample_ids = list(batch["sample_id"])
            post_emb_cpu = batch["embeddings"][1]
            keys = [extract_key_from_sample_id(sid) for sid in sample_ids]
            image_paths = [key_to_path.get(k, "") for k in keys]

            kept_idx = [i for i, p in enumerate(image_paths) if p]
            if not kept_idx:
                missing_images += len(sample_ids)
                continue

            missing_images += len(sample_ids) - len(kept_idx)
            kept_sample_ids = [sample_ids[i] for i in kept_idx]
            kept_keys = [keys[i] for i in kept_idx]
            kept_paths = [image_paths[i] for i in kept_idx]
            post_gt = post_emb_cpu[kept_idx].to(device, non_blocking=True)

            if args.normalize_recon_input:
                post_in = standardize_embeddings(
                    post_gt,
                    recon_stats["mean_post"],
                    recon_stats["std_post"],
                )
            else:
                post_in = post_gt

            with autocast(device_type=device.type, enabled=use_amp, dtype=autocast_dtype):
                pre_hat = recon_model(post_in)

            if args.normalize_recon_input:
                pre_hat = pre_hat * (recon_stats["std_pre"] + 1e-8) + recon_stats["mean_pre"]

            vlm_inputs = build_vlm_inputs(
                processor=processor,
                image_paths=kept_paths,
                prompt=args.caption_prompt,
                device=device,
                embed_model=args.embed_model,
                img_size=args.img_size,
            )

            if args.compare_post_vs_recon and args.compare_only_two_paths:
                caps_original = [""] * len(kept_paths)
            else:
                with autocast(device_type=device.type, enabled=use_amp, dtype=autocast_dtype):
                    out_original = vlm_model.generate(
                        **vlm_inputs,
                        max_new_tokens=args.caption_max_new_tokens,
                        do_sample=False,
                        use_cache=False,
                    )
                caps_original = decode_outputs(processor, out_original, vlm_inputs["input_ids"], args.embed_model)

            def run_injected(path_mode: str) -> tuple[list[str], np.ndarray, np.ndarray]:
                hook_state = {"pre_fired": False, "fwd_fired": False, "captured_post": None}
                apply_perturb = should_apply_perturbation(args, path_mode)

                def replace_connector_input(_module, hook_in):
                    x = hook_in[0]
                    repl = reshape_like(pre_hat, x).to(dtype=x.dtype, device=x.device)
                    hook_state["pre_fired"] = True
                    return (repl,) + tuple(hook_in[1:])

                def replace_connector_output(_module, _hook_in, hook_out):
                    out_tensor = tensorize_output(hook_out)
                    if path_mode == "post_direct":
                        base_post = post_gt
                    elif path_mode == "recon":
                        base_post = reshape_like(out_tensor, post_gt)
                    else:
                        raise ValueError(f"unknown path mode: {path_mode}")

                    if apply_perturb:
                        post_for_decode = apply_post_perturbation(
                            post_tokens=base_post,
                            mode=args.perturbation_mode,
                            level=args.perturbation_level,
                            sample_ids=kept_sample_ids,
                            base_seed=args.perturbation_seed,
                            orth_cache=orth_cache,
                        )
                    else:
                        post_for_decode = base_post

                    repl = reshape_like(post_for_decode, out_tensor).to(
                        dtype=out_tensor.dtype,
                        device=out_tensor.device,
                    )
                    hook_state["captured_post"] = reshape_like(repl.detach(), post_gt)
                    hook_state["fwd_fired"] = True
                    return replace_first_tensor(hook_out, repl)

                def capture_connector_output(_module, _hook_in, hook_out):
                    out_tensor = tensorize_output(hook_out)
                    hook_state["captured_post"] = out_tensor.detach()
                    hook_state["fwd_fired"] = True
                    return hook_out

                if path_mode == "recon":
                    h1 = connector.register_forward_pre_hook(replace_connector_input)
                    if apply_perturb:
                        h2 = connector.register_forward_hook(replace_connector_output)
                    else:
                        h2 = connector.register_forward_hook(capture_connector_output)
                elif path_mode == "post_direct":
                    h1 = None
                    h2 = connector.register_forward_hook(replace_connector_output)
                else:
                    raise ValueError(f"unknown path mode: {path_mode}")

                try:
                    with autocast(device_type=device.type, enabled=use_amp, dtype=autocast_dtype):
                        out_ids = vlm_model.generate(
                            **vlm_inputs,
                            max_new_tokens=args.caption_max_new_tokens,
                            do_sample=False,
                            use_cache=False,
                        )
                finally:
                    if h1 is not None:
                        h1.remove()
                    h2.remove()

                if path_mode == "recon" and not hook_state["pre_fired"]:
                    raise RuntimeError("connector replacement pre-hook did not fire")
                if not hook_state["fwd_fired"]:
                    raise RuntimeError("connector forward hook did not fire")
                if hook_state["captured_post"] is None:
                    raise RuntimeError("failed to capture connector output")

                caps = decode_outputs(processor, out_ids, vlm_inputs["input_ids"], args.embed_model)
                reproj_post = reshape_like(hook_state["captured_post"], post_gt)
                mse_vals_local = per_sample_mse(reproj_post, post_gt)
                cos_vals_local = per_sample_cosine(reproj_post, post_gt)
                return caps, mse_vals_local, cos_vals_local

            if args.compare_post_vs_recon:
                caps_post, mse_post, cos_post = run_injected("post_direct")
                caps_recon, mse_recon, cos_recon = run_injected("recon")

                for sid, key, path, c0, c_post, c_recon, mse_p, cos_p, mse_r, cos_r in zip(
                    kept_sample_ids,
                    kept_keys,
                    kept_paths,
                    caps_original,
                    caps_post,
                    caps_recon,
                    mse_post,
                    cos_post,
                    mse_recon,
                    cos_recon,
                ):
                    rows.append(
                        {
                            "sample_id": sid,
                            "key": key,
                            "image_path": path,
                            "caption_original": c0,
                            "caption_post_direct": c_post,
                            "caption_recon": c_recon,
                            "reproj_mse_post_direct": float(mse_p),
                            "reproj_cosine_post_direct": float(cos_p),
                            "reproj_mse_recon": float(mse_r),
                            "reproj_cosine_recon": float(cos_r),
                            # Keep legacy columns for compatibility with existing scripts.
                            "reproj_mse": float(mse_r),
                            "reproj_cosine": float(cos_r),
                            "perturbation_mode": args.perturbation_mode,
                            "perturbation_level": float(args.perturbation_level),
                            "perturbation_seed": int(args.perturbation_seed),
                            "perturbation_paths": args.perturbation_paths,
                            "perturbation_run_id": args.perturbation_run_id,
                        }
                    )
            else:
                caps_recon, mse_vals, cos_vals = run_injected("recon")

                for sid, key, path, c0, c1, mse_v, cos_v in zip(
                    kept_sample_ids,
                    kept_keys,
                    kept_paths,
                    caps_original,
                    caps_recon,
                    mse_vals,
                    cos_vals,
                ):
                    rows.append(
                        {
                            "sample_id": sid,
                            "key": key,
                            "image_path": path,
                            "caption_original": c0,
                            "caption_recon": c1,
                            "reproj_mse": float(mse_v),
                            "reproj_cosine": float(cos_v),
                            "perturbation_mode": args.perturbation_mode,
                            "perturbation_level": float(args.perturbation_level),
                            "perturbation_seed": int(args.perturbation_seed),
                            "perturbation_paths": args.perturbation_paths,
                            "perturbation_run_id": args.perturbation_run_id,
                        }
                    )

            processed += len(kept_keys)
            if args.caption_limit > 0 and processed >= args.caption_limit:
                break

    if args.with_bertscore and args.compare_post_vs_recon and args.compare_only_two_paths:
        print("[warn] --with_bertscore ignored in compare-only-two-paths mode")

    if args.with_bertscore and rows and not (args.compare_post_vs_recon and args.compare_only_two_paths):
        from bert_score import score as bertscore_score

        cands = [r["caption_recon"] for r in rows]
        refs = [r["caption_original"] for r in rows]
        p, r, f1 = bertscore_score(
            cands=cands,
            refs=refs,
            lang=args.bertscore_lang,
            model_type=(args.bertscore_model_type or None),
            batch_size=args.bertscore_batch_size,
            rescale_with_baseline=args.bertscore_rescale,
            verbose=True,
        )
        p_np = p.detach().cpu().numpy()
        r_np = r.detach().cpu().numpy()
        f1_np = f1.detach().cpu().numpy()
        for i, row in enumerate(rows):
            row["bertscore_precision"] = float(p_np[i])
            row["bertscore_recall"] = float(r_np[i])
            row["bertscore_f1"] = float(f1_np[i])
        print(
            f"bertscore summary | P={float(np.mean(p_np)):.4f} "
            f"R={float(np.mean(r_np)):.4f} F1={float(np.mean(f1_np)):.4f}"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, args.caption_csv_name)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"done | rows={len(rows)} missing_images={missing_images}")
    print(f"saved: {out_csv}")

    if len(df) > 0:
        print("reproj summary")
        print(df[["reproj_mse", "reproj_cosine"]].describe())


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--vec_dir", type=str, default="")
    parser.add_argument("--pre_dir", type=str, default="")
    parser.add_argument("--post_dir", type=str, default="")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./caption_compare_out")

    parser.add_argument("--embed_model", type=str, default="llava", choices=["llava", "idefics2", "qwen2.5vl"])
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "transformer"])
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--seq_length", type=int, default=576)
    parser.add_argument("--hidden_size", type=int, default=2048)

    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--vlm_model_id", type=str, default="")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cache_shards", type=int, default=1)
    parser.add_argument("--cache_fp32", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no_strip_cls", action="store_true")
    parser.add_argument("--max_items", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--normalize_recon_input", action="store_true")
    parser.add_argument("--recon_stats_path", type=str, default="")
    parser.add_argument("--compare_post_vs_recon", action="store_true")
    parser.add_argument("--compare_only_two_paths", action="store_true")

    parser.add_argument("--caption_prompt", type=str, default="Describe this image in one detailed sentence, focusing on the main objects, attributes, and scene.")
    parser.add_argument("--caption_max_new_tokens", type=int, default=128)
    parser.add_argument("--caption_csv_name", type=str, default="caption_original_vs_recon.csv")
    parser.add_argument("--caption_limit", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=336)

    parser.add_argument(
        "--perturbation_mode",
        type=str,
        default="none",
        choices=["none", "mask", "lowrank", "orthogonal"],
    )
    parser.add_argument("--perturbation_level", type=float, default=0.5)
    parser.add_argument("--perturbation_seed", type=int, default=1337)
    parser.add_argument(
        "--perturbation_paths",
        type=str,
        default="all",
        choices=["all", "post_direct", "recon"],
    )
    parser.add_argument("--perturbation_run_id", type=str, default="")
    
    parser.add_argument("--with_bertscore", action="store_true")
    parser.add_argument("--bertscore_lang", type=str, default="en")
    parser.add_argument("--bertscore_model_type", type=str, default="")
    parser.add_argument("--bertscore_batch_size", type=int, default=16)
    parser.add_argument("--bertscore_rescale", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.compare_only_two_paths and not args.compare_post_vs_recon:
        raise ValueError("--compare_only_two_paths requires --compare_post_vs_recon")

    if not args.vec_dir and (not args.pre_dir or not args.post_dir):
        raise ValueError("please provide --vec_dir or both --pre_dir and --post_dir")
    if not args.model_dir:
        args.model_dir = args.out_dir

    if args.perturbation_mode == "none" and args.perturbation_paths != "all":
        print("[warn] --perturbation_paths has no effect when --perturbation_mode=none")

    if args.perturbation_mode != "none":
        print(
            "[info] perturbation enabled | "
            f"mode={args.perturbation_mode} level={args.perturbation_level} "
            f"paths={args.perturbation_paths} seed={args.perturbation_seed}"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ShardPairVectorDataset(
        vec_dir=args.vec_dir or None,
        pre_dir=args.pre_dir or None,
        post_dir=args.post_dir or None,
        limit=args.limit,
        cache_shards=args.cache_shards,
        cache_fp32=args.cache_fp32,
        strip_cls=not args.no_strip_cls,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    recon_model = build_model(args, device)
    model_path = resolve_model_path(args)
    print("loading reconstruction model from", model_path)
    recon_model.load_state_dict(load_state_dict(model_path), strict=True)
    recon_model.eval()

    run_caption_compare(recon_model, loader, args, device)


if __name__ == "__main__":
    main()
