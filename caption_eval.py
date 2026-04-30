import argparse
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
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vlm_utils import (
    build_image_index,
    build_vlm_inputs,
    decode_outputs,
    load_vlm,
    numeric_suffix,
    resolve_connector,
)



class ShardSample(TypedDict):
    embeddings: tuple[torch.Tensor, torch.Tensor]
    sample_id: str




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



class EmbeddingTransformer(nn.Module):
    def __init__(self, input_dim=4096, output_dim=1024, hidden_dim=2048, num_layers=3):
        super().__init__()
        hidden_dim = min(hidden_dim, max(input_dim, output_dim))

        layers: list[nn.Module] = [
            nn.Linear(input_dim, input_dim), nn.LayerNorm(input_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
        ]
        for _ in range(num_layers - 3):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1)]
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
    elif args.embed_model == "qwen3.5":
        model = SeqEmbeddingTransformer(
            input_dim=4096,
            input_seq_len=100,
            output_seq_len=400,
            output_dim=1152,
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






def run_caption_compare(
    recon_model: nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if not args.image_dir:
        raise ValueError("--image_dir is required")

    processor, vlm_model = load_vlm(args.embed_model, args.img_size, device)
    connector = resolve_connector(vlm_model, args.embed_model)

    key_to_path = build_image_index(args.image_dir)
    if not key_to_path:
        raise FileNotFoundError(f"no images found under {args.image_dir}")

    # Load CLIP for inline CLIPScore
    from transformers import CLIPModel, AutoProcessor as _AutoProcessor
    clip_processor = _AutoProcessor.from_pretrained(args.clip_model_id)
    clip_model = CLIPModel.from_pretrained(args.clip_model_id, use_safetensors=True).to(device).eval()

    @torch.no_grad()
    def _clipscore(image_paths, captions):
        scores = [float("nan")] * len(image_paths)
        valid = [(i, p, c) for i, (p, c) in enumerate(zip(image_paths, captions)) if p and c]
        if not valid:
            return scores
        idxs, paths, caps = zip(*valid)
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = clip_processor(
            text=list(caps), images=images,
            padding=True, truncation=True, return_tensors="pt",
        ).to(device)
        out = clip_model(**inputs)
        img_e = F.normalize(out.image_embeds, dim=-1)
        txt_e = F.normalize(out.text_embeds, dim=-1)
        vals = torch.clamp(100.0 * (img_e * txt_e).sum(dim=-1), min=0.0).cpu().tolist()
        for i, v in zip(idxs, vals):
            scores[i] = v
        return scores

    recon_model.eval()
    recon_stats = maybe_load_recon_stats(args.recon_stats_path)
    if args.normalize_recon_input and recon_stats is None:
        raise ValueError("--normalize_recon_input requires --recon_stats_path")

    if recon_stats is not None:
        recon_stats = {k: v.to(device) for k, v in recon_stats.items()}

    use_amp = bool(args.amp and device.type == "cuda")
    autocast_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, args.caption_csv_name)

    # Resume: load already-done keys
    done_keys: set[str] = set()
    if os.path.exists(out_csv):
        try:
            done_keys = set(pd.read_csv(out_csv)["key"].astype(str).tolist())
            print(f"[resume] found {len(done_keys)} already-done keys in {out_csv}")
        except Exception:
            pass

    csv_file = open(out_csv, "a", newline="", encoding="utf-8")
    write_header = (len(done_keys) == 0)

    missing_images = 0
    processed = 0
    if args.max_items != 0:
        print(f"[info] max_items is set to {args.max_items}, will process at most {args.max_items} batches.")
        loader = itertools.islice(loader, args.max_items)

    tqdm_total = None
    if args.caption_limit > 0:
        tqdm_total = math.ceil(args.caption_limit / args.batch_size)
    elif args.max_items != 0:
        tqdm_total = args.max_items

    with torch.no_grad():
        for batch in tqdm(loader, desc="Caption Compare", total=tqdm_total):
            sample_ids = list(batch["sample_id"])
            post_emb_cpu = batch["embeddings"][1]
            keys = [extract_key_from_sample_id(sid) for sid in sample_ids]
            image_paths = [key_to_path.get(k, "") for k in keys]

            kept_idx = [i for i, p in enumerate(image_paths) if p and keys[i] not in done_keys]
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
                model_name=args.embed_model,
                img_size=args.img_size,
                device=device,
            )

            # GT captions from .txt files (image vs GT = reference CLIPScore)
            gt_caps = []
            for p in kept_paths:
                txt = Path(p).with_suffix(".txt")
                gt_caps.append(txt.read_text(encoding="utf-8").strip() if txt.exists() else "")

            def run_injected(path_mode: str) -> tuple[list[str], np.ndarray, np.ndarray]:
                hook_state = {"pre_fired": False, "fwd_fired": False, "captured_post": None}

                def replace_connector_input(_module, hook_in):
                    x = hook_in[0]
                    repl = reshape_like(pre_hat, x).to(dtype=x.dtype, device=x.device)
                    hook_state["pre_fired"] = True
                    return (repl,) + tuple(hook_in[1:])

                def replace_connector_output(_module, _hook_in, hook_out):
                    out_tensor = tensorize_output(hook_out)
                    if path_mode == "post_direct":
                        post_for_decode = post_gt
                    elif path_mode == "recon":
                        post_for_decode = reshape_like(out_tensor, post_gt)
                    else:
                        raise ValueError(f"unknown path mode: {path_mode}")

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
                cs_gt    = _clipscore(kept_paths, gt_caps)
                cs_post  = _clipscore(kept_paths, caps_post)
                cs_recon = _clipscore(kept_paths, caps_recon)
                batch_rows = [
                    {
                        "sample_id": sid, "key": key,
                        "folder_id": Path(path).parent.name if path else "",
                        "model": args.embed_model,
                        "image_path": path,
                        "caption_post_direct": c_post,
                        "caption_recon": c_recon,
                        "reproj_mse_recon": float(mse_r),
                        "reproj_cosine_recon": float(cos_r),
                        "clipscore_gt": float(cs_g),
                        "clipscore_post_direct": float(cs_p),
                        "clipscore_recon": float(cs_r),
                        "clipscore_drop": float(cs_p - cs_r),
                    }
                    for sid, key, path, c_post, c_recon, mse_p, cos_p, mse_r, cos_r, cs_g, cs_p, cs_r in zip(
                        kept_sample_ids, kept_keys, kept_paths,
                        caps_post, caps_recon,
                        mse_post, cos_post, mse_recon, cos_recon,
                        cs_gt, cs_post, cs_recon,
                    )
                ]
            else:
                caps_recon, mse_vals, cos_vals = run_injected("recon")
                cs_gt    = _clipscore(kept_paths, gt_caps)
                cs_recon = _clipscore(kept_paths, caps_recon)
                batch_rows = [
                    {
                        "sample_id": sid, "key": key,
                        "folder_id": Path(path).parent.name if path else "",
                        "model": args.embed_model,
                        "image_path": path,
                        "caption_recon": c1,
                        "reproj_mse": float(mse_v),
                        "reproj_cosine": float(cos_v),
                        "clipscore_gt": float(cs_g),
                        "clipscore_recon": float(cs_r),
                        "clipscore_drop": float(cs_g - cs_r),
                    }
                    for sid, key, path, c1, mse_v, cos_v, cs_g, cs_r in zip(
                        kept_sample_ids, kept_keys, kept_paths,
                        caps_recon, mse_vals, cos_vals,
                        cs_gt, cs_recon,
                    )
                ]

            df_batch = pd.DataFrame(batch_rows)
            df_batch.to_csv(csv_file, index=False, header=write_header)
            csv_file.flush()
            write_header = False
            done_keys.update(kept_keys)

            processed += len(kept_keys)
            if args.caption_limit > 0 and processed >= args.caption_limit:
                break

    csv_file.close()
    print(f"done | processed={processed} missing_images={missing_images}")
    print(f"saved: {out_csv}")

    if args.with_bertscore:
        df_all = pd.read_csv(out_csv)
        post_col = "caption_post_direct" if "caption_post_direct" in df_all.columns else \
                   "caption_original" if "caption_original" in df_all.columns else None
        if post_col is None or "caption_recon" not in df_all.columns:
            print("[warn] --with_bertscore: missing caption columns, skipping")
            return
        from bert_score import score as bertscore_score

        def _bscore(hyps, refs):
            valid = [(i, h, r) for i, (h, r) in enumerate(zip(hyps, refs)) if h.strip() and r.strip()]
            f_out = [float("nan")] * len(hyps)
            if not valid:
                return f_out
            idxs, hs, rs = zip(*valid)
            _, _, f1 = bertscore_score(
                list(hs), list(rs),
                lang=args.bertscore_lang,
                model_type=(args.bertscore_model_type or None),
                batch_size=args.bertscore_batch_size,
                rescale_with_baseline=args.bertscore_rescale,
                verbose=False,
            )
            for i, v in zip(idxs, f1.tolist()):
                f_out[i] = v
            return f_out

        caps_post  = df_all[post_col].fillna("").astype(str).tolist()
        caps_recon = df_all["caption_recon"].fillna("").astype(str).tolist()
        gt_caps = [
            Path(p).with_suffix(".txt").read_text(encoding="utf-8").strip()
            if Path(p).with_suffix(".txt").exists() else ""
            for p in df_all["image_path"].fillna("").astype(str).tolist()
        ]

        print("BERTScore: recon vs post ...")
        df_all["bertscore_recon_vs_post_f1"] = _bscore(caps_recon, caps_post)
        print("BERTScore: post vs GT ...")
        df_all["bertscore_post_vs_gt_f1"]    = _bscore(caps_post, gt_caps)
        print("BERTScore: recon vs GT ...")
        df_all["bertscore_recon_vs_gt_f1"]   = _bscore(caps_recon, gt_caps)

        df_all.to_csv(out_csv, index=False)
        print(f"bertscore_recon_vs_post F1 : {np.nanmean(df_all['bertscore_recon_vs_post_f1']):.4f}")
        print(f"bertscore_post_vs_gt    F1 : {np.nanmean(df_all['bertscore_post_vs_gt_f1']):.4f}")
        print(f"bertscore_recon_vs_gt   F1 : {np.nanmean(df_all['bertscore_recon_vs_gt_f1']):.4f}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--vec_dir", type=str, default="")
    parser.add_argument("--pre_dir", type=str, default="")
    parser.add_argument("--post_dir", type=str, default="")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--shards", type=str, default="",
                        help="comma-separated shard folders to use, e.g. 00001,00002,00003. "
                             "If set, vec_dir and image_dir are treated as parent dirs.")
    parser.add_argument("--shard_limit", type=int, default=0,
                        help="max samples to take from each shard (0=all)")
    parser.add_argument("--out_dir", type=str, default="./caption_compare_out")
    
    parser.add_argument("--embed_model", type=str, default="llava", choices=["llava", "idefics2", "qwen2.5vl", "qwen3.5"])
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
    parser.add_argument("--caption_max_new_tokens", type=int, default=256)
    parser.add_argument("--caption_csv_name", type=str, default="caption_original_vs_recon.csv")
    parser.add_argument("--caption_limit", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=336)

    parser.add_argument("--clip_model_id", type=str, default="openai/clip-vit-base-patch16",
                        help="CLIP model for inline CLIPScore computation")
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

    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.shards:
        shard_names = [s.strip() for s in args.shards.split(",") if s.strip()]
        datasets = []
        for shard in shard_names:
            vec_dir = os.path.join(args.vec_dir, shard) if args.vec_dir else None
            pre_dir = os.path.join(args.pre_dir, shard) if args.pre_dir else None
            post_dir = os.path.join(args.post_dir, shard) if args.post_dir else None
            datasets.append(ShardPairVectorDataset(
                vec_dir=vec_dir,
                pre_dir=pre_dir,
                post_dir=post_dir,
                limit=args.shard_limit,
                cache_shards=args.cache_shards,
                cache_fp32=args.cache_fp32,
                strip_cls=not args.no_strip_cls,
            ))
        dataset = torch.utils.data.ConcatDataset(datasets)
        # build a merged image index from all shard image dirs
        args.image_dir = args.image_dir  # keep as parent; build_image_index recurses
    else:
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

    model_path = resolve_model_path(args)
    print("loading reconstruction model from", model_path)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "meta" in ckpt:
        meta = ckpt["meta"]
        args.model_type  = meta.get("model_type",  args.model_type)
        args.hidden_size = meta.get("hidden_size",  args.hidden_size)
        args.num_layers  = meta.get("num_layers",   args.num_layers)
        args.num_heads   = meta.get("num_heads",    args.num_heads)
        args.seq_length  = meta.get("seq_length",   args.seq_length)
        if meta.get("normalize", False):
            args.normalize_recon_input = True
        print(f"[meta] model_type={args.model_type} hidden={args.hidden_size} "
              f"layers={args.num_layers} heads={args.num_heads}")
    recon_model = build_model(args, device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt \
        else {k.replace("module.", ""): v for k, v in ckpt.items()}
    recon_model.load_state_dict(state_dict, strict=True)
    recon_model.eval()

    run_caption_compare(recon_model, loader, args, device)


if __name__ == "__main__":
    main()
