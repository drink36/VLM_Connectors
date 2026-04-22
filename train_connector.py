"""
Train alternative connector architectures for LLaVA and Idefics2.

Two connector types:
  invertible  — RealNVP affine coupling layers (LLaVA; dimension-preserving in 4096-d space)
  crossattn   — Perceiver resampler (Idefics2; compresses 576 → 64 tokens)

Qwen is excluded: its ViT and connector are jointly trained, making
connector-only replacement ill-defined.

Training uses saved pre_vectors_*.pt shards + captions. The frozen LLM
provides CE loss; no vision encoder is needed at train time.

Data loading:
  --vec_dir points to the parent model dir (e.g. data/vector/llava).
  --shard_folders lists which subfolders to use (e.g. 00000,00001,...,00005).
  --limit_per_shard / --val_per_shard control how many samples per shard go to
  train vs val (non-overlapping slices: train = [0, limit), val = [limit, limit+val)).

Usage:
  # smoke test (1 shard, 100 train + 20 val, 2 epochs)
  python train_connector.py \
      --embed_model llava \
      --connector_type invertible \
      --vec_dir data/vector/llava \
      --shard_folders 00000 \
      --limit_per_shard 100 --val_per_shard 20 \
      --epochs 2 --batch_size 4 --amp \
      --out_dir train_connector_out/test

  # full run (6 shards x 500 train + 200 val = 3000 train / 1200 val)
  python train_connector.py \
      --embed_model llava \
      --connector_type invertible \
      --vec_dir data/vector/llava \
      --shard_folders 00000,00001,00002,00003,00004,00005 \
      --limit_per_shard 500 --val_per_shard 200 \
      --gt_caption_dir data/mtf2025_web_images \
      --epochs 10 --batch_size 4 --accumulation_steps 4 --amp \
      --wandb --wandb_project vlm-connectors \
      --out_dir train_connector_out/llava_invertible \
      --export_post_vectors --export_dir data/new_post/llava_invertible
"""
import argparse
import math
import re
import random
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Connector architectures
# ---------------------------------------------------------------------------

class AffineCouplingLayer(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, mask_first_half: bool = True):
        super().__init__()
        half = dim // 2
        in_ch = half if mask_first_half else dim - half
        out_ch = dim - half if mask_first_half else half
        self.mask_first_half = mask_first_half
        self.half = half
        self.scale_net = nn.Sequential(
            nn.Linear(in_ch, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_ch),
        )
        self.shift_net = nn.Sequential(
            nn.Linear(in_ch, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        if self.mask_first_half:
            x_fixed, x_change = x[..., :self.half], x[..., self.half:]
        else:
            x_fixed, x_change = x[..., self.half:], x[..., :self.half]
        s = torch.tanh(self.scale_net(x_fixed)) * 2.0
        t = self.shift_net(x_fixed)
        y_change = x_change * torch.exp(s) + t
        if self.mask_first_half:
            return torch.cat([x_fixed, y_change], dim=-1)
        else:
            return torch.cat([y_change, x_fixed], dim=-1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if self.mask_first_half:
            y_fixed, y_change = y[..., :self.half], y[..., self.half:]
        else:
            y_fixed, y_change = y[..., self.half:], y[..., :self.half]
        s = torch.tanh(self.scale_net(y_fixed)) * 2.0
        t = self.shift_net(y_fixed)
        x_change = (y_change - t) * torch.exp(-s)
        if self.mask_first_half:
            return torch.cat([y_fixed, x_change], dim=-1)
        else:
            return torch.cat([x_change, y_fixed], dim=-1)


class InvertibleConnector(nn.Module):
    """LLaVA: [B,576,1024] -> [B,576,4096] via linear projection + RealNVP coupling."""
    def __init__(self, in_dim: int = 1024, out_dim: int = 4096,
                 hidden_dim: int = 512, num_coupling_layers: int = 4):
        super().__init__()
        self.linear_proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.coupling_layers = nn.ModuleList([
            AffineCouplingLayer(out_dim, hidden_dim, mask_first_half=(i % 2 == 0))
            for i in range(num_coupling_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D_pre]
        z = self.norm(self.linear_proj(x))
        for layer in self.coupling_layers:
            z = layer(z)
        return z  # [B, T, D_post]

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        for layer in reversed(self.coupling_layers):
            y = layer.inverse(y)
        return y


class CrossAttnLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: [B, num_queries, D], kv: [B, T, D]
        q2, _ = self.cross_attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + q2
        q = q + self.ff(self.norm_ff(q))
        return q


class CrossAttnConnector(nn.Module):
    """Idefics2: [B,576,1152] -> [B,64,4096] via perceiver resampler."""
    def __init__(self, in_dim: int = 1152, out_dim: int = 4096,
                 num_queries: int = 64, num_heads: int = 8,
                 num_layers: int = 2, ff_mult: int = 4):
        super().__init__()
        self.modality_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.SiLU(),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.latent_queries = nn.Parameter(torch.randn(num_queries, out_dim) * 0.02)
        self.layers = nn.ModuleList([
            CrossAttnLayer(out_dim, num_heads, ff_mult) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D_pre]
        context = self.modality_proj(x)                         # [B, T, D_out]
        q = self.latent_queries.unsqueeze(0).expand(x.shape[0], -1, -1)  # [B, num_q, D_out]
        for layer in self.layers:
            q = layer(q, context)
        return self.norm(q)  # [B, num_queries, D_out]


def build_connector(args) -> nn.Module:
    if args.connector_type == "invertible":
        dims = _connector_dims(args.embed_model)
        return InvertibleConnector(
            in_dim=dims["pre"],
            out_dim=dims["post"],
            hidden_dim=args.coupling_hidden_dim,
            num_coupling_layers=args.num_coupling_layers,
        )
    elif args.connector_type == "crossattn":
        dims = _connector_dims(args.embed_model)
        return CrossAttnConnector(
            in_dim=dims["pre"],
            out_dim=dims["post"],
            num_queries=args.num_query_tokens,
            num_heads=args.num_crossattn_heads,
            num_layers=args.num_crossattn_layers,
        )
    raise ValueError(f"unknown connector_type: {args.connector_type}")


def _connector_dims(embed_model: str) -> dict:
    return {
        "llava":    {"pre": 1024, "post": 4096},
        "idefics2": {"pre": 1152, "post": 4096},
    }[embed_model]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _numeric_suffix(p: Path) -> int:
    m = re.search(r"_(\d+)\.pt$", p.name)
    return int(m.group(1)) if m else 10 ** 18


class ConnectorTrainDataset(Dataset):
    """Loads pre_vectors_*.pt shards + captions. Optionally loads post vecs for geo reg.

    vec_dirs: list of shard directories (e.g. [data/vector/llava/00000, .../00001, ...]).
    limit_per_shard: max samples taken from each dir (0 = all).
    If gt_caption_dir is given, captions are loaded from <gt_caption_dir>/<key>.txt
    instead of the shard's own 'caps' field. Samples missing a GT file are skipped.
    """

    def __init__(self, vec_dirs: list, limit_per_shard: int = 0,
                 offset_per_shard: int = 0,
                 cache_shards: int = 2, load_post: bool = False,
                 gt_caption_dir: Optional[str] = None):
        self.load_post = load_post
        self.cache_shards = max(1, cache_shards)
        self.gt_dir = Path(gt_caption_dir) if gt_caption_dir else None

        self.pre_map: dict[int, Path] = {}
        self.post_map: dict[int, Path] = {}
        self.shard_gt_map: dict[int, Optional[Path]] = {}  # unique_tag -> shard gt dir
        self.items: list[tuple[int, int, str]] = []

        global_tag = 0
        skipped = 0
        for vec_dir in vec_dirs:
            root = Path(vec_dir)
            shard_name = root.name  # e.g. "00000"
            pre_files = sorted(root.glob("pre_vectors_*.pt"), key=_numeric_suffix)
            post_files = sorted(root.glob("post_vectors_*.pt"), key=_numeric_suffix) if load_post else []
            if not pre_files:
                print(f"[dataset] WARNING: no pre_vectors_*.pt in {vec_dir}, skipping")
                continue

            # GT captions live at gt_dir/<shard>/<key>.txt
            shard_gt_dir = (self.gt_dir / shard_name) if self.gt_dir is not None else None

            pre_map_local = {_numeric_suffix(p): p for p in pre_files}
            post_map_local = {_numeric_suffix(p): p for p in post_files} if load_post else {}
            tags = sorted(pre_map_local.keys())
            if load_post:
                tags = sorted(set(tags) & set(post_map_local.keys()))

            seen_this_dir = 0
            for local_tag in tags:
                unique_tag = global_tag
                global_tag += 1
                self.pre_map[unique_tag] = pre_map_local[local_tag]
                self.shard_gt_map[unique_tag] = shard_gt_dir
                if load_post:
                    self.post_map[unique_tag] = post_map_local[local_tag]

                obj = torch.load(pre_map_local[local_tag], map_location="cpu", weights_only=False)
                skipped_offset = 0
                for row_idx, key in enumerate(obj["keys"]):
                    key = str(key)
                    if shard_gt_dir is not None:
                        gt_path = shard_gt_dir / f"{key}.txt"
                        if not gt_path.exists():
                            skipped += 1
                            continue
                    if skipped_offset < offset_per_shard:
                        skipped_offset += 1
                        continue
                    self.items.append((unique_tag, row_idx, key))
                    seen_this_dir += 1
                    if limit_per_shard and seen_this_dir >= limit_per_shard:
                        break
                if limit_per_shard and seen_this_dir >= limit_per_shard:
                    break

        if skipped:
            print(f"[dataset] skipped {skipped} samples with no GT caption file")
        dirs_str = ", ".join(str(Path(d).name) for d in vec_dirs)
        print(f"[dataset] {len(self.items)} samples from [{dirs_str}]"
              + (" (GT captions)" if self.gt_dir else " (shard captions)"))

        self.pre_cache: OrderedDict = OrderedDict()
        self.post_cache: OrderedDict = OrderedDict()
        self.cap_cache: OrderedDict = OrderedDict()

    def __len__(self) -> int:
        return len(self.items)

    def _load_tag(self, tag: int) -> None:
        if tag in self.pre_cache:
            self.pre_cache.move_to_end(tag)
            return
        while len(self.pre_cache) >= self.cache_shards:
            old = next(iter(self.pre_cache))
            self.pre_cache.pop(old)
            self.cap_cache.pop(old, None)
            if self.load_post:
                self.post_cache.pop(old, None)

        pre_obj = torch.load(self.pre_map[tag], map_location="cpu", weights_only=False)
        self.pre_cache[tag] = pre_obj["vecs"].float()
        self.cap_cache[tag] = pre_obj.get("caps", [""] * len(pre_obj["keys"]))
        if self.load_post:
            post_obj = torch.load(self.post_map[tag], map_location="cpu", weights_only=False)
            self.post_cache[tag] = post_obj["vecs"].float()

    def _get_cap(self, tag: int, row: int, key: str) -> str:
        shard_gt = self.shard_gt_map.get(tag)
        if shard_gt is not None:
            return (shard_gt / f"{key}.txt").read_text(encoding="utf-8").strip()
        return self.cap_cache[tag][row]

    def __getitem__(self, idx: int) -> dict:
        tag, row, key = self.items[idx]
        self._load_tag(tag)
        pre = self.pre_cache[tag][row]   # [T, D_pre]
        cap = self._get_cap(tag, row, key)
        item = {"pre": pre, "cap": cap, "sample_id": key}
        if self.load_post:
            item["post"] = self.post_cache[tag][row]
        return item


def collate_fn(batch: list[dict]) -> dict:
    out = {
        "pre": torch.stack([b["pre"] for b in batch]),
        "cap": [b["cap"] for b in batch],
        "sample_id": [b["sample_id"] for b in batch],
    }
    if "post" in batch[0]:
        out["post"] = torch.stack([b["post"] for b in batch])
    return out


# ---------------------------------------------------------------------------
# LLM loading and input building
# ---------------------------------------------------------------------------

def load_vlm(embed_model: str, model_id: str, device: torch.device, amp_dtype: torch.dtype):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if embed_model == "idefics2" and hasattr(processor, "image_processor"):
        processor.image_processor.size = {"longest_edge": 336, "shortest_edge": 336}
        if hasattr(processor.image_processor, "do_image_splitting"):
            processor.image_processor.do_image_splitting = False

    dtype = amp_dtype if device.type == "cuda" else torch.float32
    if embed_model == "llava":
        from transformers import LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    elif embed_model == "idefics2":
        from transformers import Idefics2ForConditionalGeneration
        model = Idefics2ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True)
    else:
        raise ValueError(f"unsupported embed_model: {embed_model}")

    model = model.to(device).eval()
    model.requires_grad_(False)
    return processor, model


def _make_dummy_image(img_size: int = 336):
    from PIL import Image
    return Image.new("RGB", (img_size, img_size), color=(0, 0, 0))


def build_llava_inputs(processor, caps: list[str], prompt: str,
                       device: torch.device, img_size: int = 336) -> dict:
    dummy = _make_dummy_image(img_size)
    template = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    prompts = [processor.apply_chat_template(template, add_generation_prompt=True)] * len(caps)

    # Build full sequences: prompt + caption (for CE loss target)
    full_prompts = [p + c for p, c in zip(prompts, caps)]
    inputs = processor(
        images=[dummy] * len(caps),
        text=full_prompts,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Build labels: mask everything up to and including the ASSISTANT turn prefix
    prompt_only = processor(
        images=[dummy] * len(caps),
        text=prompts,
        padding=True,
        return_tensors="pt",
    )
    prompt_lens = (prompt_only["attention_mask"]).sum(dim=1)  # [B]

    labels = inputs["input_ids"].clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100
    labels[inputs["attention_mask"] == 0] = -100
    inputs["labels"] = labels
    return inputs


def build_idefics2_inputs(processor, caps: list[str], prompt: str,
                          device: torch.device, img_size: int = 336) -> dict:
    dummy = _make_dummy_image(img_size)
    template = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    prompts = [processor.apply_chat_template(template, add_generation_prompt=True)] * len(caps)
    full_prompts = [p + c for p, c in zip(prompts, caps)]

    inputs = processor(
        images=[[dummy]] * len(caps),
        text=full_prompts,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    prompt_only = processor(
        images=[[dummy]] * len(caps),
        text=prompts,
        padding=True,
        return_tensors="pt",
    )
    prompt_lens = prompt_only["attention_mask"].sum(dim=1)

    labels = inputs["input_ids"].clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100
    labels[inputs["attention_mask"] == 0] = -100
    inputs["labels"] = labels
    return inputs


# ---------------------------------------------------------------------------
# reshape_like (copied from caption_eval.py)
# ---------------------------------------------------------------------------

def reshape_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    out = src
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
            out = out.repeat_interleave(ref.shape[0] // out.shape[0], dim=0)
        elif out.shape[0] % ref.shape[0] == 0:
            group = out.shape[0] // ref.shape[0]
            out = out.reshape(ref.shape[0], group, out.shape[1], out.shape[2]).mean(dim=1)
    if ref.dim() == 3 and out.dim() == 3 and out.shape[1] != ref.shape[1]:
        out = F.interpolate(out.transpose(1, 2), size=ref.shape[1],
                            mode="linear", align_corners=False).transpose(1, 2)
    if out.shape != ref.shape:
        raise RuntimeError(f"shape mismatch: expected {tuple(ref.shape)}, got {tuple(out.shape)}")
    return out


# ---------------------------------------------------------------------------
# Forward with injection
# ---------------------------------------------------------------------------

def forward_with_injection(vlm, connector, pre: torch.Tensor,
                            inputs: dict, embed_model: str,
                            use_amp: bool, amp_dtype: torch.dtype) -> torch.Tensor:
    with autocast(device_type=pre.device.type, dtype=amp_dtype, enabled=use_amp):
        new_E_post = connector(pre)  # trainable

        if embed_model == "llava":
            # Hook replaces multi_modal_projector output with new_E_post
            _injected = [None]

            def _hook(_module, _inp, output):
                _injected[0] = reshape_like(new_E_post, output).to(dtype=output.dtype)
                return _injected[0]

            mods = dict(vlm.named_modules())
            proj = mods.get("multi_modal_projector") or mods.get("model.multi_modal_projector")
            h = proj.register_forward_hook(_hook)
            try:
                outputs = vlm(**inputs)
            finally:
                h.remove()

        elif embed_model == "idefics2":
            # Remove pixel_values; pass image_hidden_states directly
            idef_inputs = {k: v for k, v in inputs.items()
                           if k not in ("pixel_values", "pixel_attention_mask")}
            outputs = vlm(**idef_inputs, image_hidden_states=new_E_post)

    return outputs.loss, new_E_post


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_amp = args.amp and device.type == "cuda"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # wandb
    use_wandb = args.wandb and _WANDB_AVAILABLE
    if args.wandb and not _WANDB_AVAILABLE:
        print("WARNING: wandb not installed, skipping logging")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"{args.embed_model}_{args.connector_type}",
            config=vars(args),
        )

    # Dataset
    load_post = args.geo_reg_weight > 0
    if args.shard_folders:
        base = Path(args.vec_dir)
        vec_dirs = [str(base / s.strip()) for s in args.shard_folders.split(",")]
    else:
        vec_dirs = [args.vec_dir]
    train_ds = ConnectorTrainDataset(vec_dirs, limit_per_shard=args.limit_per_shard,
                                     offset_per_shard=0,
                                     cache_shards=args.cache_shards, load_post=load_post,
                                     gt_caption_dir=args.gt_caption_dir or None)
    val_ds = ConnectorTrainDataset(vec_dirs, limit_per_shard=args.val_per_shard,
                                   offset_per_shard=args.limit_per_shard,
                                   cache_shards=args.cache_shards, load_post=load_post,
                                   gt_caption_dir=args.gt_caption_dir or None)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers)

    # Connector
    connector = build_connector(args).to(device)
    n_params = sum(p.numel() for p in connector.parameters() if p.requires_grad)
    print(f"Connector: {args.connector_type} | trainable params: {n_params:,}")

    # VLM (frozen)
    model_id = args.vlm_model_id or _default_model_id(args.embed_model)
    print(f"Loading frozen VLM: {model_id}")
    processor, vlm = load_vlm(args.embed_model, model_id, device, amp_dtype)
    print("VLM loaded and frozen.")

    # Optimizer & scheduler
    optimizer = optim.AdamW(connector.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    warmup = min(args.warmup_steps, max(1, total_steps - 1))
    sched_warm = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: min(1.0, s / warmup))
    sched_cos = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup), eta_min=args.lr * 0.01)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[sched_warm, sched_cos], milestones=[warmup])
    scaler = GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    best_val_loss = float("inf")
    patience_counter = 0

    _build_inputs = build_llava_inputs if args.embed_model == "llava" else build_idefics2_inputs

    for epoch in range(args.epochs):
        # --- Train ---
        connector.train()
        train_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]")
        for step, batch in enumerate(pbar):
            pre = batch["pre"].to(device)
            caps = batch["cap"]

            inputs = _build_inputs(processor, caps, args.caption_prompt, device)
            ce_loss, new_E_post = forward_with_injection(
                vlm, connector, pre, inputs, args.embed_model, use_amp, amp_dtype)

            loss = ce_loss
            if args.geo_reg_weight > 0 and "post" in batch:
                post_gt = batch["post"].to(device)
                # align shapes (crossattn compresses tokens)
                post_new = new_E_post if new_E_post.shape == post_gt.shape \
                    else F.interpolate(new_E_post.transpose(1, 2),
                                       size=post_gt.shape[1],
                                       mode="linear", align_corners=False).transpose(1, 2)
                loss = loss + args.geo_reg_weight * F.mse_loss(post_new, post_gt)

            loss = loss / args.accumulation_steps
            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(connector.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss += loss.item() * args.accumulation_steps
            pbar.set_postfix(loss=f"{train_loss / (step + 1):.4f}")
            if use_wandb and (step + 1) % args.accumulation_steps == 0:
                global_step = epoch * len(train_loader) + step + 1
                wandb.log({"train/loss": loss.item() * args.accumulation_steps,
                           "train/lr": scheduler.get_last_lr()[0]}, step=global_step)

        # --- Validate ---
        connector.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pre = batch["pre"].to(device)
                caps = batch["cap"]
                inputs = _build_inputs(processor, caps, args.caption_prompt, device)
                ce_loss, _ = forward_with_injection(
                    vlm, connector, pre, inputs, args.embed_model, use_amp, amp_dtype)
                val_loss += ce_loss.item()
        val_loss /= max(1, len(val_loader))
        train_epoch_loss = train_loss / max(1, len(train_loader))
        print(f"  val_loss={val_loss:.4f}")
        if use_wandb:
            wandb.log({"epoch/train_loss": train_epoch_loss,
                       "epoch/val_loss": val_loss}, step=(epoch + 1) * len(train_loader))

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"state_dict": connector.state_dict(),
                        "args": vars(args), "epoch": epoch + 1,
                        "val_loss": val_loss},
                       out_dir / "best_connector.pt")
            print(f"  -> saved best (val_loss={val_loss:.4f})")
            if use_wandb:
                wandb.summary["best_val_loss"] = best_val_loss
                wandb.summary["best_epoch"] = epoch + 1
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Training done. Best val_loss={best_val_loss:.4f}")
    if use_wandb:
        wandb.finish()

    if args.export_post_vectors:
        ckpt = torch.load(out_dir / "best_connector.pt", map_location="cpu", weights_only=False)
        connector.load_state_dict(ckpt["state_dict"])
        for vd in vec_dirs:
            shard_name = Path(vd).name
            export_dir = args.export_dir or str(out_dir / "new_post")
            export_new_post_vectors(connector, vd, str(Path(export_dir) / shard_name),
                                    device, args.batch_size)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_new_post_vectors(connector: nn.Module, vec_dir: str, export_dir: str,
                             device: torch.device, batch_size: int = 32) -> None:
    connector.eval()
    root = Path(vec_dir)
    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)

    pre_files = sorted(root.glob("pre_vectors_*.pt"), key=_numeric_suffix)
    print(f"Exporting {len(pre_files)} shards to {out}")

    with torch.no_grad():
        for f in tqdm(pre_files, desc="Export"):
            obj = torch.load(f, map_location="cpu", weights_only=False)
            pre_all = obj["vecs"].float()  # [N, T, D_pre]
            keys = obj["keys"]
            caps = obj.get("caps", [""] * len(keys))

            new_vecs = []
            for i in range(0, len(keys), batch_size):
                pre_b = pre_all[i: i + batch_size].to(device)
                new_vecs.append(connector(pre_b).cpu())
            new_vecs = torch.cat(new_vecs, dim=0)

            tag = _numeric_suffix(f)
            torch.save({"keys": keys, "vecs": new_vecs, "caps": caps},
                       out / f"post_vectors_{tag}.pt")

    torch.save({"source": vec_dir, "connector_type": "exported"}, out / "manifest.pt")
    print(f"Export done -> {out}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_model_id(embed_model: str) -> str:
    return {
        "llava": "llava-hf/llava-1.5-7b-hf",
        "idefics2": "HuggingFaceM4/idefics2-8b",
    }[embed_model]


def parse_args():
    p = argparse.ArgumentParser("Train alternative VLM connector (LLaVA / Idefics2)")
    p.add_argument("--embed_model", type=str, required=True, choices=["llava", "idefics2"])
    p.add_argument("--connector_type", type=str, required=True, choices=["invertible", "crossattn"])
    p.add_argument("--vec_dir", type=str, required=True,
                   help="shard dir (single) or parent dir (with --shard_folders)")
    p.add_argument("--shard_folders", type=str, default="",
                   help="comma-separated shard subfolder names, e.g. 00000,00001,00002,00003,00004,00005")
    p.add_argument("--limit_per_shard", type=int, default=500,
                   help="max samples per shard folder (0 = all)")
    p.add_argument("--vlm_model_id", type=str, default="",
                   help="HuggingFace model id (default: model-family default)")

    # Connector hyperparams
    p.add_argument("--num_coupling_layers", type=int, default=4)
    p.add_argument("--coupling_hidden_dim", type=int, default=512)
    p.add_argument("--num_query_tokens", type=int, default=64)
    p.add_argument("--num_crossattn_layers", type=int, default=2)
    p.add_argument("--num_crossattn_heads", type=int, default=8)

    # Training
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--val_per_shard", type=int, default=200,
                   help="val samples taken per shard folder (from offset limit_per_shard)")
    p.add_argument("--accumulation_steps", type=int, default=4)
    p.add_argument("--geo_reg_weight", type=float, default=0.0,
                   help="weight for MSE geometry regularization against original E_post")
    p.add_argument("--train_split", type=float, default=0.9)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="0 = all samples")
    p.add_argument("--cache_shards", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)

    # Caption source
    p.add_argument("--caption_prompt", type=str, default="Describe this image in one detailed sentence, focusing on the main objects, attributes, and scene.")
    p.add_argument("--gt_caption_dir", type=str, default="",
                   help="directory with <key>.txt GT captions; if set, used instead of shard captions")
    p.add_argument("--image_dir", type=str, default="",
                   help="root image directory for CLIPScore and GT caption .txt lookup during inference")

    # Logging
    p.add_argument("--wandb", action="store_true", help="log to wandb")
    p.add_argument("--wandb_project", type=str, default="vlm-connectors")
    p.add_argument("--wandb_run_name", type=str, default="",
                   help="run name (default: <embed_model>_<connector_type>)")

    # Output
    p.add_argument("--out_dir", type=str, default="train_connector_out")
    p.add_argument("--export_post_vectors", action="store_true",
                   help="after training, export new post-vectors for knn.py analysis")
    p.add_argument("--export_dir", type=str, default="",
                   help="where to save exported post shards (default: out_dir/new_post)")

    # Inference
    p.add_argument("--infer", action="store_true",
                   help="run caption inference with a trained connector (skip training)")
    p.add_argument("--infer_checkpoint", type=str, default="",
                   help="path to best_connector.pt (default: out_dir/best_connector.pt)")
    p.add_argument("--infer_limit", type=int, default=100,
                   help="number of samples to run inference on")
    p.add_argument("--infer_out", type=str, default="",
                   help="output CSV path (default: out_dir/infer_captions.csv)")
    p.add_argument("--max_new_tokens", type=int, default=256)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _resolve_proj_module(vlm):
    mods = dict(vlm.named_modules())
    return mods.get("multi_modal_projector") or mods.get("model.multi_modal_projector")


def infer(args):
    import csv
    from PIL import Image as PILImage
    from transformers import AutoProcessor as _AutoProcessor, CLIPModel
    from transformers.image_utils import load_image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_amp = args.amp and device.type == "cuda"

    out_dir = Path(args.out_dir)
    ckpt_path = args.infer_checkpoint or str(out_dir / "best_connector.pt")
    out_csv = args.infer_out or str(out_dir / "infer_captions.csv")

    print(f"Loading connector from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    connector = build_connector(args).to(device).eval()
    connector.load_state_dict(ckpt["state_dict"])

    model_id = args.vlm_model_id or _default_model_id(args.embed_model)
    print(f"Loading frozen VLM: {model_id}")
    processor, vlm = load_vlm(args.embed_model, model_id, device, amp_dtype)

    clip_model_id = "openai/clip-vit-base-patch16"
    print(f"Loading CLIP: {clip_model_id}")
    clip_proc = _AutoProcessor.from_pretrained(clip_model_id)
    clip_mdl = CLIPModel.from_pretrained(clip_model_id, use_safetensors=True).to(device).eval()

    key_to_path: dict[str, str] = {}
    if args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        for p in Path(args.image_dir).rglob("*"):
            if p.suffix.lower() in exts:
                key_to_path.setdefault(p.stem, str(p))

    if args.shard_folders:
        base = Path(args.vec_dir)
        vec_dirs = [str(base / s.strip()) for s in args.shard_folders.split(",")]
    else:
        vec_dirs = [args.vec_dir]

    ds = ConnectorTrainDataset(vec_dirs, limit_per_shard=args.infer_limit,
                               offset_per_shard=0, cache_shards=args.cache_shards,
                               load_post=False, gt_caption_dir=None)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    proj_module = _resolve_proj_module(vlm) if args.embed_model == "llava" else None

    @torch.no_grad()
    def _clipscore(image_paths, captions):
        scores = [float("nan")] * len(image_paths)
        valid = [(i, p, c) for i, (p, c) in enumerate(zip(image_paths, captions))
                 if p and Path(p).exists() and str(c).strip()]
        if not valid:
            return scores
        idxs, paths, caps = zip(*valid)
        images = [PILImage.open(p).convert("RGB") for p in paths]
        inp = clip_proc(text=list(caps), images=images,
                        padding=True, truncation=True, return_tensors="pt").to(device)
        out = clip_mdl(**inp)
        img_e = F.normalize(out.image_embeds, dim=-1)
        txt_e = F.normalize(out.text_embeds, dim=-1)
        vals = torch.clamp(100.0 * (img_e * txt_e).sum(dim=-1), min=0.0).cpu().tolist()
        for i, v in zip(idxs, vals):
            scores[i] = v
        return scores

    COLS = ["sample_id", "key", "folder_id", "model", "connector_type",
            "image_path", "caption_connector", "clipscore_connector"]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    total = 0

    _template = [{"role": "user", "content": [{"type": "image"},
                                               {"type": "text", "text": args.caption_prompt}]}]
    _prompt_text = processor.apply_chat_template(_template, add_generation_prompt=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()

        with torch.no_grad():
            for batch in tqdm(loader, desc="Inference"):
                pre   = batch["pre"].to(device)
                s_ids = batch["sample_id"]
                keys  = [sid.split(":")[-1] if ":" in sid else sid for sid in s_ids]
                image_paths = [key_to_path.get(k, "") for k in keys]

                # Load real images per batch — exactly like caption_eval.py
                images = [
                    load_image(p) if p and Path(p).exists()
                    else PILImage.new("RGB", (336, 336))
                    for p in image_paths
                ]
                prompts = [_prompt_text] * len(images)
                if args.embed_model == "llava":
                    vlm_inputs = processor(images=images, text=prompts,
                                           padding=True, return_tensors="pt")
                else:  # idefics2
                    vlm_inputs = processor(images=[[im] for im in images], text=prompts,
                                           padding=True, return_tensors="pt")
                vlm_inputs = {k: v.to(device) for k, v in vlm_inputs.items()}

                new_e_post = connector(pre)

                if args.embed_model == "llava":
                    def _hook(_module, _inp, output):
                        return reshape_like(new_e_post, output).to(dtype=output.dtype)
                    h = proj_module.register_forward_hook(_hook)
                    try:
                        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                            out_ids = vlm.generate(**vlm_inputs,
                                                   max_new_tokens=args.max_new_tokens,
                                                   do_sample=False)
                    finally:
                        h.remove()
                else:  # idefics2
                    idef_inputs = {k: v for k, v in vlm_inputs.items()
                                   if k not in ("pixel_values", "pixel_attention_mask")}
                    with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        out_ids = vlm.generate(**idef_inputs,
                                               image_hidden_states=new_e_post,
                                               max_new_tokens=args.max_new_tokens,
                                               do_sample=False)

                prompt_len = vlm_inputs["input_ids"].shape[1]
                caps_conn = processor.batch_decode(out_ids[:, prompt_len:],
                                                   skip_special_tokens=True)
                cs_conn = _clipscore(image_paths, caps_conn)

                for sid, key, path, cap, cs in zip(s_ids, keys, image_paths, caps_conn, cs_conn):
                    writer.writerow({
                        "sample_id": sid,
                        "key": key,
                        "folder_id": Path(path).parent.name if path else "",
                        "model": args.embed_model,
                        "connector_type": args.connector_type,
                        "image_path": path,
                        "caption_connector": cap.strip(),
                        "clipscore_connector": cs,
                    })
                total += len(s_ids)
                f.flush()

    print(f"Saved {total} rows -> {out_csv}")


if __name__ == "__main__":
    args = parse_args()
    if args.infer:
        infer(args)
    else:
        train(args)
