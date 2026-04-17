"""
Train alternative connector architectures for LLaVA and Idefics2.

Two connector types:
  invertible  — RealNVP affine coupling layers (LLaVA; dimension-preserving in 4096-d space)
  crossattn   — Perceiver resampler (Idefics2; compresses 576 → 64 tokens)

Qwen is excluded: its ViT and connector are jointly trained, making
connector-only replacement ill-defined.

Training uses saved pre_vectors_*.pt shards + captions. The frozen LLM
provides CE loss; no vision encoder is needed at train time.

Usage:
  # LLaVA invertible connector (smoke test)
  python train_connector.py \
      --embed_model llava \
      --connector_type invertible \
      --vec_dir data/vector/llava/00000 \
      --limit 500 --epochs 2 --batch_size 4 --amp \
      --out_dir train_connector_out/llava_inv_test

  # Idefics2 cross-attention connector
  python train_connector.py \
      --embed_model idefics2 \
      --connector_type crossattn \
      --vec_dir data/vector/idefics2/00000 \
      --epochs 10 --batch_size 4 --amp \
      --out_dir train_connector_out/idefics2_ca \
      --export_post_vectors \
      --export_dir data/new_post/idefics2_ca/00000
"""
import argparse
import math
import re
import random
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
    """Loads pre_vectors_*.pt shards + captions. Optionally loads post vecs for geo reg."""

    def __init__(self, vec_dir: str, limit: int = 0,
                 cache_shards: int = 2, load_post: bool = False):
        root = Path(vec_dir)
        self.pre_files = sorted(root.glob("pre_vectors_*.pt"), key=_numeric_suffix)
        self.post_files = sorted(root.glob("post_vectors_*.pt"), key=_numeric_suffix) if load_post else []
        if not self.pre_files:
            raise FileNotFoundError(f"No pre_vectors_*.pt found in {vec_dir}")

        self.load_post = load_post
        self.cache_shards = max(1, cache_shards)

        pre_map = {_numeric_suffix(p): p for p in self.pre_files}
        post_map = {_numeric_suffix(p): p for p in self.post_files} if load_post else {}
        tags = sorted(pre_map.keys())
        if load_post:
            tags = sorted(set(tags) & set(post_map.keys()))

        self.pre_map = pre_map
        self.post_map = post_map

        self.items: list[tuple[int, int, str]] = []
        seen = 0
        for tag in tags:
            obj = torch.load(pre_map[tag], map_location="cpu", weights_only=False)
            for row_idx, key in enumerate(obj["keys"]):
                self.items.append((tag, row_idx, str(key)))
                seen += 1
                if limit and seen >= limit:
                    break
            if limit and seen >= limit:
                break

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

    def __getitem__(self, idx: int) -> dict:
        tag, row, key = self.items[idx]
        self._load_tag(tag)
        pre = self.pre_cache[tag][row]   # [T, D_pre]
        cap = self.cap_cache[tag][row]
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

            def _hook(module, inp, output):
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

    # Dataset
    load_post = args.geo_reg_weight > 0
    ds = ConnectorTrainDataset(args.vec_dir, limit=args.limit,
                               cache_shards=args.cache_shards, load_post=load_post)
    n_val = max(1, int(len(ds) * (1 - args.train_split)))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
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
        print(f"  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"state_dict": connector.state_dict(),
                        "args": vars(args), "epoch": epoch + 1,
                        "val_loss": val_loss},
                       out_dir / "best_connector.pt")
            print(f"  -> saved best (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Training done. Best val_loss={best_val_loss:.4f}")

    if args.export_post_vectors:
        ckpt = torch.load(out_dir / "best_connector.pt", map_location="cpu", weights_only=False)
        connector.load_state_dict(ckpt["state_dict"])
        export_new_post_vectors(connector, args.vec_dir, args.export_dir or str(out_dir / "new_post"),
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
                   help="directory containing pre_vectors_*.pt shards")
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

    # Caption prompt
    p.add_argument("--caption_prompt", type=str, default="What do you see in this image?")

    # Output
    p.add_argument("--out_dir", type=str, default="train_connector_out")
    p.add_argument("--export_post_vectors", action="store_true",
                   help="after training, export new post-vectors for knn.py analysis")
    p.add_argument("--export_dir", type=str, default="",
                   help="where to save exported post shards (default: out_dir/new_post)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
