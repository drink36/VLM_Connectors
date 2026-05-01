import argparse
import math
import os
import pickle
import random
import re
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
try:
    import wandb
except ImportError:
    wandb = None
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler, random_split
from tqdm import tqdm


class ShardSample(TypedDict):
    embeddings: tuple[torch.Tensor, torch.Tensor]
    sample_id: str


class ShardBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        items: list[tuple[int, int, str]],
        batch_size: int,
        drop_last: bool = True,
        shuffle_tags: bool = True,
        shuffle_within_tag: bool = False,
    ):
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.shuffle_tags = shuffle_tags
        self.shuffle_within_tag = shuffle_within_tag

        tag_to_indices: OrderedDict[int, list[int]] = OrderedDict()
        for local_idx, (tag, _row_idx, _key) in enumerate(items):
            tag_to_indices.setdefault(tag, []).append(local_idx)
        self.tag_to_indices = tag_to_indices

    def __iter__(self):
        rng = random.Random()
        tags = list(self.tag_to_indices.keys())
        if self.shuffle_tags:
            rng.shuffle(tags)

        batch: list[int] = []
        for tag in tags:
            idxs = list(self.tag_to_indices[tag])
            if self.shuffle_within_tag:
                rng.shuffle(idxs)
            for idx in idxs:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        total = sum(len(v) for v in self.tag_to_indices.values())
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size


def numeric_suffix(path: Path) -> int:
    match = re.search(r"_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else 10**18


class ShardPairVectorDataset(Dataset):
    """
    Dataset for shard .pt vectors written by extract scripts:
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

        self._warned_pre_2d = False
        self._warned_post_2d = False
        self._warned_pre_cls = False
        self._warned_post_cls = False

    def __len__(self) -> int:
        return len(self.items)

    def _maybe_strip_cls(self, vecs: torch.Tensor, name: str) -> torch.Tensor:
        # vecs: [N, T, D]
        token_len = vecs.shape[1]
        s = int(math.isqrt(token_len))
        if s * s == token_len:
            return vecs

        if self.strip_cls and token_len > 1:
            s1 = int(math.isqrt(token_len - 1))
            if s1 * s1 == (token_len - 1):
                if name == "pre" and not self._warned_pre_cls:
                    print(f"[warn] pre token len {token_len} -> strip CLS -> {token_len - 1}")
                    self._warned_pre_cls = True
                if name == "post" and not self._warned_post_cls:
                    print(f"[warn] post token len {token_len} -> strip CLS -> {token_len - 1}")
                    self._warned_post_cls = True
                return vecs[:, 1:, :]

        raise ValueError(
            f"{name} token len={token_len} is not square and not (square+1). "
            "Use --no_strip_cls to disable auto CLS handling only if your data is already aligned."
        )

    def _load_tag(self, tag: int) -> None:
        if tag in self.pre_cache and tag in self.post_cache and tag in self.key_cache:
            self.pre_cache.move_to_end(tag)
            self.post_cache.move_to_end(tag)
            self.key_cache.move_to_end(tag)
            return

        # Evict first to avoid transient peak RAM when loading a new shard.
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
            if not self._warned_pre_2d:
                print("[warn] pre vecs are [N,D], auto convert to [N,1,D].")
                self._warned_pre_2d = True
            pre_vecs = pre_vecs.unsqueeze(1)

        if post_vecs.dim() == 2:
            if not self._warned_post_2d:
                print("[warn] post vecs are [N,D], auto convert to [N,1,D].")
                self._warned_post_2d = True
            post_vecs = post_vecs.unsqueeze(1)

        pre_vecs = self._maybe_strip_cls(pre_vecs, "pre")
        post_vecs = self._maybe_strip_cls(post_vecs, "post")

        self.key_cache[tag] = keys
        self.pre_cache[tag] = pre_vecs
        self.post_cache[tag] = post_vecs

    def __getitem__(self, index: int) -> ShardSample:
        tag, row_idx, key = self.items[index]
        self._load_tag(tag)

        pre = self.pre_cache[tag][row_idx]   # [T, D]
        post = self.post_cache[tag][row_idx]  # [T, D]
        sample_id = f"{tag}:{row_idx}:{key}"

        return {
            "embeddings": (pre, post),
            "sample_id": sample_id,
        }


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


def standardize_embeddings(embeddings: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (embeddings - mean) / (std + 1e-8)


def compute_dataset_stats(dataloader: DataLoader) -> dict[str, torch.Tensor]:
    sum_pre, sum_sq_pre, num_samples_pre = None, None, 0
    sum_post, sum_sq_post, num_samples_post = None, None, 0

    for batch in tqdm(dataloader, total=len(dataloader), desc="Computing Dataset Stats"):
        pre_emb = batch["embeddings"][0]
        post_emb = batch["embeddings"][1]

        _, _, d_pre = pre_emb.shape
        pre_flat = pre_emb.reshape(-1, d_pre)
        curr_num_pre = pre_flat.size(0)

        _, _, d_post = post_emb.shape
        post_flat = post_emb.reshape(-1, d_post)
        curr_num_post = post_flat.size(0)

        if sum_pre is None:
            device = pre_emb.device
            sum_pre = torch.zeros(d_pre, device=device)
            sum_sq_pre = torch.zeros(d_pre, device=device)
            sum_post = torch.zeros(d_post, device=device)
            sum_sq_post = torch.zeros(d_post, device=device)

        sum_pre += torch.sum(pre_flat, dim=0)
        sum_sq_pre += torch.sum(pre_flat ** 2, dim=0)
        num_samples_pre += curr_num_pre

        sum_post += torch.sum(post_flat, dim=0)
        sum_sq_post += torch.sum(post_flat ** 2, dim=0)
        num_samples_post += curr_num_post

    assert sum_pre is not None and sum_post is not None

    mean_pre = sum_pre / num_samples_pre
    std_pre = torch.sqrt((sum_sq_pre / num_samples_pre) - (mean_pre ** 2) + 1e-8)

    mean_post = sum_post / num_samples_post
    std_post = torch.sqrt((sum_sq_post / num_samples_post) - (mean_post ** 2) + 1e-8)

    return {
        "mean_pre": mean_pre,
        "std_pre": std_pre,
        "mean_post": mean_post,
        "std_post": std_post,
    }


def get_model_params(module: nn.Module) -> tuple[int, int]:
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total_params, trainable_params


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


def save_checkpoint(state_dict: dict[str, torch.Tensor], args: argparse.Namespace, config: SimpleNamespace) -> None:
    ckpt = {
        "state_dict": state_dict,
        "meta": {
            "embed_model": args.embed_model,
            "model_type": args.model_type,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "seq_length": args.seq_length,
            "normalize": args.normalize,
            "learning_rate": config.learning_rate,
        },
    }
    torch.save(ckpt, os.path.join(args.out_dir, "best_model.pt"))
    torch.save(ckpt, os.path.join(args.out_dir, "best_embedding_transformer.pth"))


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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: SimpleNamespace,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    total_train_steps = config.num_epochs * len(train_loader)
    warmup_steps = min(config.warmup_steps, max(1, total_train_steps - 1))

    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_train_steps - warmup_steps),
        eta_min=config.learning_rate * 0.01,
    )

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / warmup_steps),
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )

    use_amp = bool(config.mixed_precision and device.type == "cuda")
    autocast_dtype = get_autocast_dtype(args)
    scaler = GradScaler(enabled=use_amp and autocast_dtype == torch.float16)

    best_val_loss = float("inf")
    best_model = None
    patience_counter = 0

    stats_dict = None
    if args.normalize:
        print("calculate dataset mean and std on train split")
        stats_dict = compute_dataset_stats(train_loader)
        with open(os.path.join(args.out_dir, "dataset_stats.pkl"), "wb") as f:
            pickle.dump(stats_dict, f)

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Training]", ncols=100)

        for batch_idx, batch in enumerate(train_loader_tqdm):
            if args.normalize:
                post_emb = standardize_embeddings(
                    batch["embeddings"][1], stats_dict["mean_post"], stats_dict["std_post"]
                ).to(device, non_blocking=True)
                pre_emb = standardize_embeddings(
                    batch["embeddings"][0], stats_dict["mean_pre"], stats_dict["std_pre"]
                ).to(device, non_blocking=True)
            else:
                post_emb = batch["embeddings"][1].to(device, non_blocking=True)
                pre_emb = batch["embeddings"][0].to(device, non_blocking=True)

            with autocast(enabled=use_amp, dtype=autocast_dtype, device_type=device.type):
                output = model(post_emb)
                loss = criterion(output, pre_emb)
                loss = loss / config.accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss += loss.item() * config.accumulation_steps
            train_loader_tqdm.set_postfix({"Batch Loss": loss.item() * config.accumulation_steps})

            if getattr(config, "use_wandb", False) and batch_idx % config.log_interval == 0:
                wandb.log(
                    {
                        "batch_loss": loss.item() * config.accumulation_steps,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "batch": batch_idx,
                    }
                )

        # Step once if the last few mini-batches did not hit accumulation boundary.
        if len(train_loader) % config.accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Validation]", ncols=100)

        with torch.no_grad():
            for batch in val_loader_tqdm:
                if args.normalize:
                    post_emb = standardize_embeddings(
                        batch["embeddings"][1], stats_dict["mean_post"], stats_dict["std_post"]
                    ).to(device, non_blocking=True)
                    pre_emb = standardize_embeddings(
                        batch["embeddings"][0], stats_dict["mean_pre"], stats_dict["std_pre"]
                    ).to(device, non_blocking=True)
                else:
                    post_emb = batch["embeddings"][1].to(device, non_blocking=True)
                    pre_emb = batch["embeddings"][0].to(device, non_blocking=True)

                with autocast(enabled=use_amp, dtype=autocast_dtype, device_type=device.type):
                    output = model(post_emb)
                    loss = criterion(output, pre_emb)

                val_loss += loss.item()
                val_loader_tqdm.set_postfix({"Val Loss": loss.item()})

        val_loss /= max(1, len(val_loader))

        if val_loss < best_val_loss - config.min_delta:
            print(f"Validation loss decreased ({best_val_loss:.6f} --> {val_loss:.6f})")
            best_val_loss = val_loss
            best_model = model.state_dict()
            save_checkpoint(best_model, args, config)
            if getattr(config, "use_wandb", False):
                wandb.save(os.path.join(args.out_dir, "best_model.pt"))
            patience_counter = 0
            print("model saved to", os.path.join(args.out_dir, "best_model.pt"))
        else:
            patience_counter += 1

        if getattr(config, "use_wandb", False):
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "patience_counter": patience_counter,
                }
            )

        print(f"Epoch {epoch+1}/{config.num_epochs}:")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Patience Counter: {patience_counter}/{config.patience}")
        print("-" * 30)

        if patience_counter >= config.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    return best_model


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, np.ndarray | list[str]]]:
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    total_loss = 0.0

    sample_results: dict[str, list] = {
        "sample_ids": [],
        "sample_losses": [],
        "feature_losses": [],
        "predictions": [],
        "targets": [],
    }

    stats_dict = None
    if args.normalize:
        print("calculate dataset mean and std")
        stats_dict = compute_dataset_stats(test_loader)
        with open(os.path.join(args.out_dir, "dataset_stats.pkl"), "wb") as f:
            pickle.dump(stats_dict, f)

    use_amp = bool(args.amp and device.type == "cuda")
    autocast_dtype = get_autocast_dtype(args)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if args.normalize:
                post_emb = standardize_embeddings(
                    batch["embeddings"][1], stats_dict["mean_post"], stats_dict["std_post"]
                ).to(device, non_blocking=True)
                pre_emb = standardize_embeddings(
                    batch["embeddings"][0], stats_dict["mean_pre"], stats_dict["std_pre"]
                ).to(device, non_blocking=True)
            else:
                post_emb = batch["embeddings"][1].to(device, non_blocking=True)
                pre_emb = batch["embeddings"][0].to(device, non_blocking=True)

            sample_ids = batch["sample_id"]

            with autocast(enabled=use_amp, dtype=autocast_dtype, device_type=device.type):
                output = model(post_emb)
                losses = criterion(output, pre_emb)
                sample_losses = losses.mean(dim=2)
            per_sample_loss = sample_losses.mean(dim=1)
            total_loss += per_sample_loss.sum().item()

            sample_results["sample_ids"].extend(sample_ids)
            sample_results["sample_losses"].extend(sample_losses.cpu().numpy())
            sample_results["feature_losses"].extend(losses.cpu().numpy())
            # sample_results["predictions"].extend(output.cpu().numpy())
            # sample_results["targets"].extend(pre_emb.cpu().numpy())

    avg_loss = total_loss / (max(1, len(test_loader.dataset)))

    sample_results["sample_losses"] = np.array(sample_results["sample_losses"])
    sample_results["feature_losses"] = np.array(sample_results["feature_losses"])
    # sample_results["predictions"] = np.array(sample_results["predictions"])
    # sample_results["targets"] = np.array(sample_results["targets"])

    stats = {
        "average_loss": avg_loss,
        "median_loss": float(np.median(sample_results["sample_losses"])),
        "std_loss": float(np.std(sample_results["sample_losses"])),
        "min_loss": float(np.min(sample_results["sample_losses"])),
        "max_loss": float(np.max(sample_results["sample_losses"])),
        "percentile_25": float(np.percentile(sample_results["sample_losses"], 25)),
        "percentile_75": float(np.percentile(sample_results["sample_losses"], 75)),
    }

    return stats, sample_results


def save_eval_outputs(args: argparse.Namespace, stats: dict[str, float], sample_results: dict) -> None:
    print("\nEvaluation Results:")
    print("-" * 50)
    for metric, value in stats.items():
        print(f"{metric}: {value:.6f}")

    # # Keep artifact behavior same as previous script: store stats and losses, not full vectors.
    # del sample_results["predictions"]
    # del sample_results["targets"]

    results = {
        "statistics": stats,
        "sample_results": sample_results,
    }

    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)

    full_results_file = os.path.join(output_dir, "full_evaluation_results.pkl")
    with open(full_results_file, "wb") as f:
        pickle.dump(results, f)

    per_sample_loss = np.mean(sample_results["sample_losses"], axis=1).tolist()
    per_sample_results = pd.DataFrame(
        {
            "sample_id": sample_results["sample_ids"],
            "loss": per_sample_loss,
        }
    )
    csv_path = os.path.join(output_dir, f"{args.embed_model}_per_sample_losses.csv")
    per_sample_results.to_csv(csv_path, index=False)

    print(f"\nFull results saved to {full_results_file}")
    print(f"Per-sample losses saved to {csv_path}")

    plt.figure(figsize=(10, 6))
    plt.hist(per_sample_loss, bins=50)
    plt.title("Distribution of Reconstruction Losses")
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    hist_path = os.path.join(output_dir, f"{args.embed_model}_loss_distribution.png")
    plt.savefig(hist_path)
    plt.close()


def build_config(args: argparse.Namespace) -> SimpleNamespace:
    config = SimpleNamespace()
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.hidden_dim = args.hidden_size
    config.log_interval = args.log_interval
    config.warmup_steps = args.warmup_steps
    config.mixed_precision = args.amp
    config.accumulation_steps = args.accumulation_steps
    config.patience = args.patience
    config.min_delta = args.min_delta
    config.use_wandb = bool(getattr(args, "use_wandb", False))
    return config


def get_autocast_dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--vec_dir", type=str, default="", help="directory containing pre/post_vectors_*.pt")
    parser.add_argument("--pre_dir", type=str, default="", help="optional pre shard directory")
    parser.add_argument("--post_dir", type=str, default="", help="optional post shard directory")
    parser.add_argument("--out_dir", type=str, default="./embed_reconstruction_single_gpu")

    parser.add_argument("--embed_model", type=str, default="llava", choices=["llava", "idefics2", "qwen3.5", "qwen2.5vl"])
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "transformer"])
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--seq_length", type=int, default=576, help="used by llava transformer model")
    parser.add_argument("--hidden_size", type=int, default=2048)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-3)

    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cache_shards", type=int, default=1)
    parser.add_argument("--cache_fp32", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no_strip_cls", action="store_true", help="disable auto strip CLS on square+1 token lengths")

    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--shuffle_mode", type=str, default="shard", choices=["random", "shard"])
    parser.add_argument("--shuffle_within_tag", action="store_true")

    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--eval_split", type=str, default="val", choices=["val", "full"])

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.vec_dir and (not args.pre_dir or not args.post_dir):
        raise ValueError("please provide --vec_dir or both --pre_dir and --post_dir")

    if not args.model_dir:
        args.model_dir = args.out_dir

    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.use_wandb = (not args.eval_only) and wandb is not None
    if not args.eval_only and wandb is None:
        print("WARNING: wandb not installed, continuing without wandb logging")
    if args.use_wandb:
        run_name = (
            f"{args.embed_model}_reconstruction_{args.model_type}"
            f"_bs{args.batch_size}_lr{args.lr}_hidden{args.hidden_size}"
        )
        wandb.init(project="embedding-transformer", name=run_name)

    dataset = ShardPairVectorDataset(
        vec_dir=args.vec_dir or None,
        pre_dir=args.pre_dir or None,
        post_dir=args.post_dir or None,
        limit=args.limit,
        cache_shards=args.cache_shards,
        cache_fp32=args.cache_fp32,
        strip_cls=not args.no_strip_cls,
    )

    total_length = len(dataset)
    train_length = int(args.train_split * total_length)
    val_length = total_length - train_length

    if total_length < 2:
        raise ValueError("dataset must contain at least 2 samples for split-based training/eval")
    if train_length == 0 or val_length == 0:
        raise ValueError(
            f"invalid split: total={total_length}, train={train_length}, val={val_length}. "
            "adjust --train_split or provide more data"
        )

    split_generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_length, val_length], generator=split_generator)

    if args.eval_only:
        eval_dataset = val_dataset if args.eval_split == "val" else dataset
        if args.shuffle_mode == "shard" and args.eval_split == "val":
            eval_subset_items = [dataset.items[i] for i in val_dataset.indices]
            eval_batch_sampler = ShardBatchSampler(
                items=eval_subset_items,
                batch_size=args.batch_size,
                drop_last=False,
                shuffle_tags=False,
                shuffle_within_tag=False,
            )
            eval_loader = DataLoader(
                eval_dataset,
                batch_sampler=eval_batch_sampler,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
        else:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )

        model = build_model(args, device)
        model_path = resolve_model_path(args)
        print("loading model from", model_path)
        model.load_state_dict(load_state_dict(model_path), strict=True)
        print("model loaded")

        stats, sample_results = evaluate_model(model, eval_loader, args, device)
        save_eval_outputs(args, stats, sample_results)
        return

    print("total vectors:", total_length)
    print("train vectors:", train_length)
    print("val vectors:", val_length)

    if args.num_workers > 0 and args.cache_shards > 1:
        print(
            f"[warn] num_workers={args.num_workers} and cache_shards={args.cache_shards}: "
            "each worker has its own dataset cache, RAM usage may increase significantly."
        )

    if args.shuffle_mode == "shard":
        train_subset_items = [dataset.items[i] for i in train_dataset.indices]
        batch_sampler = ShardBatchSampler(
            items=train_subset_items,
            batch_size=args.batch_size,
            drop_last=True,
            shuffle_tags=True,
            shuffle_within_tag=args.shuffle_within_tag,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_subset_items = [dataset.items[i] for i in val_dataset.indices]
        val_batch_sampler = ShardBatchSampler(
            items=val_subset_items,
            batch_size=args.batch_size,
            drop_last=False,
            shuffle_tags=False,
            shuffle_within_tag=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    model = build_model(args, device)
    total_params, trainable_params = get_model_params(model)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    config = build_config(args)
    best_model_state = train_model(model, train_loader, val_loader, config, args, device=device)
    if best_model_state is not None:
        save_checkpoint(best_model_state, args, config)


if __name__ == "__main__":
    main()
