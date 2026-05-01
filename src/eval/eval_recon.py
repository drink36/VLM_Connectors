"""
Compute per-sample MSE and cosine similarity between E_pre_hat and E_pre.

Loads the _norm reconstruction checkpoint, runs inference on all shard folders,
and saves per-shard CSVs plus a combined CSV.

Output columns:
  key, shard, model, mse_pre, cosine_pre

Per-shard output:  data/output/recon_pre_eval/<model>/<shard>.csv
Combined output:   data/output/recon_pre_eval/<model>.csv

Usage:
  python eval_recon_pre.py --embed_model llava
  python eval_recon_pre.py --embed_model idefics2
  python eval_recon_pre.py --embed_model qwen2.5vl
  python eval_recon_pre.py --embed_model qwen3.5
  python eval_recon_pre.py --embed_model llava --shards 00000,00001 --limit 500
"""
import argparse
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_DEFAULTS = {
    "llava":     "data/output/out/out_llava_mlp_norm",
    "idefics2":  "data/output/out/out_idefics2_norm",
    "qwen2.5vl": "data/output/out/out_qwen2.5vl_norm",
    "qwen3.5":   "data/output/out/out_qwen3.5",
}

VEC_BASE = "data/vector"

ALL_SHARDS = [f"{i:05d}" for i in range(14)]


def load_recon_model(model_dir: str, device: torch.device):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train.train_recon import build_model

    ckpt_path = Path(model_dir) / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})

    import argparse as ap
    dummy = ap.Namespace(
        embed_model=meta["embed_model"],
        model_type=meta.get("model_type", "transformer"),
        hidden_size=meta.get("hidden_size", 2048),
        num_layers=meta.get("num_layers", 8),
        num_heads=meta.get("num_heads", 16),
        seq_length=meta.get("seq_length", 576),
    )
    model = build_model(dummy, device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded model from {ckpt_path}  meta={meta}")

    stats = None
    if meta.get("normalize", False):
        stats_path = Path(model_dir) / "dataset_stats.pkl"
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        stats = {k: v.to(device) for k, v in stats.items()}
        print(f"Loaded normalization stats from {stats_path}")

    return model, stats


@torch.no_grad()
def run_shard(model, stats, vec_dir: Path, batch_size: int, device: torch.device,
              limit: int = 0) -> list[dict]:
    import re

    def _numeric_suffix(p):
        m = re.search(r"_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else 10 ** 18

    pre_files  = sorted(vec_dir.glob("pre_vectors_*.pt"),  key=_numeric_suffix)
    post_files = sorted(vec_dir.glob("post_vectors_*.pt"), key=_numeric_suffix)
    pre_map    = {_numeric_suffix(p): p for p in pre_files}
    post_map   = {_numeric_suffix(p): p for p in post_files}
    tags = sorted(set(pre_map) & set(post_map))

    rows = []
    for tag in tags:
        if limit > 0 and len(rows) >= limit:
            break

        pre_obj  = torch.load(pre_map[tag],  map_location="cpu", weights_only=False)
        post_obj = torch.load(post_map[tag], map_location="cpu", weights_only=False)

        keys      = [str(k) for k in pre_obj["keys"]]
        pre_vecs  = pre_obj["vecs"].float()
        post_vecs = post_obj["vecs"].float()

        for start in range(0, len(keys), batch_size):
            if limit > 0 and len(rows) >= limit:
                break

            k_b    = keys[start:start + batch_size]
            pre_b  = pre_vecs[start:start + batch_size].to(device)
            post_b = post_vecs[start:start + batch_size].to(device)

            if stats is not None:
                post_in = (post_b - stats["mean_post"]) / (stats["std_post"] + 1e-8)
                pre_b_norm = (pre_b - stats["mean_pre"]) / (stats["std_pre"] + 1e-8)
            else:
                post_in = post_b
                pre_b_norm = pre_b

            pre_hat_norm = model(post_in)

            # MSE in normalized space (comparable across models)
            mse = (pre_hat_norm - pre_b_norm).pow(2).mean(dim=(1, 2)).cpu().numpy()

            # cosine in original space (denormalize first)
            if stats is not None:
                pre_hat = pre_hat_norm * (stats["std_pre"] + 1e-8) + stats["mean_pre"]
            else:
                pre_hat = pre_hat_norm
            cos = F.cosine_similarity(pre_hat, pre_b, dim=2).mean(dim=1).cpu().numpy()

            for key, m, c in zip(k_b, mse, cos):
                rows.append({"key": key, "mse_pre": float(m), "cosine_pre": float(c)})

    return rows


def main():
    import pandas as pd
    from tqdm import tqdm

    p = argparse.ArgumentParser()
    p.add_argument("--embed_model", required=True,
                   choices=["llava", "idefics2", "qwen2.5vl", "qwen3.5"])
    p.add_argument("--model_dir", default="",
                   help="path to _norm checkpoint dir (auto-detected if empty)")
    p.add_argument("--vec_base", default=VEC_BASE,
                   help="root vector directory (default: data/vector)")
    p.add_argument("--shards", default="",
                   help="comma-separated shards to process (default: all 00000-00013)")
    p.add_argument("--out", default="",
                   help="combined output CSV (default: data/output/recon_pre_eval/<model>.csv)")
    p.add_argument("--out_dir", default="",
                   help="per-shard output dir (default: data/output/recon_pre_eval/<model>/)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--limit", type=int, default=0,
                   help="max samples per shard (0 = all)")
    p.add_argument("--resume", action="store_true",
                   help="skip shards whose per-shard CSV already exists")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir  = args.model_dir or MODEL_DEFAULTS[args.embed_model]
    out_path   = args.out     or f"data/output/recon_pre_eval/{args.embed_model}.csv"
    shard_dir  = Path(args.out_dir or f"data/output/recon_pre_eval/{args.embed_model}")
    shards     = [s.strip() for s in args.shards.split(",")] if args.shards else ALL_SHARDS

    shard_dir.mkdir(parents=True, exist_ok=True)

    model, stats = load_recon_model(model_dir, device)

    all_rows = []
    for shard in tqdm(shards, desc="Shards"):
        shard_csv = shard_dir / f"{shard}.csv"

        if args.resume and shard_csv.exists():
            df_s = pd.read_csv(shard_csv)
            all_rows.extend(df_s.to_dict("records"))
            print(f"  {shard}: {len(df_s)} samples  [resumed from {shard_csv}]")
            continue

        vec_dir = Path(args.vec_base) / args.embed_model / shard
        if not vec_dir.exists():
            print(f"  [skip] {vec_dir} not found")
            continue

        rows = run_shard(model, stats, vec_dir, args.batch_size, device, limit=args.limit)
        for r in rows:
            r["shard"] = shard
            r["model"] = args.embed_model

        # save per-shard CSV immediately
        df_s = pd.DataFrame(rows)[["key", "shard", "model", "mse_pre", "cosine_pre"]]
        df_s.to_csv(shard_csv, index=False, float_format="%.6f")

        all_rows.extend(rows)
        print(f"  {shard}: {len(rows)} samples  -> {shard_csv}")

    if not all_rows:
        print("No data collected.")
        return

    df = pd.DataFrame(all_rows)
    df["model"] = args.embed_model
    df = df[["key", "shard", "model", "mse_pre", "cosine_pre"]]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\nSaved {len(df)} rows -> {out_path}")

    print(f"\n{'='*50}")
    print(f"  {args.embed_model}  (n={len(df)}, shards processed={len(shards)})")
    print(f"{'='*50}")
    for col, label in [("mse_pre", "MSE(E_pre_hat, E_pre)"),
                       ("cosine_pre", "Cosine(E_pre_hat, E_pre)")]:
        s = df[col]
        q25, q75 = s.quantile(0.25), s.quantile(0.75)
        print(f"  {label}")
        print(f"    mean={s.mean():.4f}  median={s.median():.4f}  "
              f"std={s.std():.4f}  IQR={q25:.4f}–{q75:.4f}")


if __name__ == "__main__":
    main()
