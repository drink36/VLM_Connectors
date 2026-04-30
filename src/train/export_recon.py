"""
Export reconstructed vectors via two-step round-trip:
  Step 1: E_post  -> recon_model  -> E_pre_hat
  Step 2: E_pre_hat -> VLM connector -> E_post_hat

Output per shard:
  <out_base>/<embed_model>/<shard>/pre_vectors_*.pt   <- E_pre_hat  (step 1)
  <out_base>/<embed_model>/<shard>/post_vectors_*.pt  <- E_post_hat (step 2)

Usage:
  python export_recon_vectors.py --embed_model llava
  python export_recon_vectors.py --embed_model llava --shards 00000,00001
  python export_recon_vectors.py --embed_model qwen3.5 --shards 00000
"""
import argparse
import pickle
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

ALL_SHARDS = [f"{i:05d}" for i in range(14)]

MODEL_DEFAULTS = {
    "llava":     "data/output/out/out_llava_mlp_norm",
    "idefics2":  "data/output/out/out_idefics2_norm",
    "qwen2.5vl": "data/output/out/out_qwen2.5vl_norm",
    "qwen3.5":   "data/output/out/out_qwen3.5",
}

VLM_IDS = {
    "llava":     "llava-hf/llava-1.5-7b-hf",
    "idefics2":  "HuggingFaceM4/idefics2-8b",
    "qwen2.5vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3.5":   "Qwen/Qwen3.5-9B",
}


# ── recon model ───────────────────────────────────────────────────────────────

def load_recon_model(model_path, stats_path, device):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from reconstruct_embeddings import build_model
    import argparse as ap

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})

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
    model.eval().to(device)
    print(f"Loaded recon model: {meta}")

    stats = None
    if meta.get("normalize", False):
        if stats_path is None:
            raise ValueError("--stats_path required when model was trained with --normalize")
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        for k in stats:
            stats[k] = stats[k].to(device)
        print(f"Loaded stats from {stats_path}")

    return model, stats


# ── VLM connector ─────────────────────────────────────────────────────────────

def load_connector(embed_model: str, device: torch.device) -> nn.Module:
    """Load the full VLM, extract and return only the connector module."""
    from transformers import AutoProcessor

    model_id = VLM_IDS[embed_model]
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading VLM connector from {model_id} ...")
    if embed_model == "llava":
        from transformers import LlavaForConditionalGeneration
        vlm = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    elif embed_model == "idefics2":
        from transformers import Idefics2ForConditionalGeneration
        vlm = Idefics2ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True)
    else:
        from transformers import AutoModelForImageTextToText
        vlm = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True)

    mods = dict(vlm.named_modules())
    if embed_model == "llava":
        connector = mods.get("multi_modal_projector") or mods.get("model.multi_modal_projector")
    elif embed_model == "idefics2":
        connector = mods.get("model.connector") or mods.get("connector")
    else:  # qwen2.5vl / qwen3.5
        connector = mods.get("visual.merger") or mods.get("model.visual.merger")

    if connector is None:
        raise RuntimeError(f"Could not find connector module for {embed_model}")

    connector = connector.to(device).eval()
    del vlm  # free rest of VLM memory
    torch.cuda.empty_cache()
    print(f"Connector loaded: {type(connector).__name__}")
    return connector


def _tensorize(obj) -> torch.Tensor:
    """Extract first tensor from connector output (may be tuple)."""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)):
        for x in obj:
            if isinstance(x, torch.Tensor):
                return x
    raise RuntimeError(f"Cannot extract tensor from connector output: {type(obj)}")


# ── per-shard export ──────────────────────────────────────────────────────────

@torch.no_grad()
def export_shard(recon_model, stats, connector, shard_dir: Path, out_dir: Path,
                 batch_size: int, device: torch.device):
    post_files = sorted(shard_dir.glob("post_vectors_*.pt"),
                        key=lambda f: int(f.stem.split("_")[-1]))
    if not post_files:
        print(f"  [warn] no post_vectors_*.pt in {shard_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    con_dtype = next(connector.parameters()).dtype

    for pt_file in tqdm(post_files, desc=f"  {shard_dir.name}", unit="file"):
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        post_vecs = data["vecs"].float()   # [N, T_post, D_post]
        keys = data["keys"]
        caps = data.get("caps", [""] * len(keys))

        N = post_vecs.shape[0]
        pre_hat_list  = []
        post_hat_list = []

        for start in range(0, N, batch_size):
            post_b = post_vecs[start:start + batch_size].to(device)  # [B, T, D]

            # ── step 1: E_post → recon → E_pre_hat ──────────────────────────
            post_in = post_b
            if stats is not None:
                post_in = (post_b - stats["mean_post"]) / (stats["std_post"] + 1e-8)

            pre_hat = recon_model(post_in)  # [B, T_pre, D_pre]

            if stats is not None:
                pre_hat = pre_hat * (stats["std_pre"] + 1e-8) + stats["mean_pre"]

            pre_hat_list.append(pre_hat.cpu())

            # ── step 2: E_pre_hat → connector → E_post_hat ──────────────────
            pre_in = pre_hat.to(con_dtype)
            B, T, _ = pre_in.shape
            try:
                post_hat = _tensorize(connector(pre_in))
            except TypeError:
                # Idefics2Connector requires attention_mask
                mask = torch.ones(B, T, dtype=torch.bool, device=device)
                post_hat = _tensorize(connector(pre_in, attention_mask=mask))
            post_hat_list.append(post_hat.float().cpu())

        recon_pre  = torch.cat(pre_hat_list,  dim=0)  # [N, T_pre,  D_pre]
        recon_post = torch.cat(post_hat_list, dim=0)  # [N, T_post, D_post]

        tag = pt_file.stem.split("_")[-1]

        torch.save({"keys": keys, "vecs": recon_pre,  "caps": caps},
                   out_dir / f"pre_vectors_{tag}.pt")
        torch.save({"keys": keys, "vecs": recon_post, "caps": caps},
                   out_dir / f"post_vectors_{tag}.pt")

    print(f"  -> saved pre+post to {out_dir}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embed_model", required=True,
                   choices=["llava", "idefics2", "qwen2.5vl", "qwen3.5"])
    p.add_argument("--vec_base", default="data/vector")
    p.add_argument("--shards", default="",
                   help="comma-separated shard folders (default: all 00000-00013)")
    p.add_argument("--model_path", default="",
                   help="path to best_model.pt (auto-detected if empty)")
    p.add_argument("--stats_path", default="",
                   help="path to dataset_stats.pkl (auto-detected if empty)")
    p.add_argument("--out_base", default="data/vector_recon",
                   help="output root; shards saved to out_base/<embed_model>/<shard>/")
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.model_path or f"{MODEL_DEFAULTS[args.embed_model]}/best_model.pt"
    stats_path = args.stats_path or f"{MODEL_DEFAULTS[args.embed_model]}/dataset_stats.pkl"

    recon_model, stats = load_recon_model(model_path, stats_path, device)
    connector = load_connector(args.embed_model, device)

    shards = [s.strip() for s in args.shards.split(",")] if args.shards else ALL_SHARDS
    for shard in shards:
        shard_dir = Path(args.vec_base) / args.embed_model / shard
        out_dir   = Path(args.out_base) / args.embed_model / shard
        if not shard_dir.exists():
            print(f"[skip] {shard_dir} not found")
            continue
        print(f"Exporting shard {shard} -> {out_dir}")
        export_shard(recon_model, stats, connector, shard_dir, out_dir,
                     args.batch_size, device)

    print("Done.")


if __name__ == "__main__":
    main()
