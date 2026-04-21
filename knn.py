# knn_overlap_torch.py
import argparse, json, re, gc
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _find_shard_folders(root: Path, prefix: str) -> list[Path]:
    """Return sorted list of folders that contain {prefix}_vectors_*.pt files."""
    folders = sorted([
        d for d in root.iterdir()
        if d.is_dir() and list(d.glob(f"{prefix}_vectors_*.pt"))
    ])
    return folders


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pre_pt", type=str, required=True,
                   help="pre .pt file, shard dir, or parent dir containing shard subdirs (e.g. data/vector/llava)")
    p.add_argument("--post_pt", type=str, required=True,
                   help="post .pt file, shard dir, or parent dir containing shard subdirs")
    p.add_argument("--out_dir", type=str, default="knn_overlap_out")
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2"])

    # safer defaults
    p.add_argument("--q_chunk", type=int, default=256, help="query chunk size")
    p.add_argument("--g_chunk", type=int, default=4096, help="gallery chunk size")
    p.add_argument("--pool", type=str, default="mean", choices=["mean", "cls", "none"], help="pooling for [N,T,D] vecs")

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda"])
    p.add_argument("--safe_mode", action="store_true", help="auto reduce chunk on CUDA OOM")
    p.add_argument("--min_q_chunk", type=int, default=8)
    p.add_argument("--min_g_chunk", type=int, default=64)
    p.add_argument("--sv_topn", type=int, default=128, help="store only top-N singular values in json")
    p.add_argument("--ref", type=str, action="append", default=[],
                   metavar="LABEL:PATH",
                   help="reference dir with label, e.g. --ref clip:data/clip_ref/00000 --ref dino:data/dino_ref/00000 (repeatable)")
    # legacy single-ref args kept for backward compatibility
    p.add_argument("--ref_pt", type=str, default="",
                   help="(legacy) path to first reference dir; use --ref label:path instead")
    p.add_argument("--ref_pt2", type=str, default="",
                   help="(legacy) path to second reference dir; use --ref label:path instead")
    p.add_argument("--ref_label", type=str, default="clip",
                   help="(legacy) label for --ref_pt")
    p.add_argument("--ref2_label", type=str, default="dino",
                   help="(legacy) label for --ref_pt2")
    p.add_argument("--svd_per_image_csv", action="store_true",
                   help="write per-image SVD scalars to svd_per_image.csv for correlation analysis")
    return p.parse_args()


def _numeric_suffix(p: Path):
    m = re.search(r"_(\d+)\.pt$", p.name)
    return int(m.group(1)) if m else 10**18


def _torch_load_device(fp: Path, device: str = "cuda"):
    return torch.load(fp, map_location=device)


def _resolve_files(path_str: str, prefix: str):
    path = Path(path_str)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob(f"{prefix}_vectors_*.pt"), key=_numeric_suffix)
        if not files:
            maybe = path / f"{prefix}_vectors.pt"
            if maybe.exists():
                files = [maybe]
        if files:
            return files
    raise FileNotFoundError(f"No {prefix} vector files under: {path}")


def _pool_if_needed(vecs: torch.Tensor, pool: str):
    # expect [N,D] or [N,T,D]
    if vecs.dim() == 2:
        return vecs.float()
    if vecs.dim() == 3:
        if pool == "mean":
            return vecs.mean(dim=1).float()
        if pool == "cls":
            return vecs[:, 0, :].float()
        if pool == "none":
            return vecs.float()   # keep [N,T,D] for token-level analysis
    raise ValueError(f"Unsupported vec shape: {tuple(vecs.shape)}")


def _build_key_lut(path_str: str, prefix: str, load_device: str = "cuda"):
    """
    Build key lookup without concatenating all vecs:
    lut[key] = (file_path, row_idx_in_file)
    """
    files = _resolve_files(path_str, prefix)
    lut = {}
    for fp in tqdm(files, desc=f"index {prefix} shards", unit="file"):
        obj = _torch_load_device(fp, load_device)
        if "keys" not in obj or "vecs" not in obj:
            raise RuntimeError(f"Bad format: {fp}")
        for i, k in enumerate(obj["keys"]):
            lut[k] = (fp, i)
        del obj
        gc.collect()
    return lut


def _materialize_aligned(common_keys, lut, pool="mean", load_device="cuda", out_device="cuda"):
    """
    Materialize aligned tensor by loading one source file at a time.
    Returns:
      - [Nc, D]   if pooled
      - [Nc, T, D] if pool == "none"
    """
    by_file = {}
    for out_i, k in enumerate(common_keys):
        fp, local_i = lut[k]
        by_file.setdefault(fp, []).append((out_i, local_i))

    out = None
    for fp, pairs in tqdm(by_file.items(), total=len(by_file), desc="materialize aligned", unit="file"):
        obj = _torch_load_device(fp, load_device)
        vecs = _pool_if_needed(obj["vecs"], pool).contiguous()

        if out is None:
            if vecs.dim() == 2:
                out = torch.empty(
                    (len(common_keys), vecs.size(1)),
                    dtype=torch.float32,
                    device=out_device
                )
            elif vecs.dim() == 3:
                out = torch.empty(
                    (len(common_keys), vecs.size(1), vecs.size(2)),
                    dtype=torch.float32,
                    device=out_device
                )
            else:
                raise ValueError(f"Unsupported vec shape after pooling: {tuple(vecs.shape)}")

        out_pos = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=out_device)
        in_pos = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=out_device)
        out.index_copy_(0, out_pos, vecs.index_select(0, in_pos))

        del obj, vecs, out_pos, in_pos
        gc.collect()

    if out is None:
        raise RuntimeError("Failed to materialize aligned vectors.")
    return out

@torch.inference_mode()
def compute_svd_stats_per_image(vecs_3d, topn=128, center=True):
    if vecs_3d.dim() != 3:
        raise ValueError(f"Expected [N,T,D], got {tuple(vecs_3d.shape)}")

    out = []
    for i in tqdm(range(vecs_3d.size(0)), desc="svd per-image", unit="img"):
        out.append(
            singular_spectrum_and_effective_rank(
                vecs_3d[i], topn=topn, center=center
            )
        )
    return out


@torch.inference_mode()
def compute_svd_stats_streaming(common_keys, lut, topn=128, center=True, load_device="cuda"):
    """
    Stream token-level vectors shard-by-shard to avoid materializing full [N,T,D] on GPU.
    Returns one stats dict per image in common_keys order.
    """
    by_file = {}
    for out_i, k in enumerate(common_keys):
        fp, local_i = lut[k]
        by_file.setdefault(fp, []).append((out_i, local_i))

    stats = [None] * len(common_keys)
    pbar = tqdm(total=len(common_keys), desc="svd per-image", unit="img")

    for fp, pairs in tqdm(by_file.items(), total=len(by_file), desc="stream token shards", unit="file"):
        obj = _torch_load_device(fp, load_device)
        vecs = _pool_if_needed(obj["vecs"], pool="none").contiguous()  # [N,T,D]

        for out_i, local_i in pairs:
            stats[out_i] = singular_spectrum_and_effective_rank(
                vecs[local_i], topn=topn, center=center
            )
            pbar.update(1)

        del obj, vecs
        gc.collect()
        if load_device == "cuda":
            torch.cuda.empty_cache()

    pbar.close()
    if any(x is None for x in stats):
        raise RuntimeError("Failed to compute all per-image SVD stats.")
    return stats


def summarize_spectrum_list(stats_list):
    scalar_keys = [
        "effective_rank",
        "effective_rank_normalized",
        "top1_energy_ratio",
        "top5_energy_ratio",
        "top10_energy_ratio",
        "k80_energy",
        "k90_energy",
    ]

    summary = {}
    for key in scalar_keys:
        vals = torch.tensor([x[key] for x in stats_list], dtype=torch.float32)
        summary[f"{key}_mean"] = float(vals.mean().item())
        summary[f"{key}_median"] = float(vals.median().item())
        summary[f"{key}_std"] = float(vals.std(unbiased=False).item())

    non_empty_sv = [x for x in stats_list if len(x["singular_values_topn"]) > 0]
    non_empty_en = [x for x in stats_list if len(x["energy_spectrum_topn"]) > 0]
    min_sv_len = min((len(x["singular_values_topn"]) for x in non_empty_sv), default=0)
    min_en_len = min((len(x["energy_spectrum_topn"]) for x in non_empty_en), default=0)

    if min_sv_len > 0:
        sv = torch.tensor(
            [x["singular_values_topn"][:min_sv_len] for x in non_empty_sv],
            dtype=torch.float32
        )
        summary["mean_singular_values_topn"] = sv.mean(dim=0).tolist()

    if min_en_len > 0:
        en = torch.tensor(
            [x["energy_spectrum_topn"][:min_en_len] for x in non_empty_en],
            dtype=torch.float32
        )
        summary["mean_energy_spectrum_topn"] = en.mean(dim=0).tolist()

    return summary

def build_common(pre_pt, post_pt, load_device="cuda"):
    pre_lut = _build_key_lut(pre_pt, "pre", load_device=load_device)
    post_lut = _build_key_lut(post_pt, "post", load_device=load_device)
    common = sorted(set(pre_lut) & set(post_lut))
    if not common:
        raise RuntimeError("No overlapping keys.")
    return common, pre_lut, post_lut


def _topk_ids_once(vecs, topk, metric="cosine", q_chunk=64, g_chunk=256, device="cuda"):
    X = vecs.contiguous().to(device, non_blocking=True)
    N, D = X.shape
    if N <= 1:
        return torch.empty((N, 0), dtype=torch.long, device=device)

    k_eff = min(topk, N - 1)
    out = torch.empty((N, k_eff), dtype=torch.long, device=device)

    for qs in tqdm(range(0, N, q_chunk), desc=f"topk ({metric})", unit="chunk"):
        qe = min(N, qs + q_chunk)
        Q = X[qs:qe].to(device, non_blocking=True)
        B = Q.size(0)

        if metric == "cosine":
            Q = F.normalize(Q, dim=1)

        best_scores = torch.full((B, k_eff), -float("inf"), device=device)
        best_ids = torch.full((B, k_eff), -1, dtype=torch.long, device=device)

        q_ids = torch.arange(qs, qe, device=device).unsqueeze(1)

        for gs in range(0, N, g_chunk):
            ge = min(N, gs + g_chunk)
            G = X[gs:ge].to(device, non_blocking=True)

            if metric == "cosine":
                G = F.normalize(G, dim=1)
                scores = Q @ G.T
            else:
                q2 = (Q * Q).sum(dim=1, keepdim=True)
                g2 = (G * G).sum(dim=1).unsqueeze(0)
                scores = -(q2 + g2 - 2.0 * (Q @ G.T))

            g_ids = torch.arange(gs, ge, device=device).unsqueeze(0)
            scores = scores.masked_fill(q_ids == g_ids, -float("inf"))

            kk = min(k_eff, ge - gs)
            chunk_vals, chunk_idx_local = torch.topk(scores, k=kk, dim=1, largest=True)
            chunk_idx = chunk_idx_local + gs

            merged_vals = torch.cat([best_scores, chunk_vals], dim=1)
            merged_idx = torch.cat([best_ids, chunk_idx], dim=1)

            new_vals, sel = torch.topk(merged_vals, k=k_eff, dim=1, largest=True)
            new_idx = torch.gather(merged_idx, 1, sel)

            best_scores, best_ids = new_vals, new_idx

            del G, scores, g_ids, chunk_vals, chunk_idx_local, chunk_idx, merged_vals, merged_idx, new_vals, sel, new_idx
            if device == "cuda":
                torch.cuda.empty_cache()

        out[qs:qe] = best_ids
        del Q, q_ids, best_scores, best_ids
        if device == "cuda":
            torch.cuda.empty_cache()

    return out


def _is_cuda_oom(err: RuntimeError) -> bool:
    s = str(err).lower()
    return ("out of memory" in s) or ("cuda error: out of memory" in s)


@torch.inference_mode()
def topk_ids(
    vecs, topk, metric="cosine", q_chunk=64, g_chunk=256, device=None,
    safe_mode=False, min_q_chunk=8, min_g_chunk=64
):
    device = device or "cuda"
    if device == "auto":
        device = "cuda"

    if device != "cuda":
        raise ValueError("This script is configured for CUDA-only execution.")

    if not safe_mode:
        return _topk_ids_once(vecs, topk, metric, q_chunk, g_chunk, device)

    cur_q, cur_g = q_chunk, g_chunk
    while True:
        try:
            return _topk_ids_once(vecs, topk, metric, cur_q, cur_g, "cuda")
        except RuntimeError as e:
            if not _is_cuda_oom(e):
                raise
            torch.cuda.empty_cache()

            if cur_g > min_g_chunk:
                cur_g = max(min_g_chunk, cur_g // 2)
                print(f"[safe_mode] CUDA OOM -> reduce g_chunk to {cur_g}")
                continue
            if cur_q > min_q_chunk:
                cur_q = max(min_q_chunk, cur_q // 2)
                print(f"[safe_mode] CUDA OOM -> reduce q_chunk to {cur_q}")
                continue

            raise


def overlap(pre_I, post_I, k):
    N = pre_I.size(0)
    out = torch.empty(N, dtype=torch.float32, device=pre_I.device)
    for i in tqdm(range(N), desc=f"overlap@{k}", unit="sample"):
        a = pre_I[i, :k]
        b = post_I[i, :k]
        inter_cnt = torch.isin(a, b).sum().item()
        out[i] = inter_cnt / float(k)
    return out

def mean_rank_displacement(pre_I, post_I, k):
    N = pre_I.size(0)
    disp = torch.empty(N, dtype=torch.float32, device=pre_I.device)
    for i in tqdm(range(N), desc=f"rank-disp@{k}", unit="sample"):
        pre = pre_I[i, :k]
        post = post_I[i, :k]
        rank = {int(post[j]): j for j in range(k)}
        d = 0.0
        for j in range(k):
            pid = int(pre[j])
            d += abs(j - rank[pid]) if pid in rank else k
        disp[i] = d / k
    return disp


@torch.inference_mode()
def within_image_cosine_stats(x, sample_pairs=None, seed=42, eps=1e-12):
    """
    x: [T, D] for one image
    Returns within-image pairwise cosine statistics over token pairs.
    """
    z = F.normalize(x.float(), dim=1, eps=eps)
    T = z.size(0)

    if T < 2:
        return {
            "mean_pairwise_cosine": 0.0,
            "std_pairwise_cosine": 0.0,
            "median_pairwise_cosine": 0.0,
            "num_pairs_used": 0,
        }

    if sample_pairs is None:
        # full off-diagonal cosine matrix
        C = z @ z.T
        mask = ~torch.eye(T, dtype=torch.bool, device=z.device)
        vals = C[mask]
    else:
        g = torch.Generator(device=z.device)
        g.manual_seed(seed)

        i = torch.randint(0, T, (sample_pairs,), generator=g, device=z.device)
        j = torch.randint(0, T, (sample_pairs,), generator=g, device=z.device)
        keep = i != j
        i = i[keep]
        j = j[keep]
        vals = (z[i] * z[j]).sum(dim=1)

    return {
        "mean_pairwise_cosine": float(vals.mean().item()),
        "std_pairwise_cosine": float(vals.std(unbiased=False).item()),
        "median_pairwise_cosine": float(vals.median().item()),
        "num_pairs_used": int(vals.numel()),
    }

def summarize_cosine_list(stats_list):
    keys = [
        "mean_pairwise_cosine",
        "std_pairwise_cosine",
        "median_pairwise_cosine",
    ]
    out = {}
    for k in keys:
        vals = torch.tensor([x[k] for x in stats_list], dtype=torch.float32)
        out[f"{k}_mean"] = float(vals.mean().item())
        out[f"{k}_median"] = float(vals.median().item())
        out[f"{k}_std"] = float(vals.std(unbiased=False).item())
    return out


@torch.inference_mode()
def compute_cosine_stats_streaming(common_keys, lut, load_device="cuda"):
    by_file = {}
    for out_i, k in enumerate(common_keys):
        fp, local_i = lut[k]
        by_file.setdefault(fp, []).append((out_i, local_i))

    stats = [None] * len(common_keys)
    pbar = tqdm(total=len(common_keys), desc="cosine per-image", unit="img")

    for fp, pairs in tqdm(by_file.items(), desc="stream cosine shards", unit="file"):
        obj = torch.load(fp, map_location=load_device)
        vecs = _pool_if_needed(obj["vecs"], pool="none").contiguous()  # [N,T,D]

        for out_i, local_i in pairs:
            stats[out_i] = within_image_cosine_stats(vecs[local_i])
            pbar.update(1)

        del obj, vecs
        gc.collect()
        if load_device == "cuda":
            torch.cuda.empty_cache()

    pbar.close()
    return stats

@torch.inference_mode()
def singular_spectrum_and_effective_rank(x, topn=128, center=True, eps=1e-12):
    z = x.float()
    if z.dim() != 2:
        raise ValueError(f"Expected [T,D], got {tuple(z.shape)}")

    # Some samples can have zero visual tokens after preprocessing.
    if z.size(0) == 0 or z.size(1) == 0:
        return {
            "effective_rank": 0.0,
            "effective_rank_normalized": 0.0,
            "top1_energy_ratio": 0.0,
            "top5_energy_ratio": 0.0,
            "top10_energy_ratio": 0.0,
            "k80_energy": 0,
            "k90_energy": 0,
            "singular_values_topn": [],
            "energy_spectrum_topn": [],
        }

    if center:
        z = z - z.mean(dim=0, keepdim=True)

    s = torch.linalg.svdvals(z)  # shape: [r], sorted descending
    if s.numel() == 0:
        return {
            "effective_rank": 0.0,
            "effective_rank_normalized": 0.0,
            "top1_energy_ratio": 0.0,
            "top5_energy_ratio": 0.0,
            "top10_energy_ratio": 0.0,
            "k80_energy": 0,
            "k90_energy": 0,
            "singular_values_topn": [],
            "energy_spectrum_topn": [],
        }

    energy = s.pow(2)
    total_energy = energy.sum() + eps

    # normalized energy spectrum
    p = energy / total_energy

    entropy = -(p * (p + eps).log()).sum()
    effective_rank = torch.exp(entropy)

    max_rank = s.numel()
    effective_rank_normalized = effective_rank / max_rank

    topn = min(topn, max_rank)

    cumsum_energy = torch.cumsum(p, dim=0)
    idx80 = (cumsum_energy >= 0.80).nonzero(as_tuple=True)[0]
    idx90 = (cumsum_energy >= 0.90).nonzero(as_tuple=True)[0]
    k80 = int(idx80[0].item() + 1) if idx80.numel() > 0 else int(max_rank)
    k90 = int(idx90[0].item() + 1) if idx90.numel() > 0 else int(max_rank)

    return {
        "effective_rank": float(effective_rank.item()),
        "effective_rank_normalized": float(effective_rank_normalized.item()),
        "top1_energy_ratio": float(p[:1].sum().item()),
        "top5_energy_ratio": float(p[:5].sum().item()),
        "top10_energy_ratio": float(p[:10].sum().item()),
        "k80_energy": k80,
        "k90_energy": k90,
        "singular_values_topn": s[:topn].detach().cpu().tolist(),
        "energy_spectrum_topn": p[:topn].detach().cpu().tolist(),
    }

def _run_ref_knor(label, ref_pt, keys, pre_lut, post_lut, args, device, safe_mode, results):
    """
    Run 3-way KNOR for one reference space (e.g. CLIP or DINOv2).

    Results are stored under:
      results["{label}_pre_top{k}"]  — KNOR(ref→pre)
      results["{label}_post_top{k}"] — KNOR(ref→post)
      results["global"]["{label}_analysis"] — pairwise correlations and n_common

    label: short string identifying this reference (e.g. "clip", "dino")
    ref_pt: directory containing ref_vectors_*.pt
    """
    ref_lut = _build_key_lut(ref_pt, "ref", load_device=device)
    common_3way = sorted(set(keys) & set(ref_lut))
    n_drop = len(keys) - len(common_3way)
    print(f"\n[{label}] 3-way intersection: {len(common_3way)} keys (dropped {n_drop} not in ref)")

    if len(common_3way) < 2:
        print(f"[{label}] Too few common keys; skipping ref analysis")
        return

    ref_vec = _materialize_aligned(
        common_3way, ref_lut, pool=args.pool, load_device=device, out_device=device
    )
    pre_vec_3 = _materialize_aligned(
        common_3way, pre_lut, pool=args.pool, load_device=device, out_device=device
    )
    post_vec_3 = _materialize_aligned(
        common_3way, post_lut, pool=args.pool, load_device=device, out_device=device
    )

    ref_I = topk_ids(
        ref_vec, args.topk, metric=args.metric,
        q_chunk=args.q_chunk, g_chunk=args.g_chunk, device=device,
        safe_mode=safe_mode, min_q_chunk=args.min_q_chunk, min_g_chunk=args.min_g_chunk,
    )
    pre_I_3 = topk_ids(
        pre_vec_3, args.topk, metric=args.metric,
        q_chunk=args.q_chunk, g_chunk=args.g_chunk, device=device,
        safe_mode=safe_mode, min_q_chunk=args.min_q_chunk, min_g_chunk=args.min_g_chunk,
    )
    post_I_3 = topk_ids(
        post_vec_3, args.topk, metric=args.metric,
        q_chunk=args.q_chunk, g_chunk=args.g_chunk, device=device,
        safe_mode=safe_mode, min_q_chunk=args.min_q_chunk, min_g_chunk=args.min_g_chunk,
    )

    ps_ref_pre = pairwise_corr(ref_vec, pre_vec_3, metric=args.metric)
    ps_ref_post = pairwise_corr(ref_vec, post_vec_3, metric=args.metric)

    results["global"][f"{label}_analysis"] = {
        "n_common_3way": len(common_3way),
        "pairwise_corr_ref_pre": ps_ref_pre,
        "pairwise_corr_ref_post": ps_ref_post,
    }

    per_image_ref: dict[str, dict] = {k: {} for k in common_3way}

    for k in [10, 50, 100]:
        if ref_I.size(1) >= k and pre_I_3.size(1) >= k:
            ov_rp = overlap(ref_I, pre_I_3, k)
            md_rp = mean_rank_displacement(ref_I, pre_I_3, k)
            results[f"{label}_pre_top{k}"] = {
                "mean": float(ov_rp.mean().item()),
                "median": float(ov_rp.median().item()),
                "std": float(ov_rp.std(unbiased=False).item()),
                "mean_rank_displacement": float(md_rp.mean().item()),
                "pairwise_corr_ref_pre": ps_ref_pre,
            }
            for i, key in enumerate(common_3way):
                per_image_ref[key][f"{label}_pre_knor{k}"] = float(ov_rp[i].item())
        if ref_I.size(1) >= k and post_I_3.size(1) >= k:
            ov_rq = overlap(ref_I, post_I_3, k)
            md_rq = mean_rank_displacement(ref_I, post_I_3, k)
            results[f"{label}_post_top{k}"] = {
                "mean": float(ov_rq.mean().item()),
                "median": float(ov_rq.median().item()),
                "std": float(ov_rq.std(unbiased=False).item()),
                "mean_rank_displacement": float(md_rq.mean().item()),
                "pairwise_corr_ref_post": ps_ref_post,
            }
            for i, key in enumerate(common_3way):
                per_image_ref[key][f"{label}_post_knor{k}"] = float(ov_rq[i].item())

    results.setdefault("_per_image_ref", {})
    for key, scores in per_image_ref.items():
        results["_per_image_ref"].setdefault(key, {}).update(scores)

    del ref_vec, pre_vec_3, post_vec_3, ref_I, pre_I_3, post_I_3
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[{label}] 3-way KNOR ({label}=ref, pre=pre-connector, post=post-connector):")
    for k in [10, 50, 100]:
        rp = results.get(f"{label}_pre_top{k}", {})
        rq = results.get(f"{label}_post_top{k}", {})
        pp = results.get(f"top{k}", {})
        if rp and rq and pp:
            print(
                f"  k={k:3d}: KNOR({label}→pre)={rp['mean']:.4f} | "
                f"KNOR({label}→post)={rq['mean']:.4f} | "
                f"KNOR(pre→post)={pp['mean']:.4f}"
            )


def _run_one_shard(args, pre_pt: str, post_pt: str, ref_list: list, out_dir: str):
    """Run KNN analysis on a single shard folder. Mutates args temporarily."""
    _orig_pre, _orig_post, _orig_out, _orig_ref = args.pre_pt, args.post_pt, args.out_dir, args.ref
    args.pre_pt = pre_pt
    args.post_pt = post_pt
    args.out_dir = out_dir
    args.ref = ref_list
    try:
        _main_single(args)
    finally:
        args.pre_pt, args.post_pt, args.out_dir, args.ref = _orig_pre, _orig_post, _orig_out, _orig_ref


def main():
    args = parse_args()

    pre_root = Path(args.pre_pt)
    post_root = Path(args.post_pt)

    # Detect if pre_pt is a parent directory of shard subdirs
    pre_shard_dirs = _find_shard_folders(pre_root, "pre") if pre_root.is_dir() else []
    post_shard_dirs = _find_shard_folders(post_root, "post") if post_root.is_dir() else []

    is_multi = bool(pre_shard_dirs and post_shard_dirs)

    if is_multi:
        # Match shard folders by name
        pre_map = {d.name: d for d in pre_shard_dirs}
        post_map = {d.name: d for d in post_shard_dirs}
        common_shards = sorted(set(pre_map) & set(post_map))
        print(f"Multi-shard mode: {len(common_shards)} shards found under {pre_root}")

        base_out = Path(args.out_dir)
        base_ref_list = []
        for entry in args.ref:
            if ":" not in entry:
                raise ValueError(f"--ref must be label:path, got: {entry!r}")
            label, path = entry.split(":", 1)
            base_ref_list.append((label.strip(), Path(path.strip())))

        for shard in common_shards:
            print(f"\n{'='*60}\nProcessing shard: {shard}\n{'='*60}")
            shard_ref_list = []
            for label, ref_root_path in base_ref_list:
                shard_ref_dir = ref_root_path / shard
                if shard_ref_dir.exists():
                    shard_ref_list.append(f"{label}:{shard_ref_dir}")
                else:
                    print(f"[warn] ref {label} has no folder for shard {shard}, skipping")
            _run_one_shard(
                args,
                pre_pt=str(pre_map[shard]),
                post_pt=str(post_map[shard]),
                ref_list=shard_ref_list,
                out_dir=str(base_out / shard),
            )
        return

    # Single shard / file path — original behavior
    _main_single(args)


def _main_single(args):
    pre_files = _resolve_files(args.pre_pt, "pre")
    post_files = _resolve_files(args.post_pt, "post")
    if args.device == "auto":
        device = "cuda"
    else:
        device = args.device

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but this script now runs in CUDA-only mode.")

    pre_raw0 = _torch_load_device(pre_files[0], device=device)["vecs"]
    post_raw0 = _torch_load_device(post_files[0], device=device)["vecs"]
    print(f"[raw shard] pre vecs shape: {tuple(pre_raw0.shape)}")
    print(f"[raw shard] post vecs shape: {tuple(post_raw0.shape)}")

    if args.pool == "none":
        raise ValueError("--pool none is invalid for KNN. Use flatten, mean, or cls. SVD uses token-level vecs internally.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_mode = args.safe_mode or (device == "cuda")

    keys, pre_lut, post_lut = build_common(args.pre_pt, args.post_pt, load_device=device)

    # -------------------------
    # Branch 1: KNN / overlap (pooled image embeddings)
    # -------------------------
    pre_vec_knn = _materialize_aligned(
        keys, pre_lut, pool=args.pool, load_device=device, out_device=device
    )
    pre_I = topk_ids(
        pre_vec_knn, args.topk, metric=args.metric,
        q_chunk=args.q_chunk, g_chunk=args.g_chunk, device=device,
        safe_mode=safe_mode, min_q_chunk=args.min_q_chunk, min_g_chunk=args.min_g_chunk
    )

    post_vec_knn = _materialize_aligned(
        keys, post_lut, pool=args.pool, load_device=device, out_device=device
    )
    post_I = topk_ids(
        post_vec_knn, args.topk, metric=args.metric,
        q_chunk=args.q_chunk, g_chunk=args.g_chunk, device=device,
        safe_mode=safe_mode, min_q_chunk=args.min_q_chunk, min_g_chunk=args.min_g_chunk
    )

    ps = pairwise_corr(pre_vec_knn, post_vec_knn, metric=args.metric, sample_n=10000, seed=42)

    del pre_raw0, post_raw0
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------
    # Branch 2: SVD / effective rank (token-level, per-image, streaming)
    # -------------------------
    sv_pre_list = compute_svd_stats_streaming(
        keys, pre_lut, topn=args.sv_topn, center=True, load_device=device
    )
    sv_post_list = compute_svd_stats_streaming(
        keys, post_lut, topn=args.sv_topn, center=True, load_device=device
    )
    sv_pre_summary = summarize_spectrum_list(sv_pre_list)
    sv_post_summary = summarize_spectrum_list(sv_post_list)

    # stash for CSV write after ref KNOR
    _sv_pre_list = sv_pre_list
    _sv_post_list = sv_post_list
    del sv_pre_list, sv_post_list
    gc.collect()
    cos_pre_list = compute_cosine_stats_streaming(keys, pre_lut, load_device=device)
    cos_post_list = compute_cosine_stats_streaming(keys, post_lut, load_device=device)
    cos_pre_summary = summarize_cosine_list(cos_pre_list)
    cos_post_summary = summarize_cosine_list(cos_post_list)
    del cos_pre_list, cos_post_list
    gc.collect()
    results = {}
    results["global"] = {
        "knn_pool": args.pool,
        "pairwise_corr": ps,
        "spectrum_and_rank": {
            "pre_summary": sv_pre_summary,
            "post_summary": sv_post_summary,
            "effective_rank_mean_delta": (
                sv_post_summary["effective_rank_mean"] - sv_pre_summary["effective_rank_mean"]
            ),
            "effective_rank_normalized_mean_delta": (
                sv_post_summary["effective_rank_normalized_mean"] - sv_pre_summary["effective_rank_normalized_mean"]
            ),
            "top10_energy_ratio_mean_delta": (
                sv_post_summary["top10_energy_ratio_mean"] - sv_pre_summary["top10_energy_ratio_mean"]
            ),
            "top1_energy_ratio_mean_delta": (
                sv_post_summary["top1_energy_ratio_mean"] - sv_pre_summary["top1_energy_ratio_mean"]
            ),
            "k80_energy_mean_delta": (
                sv_post_summary["k80_energy_mean"] - sv_pre_summary["k80_energy_mean"]
            ),
            "k90_energy_mean_delta": (
                sv_post_summary["k90_energy_mean"] - sv_pre_summary["k90_energy_mean"]
            ),
        },
        "cosine_concentration": {
            "pre_summary": cos_pre_summary,
            "post_summary": cos_post_summary,
            "mean_pairwise_cosine_mean_delta": (
                cos_post_summary["mean_pairwise_cosine_mean"] - cos_pre_summary["mean_pairwise_cosine_mean"]
            ),
            "std_pairwise_cosine_mean_delta": (
                cos_post_summary["std_pairwise_cosine_mean"] - cos_pre_summary["std_pairwise_cosine_mean"]
            ),
            "median_pairwise_cosine_mean_delta": (
                cos_post_summary["median_pairwise_cosine_mean"] - cos_pre_summary["median_pairwise_cosine_mean"]
            ),
        },
    }

    per_image_knor: dict[str, dict] = {k: {} for k in keys}
    for k in [10, 50, 100]:
        if pre_I.size(1) >= k:
            ov = overlap(pre_I, post_I, k)
            md = mean_rank_displacement(pre_I, post_I, k)
            results[f"top{k}"] = {
                "mean": float(ov.mean().item()),
                "median": float(ov.median().item()),
                "std": float(ov.std(unbiased=False).item()),
                "mean_rank_displacement": float(md.mean().item()),
                "median_rank_displacement": float(md.median().item()),
                "std_rank_displacement": float(md.std(unbiased=False).item()),
                "pairwise_corr": ps,
            }
            for i, key in enumerate(keys):
                per_image_knor[key][f"pre_post_knor{k}"] = float(ov[i].item())
    results["_per_image_knor"] = per_image_knor

    # -------------------------
    # Optional: N-way KNOR with image references (CLIP, DINOv2, etc.)
    # Motivation: Qwen trains its vision encoder jointly with the connector, so the
    # pre-connector space may already deviate from a standard image representation.
    # Comparing KNN against neutral image references reveals whether low pre→post KNOR
    # is caused by an unusual pre-space (joint training) vs genuine information loss.
    # -------------------------
    # Build unified ref list from --ref entries + legacy --ref_pt/--ref_pt2
    ref_list = []
    for entry in args.ref:
        if ":" not in entry:
            raise ValueError(f"--ref must be label:path, got: {entry!r}")
        label, path = entry.split(":", 1)
        ref_list.append((label.strip(), path.strip()))
    if args.ref_pt:
        ref_list.append((args.ref_label, args.ref_pt))
    if args.ref_pt2:
        ref_list.append((args.ref2_label, args.ref_pt2))

    for label, path in ref_list:
        if args.pool == "none":
            print(f"[{label}] --pool none is not compatible with image-level reference; skipping")
        else:
            _run_ref_knor(label, path, keys, pre_lut, post_lut, args, device, safe_mode, results)

    if args.svd_per_image_csv:
        import csv
        csv_path = out_dir / "per_image.csv"
        scalar_cols = ["effective_rank", "effective_rank_normalized",
                       "top1_energy_ratio", "top5_energy_ratio", "top10_energy_ratio",
                       "k80_energy", "k90_energy"]
        per_image_ref = results.pop("_per_image_ref", {})
        per_image_knor = results.pop("_per_image_knor", {})
        # collect all extra column names from ref KNOR
        ref_cols: list[str] = []
        for v in per_image_ref.values():
            for c in v:
                if c not in ref_cols:
                    ref_cols.append(c)
        knor_cols = [f"pre_post_knor{k}" for k in [10, 50, 100]]
        fieldnames = (["key"]
                      + [f"pre_{c}" for c in scalar_cols]
                      + [f"post_{c}" for c in scalar_cols]
                      + knor_cols + ref_cols)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for key, pre_s, post_s in zip(keys, _sv_pre_list, _sv_post_list):
                row = {"key": key}
                for c in scalar_cols:
                    row[f"pre_{c}"] = pre_s[c]
                    row[f"post_{c}"] = post_s[c]
                row.update(per_image_knor.get(key, {}))
                row.update(per_image_ref.get(key, {}))
                w.writerow(row)
        print(f"Per-image CSV saved: {csv_path}")
    else:
        results.pop("_per_image_ref", None)
        results.pop("_per_image_knor", None)

    with open(out_dir / f"overlap_top{args.topk}_{args.metric}_torch.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    spec = results["global"]["spectrum_and_rank"]

    print("N =", len(keys))
    print("device =", device, "| safe_mode =", safe_mode)
    print("knn pool =", args.pool, "| q_chunk =", args.q_chunk, "| g_chunk =", args.g_chunk)
    print(
        "global: "
        "| pairwise_corr={:.4f} "
        "| erank_mean={:.2f}->{:.2f} "
        "| erank_norm_mean={:.4f}->{:.4f} "
        "| top10_energy_mean={:.4f}->{:.4f} "
        "| k80_mean={:.2f}->{:.2f}".format(
            results["global"]["pairwise_corr"],
            spec["pre_summary"]["effective_rank_mean"],
            spec["post_summary"]["effective_rank_mean"],
            spec["pre_summary"]["effective_rank_normalized_mean"],
            spec["post_summary"]["effective_rank_normalized_mean"],
            spec["pre_summary"]["top10_energy_ratio_mean"],
            spec["post_summary"]["top10_energy_ratio_mean"],
            spec["pre_summary"]["k80_energy_mean"],
            spec["post_summary"]["k80_energy_mean"],
        )
    )

    for kk, vv in results.items():
        if kk == "global":
            continue
        if not isinstance(vv, dict) or "mean" not in vv:
            continue
        line = f"{kk}: mean={vv['mean']:.4f} median={vv['median']:.4f} std={vv['std']:.4f}"
        if "pairwise_corr" in vv:
            line += f" pairwise_corr={vv['pairwise_corr']:.4f}"
        print(line)

    print("saved to:", out_dir)

def pairwise_corr(pre_vec, post_vec, metric="cosine", sample_n=5000, seed=0):
    torch.manual_seed(seed)
    N = pre_vec.size(0)
    idx = torch.randperm(N)[:min(sample_n, N)]
    A = pre_vec[idx].float()
    B = post_vec[idx].float()

    if metric == "cosine":
        A = F.normalize(A, dim=1)
        B = F.normalize(B, dim=1)
        Da = 1.0 - (A @ A.T)   # cosine distance
        Db = 1.0 - (B @ B.T)
    else:
        Da = torch.cdist(A, A, p=2)
        Db = torch.cdist(B, B, p=2)

    iu = torch.triu_indices(Da.size(0), Da.size(1), offset=1)
    xa = Da[iu[0], iu[1]]
    xb = Db[iu[0], iu[1]]

    # Pearson (Spearman要自己rank；先用Pearson也很有用)
    xa = xa - xa.mean()
    xb = xb - xb.mean()
    pearson = (xa * xb).mean() / (xa.std(unbiased=False) * xb.std(unbiased=False) + 1e-8)
    return float(pearson.item())


if __name__ == "__main__":
    main()
