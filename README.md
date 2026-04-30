# VLM Connectors

**Connector-Induced Representation Distortion and Semantic Stability in Vision–Language Models**

Does projection-induced geometric distortion in VLMs actually hurt caption quality, or can language behavior stay stable under structured representation changes?

Pipeline: `E_pre = v(x)` → `E_post = p(E_pre)` → `y = d(E_post)`. Captions are a behavioral probe, not a task to optimize.

Four models: LLaVA, IDEFICS2, Qwen2.5-VL, Qwen3.5. Data: MTF 2025 Dishcovery Dataset, web subset (~139K food-domain images).

The key distinction: LLaVA and IDEFICS2 freeze the ViT so `E_pre` is a real standalone representation — projection is a translation. Qwen trains the ViT and connector jointly, so `E_pre` is an under-formed intermediate. Low KNOR for Qwen reflects representation reorganization, not information loss. That distinction drives the whole analysis.

---

## What This Measures

- **Structural distortion** — KNOR, effective rank, Top-10 spectral energy. CLIP embeddings as an external anchor.
- **Recoverability** — train `r: E_post → E_pre`, measure reprojection cosine/MSE, then run captions from reconstructed embeddings and score with CLIPScore and BERTScore.
- **Controlled perturbations** — token masking and low-rank truncation applied directly to `E_post`; geometry → caption-drop curves per model.
- **Alternative connectors** — train InvertibleMLP (LLaVA) and CrossAttn perceiver (IDEFICS2) as drop-in replacements; compare KNOR/ER on exported post-vectors.

---

## Pipeline

```bash
# 1. Extract pre/post connector vectors (one job per model)
sbatch scripts/vec.sh llava

# 2. Extract CLIP reference embeddings
sbatch scripts/emb.sh

# 3. KNOR + geometry analysis (batches all shard folders automatically)
sbatch scripts/cal.sh llava

# 4. Train reconstruction model r: E_post → E_pre
sbatch scripts/train.sh llava

# 5. Caption eval — reconstruction quality, CLIPScore, BERTScore
sbatch scripts/caption.sh llava 00001,00002,00003,00004,00005 500

# 6. Perturbation curves
sbatch scripts/perturb.sh llava

# 7. Score perturbed captions
sbatch scripts/score_perturb.sh llava

# 8. Reproduce paper figures and tables
python generate_paper_outputs.py
```

---

## Scripts

### Extraction

| Script | Purpose |
|---|---|
| `extract_multimodel_all.py` | Extract `E_pre` and `E_post` vectors for all models; writes `pre_vectors_*.pt` / `post_vectors_*.pt` shards |
| `extract_clip_ref.py` | Extract CLIP image embeddings as an external reference space |

### Analysis

| Script | Purpose |
|---|---|
| `knn.py` | KNOR (k-nearest-neighbor overlap ratio), effective rank, pairwise distance correlation; accepts parent dir to batch all shards |
| `reconstruct_embeddings.py` | Train MLP or Transformer reconstructor `r: E_post → E_pre`; exports reconstructed pre-vectors |
| `export_recon_vectors.py` | Export reconstructed pre-vectors from a trained reconstructor checkpoint |
| `eval_recon_pre.py` | Evaluate reconstruction quality (cosine, MSE) before caption generation |
| `caption_eval.py` | Generate captions from original and reconstructed `E_post`; writes per-sample CSV with CLIPScore and BERTScore columns; supports resume |
| `perturb_eval.py` | Apply token masking or low-rank truncation to `E_post` and record resulting captions |
| `score_captions.py` | Add CLIPScore and BERTScore columns to any caption CSV; `--no_clip` / `--no_bert` flags |
| `score_perturb.py` | Score perturbed caption CSVs (CLIPScore drop per perturbation level) |
| `train_connector.py` | Train InvertibleMLP (LLaVA) or CrossAttn perceiver (IDEFICS2) as alternative connectors |

### Plotting and Tables

| Script | Purpose |
|---|---|
| `plot_caption_recon.py` | Scatter plot of reconstruction quality vs caption-level metrics (Figure 2); generates Table 1 (correlations) and Table 3 (spectral diagnostics) LaTeX |
| `plot_caption_recon_pre.py` | Same scatter using `cosine_pre` (pre-reconstruction cosine) instead of reprojection cosine |
| `plot_perturb.py` | Perturbation curves: CLIPScore drop vs rank ratio or mask fraction (Figure 3) |
| `plot_erank_vs_clip.py` | Effective rank vs CLIP overlap scatter across models |
| `summary_stats.py` | Aggregate per-model summary statistics from kNN results |
| `generate_paper_outputs.py` | Copy final figures to `figures/paper/` and regenerate Table 1 + Table 3 LaTeX |

### Shared Utilities

| File | Purpose |
|---|---|
| `vlm_utils.py` | Shared VLM helpers: `load_vlm`, `build_vlm_inputs`, `decode_outputs`, `resolve_connector`, `build_image_index`, `numeric_suffix` |
| `compute_data_all.py` | Batch driver for multi-shard data aggregation |

---

## Metrics

| Metric | Definition |
|---|---|
| **KNOR** | K-nearest-neighbor overlap ratio between two embedding spaces; fraction of shared neighbors at rank k |
| **Effective rank** | Entropy-based rank of the spectral distribution; measures variance spread across dimensions |
| **Top-10 energy** | Fraction of total variance in the top-10 singular values |
| **Reprojection cosine** | `cosine(r(E_post), E_pre)` — how well the reconstructed pre-vector matches the original after inverting the connector |
| **CLIPScore** | Image × caption cosine similarity in CLIP embedding space; used as a proxy for image-text alignment |
| **CLIPScore Drop** | `CLIPScore(original) − CLIPScore(recon/perturbed)`; positive = metric decreased |
| **BERTScore** | Token-level F1 between two captions; used as a proxy for caption stability |

---

## Results Summary

- LLaVA and IDEFICS2 preserve more pre/post neighborhood structure (higher KNOR) than Qwen-family models.
- Qwen-family post-projection spaces align more strongly with CLIP neighborhoods than their pre-projection spaces — consistent with representation reorganization rather than degradation.
- IDEFICS2 shows the clearest correlation between reconstruction quality and caption stability among the four models.
- Qwen3.5 shows the strongest association between reprojection cosine and caption-level metrics, but MSE correlations remain weak.
- Low-rank truncation mainly affects Qwen-family models at very small retained-rank ratios; token masking more broadly affects Qwen2.5-VL and LLaVA.
- **Main conclusion**: connector projection can substantially distort representation geometry, but representation distortion alone does not fully explain caption-level semantic behavior. The relationship is model-dependent and metric-dependent.

---

## Paper Outputs

```
figures/paper/fig1a.png   ← figures/knn/single_dataset_topk_all.png
figures/paper/fig1b.png   ← figures/knn_clip/single_dataset_topk_with_clip.png
figures/paper/fig2.png    ← figures/caption_recon/fig_corr_scatter.png
figures/paper/fig3a.png   ← figures/lowrank_drop.png
figures/paper/fig3b.png   ← figures/mask_drop.png
results/caption_recon_correlations.tex   ← Table 1 (correlations)
results/spectral_diagnostics.tex         ← Table 3 (spectral diagnostics)
```

Regenerate all:

```bash
python generate_paper_outputs.py           # figures + tables
python generate_paper_outputs.py --tables  # tables only
python generate_paper_outputs.py --figures # figures only
```
