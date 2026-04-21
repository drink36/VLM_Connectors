# VLM Connectors

Does projection-induced geometric distortion in VLMs actually hurt caption quality, or can language behavior stay stable under structured representation changes?

Pipeline: `E_pre = v(x)` → `E_post = p(E_pre)` → `y = d(E_post)`. Captions are a behavioral probe, not a task to optimize.

Four models: LLaVA, Idefics2, Qwen2.5-VL, Qwen3.5. Data: MTF 2025 VLM Discovery Challenge, web subset (~139K images across 14 subfolders).

The key distinction: LLaVA/Idefics2 freeze the ViT so `E_pre` is a real standalone representation — projection is a translation. Qwen trains the ViT and connector jointly, so `E_pre` is an under-formed intermediate. Low KNOR for Qwen is reorganization, not information loss. That distinction drives the whole analysis.

## Analysis Sections

- **4.1 Structural distortion** — KNOR, MPC, effective rank, Top-10 energy. DINOv2 and CLIP as external anchors.
- **4.2 Recoverability** — invert `g: E_post → E_pre`, measure reprojection cosine and BERTScore vs GT captions.
- **4.3 Controlled perturbations** — token masking, low-rank PCA, orthogonal transform on `E_post`. Geometry → performance curves.
- **4.4 Alternative connectors** — train InvertibleMLP (LLaVA) and CrossAttn perceiver (Idefics2) as replacements; compare KNOR/ER on new post-vectors.

## Pipeline

```bash
# 1. Extract pre/post connector vectors (one job per model)
sbatch scripts/vec.sh llava

# 2. Extract DINO / CLIP reference embeddings
sbatch scripts/emb.sh

# 3. KNOR + geometry analysis (runs all 14 shard folders automatically)
sbatch scripts/cal.sh llava

# 4. Caption eval — reconstruction quality, CLIPScore, BERTScore
sbatch scripts/caption.sh llava 00001,00002,00003,00004,00005 500

# 5. Perturbation curves
sbatch scripts/perturb.sh llava
```

## Files

- `extract_multimodel_all.py` — pre/post connector vectors for all models
- `extract_clip_ref.py` — CLIP / DINOv2 reference embeddings
- `knn.py` — KNOR, effective rank, pairwise distance correlation; accepts parent dir to batch all shards
- `caption_eval.py` — caption generation + reconstruction eval + inline CLIPScore + BERTScore; per-batch CSV write with resume
- `perturb_eval.py` — token masking / low-rank / orthogonal perturbation on `E_post`
- `train_connector.py` — train alternative connectors (InvertibleMLP for LLaVA, CrossAttn for Idefics2)
- `clip_compare.py` — batch CLIPScore for an existing caption CSV
- `reconstruct_embeddings.py` — train reconstruction model `g: E_post → E_pre`
- `bert_compare.py`, `clip_compare.py` — semantic similarity scoring utilities
- `compute_data_all.py`, `evaluation.py` — aggregation and benchmark wrappers

## Metrics

- **KNOR** — K-nearest-neighbor overlap ratio between two embedding spaces; measures how much neighborhood structure is preserved
- **ER** — effective rank; how spread out the variance is across dimensions
- **CLIPScore** — image × caption cosine similarity in CLIP space (primary semantic metric)
- **BERTScore** — token-level F1 between generated caption and GT reference
- **Reprojection cosine** — how well the reconstructed `E_pre` matches the original after inverting the connector
