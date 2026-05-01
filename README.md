# VLM Connectors

Connector-induced representation distortion and caption-level stability in vision-language models.

This repository analyzes whether projection through a VLM connector changes representation geometry in ways that are reflected in caption-level behavior. It is an analysis project, not a model-improvement project.

For an image $x$:

$$
E_{pre} = v(x) \rightarrow E_{post} = p(E_{pre}) \rightarrow y = D(E_{post})
$$

$E_{pre}$ is a reference space, not a ground-truth semantic space.

---

## Models

- LLaVA (MLP projector)
- IDEFICS2 (Perceiver-style resampler)
- Qwen2.5-VL (patch-merging connector)
- Qwen3.5 (omnimodal, exploratory model)

Qwen-family models jointly train the vision encoder and connector, so low pre/post neighborhood overlap can reflect representation reorganization rather than simple information loss.

---

## Dataset

MTF 2025 Dishcovery dataset (web subset): ~139K food-domain images, one reference caption per image.

---

## What This Measures

- Geometry preservation (kNN overlap / KNOR) and spectral diagnostics (effective rank, top-k energy)
- Recoverability: reconstruct $E_{pre}$ from $E_{post}$ and evaluate cosine and MSE
- Caption-level behavior: CLIPScore drop and BERTScore-based caption stability
- Controlled perturbations: token masking and low-rank truncation applied to $E_{post}$

CLIPScore and BERTScore are diagnostic signals for image-text alignment and caption stability, not ground-truth semantic judgments.

---

## Key Findings (High Level)

- LLaVA and IDEFICS2 preserve more pre/post neighborhood structure than Qwen-family models.
- Qwen-family post-connector spaces align more with CLIP neighborhoods than their pre-spaces, consistent with representation reorganization.
- Reconstruction cosine is informative for some models (especially Qwen3.5), but MSE correlations remain weak.
- Low-rank truncation mainly affects Qwen-family models at very small retained ratios; token masking affects LLaVA and Qwen2.5-VL more strongly.
- Main conclusion: connector distortion alone does not explain caption-level behavior; effects are model- and metric-dependent.

---

## Limitations

- Food-domain only; results should not be generalized to other image domains.
- Single reference caption; CLIPScore and BERTScore are diagnostic proxies.
- CLIP-based neighborhoods are useful references, not ground truth.
- Reconstruction quality depends on reconstructor capacity.
- Perturbations are diagnostic, not exhaustive.

---

## Repository Layout

- data/ - full outputs (not included in sample)
- sample/ - small outputs for plotting and sanity checks
- src/extract/ - vector and reference extraction
- src/train/ - reconstructor and connector training
- src/eval/ - reconstruction, caption, and perturbation evaluation
- src/plot/ - plotting and summary scripts
- scripts/ - Slurm wrappers for full runs (assume data/ paths)
- figures/ and results/ - generated figures and tables

---

## Quick Start (Sample Outputs)

Plotting scripts default to sample/ inputs and write to figures/ and results/:

```bash
python src/plot/plot_caption_recon.py
python src/plot/plot_caption_recon_pre.py
python src/plot/plot_knn.py
python src/plot/plot_erank_vs_clip.py
python src/plot/plot_perturb.py --perturb lowrank
python src/plot/plot_perturb.py --perturb mask
python src/plot/compute_summary.py
```

---

## Full Pipeline (Requires data/)

The Python entrypoints live under src/. Each script provides --help with its options.

Extraction:

- src/extract/extract_vectors.py - extract pre/post vectors (and optional captions)
- src/extract/extract_clip.py - extract CLIP or DINO reference embeddings
- src/extract/extract_knn.py - compute KNOR, effective rank, and spectral stats

Reconstruction and connectors:

- src/train/train_recon.py - train reconstructor $r: E_{post} \rightarrow E_{pre}$
- src/train/export_recon.py - export reconstructed vectors
- src/train/train_connector.py - train alternative connector modules (LLaVA, IDEFICS2)

Evaluation:

- src/eval/eval_recon.py - reconstruction metrics (cosine, MSE)
- src/eval/eval_captions.py - generate captions from original and reconstructed embeddings
- src/eval/score_captions.py - add CLIPScore and BERTScore columns
- src/eval/eval_perturb.py - apply perturbations to $E_{post}$
- src/eval/score_perturb.py - score perturbed captions

Plotting:

- src/plot/plot_caption_recon.py
- src/plot/plot_caption_recon_pre.py
- src/plot/plot_perturb.py
- src/plot/plot_erank_vs_clip.py
- src/plot/plot_knn.py
- src/plot/compute_summary.py

scripts/ contains Slurm batch wrappers around the full-data workflow. They assume data/ paths and may need adjustment for local runs.