# VLM Connectors

This repository studies how vision-language model connectors transform image
representations, and whether those representation changes show up in caption
behavior.

For an image `x`, the project treats the VLM pipeline as:

```text
E_pre = vision_encoder(x)
E_post = connector(E_pre)
caption = language_model(E_post)
```

The main question is not "which model is best?" It is:

```text
How much geometry changes between E_pre and E_post, and does that change predict
caption-level instability?
```

`E_pre` is used as a reference space, not as semantic ground truth.

## What Is Measured

- Pre/post geometry preservation with kNN overlap and KNOR.
- Spectral structure, including effective rank and top-k energy.
- Recoverability of `E_pre` from `E_post` using a learned reconstructor.
- Caption stability with CLIPScore and BERTScore diagnostics.
- Controlled perturbations to `E_post`, including token masking and low-rank truncation.

CLIPScore and BERTScore are diagnostic signals. They are not treated as perfect
semantic judgments.

## Models

The code supports:

- LLaVA, with an MLP projector.
- Idefics2, with a Perceiver-style resampler.
- Qwen2.5-VL, with a patch-merging connector.
- Qwen3.5, used here as an exploratory omnimodal model.

One important interpretation note: Qwen-family models jointly train the vision
encoder and connector, so a low pre/post neighborhood overlap can reflect
representation reorganization rather than simple information loss.

## Repository Layout

```text
src/extract/        Vector extraction, CLIP/DINO reference extraction, kNN metrics
src/train/          Reconstructor training and connector experiments
src/eval/           Reconstruction, caption, and perturbation evaluation
src/plot/           Figure and summary generation
scripts/            Slurm wrappers for full GPU runs
sample/             Small sample outputs for quick plotting checks
data/               Full local data and generated outputs, not committed
figures/            Generated figures
results/            Generated summary tables
```

## Setup

Create an environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU runs, install a PyTorch build that matches your CUDA environment if the
default `torch` wheel is not appropriate for your machine.

The full extraction and evaluation pipeline downloads large Hugging Face models.
Make sure your environment has access to the model repositories used by the
scripts.

## Expected Data Layout

The full pipeline assumes images are stored under:

```text
data/mtf2025_web_images/
```

The code expects image files, and some scoring scripts also look for matching
caption text files with the same stem:

```text
data/mtf2025_web_images/00000/example.jpg
data/mtf2025_web_images/00000/example.txt
```

Vector extraction writes files like:

```text
data/vector/<model>/<shard>/pre_vectors_*.pt
data/vector/<model>/<shard>/post_vectors_*.pt
```

Common model names are:

```text
llava
idefics2
qwen2.5vl
qwen3.5
```

## Quick Start With Sample Outputs

If you only want to regenerate figures and summaries from the included sample
outputs, run:

```bash
python src/plot/plot_caption_recon.py
python src/plot/plot_caption_recon_pre.py
python src/plot/plot_knn.py
python src/plot/plot_erank_vs_clip.py
python src/plot/plot_perturb.py --perturb lowrank
python src/plot/plot_perturb.py --perturb mask
python src/plot/compute_summary.py
```

Outputs are written to `figures/` and `results/`.

## Full Pipeline

The Slurm wrappers in `scripts/` are the easiest way to reproduce the large GPU
runs on the original cluster setup. The commands below show the corresponding
Python entry points.

### 1. Extract VLM Pre/Post Vectors

Extract pre-connector and post-connector vectors for all supported models:

```bash
python src/extract/extract_vectors.py \
  --root data/mtf2025_web_images \
  --out_dir data/vector \
  --models all \
  --device cuda \
  --dtype bf16 \
  --batch_size 32 \
  --num_workers 4 \
  --save_every 400 \
  --skip_existing
```

Slurm wrapper:

```bash
sbatch scripts/vec.sh
```

### 2. Extract Reference Image Embeddings

Extract CLIP reference embeddings:

```bash
python src/extract/extract_clip.py \
  --root data/mtf2025_web_images \
  --out_dir data/clip_ref \
  --model_id openai/clip-vit-large-patch14 \
  --pool cls \
  --batch_size 32 \
  --device cuda
```

Slurm wrapper:

```bash
sbatch scripts/clip.sh
```

### 3. Compute Geometry Metrics

Compute pre/post kNN overlap, KNOR, and spectral diagnostics:

```bash
python src/extract/extract_knn.py \
  --pre_pt data/vector/llava \
  --post_pt data/vector/llava \
  --ref clip:data/clip_ref \
  --out_dir data/output/knn_out/llava \
  --pool mean \
  --metric l2 \
  --svd_per_image_csv
```

Slurm wrapper:

```bash
sbatch scripts/cal.sh llava
```

### 4. Train Reconstructors

Train a reconstructor from `E_post` back to `E_pre`:

```bash
python src/train/train_recon.py \
  --vec_dir data/vector/llava/00000 \
  --embed_model llava \
  --model_type mlp \
  --num_layers 3 \
  --num_heads 16 \
  --hidden_size 2048 \
  --out_dir data/output/out/out_llava_mlp_norm \
  --normalize \
  --amp \
  --batch_size 8 \
  --epochs 20
```

Slurm wrapper:

```bash
sbatch scripts/emb.sh llava
```

Evaluate a trained reconstructor without wandb:

```bash
python src/train/train_recon.py \
  --vec_dir data/vector/llava/00000 \
  --embed_model llava \
  --model_type mlp \
  --out_dir data/output/out/out_llava_mlp_norm \
  --normalize \
  --eval_only
```

### 5. Export Reconstructed Vectors

Export reconstructed `E_pre` vectors for downstream analysis:

```bash
python src/train/export_recon.py \
  --embed_model llava \
  --out_base data/vector_recon
```

Slurm wrapper:

```bash
sbatch scripts/export_recon.sh llava
```

### 6. Evaluate Reconstruction Metrics

Compute per-sample MSE and cosine similarity between reconstructed and original
`E_pre`:

```bash
python src/eval/eval_recon.py \
  --embed_model llava \
  --shards 00000,00001 \
  --batch_size 32
```

### 7. Run Caption Evaluation

Generate captions from original post-connector embeddings and reconstructed
embeddings:

```bash
python src/eval/eval_captions.py \
  --vec_dir data/vector/llava \
  --image_dir data/mtf2025_web_images \
  --embed_model llava \
  --model_path data/output/out/out_llava_mlp_norm/best_model.pt \
  --recon_stats_path data/output/out/out_llava_mlp_norm/dataset_stats.pkl \
  --out_dir data/output/caption_compare_out_n/llava \
  --batch_size 4 \
  --amp \
  --compare_post_vs_recon \
  --shards 00001,00002 \
  --shard_limit 400 \
  --max_items 0 \
  --with_bertscore
```

Score caption CSVs after generation:

```bash
python src/eval/score_captions.py \
  --caption_dir data/output/caption_compare_out_n/llava \
  --batch_size 16
```

### 8. Run Perturbation Evaluation

Apply controlled perturbations to `E_post` and generate captions:

```bash
python src/eval/eval_perturb.py \
  --vec_dir data/vector/llava/00000 \
  --image_dir data/mtf2025_web_images/00000 \
  --model_name llava \
  --mode lowrank \
  --level 0.01 \
  --out_dir data/perturb_out \
  --amp \
  --limit 5000
```

Score perturbation outputs:

```bash
python src/eval/score_perturb.py \
  --perturb_dir data/perturb_out \
  --out_dir data/perturb_out_scored \
  --batch_size 16 \
  --skip_existing
```

Slurm wrappers:

```bash
sbatch scripts/perturb.sh llava lowrank 0.01
sbatch scripts/score_perturb.sh data/perturb_out
```

## Main Python Entry Points

```text
src/extract/extract_vectors.py     Extract VLM pre/post vectors
src/extract/extract_clip.py        Extract CLIP or DINO reference embeddings
src/extract/extract_knn.py         Compute kNN overlap, KNOR, and spectra
src/train/train_recon.py           Train or evaluate E_post -> E_pre reconstructors
src/train/export_recon.py          Export reconstructed vector shards
src/train/train_connector.py       Train alternative connector modules
src/eval/eval_recon.py             Reconstruction MSE and cosine evaluation
src/eval/eval_captions.py          Caption generation from original/reconstructed embeddings
src/eval/score_captions.py         CLIPScore and BERTScore for caption outputs
src/eval/eval_perturb.py           Caption generation under E_post perturbations
src/eval/score_perturb.py          CLIPScore and BERTScore for perturbation outputs
```

Every entry point exposes its options with `--help`.

## High-Level Findings

- LLaVA and Idefics2 preserve more pre/post neighborhood structure than the
  Qwen-family models in this setup.
- Qwen-family post-connector spaces align more with CLIP neighborhoods than
  their pre-connector spaces, consistent with representation reorganization.
- Reconstruction cosine is informative for some models, especially Qwen3.5, but
  MSE correlations are weaker.
- Low-rank truncation mainly affects Qwen-family models at very small retained
  ratios, while token masking affects LLaVA and Qwen2.5-VL more strongly.
- The main conclusion is that connector distortion alone does not explain
  caption-level behavior; effects are model- and metric-dependent.

## Limitations

- The experiments are food-domain only.
- The dataset uses a single reference caption per image.
- CLIPScore and BERTScore are proxies, not ground-truth semantic labels.
- CLIP and DINO neighborhoods are reference views, not definitive semantics.
- Reconstructor results depend on reconstructor capacity and training setup.
- Perturbations are diagnostic and do not exhaust all possible failure modes.
