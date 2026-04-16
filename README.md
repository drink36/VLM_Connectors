# VLM Connectors

Does projection-induced geometric distortion in VLMs actually hurt caption quality, or can language behavior stay stable under structured representation changes?

We look at four models — LLaVA, Idefics2, Qwen2.5-VL, Qwen3.5 — and measure what the connector/projector does to vision token geometry, then ask whether that distortion shows up in captions.

The core tension: LLaVA and Idefics2 use a frozen ViT, so the pre-connector space is a real standalone representation. Qwen trains its vision encoder jointly with the connector, so "pre-connector" is more of an intermediate state than a finished embedding. Low KNOR for Qwen isn't information loss — it's reorganization. That distinction drives the whole analysis.

## What's here

- `extract_multimodel_all.py` — extract pre/post connector vectors for all four models
- `extract_clip_ref.py` — extract CLIP or DINOv2 reference embeddings (neutral image anchor)
- `knn.py` — compute KNOR, effective rank, pairwise distance correlation
- `compute_data_all.py` — aggregate results and make figures
- `caption_eval_gpt.py` — generate and evaluate captions, with perturbation support
- `evaluation.py` — lmms-eval wrapper for benchmark runs
- `bert_compare.py`, `clip_compare.py` — semantic similarity scoring

## Data

MTF 2025 VLM Discovery Challenge, web subset (~139K images across subfolders).

## Pipeline

```bash
# 1. Extract pre/post vectors
python extract_multimodel_all.py --root data/mtf2025_web_images --out_dir data/vector \
    --models all --device cuda --dtype bf16 --batch_size 32

# 2. Extract reference embeddings (DINOv2 as geometric anchor)
python extract_clip_ref.py --root data/mtf2025_web_images --out_dir data/dino_ref \
    --model_id facebook/dinov2-large --pool cls --device cuda

# 3. Run KNOR analysis
python knn.py --pre_pt data/vector/llava/00000 --post_pt data/vector/llava/00000 \
    --ref_pt2 data/dino_ref/00000 --ref2_label dino \
    --out_dir knn_out/llava --pool mean --metric cosine

# 4. Aggregate and plot
python compute_data_all.py
```
