#!/bin/bash
#SBATCH --account=PAS3272
#SBATCH --job-name=dino-ref
#SBATCH --time=1:00:00
#SBATCH --cluster=pitzer
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

python reconstruct_embeddings.py \
  --vec_dir data/vector/qwen3.5/00000 \
  --embed_model qwen3.5 \
  --model_type transformer \
  --num_layers 8 \
  --out_dir data/output/out \
  --normalize \
  --amp \
  --batch_size 8 \
  --epochs 20