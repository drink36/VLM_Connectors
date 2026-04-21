#!/bin/bash
#SBATCH --account=PAS3272
#SBATCH --job-name=clip-ref
#SBATCH --time=1:00:00
#SBATCH --cluster=pitzer
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1



conda activate nlp_p
python extract_clip_ref.py       --root data/mtf2025_web_images       --out_dir data/clip_ref       --model_id openai/clip-vit-large-patch14    --pool cls --batch_size 32 --device cuda