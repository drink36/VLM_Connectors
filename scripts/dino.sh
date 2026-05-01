#!/bin/bash
#SBATCH --account=PAS3272
#SBATCH --job-name=dino-ref
#SBATCH --time=1:00:00
#SBATCH --cluster=pitzer
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1



conda activate nlp_p
python src/extract/extract.py       --root data/mtf2025_web_images       --out_dir data/dinov3_ref       --model_id facebook/dinov3-vitl16-pretrain-lvd1689m      --pool cls --batch_size 32 --device cuda