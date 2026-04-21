#!/bin/bash
#SBATCH --account=PAS3272
#SBATCH --job-name=vec
#SBATCH --time=2:00:00
#SBATCH --cluster=ascend
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1



conda activate nlp_p
python extract_multimodel_all.py --root data/mtf2025_web_images --out_dir data/vector --models all --device cuda --dtype bf16 --batch_size 32 --num_workers 4 --save_every 400 --skip_existing