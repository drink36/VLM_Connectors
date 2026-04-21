#!/bin/bash
#SBATCH --account=PAS3272
#SBATCH --job-name=cal-knn
#SBATCH --time=8:00:00
#SBATCH --cluster=pitzer
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1


MODEL=${1:-llava}

conda activate nlp_p

python knn.py \
    --pre_pt data/vector/${MODEL} \
    --post_pt data/vector/${MODEL} \
    --ref clip:data/clip_ref \
    --ref dino:data/dino_ref \
    --ref dinov3:data/dinov3_ref \
    --out_dir data/output/knn_out/${MODEL} \
    --pool mean \
    --metric l2 \
    --svd_per_image_csv