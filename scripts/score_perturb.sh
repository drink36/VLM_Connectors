#!/bin/bash
#SBATCH --account=PAS3272
#SBATCH --job-name=score-perturb
#SBATCH --time=00:30:00
#SBATCH --cluster=ascend
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1


PERTURB_DIR=${1:-data/perturb_out}

conda activate nlp_p

python score_perturb.py \
    --perturb_dir ${PERTURB_DIR} \
    --out_dir data/perturb_out_scored \
    --batch_size 16 \
    --skip_existing
