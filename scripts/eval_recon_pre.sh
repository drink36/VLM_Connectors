#!/bin/bash
#SBATCH --account=PAS3272
#SBATCH --job-name=eval-recon-pre
#SBATCH --time=00:30:00
#SBATCH --cluster=ascend
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

MODEL=${1:-llava}

conda activate nlp_p

python eval_recon_pre.py --embed_model ${MODEL}
