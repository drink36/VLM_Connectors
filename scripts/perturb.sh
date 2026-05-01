#!/bin/bash
#SBATCH --account=PAS3272
#SBATCH --job-name=perturb-eval
#SBATCH --time=6:00:00
#SBATCH --cluster=ascend
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

MODEL=${1:-llava}
MODE=${2:-lowrank}
LEVEL=${3:-0.01}

conda activate nlp_p
python src/eval/perturb_eval.py \
    --vec_dir data/vector/${MODEL}/00000 \
    --image_dir data/mtf2025_web_images/00000 \
    --model_name ${MODEL} \
    --mode ${MODE} \
    --level ${LEVEL} \
    --out_dir ./perturb_out_a \
    --amp --limit 5000
