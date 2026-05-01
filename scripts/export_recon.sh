#!/bin/bash
#SBATCH --account=PAS3272
#SBATCH --job-name=export-recon
#SBATCH --time=1:30:00
#SBATCH --cluster=ascend
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G

# Usage:
#   sbatch scripts/export_recon.sh llava
#   sbatch scripts/export_recon.sh idefics2
#   sbatch scripts/export_recon.sh llava 00000,00001

MODEL=${1:-llava}
SHARDS=${2:-}

conda activate nlp_p

ARGS="--embed_model ${MODEL} --out_base data/vector_recon"

if [ -n "${SHARDS}" ]; then
    ARGS="${ARGS} --shards ${SHARDS}"
fi

python src/train/export_recon.py ${ARGS}
