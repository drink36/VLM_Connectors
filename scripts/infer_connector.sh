#!/bin/bash
#SBATCH --account=PAS3272
#SBATCH --job-name=infer-connector
#SBATCH --time=2:00:00
#SBATCH --cluster=ascend
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

MODEL=${1:-llava}
SHARD=${2:-00000}
LIMIT=${3:-500}

conda activate nlp_p

case ${MODEL} in
    llava)    CONNECTOR=invertible ;;
    idefics2) CONNECTOR=crossattn ;;
    *)        echo "Unknown model: ${MODEL}"; exit 1 ;;
esac

CKPT_DIR=data/output/train_connector_out/${MODEL}_${CONNECTOR}

python train_connector.py \
    --embed_model ${MODEL} \
    --connector_type ${CONNECTOR} \
    --vec_dir data/vector/${MODEL} \
    --shard_folders ${SHARD} \
    --image_dir data/mtf2025_web_images \
    --gt_caption_dir data/mtf2025_web_images \
    --infer \
    --infer_checkpoint ${CKPT_DIR}/best_connector.pt \
    --infer_limit ${LIMIT} \
    --infer_out data/output/caption_compare_out_a/${MODEL}_${CONNECTOR}/${SHARD}.csv \
    --amp \
    --out_dir ${CKPT_DIR}
