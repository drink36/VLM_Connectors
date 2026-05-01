#!/bin/bash
#SBATCH --account=PAS3272
#SBATCH --job-name=train-connector
#SBATCH --time=5:00:00
#SBATCH --cluster=ascend
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1


MODEL=${1:-llava}
EXPORT=${2:-0}

conda activate nlp_p

case ${MODEL} in
    llava)
        CONNECTOR=invertible
        VEC_DIR=data/vector/llava ;;
    idefics2)
        CONNECTOR=crossattn
        VEC_DIR=data/vector/idefics2 ;;
    *)
        echo "Unknown model: ${MODEL}"; exit 1 ;;
esac

OUT_DIR=data/output/train_connector_out/${MODEL}_${CONNECTOR}
EXPORT_DIR=data/new_post/${MODEL}_${CONNECTOR}

EXPORT_ARGS=""
if [ "${EXPORT}" = "1" ]; then
    EXPORT_ARGS="--export_post_vectors --export_dir ${EXPORT_DIR}"
fi

python src/train/train_connector.py \
    --embed_model ${MODEL} \
    --connector_type ${CONNECTOR} \
    --vec_dir ${VEC_DIR} \
    --shard_folders 00000,00001,00002,00003,00004,00005 \
    --limit_per_shard 500 \
    --val_per_shard 200 \
    --gt_caption_dir data/mtf2025_web_images \
    --batch_size 4 --accumulation_steps 4 \
    --epochs 10 --lr 1e-4 --warmup_steps 200 \
    --amp \
    --wandb --wandb_project vlm-connectors --wandb_run_name ${MODEL}_${CONNECTOR} \
    --out_dir ${OUT_DIR} \
    ${EXPORT_ARGS}
