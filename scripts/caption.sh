#!/bin/bash
#SBATCH --account=PAS3272
#SBATCH --job-name=caption-eval
#SBATCH --time=5:00:00
#SBATCH --cluster=ascend
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1


MODEL=${1:-llava}
SHARDS=${2:-00001,00002,00003,00004,00005}
SHARD_LIMIT=${3:-400}

conda activate nlp_p

EXTRA_ARGS=""
case ${MODEL} in
    llava)
        MODEL_PATH=data/output/out/out_llava_mlp/best_model.pt ;;
    idefics2)
        MODEL_PATH=data/output/out/out_idefics2/best_model.pt ;;
    qwen2.5vl)
        MODEL_PATH=data/output/out/out_qwen2.5vl/best_model.pt ;;
    qwen3.5)
        MODEL_PATH=data/output/out/out_qwen3.5/best_model.pt
        EXTRA_ARGS="--recon_stats_path data/output/out/out_qwen3.5/dataset_stats.pkl" ;;
    *)
        echo "Unknown model: ${MODEL}"; exit 1 ;;
esac

python caption_eval.py \
    --vec_dir data/vector/${MODEL} \
    --image_dir data/mtf2025_web_images \
    --embed_model ${MODEL} \
    --model_path ${MODEL_PATH} \
    --out_dir data/output/caption_compare_out_a/${MODEL} \
    --batch_size 4 --amp \
    --compare_post_vs_recon \
    --shards ${SHARDS} \
    --shard_limit ${SHARD_LIMIT} \
    --max_items 0 \
    --with_bertscore \
    ${EXTRA_ARGS}
