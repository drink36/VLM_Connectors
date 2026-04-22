#!/bin/bash
#SBATCH --account=PAS3272
#SBATCH --job-name=recon-emb
#SBATCH --time=3:00:00
#SBATCH --cluster=pitzer
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

MODEL=${1:-qwen3.5}

conda activate nlp_p

case ${MODEL} in
    llava)
        MODEL_TYPE=mlp
        NUM_LAYERS=3
        NUM_HEADS=16
        VEC_DIR=data/vector/llava/00000
        OUT_DIR=data/output/out/out_llava_mlp_norm ;;
    idefics2)
        MODEL_TYPE=transformer
        NUM_LAYERS=8
        NUM_HEADS=16
        VEC_DIR=data/vector/idefics2/00000
        OUT_DIR=data/output/out/out_idefics2_norm ;;
    qwen2.5vl)
        MODEL_TYPE=transformer
        NUM_LAYERS=8
        NUM_HEADS=16
        VEC_DIR=data/vector/qwen2.5vl/00000
        OUT_DIR=data/output/out/out_qwen2.5vl_norm ;;
    qwen3.5)
        MODEL_TYPE=transformer
        NUM_LAYERS=8
        NUM_HEADS=16
        VEC_DIR=data/vector/qwen3.5/00000
        OUT_DIR=data/output/out/out_qwen3.5_norm ;;
    *)
        echo "Unknown model: ${MODEL}"; exit 1 ;;
esac

python reconstruct_embeddings.py \
  --vec_dir ${VEC_DIR} \
  --embed_model ${MODEL} \
  --model_type ${MODEL_TYPE} \
  --num_layers ${NUM_LAYERS} \
  --num_heads ${NUM_HEADS} \
  --hidden_size 2048 \
  --out_dir ${OUT_DIR} \
  --normalize \
  --amp \
  --batch_size 8 \
  --epochs 20
