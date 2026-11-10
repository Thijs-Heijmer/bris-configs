#!/bin/bash
# =======================================
# Supports command line arguments for overrides:
# sbatch train_coherent-lama.sh training.max_steps=2000
# NOTE: ACTIVATE MAMBA ENV BEFORE SUBMITTING
# =======================================

#SBATCH --export=ALL
#SBATCH --propagate=NONE
#SBATCH --partition=gpuB-research
#SBATCH --account=yr-bris
#SBATCH --job-name=coherent_lama_finetuned-long
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --output=/lustre/storeB/users/thhei1261/bris-training/experiments/logs/%x_%j.out      # e.g. <ROOT>/logs/%x_%j.out (directory must exist)


set -euo pipefail


ROOT=/lustre/storeB/users/thhei1261/bris-training/experiments/coherent-lama-finetune
CFGDIR=/lustre/storeB/users/thhei1261/bris-configs/configs/alpaca/anemoi-training
CONFIG=${CONFIG:-coherent-lama-long.yaml}
RUN_ID=${RUN_ID:-}

export SLURM_GPUS_PER_NODE=2
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export MLFLOW_ALLOW_FILE_STORE=true
# export CUDA_LAUNCH_BLOCKING=1

HYDRA_DIR=${ROOT}/logs/hydra/${CONFIG}_${SLURM_JOB_ID}
mkdir -p "${HYDRA_DIR}" && cd "${HYDRA_DIR}"
# echo "== $(date) CONFIG=${CONFIG} nodes=${SLURM_NNODES} gpus/node=${SLURM_GPUS_PER_NODE} anemoi-core $(git -C ${ROOT}/src/anemoi-core rev-parse --short HEAD)"

srun --cpu-bind=cores \
  anemoi-training train \
    --config-path="${CFGDIR}" \
    --config-name="${CONFIG}" \
    hydra.run.dir="${HYDRA_DIR}" \
    ${RUN_ID:+training.run_id=${RUN_ID}} \
    ${EXTRA:-}\
    "$@"
echo "== $(date) done"


# anemoi-training config validate --config-name coherent-lama_w0.02_standalone.yaml 
