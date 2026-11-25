#!/bin/bash

################################################################################################
### Evaluate fixed stratified-CV checkpoints with complete permutations of RNA 3D structures.
### Submit once for "full" and once for "w/o seq"; the two jobs can run in parallel.
################################################################################################

#SBATCH --partition=main
#SBATCH --time=0-24:00:00
#SBATCH --job-name=perturbation_cv
#SBATCH --output=jobs/job-%J.out
#SBATCH --error=jobs/job-%J.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mail-user=gavrilev@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --mem=24G

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: sbatch run_perturbation_analysis.sh <full|w/o seq> [n_perturbations]' >&2
  exit 2
fi

MODEL_SETTING="$1"
N_PERTURBATIONS="${2:-50}"
FOLD="${SLURM_ARRAY_TASK_ID:-}"

if ! [[ "$N_PERTURBATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "n_perturbations must be a positive integer: $N_PERTURBATIONS" >&2
  exit 2
fi

RESULTS_DIR="results/table_results/perturbation_analysis_permutation"
if [[ "$N_PERTURBATIONS" -ne 50 ]]; then
  RESULTS_DIR="${RESULTS_DIR}_n${N_PERTURBATIONS}"
fi

case "$MODEL_SETTING" in
  full)
    OUTPUT_DIR="${RESULTS_DIR}/full"
    ;;
  "w/o seq")
    OUTPUT_DIR="${RESULTS_DIR}/w_o_seq"
    ;;
  *)
    echo "Unsupported model setting: $MODEL_SETTING" >&2
    exit 2
    ;;
esac

FOLD_ARGS=()
if [[ -n "$FOLD" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR}/fold_${FOLD}"
  FOLD_ARGS=(--folds "$FOLD")
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/home/gavrilev/Thesis/src}"
cd "$PROJECT_DIR"

module load anaconda
module load cuda/11.8

source activate chemprop
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HOME=/home/gavrilev/.cache/huggingface

nvcc --version
nvidia-smi

python main.py \
  --setting perturbation \
  --model_setting "$MODEL_SETTING" \
  --n_perturbations "$N_PERTURBATIONS" \
  --k_fold 5 \
  --dataset_seed 2 \
  --seed 1 \
  --val_setting 4 \
  --batch_size 1 \
  --num_workers 0 \
  --device cuda \
  "${FOLD_ARGS[@]}" \
  --output_dir "$OUTPUT_DIR"
