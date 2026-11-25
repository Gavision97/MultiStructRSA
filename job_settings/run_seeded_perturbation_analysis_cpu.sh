#!/bin/bash

################################################################################
### Evaluate one five-fold seeded CV model with matched RNA permutations on CPU.
################################################################################

#SBATCH --partition=main
#SBATCH --time=2-00:00:00
#SBATCH --job-name=perturb_seed_cpu
#SBATCH --output=jobs/job-%A_%a.out
#SBATCH --error=jobs/job-%A_%a.err
#SBATCH --array=1-5
#SBATCH --cpus-per-task=8
#SBATCH --constraint=cpu128
#SBATCH --mail-user=gavrilev@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --mem=24G

set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo 'Usage: sbatch run_seeded_perturbation_analysis_cpu.sh <model_seed> [n_perturbations] [results_root]' >&2
  exit 2
fi

MODEL_SEED="$1"
N_PERTURBATIONS="${2:-100}"
RESULTS_ROOT="${3:-results/table_results/perturbation_analysis_permutation}"
FOLD="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array job}"

if ! [[ "$MODEL_SEED" =~ ^[1-9][0-9]*$ ]]; then
  echo "model_seed must be a positive integer: $MODEL_SEED" >&2
  exit 2
fi
if ! [[ "$N_PERTURBATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "n_perturbations must be a positive integer: $N_PERTURBATIONS" >&2
  exit 2
fi

CHECKPOINT_DIR="saved_objects/cv/full_seed${MODEL_SEED}"
if [[ "$MODEL_SEED" -eq 1 ]]; then
  CHECKPOINT_PATTERN='model_fold{fold}.pt'
else
  CHECKPOINT_PATTERN="model_seed_${MODEL_SEED}_fold{fold}.pt"
fi
OUTPUT_DIR="${RESULTS_ROOT}/seed_${MODEL_SEED}/fold_${FOLD}"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/home/gavrilev/Thesis/src}"
cd "$PROJECT_DIR"

module load anaconda
source activate chemprop
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Force CPU execution even on a node that happens to expose GPUs.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HOME=/home/gavrilev/.cache/huggingface

python main.py \
  --setting perturbation \
  --model_setting full \
  --model_seed "$MODEL_SEED" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --checkpoint_pattern "$CHECKPOINT_PATTERN" \
  --n_perturbations "$N_PERTURBATIONS" \
  --k_fold 5 \
  --folds "$FOLD" \
  --dataset_seed 2 \
  --seed 1 \
  --val_setting 4 \
  --batch_size 1 \
  --num_workers 0 \
  --device cpu \
  --output_dir "$OUTPUT_DIR"
