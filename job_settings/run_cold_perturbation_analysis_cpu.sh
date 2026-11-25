#!/bin/bash

################################################################################
### Evaluate cold-start checkpoints with RNA-structure permutations on CPU.
### Submit once for cold_rna and once for cold_rm (the joint cold-start split).
################################################################################

#SBATCH --partition=main
#SBATCH --time=2-00:00:00
#SBATCH --job-name=perturb_cold_cpu
#SBATCH --output=jobs/job-%A_%a.out
#SBATCH --error=jobs/job-%A_%a.err
#SBATCH --array=1-5
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=gavrilev@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --mem=24G

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: sbatch run_cold_perturbation_analysis_cpu.sh <cold_rna|cold_rm> [n_perturbations]' >&2
  exit 2
fi

COLD_TYPE="$1"
N_PERTURBATIONS="${2:-100}"
FOLD="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array job}"

if ! [[ "$N_PERTURBATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "n_perturbations must be a positive integer: $N_PERTURBATIONS" >&2
  exit 2
fi

case "$COLD_TYPE" in
  cold_rna)
    RESULT_NAME="cold_rna"
    ;;
  cold_rm)
    RESULT_NAME="cold_both"
    ;;
  *)
    echo "Unsupported cold-start type: $COLD_TYPE" >&2
    exit 2
    ;;
esac

RESULTS_ROOT="results/table_results/perturbation_analysis_permutation"
if [[ "$N_PERTURBATIONS" -ne 50 ]]; then
  RESULTS_ROOT="${RESULTS_ROOT}_n${N_PERTURBATIONS}"
fi
OUTPUT_DIR="${RESULTS_ROOT}/${RESULT_NAME}/fold_${FOLD}"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/home/gavrilev/Thesis/src}"
cd "$PROJECT_DIR"

module load anaconda
source activate chemprop
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Keep this run CPU-only even if it lands on a node that also has GPUs.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HOME=/home/gavrilev/.cache/huggingface

python main.py \
  --setting perturbation \
  --perturbation_cold_type "$COLD_TYPE" \
  --model_setting full \
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
