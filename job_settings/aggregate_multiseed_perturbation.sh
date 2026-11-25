#!/bin/bash

#SBATCH --partition=main
#SBATCH --time=0-01:00:00
#SBATCH --job-name=aggregate_perturb
#SBATCH --output=jobs/job-%J.out
#SBATCH --error=jobs/job-%J.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo 'Usage: sbatch aggregate_multiseed_perturbation.sh <input_dir> <model_seed> [model_seed ...]' >&2
  exit 2
fi

INPUT_DIR="$1"
shift

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/home/gavrilev/Thesis/src}"
cd "$PROJECT_DIR"

module load anaconda
source activate chemprop

python evaluate/aggregate_multiseed_perturbation.py \
  --input-dir "$INPUT_DIR" \
  --model-seeds "$@"
