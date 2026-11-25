#!/bin/bash

#SBATCH --partition=main
#SBATCH --time=0-01:00:00
#SBATCH --job-name=aggregate_perturb
#SBATCH --output=jobs/job-%J.out
#SBATCH --error=jobs/job-%J.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo 'Usage: sbatch aggregate_perturbation_folds.sh <input_dir>' >&2
  exit 2
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/home/gavrilev/Thesis/src}"
cd "$PROJECT_DIR"

module load anaconda
source activate chemprop

python evaluate/aggregate_perturbation_folds.py --input-dir "$1" --n-folds 5
