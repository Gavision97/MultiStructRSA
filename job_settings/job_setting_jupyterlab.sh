#!/bin/bash

#SBATCH --partition=main
#SBATCH --time=0-24:00:00
#SBATCH --job-name=research_project
#SBATCH --output=jobs/job-%J.out
#SBATCH --error=jobs/job-%J.err
#SBATCH --gpus=0
#SBATCH --mail-user=gavrilev@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --mem=24G

module load anaconda

source activate chemprop

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HOME=/home/gavrilev/.cache/huggingface

jupyter lab --no-browser