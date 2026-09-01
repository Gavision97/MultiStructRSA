#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition=main
#SBATCH --time=0-48:00:00
#SBATCH --job-name=case_study_smiles_emb_gen
#SBATCH --output=jobs/job-%J.out
#SBATCH --error=jobs/job-%J.err
#SBATCH --gpus=1
#SBATCH --mail-user=gavrilev@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --mem=24G


################  Following lines will be executed by a compute node  ################

module load anaconda
module load cuda/11.8

source activate chemprop
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1

export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HOME=/home/gavrilev/.cache/huggingface

nvcc --version
nvidia-smi

python main.py