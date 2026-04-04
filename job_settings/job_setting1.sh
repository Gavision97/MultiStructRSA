#!/bin/bash

#SBATCH --partition=main
#SBATCH --time=0-72:00:00
#SBATCH --job-name=research_project
#SBATCH --output=/home/gavrilev/Thesis/src/jobs/job-%J.out
#SBATCH --error=/home/gavrilev/Thesis/src/jobs/job-%J.err
#SBATCH --gpus=1
#SBATCH --mail-user=gavrilev@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --mem=40G

SETTING="smi_emb_gen"
SMILES_PATH="/home/gavrilev/Thesis/src/saved_objects/rsim_unique_smiles.pkl"
SAVE_NAME="smiles_emb"
DEVICE="cuda"
DATE=0404
EXP_ID=1
LOG_FILE_NAME="smi_emb_gen_${DATE}_exp${EXP_ID}"

module load anaconda
module load cuda/11.8

source activate chemprop

cd /home/gavrilev/Thesis/src || exit 1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HOME=/home/gavrilev/.cache/huggingface

nvcc --version
nvidia-smi
pwd

python /home/gavrilev/Thesis/src/main.py \
  --log_file_name "$LOG_FILE_NAME" \
  --setting "$SETTING" \
  --smiles_path "$SMILES_PATH" \
  --save_name "$SAVE_NAME" \
  --device "$DEVICE"