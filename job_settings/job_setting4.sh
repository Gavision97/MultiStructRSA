#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition=main
#SBATCH --time=0-72:00:00
#SBATCH --job-name=case_study_mc_dropout
#SBATCH --output=jobs/job-%J.out
#SBATCH --error=jobs/job-%J.err
#SBATCH --gpus=rtx_4090:1
#SBATCH --mail-user=gavrilev@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --mem=24G

############################
### Experiment variables ###
############################

SETTING="screen"
MC_SAMPLES=100
NUM_WORKERS=0
DEVICE="cuda"

LOG_FILE_NAME="screen_case_study_mc${MC_SAMPLES}"
LOG_MSG="ChemDiv MC Dropout screening against SLI with ${MC_SAMPLES} stochastic samples per molecule"

################  Following lines will be executed by a compute node  ################

module load anaconda
module load cuda/11.8

source activate chemprop

PROJECT_DIR="/home/gavrilev/Thesis/src"
cd "$PROJECT_DIR" || exit 1
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1

export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HOME=/home/gavrilev/.cache/huggingface

nvcc --version
nvidia-smi

# main.py dispatches setting="screen" to evaluate/eval_case_study.py.
python main.py \
  --setting "$SETTING" \
  --mc_samples "$MC_SAMPLES" \
  --num_workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --log_file_name "$LOG_FILE_NAME" \
  --log_msg "$LOG_MSG"
