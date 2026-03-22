#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition=main
#SBATCH --time=0-72:00:00
#SBATCH --job-name=research_project
#SBATCH --output=jobs/job-%J.out
#SBATCH --error=jobs/job-%J.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mail-user=gavrilev@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --mem=24G

############################
### Experiment variables ###
############################

EXP_ID=5   # experiment ID
DATE=2103  # for log & result directory creation

MODEL_SETTING="w/o fusion" # ["full", "freezed", "w/o seq", "w/o 3D struct", "w/o fusion"]
TRAIN_SETTING=4
VAL_SETTING=4
AUG="False"
IS_SEQ_EMB="True"

EXP_NAME="exp${EXP_ID}"  

RES_FILE_NAME="cv_all_${DATE}"
PLOT_NAME="${DATE}_exp${EXP_ID}"

LOG_FILE_NAME="stratified_cv_${DATE}_exp${EXP_ID}"

EVAL_METHOD="cv"
CV_METHOD="all"
K_FOLD=10
DEVICE="cuda"

SEED=1
DATASET_SEED=2

LOG_MSG="stratified_cv_${DATE}_exp${EXP_ID} - MultiStructRSA (${MODEL_SETTING} version) - TRAIN SETTING=${TRAIN_SETTING}; VAL SETTING=${VAL_SETTING}; AUG=${AUG}; SEED=${SEED}; DATASET SEED=${DATASET_SEED}"



################  Following lines will be executed by a compute node  ################

module load anaconda
module load cuda/11.8

source activate chemprop
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HOME=/home/gavrilev/.cache/huggingface

nvcc --version
nvidia-smi

python main.py \
  --exp_name "$EXP_NAME" \
  --model_setting "$MODEL_SETTING" \
  --train_setting "$TRAIN_SETTING" \
  --val_setting "$VAL_SETTING" \
  --aug "$AUG" \
  --is_seq_emb "$IS_SEQ_EMB" \
  --res_file_name "$RES_FILE_NAME" \
  --plot_name "$PLOT_NAME" \
  --log_msg "$LOG_MSG" \
  --log_file_name "$LOG_FILE_NAME" \
  --eval_method "$EVAL_METHOD" \
  --cv_method "$CV_METHOD" \
  --k_fold "$K_FOLD" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --dataset_seed "$DATASET_SEED"
