#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition=main
#SBATCH --time=0-48:00:00
#SBATCH --job-name=cold_rm_best
#SBATCH --output=jobs/job-%J.out
#SBATCH --error=jobs/job-%J.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mail-user=gavrilev@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --mem=24G

############################
### Experiment variables ###
############################

EXP_ID=2   # experiment ID
DATE=1811  # for log & result directory creation

MODEL_SETTING="full"   # ["full", "freezed", "w/o seq", "w/o 3D struct", "w/o fusion"]
EXCLUDE_MODALITIES="morgan ernie"
FREEZE_GRNADE="False"

TRAIN_SETTING=4
VAL_SETTING=4
MOL_AUG="False"
RNA_AUG="False"
IS_SEQ_EMB="False"

RES_FILE_NAME="cold_rm_${DATE}"
SAVE_MODEL="True"
IS_PLOT="False"
PLOT_NAME="${DATE}_exp${EXP_ID}"
LOG_FILE_NAME="cold_${DATE}_exp${EXP_ID}"

########
# Best hyperparameters from the global full-model search
LR=2e-5
WD=1e-6
DROPOUT=0.1
########

########
# Best joint cold-start early-stopping hyperparameters (1905, exp4)
PATIENCE=20
WARM_UP_EPOCHS=50
DELTA=1e-6
########

#####
# RNA module hyperparameters
D_MODEL=128
FFN_MULT=4
BDCF_BLOCKS=2
NODE_POOL='mean'
TOKEN_POOL='mean'
#####

# SMILES module hyperparameters
SMILES_SEQ_SETTING='deep'
SMILES_EXPERT_SETTING='proj'

BATCH_SIZE=1
NUM_WORKERS=0

EXP_NAME="exp${EXP_ID}_cold_rm_best"

SETTING="cold"
COLD_TYPE="cold_rm"
DEVICE="cuda"

SEED=2
DATASET_SEED=2

LOG_MSG="cold_rm_${DATE}_exp${EXP_ID} - MultiStructRSA (${MODEL_SETTING} version) - TRAIN SETTING=${TRAIN_SETTING}; VAL SETTING=${VAL_SETTING}; RNA AUG=${RNA_AUG}; MOL AUG=${MOL_AUG}; PATIENCE=${PATIENCE}; WARM UP=${WARM_UP_EPOCHS}; DELTA=${DELTA}; SEED=${SEED}; DATASET SEED=${DATASET_SEED}"

################  Following lines will be executed by a compute node  ################

module load anaconda
module load cuda/11.8

source activate chemprop
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1

export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HOME=/home/gavrilev/.cache/huggingface

nvcc --version
nvidia-smi

python main.py \
  --exp_name "$EXP_NAME" \
  --model_setting "$MODEL_SETTING" \
  --exclude_modalities $EXCLUDE_MODALITIES \
  --freeze_grnade "$FREEZE_GRNADE" \
  --train_setting "$TRAIN_SETTING" \
  --val_setting "$VAL_SETTING" \
  --rna_aug "$RNA_AUG" \
  --mol_aug "$MOL_AUG" \
  --is_seq_emb "$IS_SEQ_EMB" \
  --patience "$PATIENCE" \
  --warm_up_epochs "$WARM_UP_EPOCHS" \
  --delta "$DELTA" \
  --res_file_name "$RES_FILE_NAME" \
  --save_model "$SAVE_MODEL" \
  --is_plot "$IS_PLOT" \
  --plot_name "$PLOT_NAME" \
  --log_msg "$LOG_MSG" \
  --log_file_name "$LOG_FILE_NAME" \
  --setting "$SETTING" \
  --cold_type "$COLD_TYPE" \
  --lr "$LR" \
  --wd "$WD" \
  --dropout "$DROPOUT" \
  --smiles_seq_setting "$SMILES_SEQ_SETTING" \
  --smiles_expert_setting "$SMILES_EXPERT_SETTING" \
  --d_model "$D_MODEL" \
  --ffn_mult "$FFN_MULT" \
  --bdcf_blocks "$BDCF_BLOCKS" \
  --node_pool "$NODE_POOL" \
  --token_pool "$TOKEN_POOL" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --dataset_seed "$DATASET_SEED"
