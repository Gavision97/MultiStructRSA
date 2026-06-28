#!/bin/bash

################################################################################################
### SLURM configuration
################################################################################################

#SBATCH --partition=main
#SBATCH --time=0-72:00:00
#SBATCH --job-name=external_eval
#SBATCH --output=jobs/external_eval-%J.out
#SBATCH --error=jobs/external_eval-%J.err
#SBATCH --gpus=rtx_4090:1
#SBATCH --mail-user=gavrilev@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --mem=24G

############################
### Experiment variables ###
############################

EXP_ID=7
DATE=1006

MODEL_SETTING="full"
EXCLUDE_MODALITIES="morgan ernie"

TRAIN_SETTING=4
VAL_SETTING=4

RNA_AUG="False"
MOL_AUG="False"
IS_SEQ_EMB="False"

RES_FILE_NAME="external_${DATE}"
IS_PLOT="False"
PLOT_NAME="${DATE}_external_exp${EXP_ID}"
LOG_FILE_NAME="external_${DATE}_exp${EXP_ID}"

############################
### Training parameters  ###
############################

# Same hyperparameters used for cold-start evaluation
LR=2e-5
WD=1e-6
DROPOUT=0.1

PATIENCE=20
WARM_UP_EPOCHS=15
DELTA=0.001

############################
### Model parameters     ###
############################

D_MODEL=128
FFN_MULT=4
BDCF_BLOCKS=2
NODE_POOL="mean"
TOKEN_POOL="mean"

SMILES_SEQ_SETTING="deep"
SMILES_EXPERT_SETTING="proj"

BATCH_SIZE=1
NUM_WORKERS=0

############################
### Evaluation parameters ###
############################

SETTING="external"
DEVICE="cuda"

SEED=2
DATASET_SEED=2

EXP_NAME="exp${EXP_ID}_external_pdbbind_patience=${PATIENCE}_delta=${DELTA}"

LOG_MSG="external_validation_${DATE}_exp${EXP_ID} - Train on R-SIM cold-RNA folds and evaluate on PDBBindNL2020 - MultiStructRSA (${MODEL_SETTING}) - exclude_modalities=${EXCLUDE_MODALITIES}; TRAIN_SETTING=${TRAIN_SETTING}; VAL_SETTING=${VAL_SETTING}; RNA_AUG=${RNA_AUG}; MOL_AUG=${MOL_AUG}; SEED=${SEED}; DATASET_SEED=${DATASET_SEED}"

################################################################################################
### Commands executed on the compute node
################################################################################################

module load anaconda
module load cuda/11.8

source activate chemprop

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1

export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HOME=/home/gavrilev/.cache/huggingface

nvcc --version
nvidia-smi

python main.py \
  --exp_name "$EXP_NAME" \
  --model_setting "$MODEL_SETTING" \
  --exclude_modalities $EXCLUDE_MODALITIES \
  --train_setting "$TRAIN_SETTING" \
  --val_setting "$VAL_SETTING" \
  --rna_aug "$RNA_AUG" \
  --mol_aug "$MOL_AUG" \
  --patience "$PATIENCE" \
  --warm_up_epochs "$WARM_UP_EPOCHS" \
  --delta "$DELTA" \
  --is_seq_emb "$IS_SEQ_EMB" \
  --res_file_name "$RES_FILE_NAME" \
  --is_plot "$IS_PLOT" \
  --plot_name "$PLOT_NAME" \
  --log_msg "$LOG_MSG" \
  --log_file_name "$LOG_FILE_NAME" \
  --setting "$SETTING" \
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

