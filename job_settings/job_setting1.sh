#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition=main
#SBATCH --time=0-48:00:00
#SBATCH --job-name=train_cold_models
#SBATCH --output=jobs/job-%J.out
#SBATCH --error=jobs/job-%J.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mail-user=gavrilev@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --mem=24G

############################
### Experiment variables ###
############################

EXP_ID=1
DATE=0109

MODEL_SETTING="full"   # ["full", "freezed", "w/o seq", "w/o 3D struct", "w/o fusion"]
EXCLUDE_MODALITIES="morgan ernie"   # e.g. simson / chemberta / mordred / morgan / rinalmo / ernie / grnade
FREEZE_GRNADE="False"

TRAIN_SETTING=4
VAL_SETTING=4
MOL_AUG="False"
RNA_AUG="False"
IS_SEQ_EMB="False"

IS_PLOT="True"
PLOT_NAME="${DATE}_exp${EXP_ID}"
LOG_FILE_NAME="cold_train_${DATE}_exp${EXP_ID}"

# Best hyperparameters from global hyperparameter search
LR=2e-5
WD=1e-6
DROPOUT=0.1

# Best RNA module hyperparameters
D_MODEL=128
FFN_MULT=4
BDCF_BLOCKS=2
NODE_POOL="mean"
TOKEN_POOL="mean"

# Best SMILES module hyperparameters
SMILES_SEQ_SETTING="deep"
SMILES_EXPERT_SETTING="proj"

BATCH_SIZE=1
NUM_WORKERS=0
MAX_EPOCHS=200

EXP_NAME="exp${EXP_ID}_exclude=${EXCLUDE_MODALITIES}"

SETTING="train_cold"
COLD_TYPE="cold_rna"   # ["cold_rna", "cold_mole", "cold_rm"]
MODEL_SAVE_DIR="saved_objects/trained_cold/${DATE}/exp${EXP_ID}_${COLD_TYPE}"
DEVICE="cuda"

PATIENCE=20
WARM_UP_EPOCHS=50
DELTA=1e-6

SEED=1

LOG_MSG="train_cold_${DATE}_exp${EXP_ID} - COLD_TYPE=${COLD_TYPE}; MultiStructRSA (${MODEL_SETTING}); exclude_modalities=${EXCLUDE_MODALITIES}; TRAIN_SETTING=${TRAIN_SETTING}; VAL_SETTING=${VAL_SETTING}; RNA_AUG=${RNA_AUG}; MOL_AUG=${MOL_AUG}; SEED=${SEED}"

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

python main.py \
  --exp_name "$EXP_NAME" \
  --exp_id "$EXP_ID" \
  --date "$DATE" \
  --model_setting "$MODEL_SETTING" \
  --exclude_modalities $EXCLUDE_MODALITIES \
  --freeze_grnade "$FREEZE_GRNADE" \
  --train_setting "$TRAIN_SETTING" \
  --val_setting "$VAL_SETTING" \
  --rna_aug "$RNA_AUG" \
  --mol_aug "$MOL_AUG" \
  --patience "$PATIENCE" \
  --warm_up_epochs "$WARM_UP_EPOCHS" \
  --delta "$DELTA" \
  --is_seq_emb "$IS_SEQ_EMB" \
  --is_plot "$IS_PLOT" \
  --plot_name "$PLOT_NAME" \
  --log_msg "$LOG_MSG" \
  --log_file_name "$LOG_FILE_NAME" \
  --setting "$SETTING" \
  --cold_type "$COLD_TYPE" \
  --max_epochs "$MAX_EPOCHS" \
  --model_save_dir "$MODEL_SAVE_DIR" \
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
  --seed "$SEED"
