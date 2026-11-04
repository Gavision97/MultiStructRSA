#!/bin/bash

# Reproduction of sensitivity_analysis_1904_exp4.
# Run this with the April source revision, where evaluate/cv_eval.py has
# MAX_EPOCHS = 1000. Output names use "repro1904" to preserve the old results.

#SBATCH --partition=main
#SBATCH --time=0-48:00:00
#SBATCH --job-name=reproduce_1904_exp4
#SBATCH --output=jobs/job-%J.out
#SBATCH --error=jobs/job-%J.err
#SBATCH --gpus=rtx_6000:1
#SBATCH --mail-user=gavrilev@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --mem=24G

EXP_ID=4
DATE=repro1904

MODEL_SETTING="full"
EXCLUDE_MODALITIES=("morgan" "ernie")

TRAIN_SETTING=4
VAL_SETTING=4
MOL_AUG="False"
RNA_AUG="False"
IS_SEQ_EMB="False"

RES_FILE_NAME="cv_all_${DATE}"
PLOT_NAME="${DATE}_exp${EXP_ID}"
LOG_FILE_NAME="sensitivity_${DATE}_exp${EXP_ID}"

LR=2e-5
WD=1e-6
DROPOUT=0.1

D_MODEL=128
FFN_MULT=4
BDCF_BLOCKS=2
NODE_POOL="mean"
TOKEN_POOL="mean"

SMILES_SEQ_SETTING="deep"
SMILES_EXPERT_SETTING="proj"

BATCH_SIZE=1
NUM_WORKERS=0

EXP_NAME="exp${EXP_ID}_exclude=morgan_ernie_seq=${SMILES_SEQ_SETTING}_expert=${SMILES_EXPERT_SETTING}"

SETTING="cv"
CV_METHOD="all"
K_FOLD=5
DEVICE="cuda"

SEED=1
DATASET_SEED=2

LOG_MSG="stratified_cv_${DATE}_exp${EXP_ID} - MultiStructRSA (${MODEL_SETTING}) - exclude_modalities=morgan ernie; TRAIN_SETTING=${TRAIN_SETTING}; VAL_SETTING=${VAL_SETTING}; RNA_AUG=${RNA_AUG}; MOL_AUG=${MOL_AUG}; SEED=${SEED}; DATASET_SEED=${DATASET_SEED}"

if ! grep -Eq '^MAX_EPOCHS[[:space:]]*=[[:space:]]*1000([[:space:]]*#.*)?$' evaluate/cv_eval.py; then
  echo "Refusing to run: evaluate/cv_eval.py must use MAX_EPOCHS = 1000 for the April reproduction." >&2
  exit 1
fi

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
  --exclude_modalities "${EXCLUDE_MODALITIES[@]}" \
  --train_setting "$TRAIN_SETTING" \
  --val_setting "$VAL_SETTING" \
  --rna_aug "$RNA_AUG" \
  --mol_aug "$MOL_AUG" \
  --is_seq_emb "$IS_SEQ_EMB" \
  --res_file_name "$RES_FILE_NAME" \
  --plot_name "$PLOT_NAME" \
  --log_msg "$LOG_MSG" \
  --log_file_name "$LOG_FILE_NAME" \
  --setting "$SETTING" \
  --cv_method "$CV_METHOD" \
  --k_fold "$K_FOLD" \
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
