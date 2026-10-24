#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <number_of_folds> <run_tag>" >&2
  exit 2
fi

K_FOLD="$1"
RUN_TAG="$2"

if ! [[ "$RUN_TAG" =~ ^[A-Za-z0-9.-]+$ ]]; then
  echo "run_tag may contain only letters, numbers, dots, and hyphens (no underscores)" >&2
  exit 2
fi

if ! [[ "$K_FOLD" =~ ^[0-9]+$ ]] || (( K_FOLD < 2 )); then
  echo "number_of_folds must be an integer greater than or equal to 2" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# User-controlled output labels. main.py writes them below results/.
RESULT_NAME="cv_${RUN_TAG}"
LOG_NAME="cv_${RUN_TAG}"
PLOT_NAME="${RUN_TAG}_cv"
EXP_NAME="multistructrsa_stratified_${K_FOLD}fold_${RUN_TAG}"

# The remaining values are the fixed final MultiStructRSA configuration.
python main.py \
  --setting cv \
  --cv_method all \
  --k_fold "$K_FOLD" \
  --exp_name "$EXP_NAME" \
  --res_file_name "$RESULT_NAME" \
  --save_predictions True \
  --log_file_name "$LOG_NAME" \
  --log_msg "MultiStructRSA stratified ${K_FOLD}-fold CV (${RUN_TAG})" \
  --is_plot True \
  --plot_name "$PLOT_NAME" \
  --model_setting full \
  --exclude_modalities morgan ernie \
  --train_setting 4 \
  --val_setting 4 \
  --rna_aug False \
  --mol_aug True \
  --is_seq_emb False \
  --lr 2e-5 \
  --wd 1e-6 \
  --dropout 0.1 \
  --patience 20 \
  --warm_up_epochs 50 \
  --delta 1e-6 \
  --d_model 128 \
  --ffn_mult 4 \
  --bdcf_blocks 2 \
  --node_pool mean \
  --token_pool mean \
  --smiles_seq_setting deep \
  --smiles_expert_setting proj \
  --freeze_grnade False \
  --batch_size 1 \
  --num_workers 0 \
  --device cuda \
  --seed 1 \
  --dataset_seed 2
