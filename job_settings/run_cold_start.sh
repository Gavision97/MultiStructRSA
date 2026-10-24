#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <cold_rna|cold_mole|cold_rm> <run_tag>" >&2
  exit 2
fi

COLD_TYPE="$1"
RUN_TAG="$2"

if ! [[ "$RUN_TAG" =~ ^[A-Za-z0-9.-]+$ ]]; then
  echo "run_tag may contain only letters, numbers, dots, and hyphens (no underscores)" >&2
  exit 2
fi

case "$COLD_TYPE" in
  cold_rna|cold_mole|cold_rm) ;;
  *)
    echo "Invalid cold-start type: $COLD_TYPE" >&2
    echo "Choose one of: cold_rna, cold_mole, cold_rm" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# User-controlled output labels. main.py writes them below results/.
RESULT_NAME="cold_${RUN_TAG}"
LOG_NAME="cold_${RUN_TAG}"
PLOT_NAME="${RUN_TAG}_cold"
EXP_NAME="multistructrsa_${COLD_TYPE}_${RUN_TAG}"

# The remaining values are the fixed final MultiStructRSA configuration.
python main.py \
  --setting cold \
  --cold_type "$COLD_TYPE" \
  --exp_name "$EXP_NAME" \
  --res_file_name "$RESULT_NAME" \
  --log_file_name "$LOG_NAME" \
  --log_msg "MultiStructRSA cold-start evaluation: ${COLD_TYPE} (${RUN_TAG})" \
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
  --warm_up_epochs 10 \
  --delta 1e-4 \
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
  --seed 2 \
  --dataset_seed 2
