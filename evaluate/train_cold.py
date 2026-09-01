"""Train five models on the predefined cold-start train/validation folds.

Despite their filenames, ``cold_*_test_foldN.csv`` files are used only as
validation sets for early stopping in this script. No test evaluation is run.

Example
-------
python evaluate/train_cold.py --cold-type cold_rna --device cuda
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Allow direct execution via ``python evaluate/train_cold.py``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.MultiStructRSA import choose_model_setting
from utils.MolRNACDataset import MolRNADataset
from utils.tools import set_seed
from utils.utils import (
    EarlyStopping,
    predict,
    rmse_loss,
    safe_regression_metrics,
    train_one_epoch,
)


LOGGER = logging.getLogger(__name__)
N_FOLDS = 5
MODE = "min"

COLD_TYPE_ALIASES = {
    "rna": "cold_rna",
    "cold_rna": "cold_rna",
    "mol": "cold_mole",
    "mole": "cold_mole",
    "molecule": "cold_mole",
    "cold_mol": "cold_mole",
    "cold_mole": "cold_mole",
    "rm": "cold_rm",
    "both": "cold_rm",
    "cold_rm": "cold_rm",
}

COLD_TYPE_PATHS = {
    "cold_rna": ("rna", "rna"),
    "cold_mole": ("mole", "mole"),
    "cold_rm": ("rm", "rm"),
}

DATA_COLUMNS = ["SMILES", "Target_RNA_sequence", "Molecule_name", "pKd"]


def canonical_cold_type(value: str) -> str:
    """Normalize convenient cold-start aliases to the repository names."""
    try:
        return COLD_TYPE_ALIASES[value.lower()]
    except KeyError as exc:
        valid = ", ".join(sorted(COLD_TYPE_ALIASES))
        raise argparse.ArgumentTypeError(
            f"Unknown cold-start type {value!r}. Choose one of: {valid}"
        ) from exc


def get_split_paths(cold_type: str, fold: int) -> tuple[Path, Path]:
    """Return the train and validation CSV paths for one cold-start fold."""
    directory_name, file_name = COLD_TYPE_PATHS[cold_type]
    split_dir = (
        PROJECT_ROOT
        / "datasets"
        / "blind_test"
        / directory_name
        / "train_test_splits"
    )
    return (
        split_dir / f"cold_{file_name}_train_fold{fold}.csv",
        split_dir / f"cold_{file_name}_test_fold{fold}.csv",
    )


def validate_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cold_type: str,
    fold: int,
) -> None:
    """Fail early if the selected cold entities leak into validation."""
    cold_columns = {
        "cold_rna": ["Target_RNA_sequence"],
        "cold_mole": ["Molecule_name"],
        "cold_rm": ["Target_RNA_sequence", "Molecule_name"],
    }[cold_type]

    for column in cold_columns:
        overlap = set(train_df[column].dropna()) & set(val_df[column].dropna())
        if overlap:
            raise ValueError(
                f"Fold {fold} is not a valid {cold_type} split: "
                f"{len(overlap)} {column!r} values occur in both train and validation"
            )


def model_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build the kwargs needed to create (and later reload) the model."""
    return {
        "option": args.model_setting,
        "exclude_modalities": args.exclude_modalities,
        "freeze_grnade": args.freeze_grnade,
        "smiles_seq_setting": args.smiles_seq_setting,
        "smiles_expert_setting": args.smiles_expert_setting,
        "dropout": args.dropout,
        "d_model": args.d_model,
        "ffn_mult": args.ffn_mult,
        "bdcf_blocks": args.bdcf_blocks,
        "node_pool": args.node_pool,
        "token_pool": args.token_pool,
    }


def save_checkpoint(
    path: Path,
    state_dict: dict[str, torch.Tensor],
    args: argparse.Namespace,
    fold: int,
    train_path: Path,
    val_path: Path,
    early_stopping: EarlyStopping,
) -> None:
    """Save portable CPU weights plus all information needed for reconstruction."""
    cpu_state_dict = {
        name: value.detach().cpu() for name, value in state_dict.items()
    }
    p_val, s_val, r2_val, rmse_val = early_stopping.best_metrics
    checkpoint = {
        "format_version": 1,
        "model_state_dict": cpu_state_dict,
        "model_config": model_config_from_args(args),
        "cold_type": args.cold_type,
        "fold": fold,
        "best_epoch": early_stopping.best_epoch + 1,
        "validation_metrics": {
            "pearson": float(p_val),
            "spearman": float(s_val),
            "rmse": float(rmse_val),
            "r2": float(r2_val),
        },
        "data_config": {
            "train_csv": str(train_path),
            "validation_csv": str(val_path),
            "train_setting": args.train_setting,
            "val_setting": args.val_setting,
            "is_seq_emb": args.is_seq_emb,
        },
        "training_config": {
            "seed": args.seed,
            "learning_rate": args.lr,
            "weight_decay": args.wd,
            "batch_size": args.batch_size,
            "rna_aug": args.rna_aug,
            "mol_aug": args.mol_aug,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "warm_up_epochs": args.warm_up_epochs,
            "delta": args.delta,
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def train_cold_models(args: argparse.Namespace) -> list[Path]:
    """Train one early-stopped model for each of the five predefined folds."""
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Run with --device cpu instead."
        )

    output_dir = args.output_dir or (
        PROJECT_ROOT / "saved_objects" / f"trained_{args.cold_type}"
    )
    output_dir = Path(output_dir).expanduser().resolve()
    model_config = model_config_from_args(args)

    # Check all inputs before beginning a potentially long training run.
    missing_paths = [
        path
        for fold in range(1, N_FOLDS + 1)
        for path in get_split_paths(args.cold_type, fold)
        if not path.is_file()
    ]
    if missing_paths:
        missing = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing cold-start split file(s):\n{missing}")

    saved_paths: list[Path] = []
    LOGGER.info(
        "Training %d %s models; checkpoints will be written to %s",
        N_FOLDS,
        args.cold_type,
        output_dir,
    )

    for fold in range(1, N_FOLDS + 1):
        set_seed(seed_=args.seed)
        train_path, val_path = get_split_paths(args.cold_type, fold)

        train_df = pd.read_csv(train_path).loc[:, DATA_COLUMNS].copy()
        val_df = pd.read_csv(val_path).loc[:, DATA_COLUMNS].copy()
        if train_df.empty or val_df.empty:
            raise ValueError(
                f"Fold {fold} contains an empty split: "
                f"train={len(train_df)}, validation={len(val_df)}"
            )
        validate_splits(train_df, val_df, args.cold_type, fold)

        LOGGER.info(
            "Fold %d/%d: train=%d, validation=%d (validation source: %s)",
            fold,
            N_FOLDS,
            len(train_df),
            len(val_df),
            val_path,
        )

        train_ds = MolRNADataset(
            train_df,
            setting=args.train_setting,
            rna_aug=args.rna_aug,
            mol_aug=args.mol_aug,
            task="training",
            is_seq_emb=args.is_seq_emb,
        )
        val_ds = MolRNADataset(
            val_df,
            setting=args.val_setting,
            rna_aug=False,
            mol_aug=False,
            task="validation",
            is_seq_emb=args.is_seq_emb,
        )

        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=MolRNADataset.collate_fn,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=MolRNADataset.collate_fn,
            num_workers=args.num_workers,
        )

        model = choose_model_setting(**model_config).to(args.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wd
        )
        early_stopping = EarlyStopping(
            mode=MODE,
            patience=args.patience,
            warm_up_epochs=args.warm_up_epochs,
            delta=args.delta,
        )

        for epoch in range(args.max_epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, rmse_loss, args.device
            )
            y_true_val, y_pred_val = predict(model, val_loader, args.device)
            p_val, s_val, rmse_val, r2_val = safe_regression_metrics(
                y_true_val, y_pred_val
            )
            early_stopping.check_early_stop(
                model,
                rmse_val,
                [p_val, s_val, r2_val, rmse_val],
                train_loss,
                epoch,
                [y_true_val, y_pred_val],
            )

            if early_stopping.stop_training:
                LOGGER.info(
                    "Fold %d/%d: early stopping at epoch %d; "
                    "best validation RMSE %.4f at epoch %d",
                    fold,
                    N_FOLDS,
                    epoch + 1,
                    early_stopping.best_score,
                    early_stopping.best_epoch + 1,
                )
                break

        if early_stopping.best_model is False:
            raise RuntimeError(
                f"Fold {fold} produced no finite validation RMSE; no model was saved"
            )

        checkpoint_path = output_dir / f"model{fold}.pt"
        save_checkpoint(
            checkpoint_path,
            early_stopping.best_model,
            args,
            fold,
            train_path,
            val_path,
            early_stopping,
        )
        saved_paths.append(checkpoint_path)
        LOGGER.info(
            "Fold %d/%d: saved best model to %s",
            fold,
            N_FOLDS,
            checkpoint_path,
        )

        # Release fold-specific GPU memory before constructing the next model.
        del model, optimizer, train_loader, val_loader, train_ds, val_ds
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    return saved_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train five models using predefined cold-start train/test CSVs, "
            "treating each test CSV as validation for early stopping."
        )
    )
    parser.add_argument(
        "--cold-type",
        "--cold_type",
        type=canonical_cold_type,
        default="cold_rna",
        help="Cold-start type: rna/cold_rna, mole/cold_mole, or rm/cold_rm",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--warm-up-epochs", type=int, default=10)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--train-setting", type=int, default=4)
    parser.add_argument("--val-setting", type=int, default=4)
    parser.add_argument("--rna-aug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mol-aug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--is-seq-emb", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument(
        "--model-setting",
        default="full",
        choices=[
            "full",
            "freezed",
            "w/o seq",
            "w/o 3D struct",
            "w/o fusion",
            "full batched",
        ],
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, choices=[128, 256], default=128)
    parser.add_argument("--ffn-mult", type=int, choices=[2, 4, 6], default=4)
    parser.add_argument("--bdcf-blocks", type=int, choices=[1, 2], default=1)
    parser.add_argument("--node-pool", choices=["attn", "mean"], default="attn")
    parser.add_argument("--token-pool", choices=["attn", "mean"], default="mean")
    parser.add_argument(
        "--smiles-seq-setting",
        choices=["proj", "shallow", "deep"],
        default="deep",
    )
    parser.add_argument(
        "--smiles-expert-setting",
        choices=["proj", "shallow", "deep"],
        default="proj",
    )
    parser.add_argument(
        "--exclude-modalities",
        nargs="*",
        default=[],
        choices=[
            "simson",
            "chemberta",
            "mordred",
            "morgan",
            "rinalmo",
            "ernie",
            "grnade",
        ],
    )
    parser.add_argument(
        "--freeze-grnade", action=argparse.BooleanOptionalAction, default=False
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_epochs < 1:
        raise ValueError("--max-epochs must be at least 1")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1")

    # Dataset/model asset paths elsewhere in the project are repository-relative.
    os.chdir(PROJECT_ROOT)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    saved_paths = train_cold_models(args)
    LOGGER.info("Training complete. Saved %d models:", len(saved_paths))
    for path in saved_paths:
        LOGGER.info("  %s", path)


if __name__ == "__main__":
    main()
