import os
import pandas as pd
import torch

from torch.utils.data import DataLoader

from utils.utils import *
from utils.tools import set_seed
from utils.utils import rmse_loss
from utils.plot_utils import plot_train_val_loss, plot_prediction_correlation
from utils.MolRNACDataset import MolRNADataset
from models.MultiStructRSA import *


import logging
logger = logging.getLogger(__name__)


MODE = "min"
N_FOLDS = 5

RNA_COLD_PATH = os.path.join("datasets", "blind_test", "rna", "train_test_splits")
MOLE_COLD_PATH = os.path.join("datasets", "blind_test", "mole", "train_test_splits")
BOTH_COLD_PATH = os.path.join("datasets", "blind_test", "rm", "train_test_splits")
MODEL_SAVE_ROOT = os.path.join("saved_objects", "trained_cold")

DATA_COLUMNS = ["SMILES", "Target_RNA_sequence", "Molecule_name", "pKd"]


def choose_cold_type(cold_type="cold_rna"):
    assert cold_type in ["cold_rna", "cold_mole", "cold_rm"]

    if cold_type == "cold_rna":
        return RNA_COLD_PATH, "rna", ["Target_RNA_sequence"]
    elif cold_type == "cold_mole":
        return MOLE_COLD_PATH, "mole", ["Molecule_name"]
    else:
        return BOTH_COLD_PATH, "rm", ["Target_RNA_sequence", "Molecule_name"]


def validate_cold_split(train_df, val_df, cold_columns, fold):
    for column in cold_columns:
        overlap = set(train_df[column].dropna()) & set(val_df[column].dropna())
        if overlap:
            raise ValueError(
                f"Fold {fold} is not a valid cold split: {len(overlap)} "
                f"{column} values occur in both training and validation"
            )


def train_cold_models(
    exp_name="exp",
    exp_id=1,
    date="undated",
    cold_type="cold_rna",
    model_setting="full",
    train_setting=4,
    val_setting=4,
    rna_aug=False,
    mol_aug=False,
    patience=20,
    warm_up_epochs=10,
    delta=1e-5,
    lr=5e-5,
    wd=1e-5,
    dropout=0.1,
    exclude_modalities=None,
    freeze_grnade=False,
    smiles_seq_setting="deep",
    smiles_expert_setting="proj",
    batch_size=1,
    num_workers=0,
    is_plot=False,
    is_seq_emb=False,
    d_model=128,
    ffn_mult=4,
    bdcf_blocks=1,
    node_pool="attn",
    token_pool="mean",
    max_epochs=200,
    plot_name="train_cold",
    model_save_dir=None,
    device="cuda",
    seed=1,
):
    """
    Train and save one model for each predefined cold-start fold.

    The held-out ``cold_*_test_foldN.csv`` split is used as validation for
    early stopping. This function does not report it as an independent test
    result.
    """

    if max_epochs < 1:
        raise ValueError("max_epochs must be at least 1")
    if patience < 1:
        raise ValueError("patience must be at least 1")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    cold_path, cold_name, cold_columns = choose_cold_type(cold_type=cold_type)

    if model_save_dir is None:
        experiment_name = f"{exp_name}_exp{exp_id}_{cold_type}"
        model_save_dir = os.path.join(MODEL_SAVE_ROOT, str(date), experiment_name)

    os.makedirs(model_save_dir, exist_ok=True)

    logger.info(
        f"Training cold-start models | cold type:{cold_type}, folds:{N_FOLDS}, "
        f"seed:{seed}, batch size:{batch_size}, max epochs:{max_epochs}, "
        f"lr:{lr}, weight decay:{wd}, dropout:{dropout}, num_workers:{num_workers}, "
        f"patience:{patience}, warm up epo:{warm_up_epochs}, delta:{delta}"
    )
    logger.info(f"Model checkpoints will be saved to: {model_save_dir}")

    ########################################################################
    # Validate that all fold files exist before training starts
    ########################################################################

    split_paths = []

    for fold in range(1, N_FOLDS + 1):
        train_path = os.path.join(cold_path, f"cold_{cold_name}_train_fold{fold}.csv")
        val_path = os.path.join(cold_path, f"cold_{cold_name}_test_fold{fold}.csv")
        split_paths.append((train_path, val_path))

    missing_paths = [path for paths in split_paths for path in paths if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError(f"Missing cold-start split files: {missing_paths}")

    fold_results = []
    checkpoint_paths = []

    ########################################################################
    # Train one model for each predefined cold-start fold
    ########################################################################

    for fold, (train_path, val_path) in enumerate(split_paths, start=1):
        set_seed(seed_=seed)

        logger.info(f"Fold number {fold}/{N_FOLDS}")

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        missing_columns = set(DATA_COLUMNS) - set(train_df.columns)
        missing_columns |= set(DATA_COLUMNS) - set(val_df.columns)
        if missing_columns:
            raise ValueError(
                f"Fold {fold} is missing required columns: {sorted(missing_columns)}"
            )

        train_df = train_df.loc[:, DATA_COLUMNS].copy()
        val_df = val_df.loc[:, DATA_COLUMNS].copy()

        if train_df.empty or val_df.empty:
            raise ValueError(
                f"Fold {fold} contains an empty split: "
                f"train={len(train_df)}, validation={len(val_df)}"
            )

        validate_cold_split(train_df, val_df, cold_columns, fold)

        logger.info(
            f"(|train|, |validation|) -> ({len(train_df)}, {len(val_df)})"
        )

        if cold_type == "cold_rna":
            logger.info(
                f"(unique RNA train, validation) -> "
                f"({train_df['Target_RNA_sequence'].nunique()}, "
                f"{val_df['Target_RNA_sequence'].nunique()})"
            )
        elif cold_type == "cold_mole":
            logger.info(
                f"(unique molecule names train, validation) -> "
                f"({train_df['Molecule_name'].nunique()}, "
                f"{val_df['Molecule_name'].nunique()})"
            )
        else:
            logger.info(
                f"(unique RNA train, validation) -> "
                f"({train_df['Target_RNA_sequence'].nunique()}, "
                f"{val_df['Target_RNA_sequence'].nunique()})"
            )
            logger.info(
                f"(unique molecule names train, validation) -> "
                f"({train_df['Molecule_name'].nunique()}, "
                f"{val_df['Molecule_name'].nunique()})"
            )

        ####################################################################
        # Datasets and loaders
        ####################################################################

        train_ds = MolRNADataset(
            train_df,
            setting=train_setting,
            rna_aug=rna_aug,
            mol_aug=mol_aug,
            task="training",
            is_seq_emb=is_seq_emb,
        )
        val_ds = MolRNADataset(
            val_df,
            setting=val_setting,
            rna_aug=False,
            mol_aug=False,
            task="validation",
            is_seq_emb=is_seq_emb,
        )

        generator = torch.Generator().manual_seed(seed)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            generator=generator,
            shuffle=True,
            collate_fn=MolRNADataset.collate_fn,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=MolRNADataset.collate_fn,
            num_workers=num_workers,
        )

        ####################################################################
        # Model, optimizer, and early stopping
        ####################################################################

        early_stopping = EarlyStopping(
            mode=MODE,
            patience=patience,
            warm_up_epochs=warm_up_epochs,
            delta=delta,
        )

        model = choose_model_setting(
            option=model_setting,
            exclude_modalities=exclude_modalities,
            freeze_grnade=freeze_grnade,
            dropout=dropout,
            smiles_seq_setting=smiles_seq_setting,
            smiles_expert_setting=smiles_expert_setting,
            d_model=d_model,
            ffn_mult=ffn_mult,
            bdcf_blocks=bdcf_blocks,
            node_pool=node_pool,
            token_pool=token_pool,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = rmse_loss

        train_losses = []
        val_losses = []

        ####################################################################
        # Train and select the best validation model
        ####################################################################

        for epoch in range(max_epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            train_losses.append(train_loss)

            y_true_val, y_pred_val = predict(model, val_loader, device)
            p_val, s_val, rmse_val, r2_val = safe_regression_metrics(
                y_true_val, y_pred_val
            )
            val_losses.append(rmse_val)

            early_stopping.check_early_stop(
                model,
                rmse_val,
                [p_val, s_val, r2_val, rmse_val],
                train_loss,
                epoch,
                [y_true_val, y_pred_val],
            )

            if early_stopping.stop_training:
                logger.info(
                    f"[Fold {fold}/{N_FOLDS}] Early stopping at epoch "
                    f"{epoch + 1}. Best validation RMSE="
                    f"{early_stopping.best_score:.4f} at epoch "
                    f"{early_stopping.best_epoch + 1}"
                )
                break

        if early_stopping.best_model is False:
            raise RuntimeError(
                f"No finite validation model was stored for fold {fold}"
            )

        ####################################################################
        # Save best fold checkpoint and validation metadata
        ####################################################################

        p_best, s_best, r2_best, rmse_best = early_stopping.best_metrics
        checkpoint_path = os.path.join(model_save_dir, f"model_fold_{fold}.pt")

        best_model_state = {
            name: value.detach().cpu()
            for name, value in early_stopping.best_model.items()
        }

        checkpoint = {
            "model_state_dict": best_model_state,
            "model_config": {
                "option": model_setting,
                "exclude_modalities": exclude_modalities,
                "freeze_grnade": freeze_grnade,
                "dropout": dropout,
                "smiles_seq_setting": smiles_seq_setting,
                "smiles_expert_setting": smiles_expert_setting,
                "d_model": d_model,
                "ffn_mult": ffn_mult,
                "bdcf_blocks": bdcf_blocks,
                "node_pool": node_pool,
                "token_pool": token_pool,
            },
            "training_config": {
                "exp_name": exp_name,
                "exp_id": exp_id,
                "date": date,
                "cold_type": cold_type,
                "fold": fold,
                "seed": seed,
                "train_setting": train_setting,
                "val_setting": val_setting,
                "rna_aug": rna_aug,
                "mol_aug": mol_aug,
                "is_seq_emb": is_seq_emb,
                "lr": lr,
                "wd": wd,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "max_epochs": max_epochs,
                "patience": patience,
                "warm_up_epochs": warm_up_epochs,
                "delta": delta,
            },
            "data_config": {
                "train_path": train_path,
                "validation_path": val_path,
            },
            "best_epoch": early_stopping.best_epoch + 1,
            "validation_metrics": {
                "p": float(p_best),
                "s": float(s_best),
                "rmse": float(rmse_best),
                "r2": float(r2_best),
            },
        }

        torch.save(checkpoint, checkpoint_path)
        checkpoint_paths.append(checkpoint_path)

        fold_results.append(
            {
                "fold": fold,
                "train_rows": len(train_df),
                "validation_rows": len(val_df),
                "best_epoch": early_stopping.best_epoch + 1,
                "p": float(p_best),
                "s": float(s_best),
                "rmse": float(rmse_best),
                "r2": float(r2_best),
                "checkpoint_path": checkpoint_path,
            }
        )

        logger.info(
            f"[Fold {fold}/{N_FOLDS}] Saved best model to: {checkpoint_path}"
        )

        ####################################################################
        # Optional validation plots
        ####################################################################

        if is_plot:
            plots_dir = os.path.join(
                "results", "plots", str(date), "train_cold", cold_type
            )
            os.makedirs(plots_dir, exist_ok=True)

            plot_train_val_loss(
                train_losses,
                val_losses,
                save_path=os.path.join(
                    plots_dir, f"losses_{plot_name}_fold_{fold}.png"
                ),
                title="Training and Validation RMSE",
            )
            plot_prediction_correlation(
                early_stopping.y_true,
                early_stopping.y_pred,
                save_path=os.path.join(
                    plots_dir, f"validation_{plot_name}_fold_{fold}.png"
                ),
                title="Best Validation Model",
            )

        del model, optimizer, train_loader, val_loader, train_ds, val_ds
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    ########################################################################
    # Save fold-level validation summary
    ########################################################################

    fold_results_path = os.path.join(model_save_dir, "fold_results.csv")
    pd.DataFrame(fold_results).to_csv(fold_results_path, index=False)

    logger.info(f"Finished training {len(checkpoint_paths)} cold-start models")
    logger.info(f"Fold validation results saved to: {fold_results_path}")

    return {
        "model_save_dir": model_save_dir,
        "checkpoint_paths": checkpoint_paths,
        "fold_results_path": fold_results_path,
    }
