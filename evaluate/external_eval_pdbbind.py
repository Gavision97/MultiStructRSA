import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import logging
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from utils.utils import *
from utils.tools import set_seed
from utils.utils import rmse_loss
from utils.plot_utils import plot_train_val_loss, plot_prediction_correlation
from utils.MolRNACDataset import MolRNADataset
from utils.MolRNACDatasetPDBBind import MolRNACDatasetPDBBind
from models.MultiStructRSA import *


logger = logging.getLogger(__name__)


MODE = "min"
MAX_EPOCHS = 200

RNA_COLD_PATH = "/home/gavrilev/Thesis/src/datasets/blind_test/rna/train_test_splits"
PDBBIND_PATH = "/home/gavrilev/Thesis/src/datasets/PDBBind2020/pdbbindnl2020_preprocessed.csv"
PREDICTIONS_PATH = "/home/gavrilev/Thesis/src/results/pdbbind_external_predictions.csv"
FOLD_RESULTS_PATH = "/home/gavrilev/Thesis/src/results/pdbbind_external_fold_results.csv"


def normalize_rna_sequences(df):
    df = df.copy()
    df["Target_RNA_sequence"] = df["Target_RNA_sequence"].astype(str).str.strip().str.upper().str.replace("T", "U", regex=False).str.replace("Y", "U", regex=False).str.replace("X", "U", regex=False)
    return df


def remove_train_validation_overlap(train_df, val_df, fold):
    train_rnas = set(train_df["Target_RNA_sequence"])
    val_rnas = set(val_df["Target_RNA_sequence"])
    overlap = train_rnas & val_rnas

    if overlap:
        rows_before = len(train_df)
        logger.warning(f"Fold {fold}: found {len(overlap)} normalized RNA overlap(s) between training and validation: {sorted(overlap)}")

        train_df = train_df[~train_df["Target_RNA_sequence"].isin(overlap)].copy().reset_index(drop=True)

        logger.warning(f"Fold {fold}: removed {rows_before - len(train_df)} training row(s) to preserve cold-RNA validation.")

    overlap_after = set(train_df["Target_RNA_sequence"]) & set(val_df["Target_RNA_sequence"])
    assert len(overlap_after) == 0, f"Fold {fold}: train-validation RNA overlap remains after filtering."

    return train_df


def perform_pdbbind_external_eval(model_setting="full", train_setting=4, val_setting=4, rna_aug=False, mol_aug=False, patience=20, warm_up_epochs=10, delta=1e-5, lr=5e-5, wd=1e-5, dropout=0.1, exclude_modalities=None, freeze_grnade=False, smiles_seq_setting="deep", smiles_expert_setting="proj", batch_size=1, num_workers=8, is_plot=False, is_seq_emb=False, d_model=128, ffn_mult=4, bdcf_blocks=1, node_pool="attn", token_pool="mean", plot_name="pdbbind_external", device="cuda", seed=1):

    logger.info(f"External validation on PDBBindNL2020 | seed={seed}, batch_size={batch_size}, max_epochs={MAX_EPOCHS}, lr={lr}, weight_decay={wd}, patience={patience}, warm_up_epochs={warm_up_epochs}, delta={delta}")

    ########################################################################
    # Load the fixed PDBBindNL2020 external test set
    ########################################################################

    pdbbind_df = pd.read_csv(PDBBIND_PATH).copy()

    required_columns = {"SMILES", "Target_RNA_sequence", "pKd"}
    missing_columns = required_columns - set(pdbbind_df.columns)

    if missing_columns:
        raise ValueError(f"PDBBind dataframe is missing required columns: {sorted(missing_columns)}")

    pdbbind_df = normalize_rna_sequences(pdbbind_df)

    logger.info(f"PDBBind external set | rows={len(pdbbind_df)}, unique RNAs={pdbbind_df['Target_RNA_sequence'].nunique()}, unique molecules={pdbbind_df['SMILES'].nunique()}")

    external_ds = MolRNACDatasetPDBBind(pdbbind_df, task="external evaluation", is_seq_emb=is_seq_emb)
    external_loader = DataLoader(external_ds, batch_size=1, shuffle=False, collate_fn=MolRNACDatasetPDBBind.collate_fn, num_workers=num_workers)

    ########################################################################
    # Result containers
    ########################################################################

    p_list = []
    s_list = []
    rmse_list = []
    r2_list = []
    best_epoch_list = []

    fold_results = []
    all_predictions = []

    ########################################################################
    # Train five models using R-SIM cold-RNA folds
    ########################################################################

    for fold in range(1, 6):
        set_seed(seed_=seed)

        logger.info(f"Fold number {fold}/5")

        train_path = os.path.join(RNA_COLD_PATH, f"cold_rna_train_fold{fold}.csv")
        val_path = os.path.join(RNA_COLD_PATH, f"cold_rna_test_fold{fold}.csv")

        train_df = pd.read_csv(train_path).loc[:, ["SMILES", "Target_RNA_sequence", "Molecule_name", "pKd"]].copy()
        val_df = pd.read_csv(val_path).loc[:, ["SMILES", "Target_RNA_sequence", "Molecule_name", "pKd"]].copy()

        ####################################################################
        # Verify that PDBBind remains an independent external test set
        ####################################################################

        train_rnas = set(train_df["Target_RNA_sequence"])
        val_rnas = set(val_df["Target_RNA_sequence"])
        external_rnas = set(pdbbind_df["Target_RNA_sequence"])

        train_external_overlap = train_rnas & external_rnas
        val_external_overlap = val_rnas & external_rnas
        rsim_external_overlap = (train_rnas | val_rnas) & external_rnas

        if train_external_overlap:
            logger.error(f"Fold {fold}: training-PDBBind RNA overlaps: {sorted(train_external_overlap)}")

        if val_external_overlap:
            logger.error(f"Fold {fold}: validation-PDBBind RNA overlaps: {sorted(val_external_overlap)}")

        assert len(rsim_external_overlap) == 0, f"Fold {fold}: found {len(rsim_external_overlap)} RNA overlap(s) between R-SIM and PDBBindNL2020: {sorted(rsim_external_overlap)}"
        
        train_df = normalize_rna_sequences(train_df)
        val_df = normalize_rna_sequences(val_df)
        logger.info(f"(|train|, |validation|, |external test|) -> ({len(train_df)}, {len(val_df)}, {len(pdbbind_df)})")
        logger.info(f"(unique RNA train, validation, external test) -> ({train_df['Target_RNA_sequence'].nunique()}, {val_df['Target_RNA_sequence'].nunique()}, {pdbbind_df['Target_RNA_sequence'].nunique()})")

        ####################################################################
        # Datasets and loaders
        ####################################################################

        train_ds = MolRNADataset(train_df, setting=train_setting, rna_aug=rna_aug, mol_aug=mol_aug, task="training", is_seq_emb=is_seq_emb)
        val_ds = MolRNADataset(val_df, setting=val_setting, rna_aug=False, mol_aug=False, task="evaluating", is_seq_emb=is_seq_emb)

        generator = torch.Generator().manual_seed(seed)

        train_loader = DataLoader(train_ds, batch_size=batch_size, generator=generator, shuffle=True, collate_fn=MolRNADataset.collate_fn, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=MolRNADataset.collate_fn, num_workers=num_workers)

        ####################################################################
        # Model and optimization
        ####################################################################

        early_stopping = EarlyStopping(mode=MODE, patience=patience, warm_up_epochs=warm_up_epochs, delta=delta)

        model = choose_model_setting(option=model_setting, exclude_modalities=exclude_modalities, freeze_grnade=freeze_grnade, dropout=dropout, smiles_seq_setting=smiles_seq_setting, smiles_expert_setting=smiles_expert_setting, d_model=d_model, ffn_mult=ffn_mult, bdcf_blocks=bdcf_blocks, node_pool=node_pool, token_pool=token_pool).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = rmse_loss

        train_losses = []
        val_losses = []

        ####################################################################
        # Training and early stopping on the R-SIM validation fold
        ####################################################################

        for epoch in range(MAX_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            train_losses.append(train_loss)

            y_true_val, y_pred_val = predict(model, val_loader, device)
            p_val, s_val, rmse_val, r2_val = safe_regression_metrics(y_true_val, y_pred_val)

            val_losses.append(rmse_val)

            early_stopping.check_early_stop(model, rmse_val, [p_val, s_val, r2_val, rmse_val], train_loss, epoch, [y_true_val, y_pred_val])

            if early_stopping.stop_training:
                logger.info(f"[Fold {fold}/5] Early stopping at epoch {epoch + 1}. Best validation RMSE={early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch + 1}")
                break

        if early_stopping.best_model is None:
            raise RuntimeError(f"No best model was stored for fold {fold}.")

        ####################################################################
        # Restore the best validation model
        ####################################################################

        model.load_state_dict(early_stopping.best_model)

        best_epoch = early_stopping.best_epoch + 1
        best_epoch_list.append(best_epoch)

        ####################################################################
        # Evaluate on PDBBindNL2020
        ####################################################################

        y_true_external, y_pred_external = predict(model, external_loader, device)
        p_external, s_external, rmse_external, r2_external = safe_regression_metrics(y_true_external, y_pred_external)

        logger.info(f"[Fold {fold}/5] PDBBindNL2020 external evaluation: p={p_external:.4f}, s={s_external:.4f}, rmse={rmse_external:.4f}, r2={r2_external:.4f}")

        p_list.append(p_external)
        s_list.append(s_external)
        rmse_list.append(rmse_external)
        r2_list.append(r2_external)

        fold_results.append({"fold": fold, "train_rows": len(train_df), "validation_rows": len(val_df), "external_rows": len(pdbbind_df), "best_epoch": best_epoch, "p": p_external, "s": s_external, "rmse": rmse_external, "r2": r2_external})

        ####################################################################
        # Store per-sample predictions
        ####################################################################

        prediction_columns = ["SMILES", "Target_RNA_sequence", "pKd"]

        if "pdb_file" in pdbbind_df.columns:
            prediction_columns.append("pdb_file")

        fold_predictions = pdbbind_df[prediction_columns].copy()
        fold_predictions["fold"] = fold
        fold_predictions["y_true"] = np.asarray(y_true_external).reshape(-1)
        fold_predictions["y_pred"] = np.asarray(y_pred_external).reshape(-1)

        all_predictions.append(fold_predictions)

        ####################################################################
        # Optional plots
        ####################################################################

        if is_plot:
            plots_dir = os.path.join("results", "plots", "pdbbind_external")
            os.makedirs(plots_dir, exist_ok=True)

            plot_train_val_loss(train_losses, val_losses, save_path=os.path.join(plots_dir, f"losses_{plot_name}_fold_{fold}.png"), title="R-SIM Training and Validation RMSE")
            plot_prediction_correlation(y_true_external, y_pred_external, save_path=os.path.join(plots_dir, f"predictions_{plot_name}_fold_{fold}.png"), title="PDBBindNL2020 External Evaluation")

        del model
        del optimizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ########################################################################
    # Save detailed outputs
    ########################################################################

    os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)

    pd.concat(all_predictions, ignore_index=True).to_csv(PREDICTIONS_PATH, index=False)
    pd.DataFrame(fold_results).to_csv(FOLD_RESULTS_PATH, index=False)

    ########################################################################
    # Mean and standard deviation across the five trained models
    ########################################################################

    res = {
        "p_mean": round(float(np.mean(p_list)), 5),
        "p_std": round(float(np.std(p_list)), 5),
        "s_mean": round(float(np.mean(s_list)), 5),
        "s_std": round(float(np.std(s_list)), 5),
        "rmse_mean": round(float(np.mean(rmse_list)), 5),
        "rmse_std": round(float(np.std(rmse_list)), 5),
        "r2_mean": round(float(np.mean(r2_list)), 5),
        "r2_std": round(float(np.std(r2_list)), 5),
        "best_epochs": best_epoch_list,
        "per_fold": {
            "p": p_list,
            "s": s_list,
            "rmse": rmse_list,
            "r2": r2_list,
        },
    }

    logger.info("PDBBindNL2020 external-validation results:")
    logger.info(res)
    logger.info(f"Per-sample predictions saved to: {PREDICTIONS_PATH}")
    logger.info(f"Per-fold results saved to: {FOLD_RESULTS_PATH}")

    return res

