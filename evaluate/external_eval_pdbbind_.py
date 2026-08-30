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

RSIM_PATH = "/home/gavrilev/Thesis/src/datasets/RSIM/All_sf_dataset_v1.csv"
PDBBIND_PATH = "/home/gavrilev/Thesis/src/datasets/PDBBind2020/pdbbindnl2020_preprocessed.csv"
RESULTS_ROOT = "/home/gavrilev/Thesis/src/results/external_validation_pdbbind"


def normalize_rna_sequences(df):
    df = df.copy()
    df["Target_RNA_sequence"] = df["Target_RNA_sequence"].astype(str).str.strip().str.upper().str.replace("T", "U", regex=False).str.replace("Y", "U", regex=False).str.replace("X", "U", regex=False)
    return df


def perform_pdbbind_external_eval(exp_id, date, model_setting="full", train_setting=4, val_setting=4, rna_aug=False, mol_aug=False, patience=20, warm_up_epochs=10, delta=1e-5, lr=5e-5, wd=1e-5, dropout=0.1, exclude_modalities=None, freeze_grnade=False, smiles_seq_setting="deep", smiles_expert_setting="proj", batch_size=1, num_workers=8, is_plot=False, is_seq_emb=False, d_model=128, ffn_mult=4, bdcf_blocks=1, node_pool="attn", token_pool="mean", n=5, plot_name="pdbbind_external", device="cuda", dataset_seed=2, seed=1):
    """Train on stratified R-SIM folds and evaluate each best model on PDBBind."""

    output_dir = os.path.join(RESULTS_ROOT, str(date))
    predictions_path = os.path.join(output_dir, f"pdbbind_external_predictions_exp{exp_id}.csv")
    fold_results_path = os.path.join(output_dir, f"pdbbind_external_fold_results_exp{exp_id}.csv")

    logger.info(f"External validation on PDBBindNL2020 | folds={n}, seed={seed}, dataset_seed={dataset_seed}, batch_size={batch_size}, max_epochs={MAX_EPOCHS}, lr={lr}, weight_decay={wd}, patience={patience}, warm_up_epochs={warm_up_epochs}, delta={delta}")

    if n < 2:
        raise ValueError("n must be at least 2 for stratified cross-validation")

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

    ########################################################################
    # Load R-SIM and build the same regression-stratified folds as cv_eval.py
    ########################################################################

    rsim_df = pd.read_csv(RSIM_PATH, sep="\t").copy()
    missing_columns = required_columns - set(rsim_df.columns)

    if missing_columns:
        raise ValueError(f"R-SIM dataframe is missing required columns: {sorted(missing_columns)}")

    rsim_df = normalize_rna_sequences(rsim_df)
    y = rsim_df["pKd"].to_numpy(dtype=np.float32)

    rsim_external_overlap = set(rsim_df["Target_RNA_sequence"]) & set(pdbbind_df["Target_RNA_sequence"])
    assert len(rsim_external_overlap) == 0, f"Found {len(rsim_external_overlap)} RNA overlap(s) between R-SIM and PDBBindNL2020: {sorted(rsim_external_overlap)}"

    splitter = regressor_stratified_cv(n_splits=n, n_repeats=1, random_state=dataset_seed, group_count=5, strategy="uniform")

    logger.info(f"R-SIM training data | rows={len(rsim_df)}, unique RNAs={rsim_df['Target_RNA_sequence'].nunique()}, unique molecules={rsim_df['SMILES'].nunique()}")

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
    # Train one model for each regression-stratified R-SIM fold
    ########################################################################

    for fold, (train_index, val_index) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
        set_seed(seed_=seed)

        logger.info(f"Fold number {fold}/{n}")

        train_df = rsim_df.iloc[train_index].reset_index(drop=True)
        val_df = rsim_df.iloc[val_index].reset_index(drop=True)

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
                logger.info(f"[Fold {fold}/{n}] Early stopping at epoch {epoch + 1}. Best validation RMSE={early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch + 1}")
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

        logger.info(f"[Fold {fold}/{n}] PDBBindNL2020 external evaluation: p={p_external:.4f}, s={s_external:.4f}, rmse={rmse_external:.4f}, r2={r2_external:.4f}")

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

    os.makedirs(output_dir, exist_ok=True)

    pd.concat(all_predictions, ignore_index=True).to_csv(predictions_path, index=False)
    pd.DataFrame(fold_results).to_csv(fold_results_path, index=False)

    ########################################################################
    # Mean and standard deviation across the trained models
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
    logger.info(f"Per-sample predictions saved to: {predictions_path}")
    logger.info(f"Per-fold results saved to: {fold_results_path}")

    return res
