
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from evaluate.evaluate_utils import (train_one_epoch, predict, EarlyStopping,
                                     regressor_stratified_cv, plot_train_val_loss, safe_regression_metrics)

#from utils.MolRNADataset import MolRNADataset # w/o RNA cross-attention fusion mechanism
from utils.MolRNACDataset import MolRNADataset # w RNA cross-attention fusion mechanism
from utils.neg_smoo_aug import *
from models.MultiStructRSA import *

import logging
logger = logging.getLogger(__name__)

# same seeds as in DeepRSMA
#SEED = 1
#SEED_DATASET = 2 

# early stopping hyperparameters: 
PATIENCE = 30 # for early stopping
WARM_UP_EPOCHS = 30
MODE = 'min'
DELTA = 0 # 0.0001 ;for early stopping (11/03, 1e-4 -> 5e-5 -> 1e-6)
MIN_EPOCHS = 50

# MultiStructRSA training hyperparameters:
LR = 0.00005 # from 0.00005 -> 0.0001
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 1
MAX_EPOCHS = 1000
NUM_WORKERS = 8

COLD_PATH = os.path.join("datasets", "blind_test", "processed")
RSIM_PATH = os.path.join("datasets", "RSIM")

ALL_SF_DATASET_PATH = os.path.join("datasets", "RSIM", "All_sf_dataset_v1.csv")

# same style as their code
def set_seed(seed_):
    random.seed(seed_)
    os.environ["PYTHONHASHSEED"] = str(seed_)
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    torch.cuda.manual_seed(seed_)
    torch.cuda.manual_seed_all(seed_)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# custom RMSE loss function
def rmse_loss(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

def cv_eval_ns(
    dataset=None,
    model_setting="full",
    train_setting=4,
    val_setting=4,
    aug=False,
    smoothing_factor=0.75,
    negative_factor=1,
    lr=5e-5,
    wd=1e-5,
    is_plot=False,
    is_seq_emb=False,
    n=10,
    val_ratio=0.1,
    fam_name=None,
    plot_name="loss_curve",
    device="cuda",
    dataset_seed=2,
    seed=1
):
    """
    Outer CV:
        outer train/test split
    Inside each outer fold:
        outer train -> inner train/val
        train on inner train
        early stop on inner val
        load/use best model
        evaluate once on outer test
    """
    logging.info(f'Smoothing Factor -> {smoothing_factor}, Negative Factor -> {negative_factor}')
    logger.info(
        f'K folds:{n}, seed:{seed}, dataset seed:{dataset_seed}, batch size:{BATCH_SIZE}, '
        f'max epochs:{MAX_EPOCHS}, lr:{lr}, weight decay:{wd}, '
        f'patience:{PATIENCE}, warm up epo:{WARM_UP_EPOCHS}, delta:{DELTA}, val_ratio:{val_ratio}'
    )

    set_seed(seed)
    all_df = dataset
    logging.info(f"--- Start stratified {n}-fold cross-validation; pd.DataFrame shape -> {all_df.shape}, lr={lr} ---")

    train_ds = MolRNADataset(all_df, setting=train_setting, aug=aug, task="training", is_seq_emb=is_seq_emb)
    #eval_ds = MolRNADataset(all_df, setting=val_setting, aug=False, task="evaluating", is_seq_emb=is_seq_emb)

    Y = train_ds.data["pKd"].to_numpy(dtype=np.float32)

    skf = regressor_stratified_cv(
        n_splits=n,
        n_repeats=1,
        random_state=dataset_seed,
        group_count=5,
        strategy="uniform"
    )

    p_list, s_list, rmse_list, r2_list = [], [], [], []
    per_fold_results = []

    for fold, (train_index, test_index) in enumerate(skf.split(np.zeros(len(Y)), Y)):
        logger.info(f"\nFold number {fold+1}/{n}")

        # -----------------------------
        # Outer fold split: train / test
        # -----------------------------
        train_outer_df = all_df.iloc[train_index].copy().reset_index(drop=True)
        test_df = all_df.iloc[test_index].copy().reset_index(drop=True)

        # ---------------------------------------------------------
        # Negative sampling on OUTER TRAIN only, before val split
        # ---------------------------------------------------------
        aug_train_outer_df, synthetic_neg_df = generate_synthetic_negatives_for_train_fold(
            train_df=train_outer_df,
            rna_col="Target_RNA_sequence",
            ligand_col="SMILES",
            pkd_col="pKd",
            threshold=4.0,
            alpha=smoothing_factor,
            negatives_per_positive=negative_factor,   # change as needed
            random_state=seed + fold,
            max_tries_per_negative=200,
        )

        logger.info(
            f"Outer train rows before augmentation: {len(train_outer_df)} | "
            f"synthetic negatives: {len(synthetic_neg_df)} | "
            f"outer train rows after augmentation: {len(aug_train_outer_df)}"
        )

        # ---------------------------------------------------------
        # Inner train/val split is now done on the AUGMENTED train
        # ---------------------------------------------------------
        y_aug = aug_train_outer_df["pKd"].to_numpy(dtype=np.float32)
        bins = pd.qcut(y_aug, q=5, labels=False, duplicates="drop")

        inner_train_df, val_df = train_test_split(
            aug_train_outer_df,
            test_size=val_ratio,
            random_state=seed + fold,
            shuffle=True,
            stratify=bins
        )

        logger.info(
            f"Inner split sizes -> train: {len(inner_train_df)}, "
            f"val: {len(val_df)}, test: {len(test_df)}"
        )

        # -----------------------------------
        # Build datasets from the split dfs
        # -----------------------------------
        train_ds = MolRNADataset(
            inner_train_df.reset_index(drop=True),
            setting=train_setting,
            aug=aug,
            task="training",
            is_seq_emb=is_seq_emb
        )

        val_ds = MolRNADataset(
            val_df.reset_index(drop=True),
            setting=val_setting,
            aug=False,
            task="evaluating",
            is_seq_emb=is_seq_emb
        )

        test_ds = MolRNADataset(
            test_df.reset_index(drop=True),
            setting=val_setting,
            aug=False,
            task="evaluating",
            is_seq_emb=is_seq_emb
        )

        g = torch.Generator().manual_seed(seed + fold)

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            generator=g,
            shuffle=True,
            collate_fn=MolRNADataset.collate_fn,
            num_workers=NUM_WORKERS
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=MolRNADataset.collate_fn,
            num_workers=NUM_WORKERS
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=MolRNADataset.collate_fn,
            num_workers=NUM_WORKERS
        )

        model = choose_model_setting(option=model_setting).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = rmse_loss

        early_stopping = EarlyStopping(
            mode=MODE,
            patience=PATIENCE,
            warm_up_epochs=WARM_UP_EPOCHS,
            delta=DELTA
        )

        train_losses, val_losses = [], []

        for epo in range(MAX_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            train_losses.append(train_loss)

            y_val_true, y_val_pred = predict(model, val_loader, device)
            val_p, val_s, val_rmse, val_r2 = safe_regression_metrics(y_val_true, y_val_pred)
            val_losses.append(val_rmse)

            early_stopping.check_early_stop(
                model, val_rmse, [val_p, val_s, val_r2], train_loss, epo
            )

            if early_stopping.stop_training:
                logger.info(
                    f"[Fold {fold+1}/{n}] Early stopping at epoch {epo+1}. "
                    f"Best val_rmse={early_stopping.best_score:.4f} "
                    f"@ epoch {early_stopping.best_epoch+1}"
                )
                break

        best_model = early_stopping.best_model if early_stopping.best_model is not None else model
        best_model = best_model.to(device)
        best_model.eval()

        y_test_true, y_test_pred = predict(best_model, test_loader, device)
        test_p, test_s, test_rmse, test_r2 = safe_regression_metrics(y_test_true, y_test_pred)

        logger.info(
            f"[Fold {fold+1}/{n}] TEST -> "
            f"p={test_p:.4f}, s={test_s:.4f}, rmse={test_rmse:.4f}, r2={test_r2:.4f}"
        )

        if is_plot:
            date, plot = plot_name.split('_')
            if fam_name:
                plots_dir = os.path.join("results", "plots", date, plot, fam_name)
            else:
                plots_dir = os.path.join("results", "plots", date, plot)
            os.makedirs(plots_dir, exist_ok=True)

            plot_train_val_loss(
                train_losses,
                val_losses,
                save_path=f"{plots_dir}/{plot_name}_fold_{fold+1}.png",
                title="Loss Curve (train vs. val RMSE)"
            )

        p_list.append(float(test_p))
        s_list.append(float(test_s))
        rmse_list.append(float(test_rmse))
        r2_list.append(float(test_r2))

        per_fold_results.append({
            "fold": fold + 1,
            "best_epoch": int(early_stopping.best_epoch + 1) if early_stopping.best_epoch is not None else None,
            "p": float(test_p),
            "s": float(test_s),
            "rmse": float(test_rmse),
            "r2": float(test_r2),
        })

    logger.info(f"p: {np.mean(p_list)}")
    logger.info(f"s: {np.mean(s_list)}")
    logger.info(f"rmse: {np.mean(rmse_list)}")
    logger.info(f"r2: {np.mean(r2_list)}")

    res = {
        "p_mean": round(float(np.mean(p_list)), 5),
        "p_std": round(float(np.std(p_list)), 5),
        "s_mean": round(float(np.mean(s_list)), 5),
        "s_std": round(float(np.std(s_list)), 5),
        "rmse_mean": round(float(np.mean(rmse_list)), 5),
        "rmse_std": round(float(np.std(rmse_list)), 5),
        "r2_mean": round(float(np.mean(r2_list)), 5),
        "r2_std": round(float(np.std(r2_list)), 5),
        "per_fold": per_fold_results
    }

    logger.info("\nStratified k-fold cross validation:")
    for k, v in res.items():
        logger.info(f"{k}: {v}")

    return res
