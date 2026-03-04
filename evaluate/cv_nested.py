import os
import copy
import random
import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from evaluate_utils import train_one_epoch, predict

from utils.MolRNADataset import MolRNADataset
from models.MultiMRSMACross import MultiMRSMA
logger = logging.getLogger(__name__)


# same seed as in DeepRSMA
seed = 1

# set random seed
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=20)
set_seed(seed)


# stratified CV
class regressor_stratified_cv:
    def __init__(self,n_splits=10,n_repeats=2,group_count=10,random_state=0,strategy='quantile'):
        self.group_count=group_count
        self.strategy=strategy
        self.cvkwargs=dict(n_splits=n_splits,n_repeats=n_repeats,random_state=random_state)  #Added shuffle here
        self.cv=RepeatedStratifiedKFold(**self.cvkwargs)
        self.discretizer=KBinsDiscretizer(n_bins=self.group_count,encode='ordinal',strategy=self.strategy)  
            
    def split(self,X,y,groups=None):
        kgroups=self.discretizer.fit_transform(y[:,None])[:,0]
        return self.cv.split(X,kgroups,groups)
    
    def get_n_splits(self,X,y,groups=None):
        return self.cv.get_n_splits(X,y,groups)




# -------------------------
# inner splitter for regression
# (prevents "least populated class has only 1 member" errors)
# -------------------------
def make_inner_splitter_regression(y_train, n_splits, n_bins=5, strategy="uniform", random_state=0):
    """
    Try StratifiedKFold on binned y_train.
    If bins are too sparse (some bin count < n_splits), reduce bins.
    If still impossible, fall back to KFold.
    """
    y_train = np.asarray(y_train, dtype=np.float32)
    n = len(y_train)
    n_splits = int(min(n_splits, n))

    if n_splits < 2:
        raise ValueError("Inner CV needs at least 2 splits.")

    max_bins = int(min(n_bins, n))
    for bins in range(max_bins, 1, -1):
        try:
            binner = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy=strategy)
            y_bins = binner.fit_transform(y_train[:, None])[:, 0].astype(int)
        except Exception:
            continue

        _, counts = np.unique(y_bins, return_counts=True)
        if counts.min() >= n_splits:
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), y_bins

    # fallback (no stratification)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state), None


# -------------------------
# Train with early stopping for ONE (train, val) split -> best_epoch
# -------------------------
def early_stop_best_epoch(
    ds,
    tr_idx,
    va_idx,
    device,
    batch_size=16,
    lr=1e-4,
    weight_decay=1e-4,
    max_epochs=1500,
    patience=30,
    dropout=0.1,
):
    train_loader = DataLoader(
        Subset(ds, tr_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=MolRNADataset.collate_fn,
    )
    val_loader = DataLoader(
        Subset(ds, va_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=MolRNADataset.collate_fn,
    )

    model = MultiMRSMA(dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_epoch = 1
    bad = 0

    for epoch in range(max_epochs):
        _ = train_one_epoch(model, train_loader, optimizer, criterion, device)

        yv_true, yv_pred = predict(model, val_loader, device)
        val_mse = mean_squared_error(yv_true, yv_pred)

        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch + 1  # 1-based
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    return best_epoch


# -------------------------
# INNER CV inside outer fold -> choose epoch* (median of best epochs)
# -------------------------
def choose_epoch_inner_cv(
    ds,
    train_index,
    Y,
    device,
    inner_splits=3,
    inner_bins=5,
    inner_strategy="uniform",   # match DeepRSMA backbone style if you want
    random_state=0,
    batch_size=16,
    max_epochs=500,
    patience=30,
    dropout=0.1,
    lr=1e-4,
    weight_decay=1e-4,
):
    train_index = np.asarray(train_index)
    y_train = Y[train_index]

    splitter, y_bins = make_inner_splitter_regression(
        y_train,
        n_splits=inner_splits,
        n_bins=inner_bins,
        strategy=inner_strategy,
        random_state=random_state,
    )

    X_dummy = np.zeros(len(train_index), dtype=np.float32)

    if y_bins is None:
        split_iter = splitter.split(X_dummy)
    else:
        split_iter = splitter.split(X_dummy, y_bins)

    best_epochs = []
    for inner_fold, (tr_rel, va_rel) in enumerate(split_iter, start=1):
        tr_idx = train_index[tr_rel]
        va_idx = train_index[va_rel]

        be = early_stop_best_epoch(
            ds, tr_idx, va_idx,
            device=device,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            dropout=dropout,
        )
        best_epochs.append(be)
        logger.info(f'Inner Fold -> {inner_fold+1}, Best Epoch -> {be}')
    epoch_star = int(np.median(best_epochs))
    epoch_star = max(1, epoch_star)
    return epoch_star, best_epochs


# -------------------------
# OUTER CV evaluation (same outer splitter as DeepRSMA)
# -------------------------
def cv_eval_outer_regressor_stratified_with_inner_epoch(
    df,
    name,
    device="cuda",
    seed=42,

    # outer backbone params (KEEP COMPARABLE)
    outer_splits=10,
    outer_bins=5,
    outer_strategy="uniform",   # DeepRSMA used uniform in the example

    # inner epoch selection params
    inner_splits=3,
    inner_bins=5,
    inner_strategy="uniform",

    # training params
    batch_size=16,
    max_epochs_inner=500,
    patience_inner=30,
    dropout=0.1,
    lr=1e-4,
    weight_decay=1e-4,
):
    print(f"--- Start OUTER CV (DeepRSMA backbone) for {name} --- ({df.shape})")
    logging.info(f"--- Start OUTER CV (DeepRSMA backbone) for {name} --- ({df.shape})")

    ds = MolRNADataset(df)
    Y = ds.data["pKd"].to_numpy(dtype=np.float32)

    # same data split as RSAPred & DeepRSMA
    kf = regressor_stratified_cv(n_splits=outer_splits, n_repeats=1, random_state=seed, group_count=outer_bins, strategy=outer_strategy)

    mae_val, mse_val, r2_val, pearson_val = [], [], [], []

    X_dummy = np.zeros(len(Y), dtype=np.float32)

    for fold, (train_index, test_index) in enumerate(kf.split(X_dummy, Y), start=1):
        print(f"\nOuter Fold {fold}")
        logging.info(f"\nOuter Fold {fold}")

        # ---------------------------
        # INNER CV: choose epoch*
        # ---------------------------
        epoch_star, inner_epochs = choose_epoch_inner_cv(
            ds=ds,
            train_index=train_index,
            Y=Y,
            device=device,
            inner_splits=inner_splits,
            inner_bins=inner_bins,
            inner_strategy=inner_strategy,
            random_state=seed + 1000 + fold,
            batch_size=batch_size,
            max_epochs=max_epochs_inner,
            patience=patience_inner,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
        )

        print(f"[Outer {fold}] inner best_epochs={inner_epochs} -> epoch*={epoch_star}")
        logging.info(f"[Outer {fold}] inner best_epochs={inner_epochs} -> epoch*={epoch_star}")

        # ---------------------------
        # Refit on FULL outer-train
        # ---------------------------
        full_train_loader = DataLoader(
            Subset(ds, train_index),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=MolRNADataset.collate_fn,
        )
        test_loader = DataLoader(
            Subset(ds, test_index),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=MolRNADataset.collate_fn,
        )

        model = MultiMRSMA(dropout=dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        for epoch in range(epoch_star):
            _ = train_one_epoch(model, full_train_loader, optimizer, criterion, device)

        # ---------------------------
        # Evaluate on outer test fold
        # ---------------------------
        yt_true, yt_pred = predict(model, test_loader, device)

        mae = mean_absolute_error(yt_true, yt_pred)
        mse = mean_squared_error(yt_true, yt_pred)
        r2  = r2_score(yt_true, yt_pred)
        r   = pearsonr(yt_true, yt_pred)[0]

        print(
            f"-------\n"
            f"Fold {fold} Results:\n"
            f"epoch* = {epoch_star}\n"
            f"MAE = {mae:.4f}\n"
            f"MSE = {mse:.4f}\n"
            f"R^2 = {r2:.4f}\n"
            f"Pearson r = {r:.4f}\n"
            f"-------"
        )

        mae_val.append(mae)
        mse_val.append(mse)
        r2_val.append(r2)
        pearson_val.append(r)

    print("\nOUTER CV results (DeepRSMA backbone + nested epoch selection)")
    print("-----------------------------------------------------------")
    print("Average MAE:", float(np.mean(mae_val)))
    print("Average MSE:", float(np.mean(mse_val)))
    print("Average R2 :", float(np.mean(r2_val)))
    print("Average Pearson r:", float(np.mean(pearson_val)))

    return {
        "Average MAE": float(np.mean(mae_val)),
        "Average MSE": float(np.mean(mse_val)),
        "Average R2 score": float(np.mean(r2_val)),
        "Average Pearson correlation (r)": float(np.mean(pearson_val)),
    }
