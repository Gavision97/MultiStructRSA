# Standard library
import random

import matplotlib.pyplot as plt

# Third-party libraries
import numpy as np
import pandas as pd
import statistics
from scipy.stats import pearsonr, spearmanr

# Scikit-learn
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


import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # helps determinism for some CUDA GEMM paths
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # optional, slows down, good for debugging determinism

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



# Project-specific imports
from utils.MolRNADataset import MolRNADataset
from models.MultiMRSMA import MultiMRSMA

from evaluate.evaluate_utils import train_one_epoch, predict, regressor_stratified_cv, plot_train_val_loss, EarlyStopping

# Logging setup
import logging
logger = logging.getLogger(__name__)



# same seed as in DeepRSMA
SEED = 1
DROPOUT = 0.2
DEVICE = 'cuda'
BATCH_SIZE = 16 
MAX_EPOCHS = 1000
PATIENCE = 20 # for early stopping
DELTA = 1e-4 # for early stopping
LR_START = 5e-5 # 0.0001
WEIGHT_DECAY = 1e-4
SEED_DATASET = 2 # same as in DeepRSMA


ALL_SF_DATASET_PATH = os.path.join("datasets", "RSIM", "All_sf_dataset_v1.csv")
# set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)



def inner_cross_validation(dataset, train_index, n, max_epochs, batch_size, lr,
                           weight_decay, patience, delta, device) -> int:
    logger.info('Performing epoch search with stratified cross-validation setting')
    best_loss = []
    best_epochs = [] 

    # get y only for the OUTER-TRAIN part
    Y_all = dataset.data["pKd"].to_numpy(dtype=np.float32)
    Y_train = Y_all[train_index]

    # Inner stratified splitter - for epochs search (on outer-train only)
    skf = regressor_stratified_cv(
        n_splits=n, n_repeats=1, random_state=SEED_DATASET, group_count=5, strategy="uniform"
    )

    for j, (inner_train_rel, inner_val_rel) in enumerate(skf.split(np.zeros(len(Y_train)), Y_train), start=1):
        logger.info(f'Inner cross-validation fold number {j}/{n}')

        # map inner indices (relative) back to original dataset indices
        inner_train_idx = np.array(train_index)[inner_train_rel]
        inner_val_idx   = np.array(train_index)[inner_val_rel]

        # initialize train & validation subsets
        inner_train_ds = Subset(dataset, inner_train_idx)
        inner_val_ds   = Subset(dataset, inner_val_idx)

        # initialize train & validation dataloaders
        inner_train_dataloader = DataLoader(inner_train_ds, batch_size=batch_size, shuffle=True, collate_fn=MolRNADataset.collate_fn)
        inner_val_dataloader = DataLoader(inner_val_ds, batch_size=batch_size, shuffle=False, collate_fn=MolRNADataset.collate_fn)

        # initialize model, optimizer & criterion
        m = MultiMRSMA(dropout=0.2).to(device)
        optimizer = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
        early_stopping = EarlyStopping(mode='max', patience=patience, delta=delta)
        criterion = nn.MSELoss()

        for epoch in range(1, max_epochs + 1):
            _ = train_one_epoch(m, inner_train_dataloader, optimizer, criterion, device)

            y, y_hat = predict(m, inner_val_dataloader, device)
            #val_loss = np.sqrt(mean_squared_error(y, y_hat))  # val RMSE
            val_loss = pearsonr(y, y_hat)[0] # pearson correlation (r
            # Check early stopping condition
            early_stopping.check_early_stop(val_loss, epoch)
            if early_stopping.stop_training:
                logger.info(
                    f"Stopped training inner fold {j}; "
                    f"best epoch={early_stopping.best_epoch}; best loss={early_stopping.best_score}"
                )
                best_epochs.append(early_stopping.best_epoch)
                best_loss.append(early_stopping.best_score)
                break

        # if early stopping did not trigger, still record best epoch found
        if not early_stopping.stop_training:
            best_epochs.append(early_stopping.best_epoch)

    loss = np.array(best_loss, dtype=float)
    epochs = np.array(best_epochs, dtype=float)

    w = 1.0 / (loss + 1e-8)          # smaller loss => bigger weight
    best_epoch = int(round(np.sum(w * epochs) / np.sum(w)))
    return best_epoch

def cv_eval(df,
            n: int =10,
            plot_name: str = "loss_curve",
            device: str = "cuda",
            batch_size: int = BATCH_SIZE,
            epochs: int = MAX_EPOCHS,
            lr: float = LR_START,
            weight_decay: float = WEIGHT_DECAY,
            patience: int = PATIENCE,
            delta: float = DELTA):
    logger.info(
        f'K folds:{n}, seed:{SEED}, batch size:{BATCH_SIZE}, '
        f'max epochs:{MAX_EPOCHS}, lr:{LR_START}, weight decay:{WEIGHT_DECAY}'
    )
    #all_df = pd.read_csv(ALL_SF_DATASET_PATH, sep="\t")
    all_df = df
    logging.info(f"--- Start statified {n}-fold cross-validation; pd.DataFrame shape -> {all_df.shape} ---")

    ds = MolRNADataset(all_df)
    Y = ds.data["pKd"].to_numpy(dtype=np.float32)

    # same initialization as in DeepRSMA 
    skf = regressor_stratified_cv(
        n_splits=n, n_repeats=1, random_state=SEED_DATASET, group_count=5, strategy="uniform"
    )

    # store per-fold best results
    p_list, s_list, rmse_list, r2_list = [], [], [], []
    set_seed(SEED)

    for fold, (train_index, test_index) in enumerate(skf.split(np.zeros(len(Y)), Y)):
        # determinism per fold
        g = torch.Generator().manual_seed(SEED)

        logging.info(f"\nFold number {fold+1}/{n}")

        n_epochs = inner_cross_validation(dataset=ds, train_index=train_index, n=n, max_epochs=epochs, batch_size=batch_size,
       
                                               lr=lr, weight_decay=weight_decay, patience=patience, delta=delta, device=device)
        logger.info(f"After inner 5-fold cross validation with stratified  setting, n_epochs={n_epochs}")

        # Initialize current fold train & test dataloaders
        train_loader = DataLoader(Subset(ds, train_index), batch_size=BATCH_SIZE, generator=g, shuffle=True, collate_fn=MolRNADataset.collate_fn)
        test_loader = DataLoader(Subset(ds, test_index), batch_size=BATCH_SIZE, shuffle=False, collate_fn=MolRNADataset.collate_fn)

        model = MultiMRSMA(dropout=0.2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_START, weight_decay=WEIGHT_DECAY)
        criterion = nn.MSELoss()

        # train the model with the best number of epochs found after inner cross-validation
        for _ in range(n_epochs):
            # uses your functions
            _ = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # predict on the test set
        y_true, y_pred = predict(model, test_loader, device)

        p = pearsonr(y_true, y_pred)[0]
        s = spearmanr(y_true, y_pred)[0]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        logger.info(f"pcc: {p}, scc: {s}, rmse: {rmse}, r2: {r2}")

        p_list.append(p)
        s_list.append(s)
        rmse_list.append(rmse)
        r2_list.append(r2)

    logger.info(f"p: {np.mean(p_list)}")
    logger.info(f"s: {np.mean(s_list)}")
    logger.info(f"rmse: {np.mean(rmse_list)}")
    logger.info(f"r2: {np.mean(r2_list)}")

    res = {
        "p_mean": float(np.mean(p_list)),
        "p_std": float(np.std(p_list)),
        "s_mean": float(np.mean(s_list)),
        "s_std": float(np.std(s_list)),
        "rmse_mean": float(np.mean(rmse_list)),
        "rmse_std": float(np.std(rmse_list)),
        "r2_mean": float(np.mean(r2_list)),
        "r2_std": float(np.std(r2_list)),
        "per_fold": {
            "p": p_list,
            "s": s_list,
            "rmse": rmse_list,
            "r2": r2_list,
        }
    }

    logger.info("\nStratified k-fold cross validation:")
    for k, v in res.items():
        logger.info(f"{k}: {v}")

    return res
