# Standard library
import random
import logging

import matplotlib.pyplot as plt

# Third-party libraries
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

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
from models.MultiMRSMA import MultiMRSMA, MultiMRSMA_, MultiMRSMAChunk

# Logging setup
logger = logging.getLogger(__name__)



# same seed as in DeepRSMA
SEED = 1
BATCH_SIZE = 16 
PATIENCE = 100 # for training patience ..
LR_PATIENCE = 10 # for scheduler ..
MAX_EPOCHS = 1000
WARMUP_EPOCHS = 100
LR_START = 1e-4
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-4


# set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)

import matplotlib.pyplot as plt

def plot_train_val_loss(
    train_losses,
    val_losses,
    save_path=None,
    title="Loss Curve"
):
    steps = range(len(train_losses))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label="Training Loss", linewidth=2, color="blue")
    plt.plot(steps, val_losses, label="Validation Loss", linewidth=2, color="green")

    plt.title(title, fontsize=16)
    plt.xlabel("Training Steps", fontsize=14)
    plt.ylabel("Loss", fontsize=14)

    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()



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



def split_train_val(train_index, y, val_size=0.1, random_state=42, n_bins=5, strategy="uniform"):
    """
    Split *within* the outer-train fold into train_inner / val_inner using stratification over binned y.
    This avoids leakage: test fold is untouched.
    """
    y_train = y[train_index]
    binner = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    y_bins = binner.fit_transform(y_train[:, None])[:, 0]

    tr_idx, va_idx = train_test_split(
        train_index,
        test_size=val_size,
        random_state=random_state,
        stratify=y_bins,
    )
    return tr_idx, va_idx


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0

    for graphs, ernie, morgan, rdkit, chemberta, y in loader:
        graphs = graphs.to(device)
        #chunk_to_sample = chunk_to_sample.to(device)
        ernie  = ernie.to(device)
        morgan = morgan.to(device)
        rdkit  = rdkit.to(device)
        chemberta = chemberta.to(device)
        y      = y.to(device).view(-1, 1)

        optimizer.zero_grad()
        pred = model(graphs, ernie, morgan, rdkit, chemberta)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        n += y.size(0)

        # ---- IMPORTANT: release big tensors for this iteration ----
        del graphs, ernie, morgan, rdkit, chemberta, y, pred, loss

    return total_loss / max(1, n)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    for graphs,  ernie, morgan, rdkit, chemberta, y in loader:
        graphs = graphs.to(device)
        #chunk_to_sample = chunk_to_sample.to(device)
        ernie  = ernie.to(device)
        morgan = morgan.to(device)
        rdkit  = rdkit.to(device)
        chemberta = chemberta.to(device)

        pred = model(graphs, ernie, morgan, rdkit, chemberta).squeeze(1).detach().cpu().numpy()
        y_true.append(y.numpy())
        y_pred.append(pred)

    return np.concatenate(y_true).astype(np.float32), np.concatenate(y_pred).astype(np.float32)


def cv_eval(df, name, n=10, plot_name="loss_curve", device="cuda"):
    logger.info(f'CV hyperparameters: K folds:{n}, seed:{SEED}, batch size:{BATCH_SIZE}, warmup epochs:{WARMUP_EPOCHS}, scheduler patience:{LR_PATIENCE}, training patience:{PATIENCE}, lr (start):{LR_START}, lr (max):{LR_MIN}, weight decay:{WEIGHT_DECAY}')
    logging.info(f"--- Start 10 fold cv for {name} --- ({df.shape})")

    ds = MolRNADataset(df)
    Y = ds.data["pKd"].to_numpy(dtype=np.float32)

    skf = regressor_stratified_cv(
        n_splits=n, n_repeats=1, random_state=42, group_count=5, strategy="uniform"
    )

    mae_val, mse_val, r2_val, pearson_val = [], [], [], []

    for fold, (train_index, test_index) in enumerate(skf.split(np.zeros(len(Y)), Y)):
        set_seed(42 + fold)
        logging.info(f"\nFold {fold+1}")
        # --- fold stats (your debug) ---
        y_tr = Y[train_index]
        y_te = Y[test_index]
        logger.info(f"train n={len(train_index)} y: mean={y_tr.mean():.3f} std={y_tr.std():.3f} min={y_tr.min():.3f} max={y_tr.max():.3f}")
        logger.info(f"test  n={len(test_index)}  y: mean={y_te.mean():.3f} std={y_te.std():.3f} min={y_te.min():.3f} max={y_te.max():.3f}")

        # --- inner val split (ONLY to pick best_epoch) ---
        tr_idx, va_idx = split_train_val(train_index, Y, val_size=0.1, random_state=42 + fold, n_bins=5, strategy="quantile")

        set_seed(SEED + fold)
        g = torch.Generator().manual_seed(SEED + fold)

        train_loader = DataLoader(Subset(ds, tr_idx), batch_size=BATCH_SIZE, generator=g ,shuffle=True, collate_fn=MolRNADataset.collate_fn)
        val_loader = DataLoader(Subset(ds, va_idx), batch_size=BATCH_SIZE, shuffle=False, collate_fn=MolRNADataset.collate_fn)
        test_loader = DataLoader(Subset(ds, test_index), batch_size=BATCH_SIZE, shuffle=False, collate_fn=MolRNADataset.collate_fn)

        # ---------------------------
        # Stage A: early-stop search
        # ---------------------------
        model = MultiMRSMA(dropout=0.2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_START, weight_decay=WEIGHT_DECAY)
        train_criterion = nn.MSELoss()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=LR_PATIENCE, min_lr=LR_MIN)

        best_epoch = -1
        best_val_loss = float("inf")
        bad = 0
        train_losses, val_losses = [], [] # track train and validation losses in order to plot train vs. val losses over time (epochs)
        
        for epoch in range(MAX_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, train_criterion, device)
            train_losses.append(train_loss)

            yv_true, yv_pred = predict(model, val_loader, device)
            val_loss = mean_squared_error(yv_true, yv_pred)
            val_losses.append(val_loss)

            val_r = pearsonr(yv_true, yv_pred)[0]
            scheduler.step(val_loss)

            # ---- warmup + early stopping ---- #
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                if epoch >= WARMUP_EPOCHS:
                    bad = 0  # reset patience only after warmup
            else:
                if epoch >= WARMUP_EPOCHS:
                    bad += 1

            if (epoch + 1) % 10 == 0:
                lr = optimizer.param_groups[0]["lr"]
                logging.info(
                    f"Epoch {epoch+1:4d} | train_mse {train_loss:.4f} | "
                    f"val_mse {val_loss:.4f} | val_r {val_r:.4f} | lr {lr:.2e}"
                )

            if epoch >= WARMUP_EPOCHS and bad >= PATIENCE:
                break
        
        # --- sanity check --- #
        if best_epoch < 0:
            raise RuntimeError("No best_state/best_epoch was saved (unexpected).")
        
        if best_epoch < WARMUP_EPOCHS: # train for at least n=WARMUP_EPOCHS (default 100)
            best_epoch = WARMUP_EPOCHS
        logging.info(f"[Fold {fold+1}] Selected best_epoch={best_epoch+1} based on inner-val MSE={best_val_loss:.4f} and inner-train MSE={train_loss:.4f}")
        
        # ---- plot train vs. validation losses over time (epoch) ---- #
        plots_dir = os.path.join("results", "plots", f"{plot_name}")
        os.makedirs(plots_dir, exist_ok=True)
        plot_train_val_loss(
            train_losses,
            val_losses,
            save_path=f"{plots_dir}/{plot_name}_fold_{fold+1}.png",
            title="Loss Curve (train vs. val MSE)"
        )
        
        # ---------------------------
        # Stage B: refit on FULL outer-train fold (uses ALL training data)
        # ---------------------------
        full_train_loader = DataLoader(Subset(ds, train_index), batch_size=BATCH_SIZE, generator=g ,shuffle=True, pin_memory=True, collate_fn=MolRNADataset.collate_fn)

        # re-init model fresh (common + clean)
        model = MultiMRSMA(dropout=0.2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_START, weight_decay=WEIGHT_DECAY)

        # optional: schedule not needing val; keep simple
        # trains exactly best_epoch+1 epochs
        for epoch in range(best_epoch + 1):
            _ = train_one_epoch(model, full_train_loader, optimizer, train_criterion, device)

        # ---------------------------
        # Evaluate on outer test fold
        # ---------------------------
        yt_true, yt_pred = predict(model, test_loader, device)

        mae = mean_absolute_error(yt_true, yt_pred)
        mse = mean_squared_error(yt_true, yt_pred)
        r2  = r2_score(yt_true, yt_pred)
        r   = pearsonr(yt_true, yt_pred)[0]

 
        logging.info(f"-------\nFold {fold + 1} Results:\nMAE = {mae:.4f}\nMSE = {mse:.4f}\nR^2 = {r2:.4f}\nR = {r:.4f}\n-------")
        mae_val.append(mae)
        mse_val.append(mse)
        r2_val.append(r2)
        pearson_val.append(r)

    logger.info("\n10-fold CV results")
    logger.info("-------------------")
    logger.info("Average MAE:", float(np.mean(mae_val)))
    logger.info("Average MSE:", float(np.mean(mse_val)))
    logger.info("Average R2 score:", float(np.mean(r2_val)))
    logger.info("Average Pearson correlation:", float(np.mean(pearson_val)))

    return {
        "Average MAE": float(np.mean(mae_val)),
        "Average MSE": float(np.mean(mse_val)),
        "Average R2 score": float(np.mean(r2_val)),
        "Average Pearson correlation (r)": float(np.mean(pearson_val)),
    }
