import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from evaluate.evaluate_utils import (train_one_epoch, predict, EarlyStopping,
                                     regressor_stratified_cv, plot_train_val_loss, safe_regression_metrics)

#from utils.MolRNADataset import MolRNADataset # w/o RNA cross-attention fusion mechanism
from utils.MolRNACDataset import MolRNADataset # w RNA cross-attention fusion mechanism

from models.MultiMRSMA import MultiMRSMA # w/o RNA cross-attention fusion mechanism
#from models.MultiMRSMACross import MultiMRSMA # w RNA cross-attention fusion mechanism


import logging
logger = logging.getLogger(__name__)

# same seeds as in DeepRSMA
SEED = 1
SEED_DATASET = 2 

# early stopping hyperparameters: 
PATIENCE = 30 # for early stopping
WARM_UP_EPOCHS = 30
MODE = 'min'
DELTA = 1e-6 # 0.0001 ;for early stopping (11/03, 1e-4 -> 5e-5 -> 1e-6)

# MultiStructRSA training hyperparameters:
LR = 0.00005 # from 0.00005 -> 0.0001
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 1
MAX_EPOCHS = 1000
NUM_WORKERS = 8

# MolRNACDataset hyperparameters:
TRAIN_SETTING = 4
VAL_SETTING = 4
AUG = False

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

def cv_eval(dataset: pd.DataFrame = None, n: int = 10, plot_name: str = "loss_curve", device: str = "cuda"):
    logger.info(
        f'K folds:{n}, seed:{SEED}, batch size:{BATCH_SIZE}, '
        f'max epochs:{MAX_EPOCHS}, lr:{LR}, weight decay:{WEIGHT_DECAY}, '
        f'patience:{PATIENCE}, warm up epo:{WARM_UP_EPOCHS}, delta:{DELTA}'
    )
    set_seed(SEED) # set seed (same as in DeepRSMA and others)
    all_df = dataset
    logging.info(f"--- Start statified {n}-fold cross-validation; pd.DataFrame shape -> {all_df.shape}, lr={LR} ---")

    train_ds = MolRNADataset(all_df, setting=TRAIN_SETTING, aug=AUG, task="training")
    val_ds = MolRNADataset(all_df, setting=VAL_SETTING, aug=False, task='evaluating')
    Y = train_ds.data["pKd"].to_numpy(dtype=np.float32)

    # same initialization as in DeepRSMA and others
    skf = regressor_stratified_cv(n_splits=n, n_repeats=1, random_state=SEED_DATASET, group_count=5, strategy="uniform")

    p_list, s_list, rmse_list, r2_list = [], [], [], []

    # extract date & fold name in order to create experiment 
    # plot directory at -> results/plots/date/plot_name/
    date, plot = plot_name.split('_') # format -> 'date_name' -> ['date', 'name']

    for fold, (train_index, test_index) in enumerate(skf.split(np.zeros(len(Y)), Y)):
        g = torch.Generator().manual_seed(SEED) # determinism per fold

        logger.info(f"\nFold number {fold+1}/{n}")

        # initialize current fold train & test dataloaders
        train_loader = DataLoader(Subset(train_ds, train_index), batch_size=BATCH_SIZE, generator=g, shuffle=True, collate_fn=MolRNADataset.collate_fn, num_workers=NUM_WORKERS)
        test_loader = DataLoader(Subset(val_ds, test_index), batch_size=BATCH_SIZE, shuffle=False, collate_fn=MolRNADataset.collate_fn, num_workers=NUM_WORKERS)
        logger.info(f'Train len -> {len(train_loader.dataset)}, Test len -> {len(test_loader.dataset)}')


        early_stopping = EarlyStopping(mode=MODE, patience=PATIENCE, warm_up_epochs=WARM_UP_EPOCHS, delta=DELTA) # early stopping object 
        
        model = MultiMRSMA().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = rmse_loss

        train_losses, val_losses = [], []
        for epo in range(MAX_EPOCHS):
            # uses your functions
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            train_losses.append(train_loss)
            # predict on the test set
            y_true, y_pred = predict(model, test_loader, device)

            p, s, rmse, r2 = safe_regression_metrics(y_true, y_pred)
            val_losses.append(rmse)

            early_stopping.check_early_stop(rmse, [p, s, r2], train_loss, epo)   
            if early_stopping.stop_training:
                logger.info(f"[Fold {fold+1}/{n}] Early stopping at epoch {epo+1}. "
                            f"Best rmse={early_stopping.best_score:.4f} @ epoch {early_stopping.best_epoch+1}")
                break


        # ---- plot train vs. validation losses over time (epoch) ---- # 
        plots_dir = os.path.join("results", "plots", date, plot)
        os.makedirs(plots_dir, exist_ok=True)
        plot_train_val_loss(
            train_losses,
            val_losses,
            save_path=f"{plots_dir}/{plot_name}_fold_{fold+1}.png",
            title="Loss Curve (train vs. val MSE)"
        )


        rmse_list.append(early_stopping.best_score)
        p_list.append(early_stopping.best_metrics[0])
        s_list.append(early_stopping.best_metrics[1])
        r2_list.append(early_stopping.best_metrics[2])

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