import os
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from evaluate.evaluate_utils import (train_one_epoch, predict, EarlyStopping, 
                                     safe_regression_metrics)

from utils.MolRNADataset import MolRNADataset
from models.MultiMRSMA import MultiMRSMA


import logging
logger = logging.getLogger(__name__)


# same seed as in DeepRSMA
SEED = 1
#DROPOUT = 0.2
DEVICE = 'cuda'
BATCH_SIZE = 1
MAX_EPOCHS = 1000
PATIENCE = 30 # for early stopping
WARM_UP_EPOCHS = 50
MODE = 'min'
DELTA = 5e-5 # 0.0001 ;for early stopping (11/03, 1e-4 -> 5e-5)
LR = 0.00005 # 0.00005
WEIGHT_DECAY = 5e-5
SEED_DATASET = 2 # same as in DeepRSMA
NUM_WORKERS = 8


TRAIN_SETTING = 4
VAL_SETTING = 4
AUG = False

COLD_PATH = os.path.join("datasets", "blind_test", "processed")
RSIM_PATH = os.path.join("datasets", "RSIM")

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

def preform_cold_blind_cv(cold_type: str = "rna", device: str = "cuda"):

    set_seed(SEED) # SEED = 2 as in DeepRSMA setting
    assert cold_type in ['rna', 'mol', 'rm'] # senaty check for cold type input

    logger.info(f"Performing Blind Test with Cold Start Setting = {cold_type}, lr={LR}")
    p_list, s_list, rmse_list, r2_list = [], [], [], []

    # Outer 5 fold cold start setting loop
    for fold in range(1,6):
        logger.info(f"Fold number {i}/5")
        curr_fold_path = os.path.join(COLD_PATH, cold_type, f'outer_fold_{i}')
        
        cold_train_df = pd.read_csv(os.path.join(curr_fold_path, 'train.csv'))
        cold_test_df = pd.read_csv(os.path.join(curr_fold_path, 'test.csv'))
        logger.info(f"(|train|, |test|) -> ({cold_train_df.shape[0]}, {cold_test_df.shape[0]})")


        train_ds = MolRNADataset(cold_train_df, setting=TRAIN_SETTING, aug=AUG, task="training")
        val_ds = MolRNADataset(cold_test_df, setting=VAL_SETTING, aug=False, task='evaluating')

        g = torch.Generator().manual_seed(SEED) # determinism per fold
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, generator=g, shuffle=True, collate_fn=MolRNADataset.collate_fn, num_workers=NUM_WORKERS)
        test_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=MolRNADataset.collate_fn, num_workers=NUM_WORKERS)

        early_stopping = EarlyStopping(mode=MODE, patience=PATIENCE, warm_up_epochs=WARM_UP_EPOCHS, delta=DELTA) # early stopping object 

        model = MultiMRSMA().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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
                    logger.info(f"[Fold {fold}/5] Early stopping at epoch {epo+1}. "
                                f"Best rmse={early_stopping.best_score:.4f} @ epoch {early_stopping.best_epoch+1}")
                    break

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