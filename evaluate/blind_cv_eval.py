import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
import statistics
from sklearn.metrics import mean_squared_error, r2_score

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from evaluate.evaluate_utils import train_one_epoch, predict, EarlyStopping

# Project-specific imports
from utils.MolRNADataset import MolRNADataset
from models.MultiMRSMA import MultiMRSMA

# Logging setup
import logging
logger = logging.getLogger(__name__)



# same seed as in DeepRSMA
SEED = 2
DROPOUT = 0.2
DEVICE = 'cuda'
BATCH_SIZE = 1 
MAX_EPOCHS = 100
PATIENCE = 20 # for early stopping
DELTA = 5e-4 # for early stopping
LR_START = 1e-4 # 0.0001 
WEIGHT_DECAY = 1e-5
SEED_DATASET = 2 # same as in DeepRSMA

# 31/1 experiments - lr -> 1e-4 (0.0001)
# 07/2 experiments - lr -> 5e-4 (0.0005)

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
def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

def preform_cold_blind_cv(
    cold_type: str,                 # 'rna' | 'mole' | 'rm'
    folds_dir: str = COLD_PATH,
    device: str = "cuda",
    batch_size: int = BATCH_SIZE,
    epochs: int = MAX_EPOCHS,
    lr: float = LR_START,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = PATIENCE,
    delta: float = DELTA
):

    set_seed(SEED) # SEED = 2 as in DeepRSMA setting
    assert cold_type in ['rna', 'mol', 'rm'] # senaty check for cold type input

    logger.info(f"Performing Blind Test with Cold Start Setting = {cold_type}, lr={LR_START}")
    p_list, s_list, rmse_list, r2_list = [], [], [], []

    # Outer 5 fold cold start setting loop
    for i in range(1,6):
        logger.info(f"Fold number {i}/5")
        curr_fold_path = os.path.join(COLD_PATH, cold_type, f'outer_fold_{i}')
        
        cold_train_df = pd.read_csv(os.path.join(curr_fold_path, 'train.csv'))
        cold_test_df = pd.read_csv(os.path.join(curr_fold_path, 'test.csv'))
        logger.info(f"(|train|, |test|) -> ({cold_train_df.shape[0]}, {cold_test_df.shape[0]})")

        best_p = -1
        best_s = 0
        best_rmse = 0

        train_dataset = MolRNADataset(cold_train_df)
        test_dataset  = MolRNADataset(cold_test_df)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=MolRNADataset.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=MolRNADataset.collate_fn)

        model = MultiMRSMA(dropout=0.2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        for epoch in range(MAX_EPOCHS):
            # train & evaluate
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            y_true, y_pred = predict(model, test_loader, device)

            p = pearsonr(y_true, y_pred)[0]
            s = spearmanr(y_true, y_pred)[0]
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

            if p > best_p:
                best_p = p
                best_s = s
                best_rmse = rmse
                best_r2  = r2

                logger.info(
                    f"[Fold {i}/5] NEW BEST @ epoch {epoch+1}: "
                    f"ppc={best_p:.4f}, ssc={best_s:.4f}, rmse={best_rmse:.4f}, r2={best_r2:.4f}; train loss={train_loss}"
                )

     

        p_list.append(best_p)
        s_list.append(best_s)
        rmse_list.append(best_rmse)
        r2_list.append(best_r2)

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

    logger.info("\nBlind Test (cold start setting):")
    for k, v in res.items():
        logger.info(f"{k}: {v}")

    return res
