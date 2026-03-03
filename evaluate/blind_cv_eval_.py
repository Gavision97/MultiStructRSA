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
from models.MultiMRSMACross import MultiMRSMA

# Logging setup
import logging
logger = logging.getLogger(__name__)



# same seed as in DeepRSMA
SEED = 2
DROPOUT = 0.2
DEVICE = 'cuda'
BATCH_SIZE = 16 
MAX_EPOCHS = 1000
PATIENCE = 20 # for early stopping
DELTA = 1e-4 # for early stopping
LR_START = 1e-4 # 0.0001 
WEIGHT_DECAY = 1e-4
SEED_DATASET = 2 # same as in DeepRSMA

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


def load_inner_train_df(fold_path, fold_n) -> pd.DataFrame:
    """
    Load and concatenate all inner-fold DataFrames except the validation fold.

    Parameters
    ----------
    fold_path : str
        Path to the outer fold directory (e.g., ".../outer_fold_1").

    fold_n : int
        Index of the validation fold (1-based, in [1,2,3,4,5]).

    Returns
    -------
    pd.DataFrame
        Concatenated inner training DataFrame composed of the 4 non-validation folds.
    """
    assert fold_n in [1, 2, 3, 4, 5] # senaty check
    inner_train_dfs = []
    for i in range(1,6):
        if i == fold_n: # skip if its validation fold
            continue
        curr_df = pd.read_csv(os.path.join(fold_path, f'inner_fold_{i}.csv'))
        inner_train_dfs.append(curr_df)

    assert len(inner_train_dfs) == 4 # senaty check
    inner_train_df = pd.concat(inner_train_dfs, axis=0, ignore_index=True)
    return inner_train_df


def inner_cold_cross_validation(curr_fold_path, max_epochs, batch_size, lr,
                                 weight_decay, patience, delta, device) -> int:
    logger.info('Performing epoch search with cold start cross-validation setting')
    best_loss = []
    best_epochs = [] 
    
    for j in range(1,6): 
        logger.info(f'Inner cross-validation fold number {j}/5')
        
        inner_val_df = pd.read_csv(os.path.join(curr_fold_path, f'inner_fold_{j}.csv'))
        inner_train_df = load_inner_train_df(curr_fold_path, j) # get inner training set

        logger.info(f"(|train|, |validation|) -> ({inner_train_df.shape[0]}, {inner_val_df.shape[0]})")
        # initialize train & validation datasets
        inner_train_ds = MolRNADataset(inner_train_df)
        inner_val_ds = MolRNADataset(inner_val_df)

        # initialize train & validation dataloaders
        inner_train_dataloader = DataLoader(inner_train_ds, batch_size=batch_size, shuffle=True, collate_fn=MolRNADataset.collate_fn)
        inner_val_dataloader = DataLoader(inner_val_ds, batch_size=batch_size, collate_fn=MolRNADataset.collate_fn)
        
        # initialize model, optimizer & criterion
        m = MultiMRSMA(dropout=0.2).to(device)
        optimizer = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
        early_stopping = EarlyStopping(mode='max', patience=patience, delta=delta)
        criterion = nn.MSELoss()

        for epoch in range(1, max_epochs+1):
            _ = train_one_epoch(m, inner_train_dataloader, optimizer, criterion, device)

            y, y_hat = predict(m, inner_val_dataloader, device)
            #val_loss = np.sqrt(mean_squared_error(y, y_hat)) # val RMSE
            val_loss = pearsonr(y, y_hat)[0] # pearson correlation (r)

            # Check early stopping condition
            early_stopping.check_early_stop(val_loss, epoch)
            if early_stopping.stop_training:
                logger.info(f"Stopped training inner fold {j}; best epoch={early_stopping.best_epoch}; best loss={early_stopping.best_loss}")
                best_epochs.append(early_stopping.best_epoch)
                best_loss.append(early_stopping.best_loss)
                break

    loss = np.array(best_loss, dtype=float)
    epochs = np.array(best_epochs, dtype=float)

    w = 1.0 / (loss + 1e-8)          # smaller loss => bigger weight
    best_epoch = int(round(np.sum(w * epochs) / np.sum(w)))
    return best_epoch


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

    logger.info(f"Performing Blind Test with Cold Start Setting = {cold_type}")
    p_list, s_list, rmse_list, r2_list = [], [], [], []

    # Outer 5 fold cold start setting loop
    for i in range(1,6):
        logger.info(f"Fold number {i}/5")
        curr_fold_path = os.path.join(COLD_PATH, cold_type, f'outer_fold_{i}')
        
        cold_train_df = pd.read_csv(os.path.join(curr_fold_path, 'train.csv'))
        cold_test_df = pd.read_csv(os.path.join(curr_fold_path, 'test.csv'))
        logger.info(f"(|train|, |test|) -> ({cold_train_df.shape[0]}, {cold_test_df.shape[0]})")

        # Inner cross validation loop for optimal n_epochs search with early stopping
        n_epochs = inner_cold_cross_validation(curr_fold_path=curr_fold_path, max_epochs=epochs, batch_size=batch_size,
                                               lr=lr, weight_decay=weight_decay, patience=patience, delta=delta, device=device)
        logger.info(f"After inner 5-fold cross validation with cold start setting, n_epochs={n_epochs}")

        train_dataset = MolRNADataset(cold_train_df)
        test_dataset  = MolRNADataset(cold_test_df)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=MolRNADataset.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=MolRNADataset.collate_fn)

        model = MultiMRSMA(dropout=0.2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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

    logger.info("\nBlind Test (cold start setting):")
    for k, v in res.items():
        logger.info(f"{k}: {v}")

    return res
