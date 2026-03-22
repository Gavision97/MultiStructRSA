import os
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from evaluate.evaluate_utils import (train_one_epoch, predict, EarlyStopping, 
                                     safe_regression_metrics)

from utils.tools import load_cold_data, get_cold_splits, set_seed
from utils.utils import rmse_loss
from evaluate.evaluate_utils import plot_train_val_loss

from utils.MolRNACDataset import MolRNADataset
from models.MultiStructRSA import *


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


def preform_cold_blind_cv(cold_type="cold_rna", model_setting="full", train_setting=4, val_setting=4, aug=False, is_plot=False,
                          is_seq_emb=False, plot_name="loss_curve", device="cuda", seed=1):

    set_seed(seed_=seed) # SEED = 2 as in DeepRSMA setting
    
    assert cold_type in ['cold_rna', 'cold_mol', 'cold_rm'] # senaty check for cold type input
    all_df = pd.read_csv(ALL_SF_DATASET_PATH, delimiter='\t') # full R-SIM dataset
    cold_dfs = load_cold_data(cold_type=cold_type) # get all 5 cold type dataframes in list
    
    logger.info(
         f'Performing Blind Test with Cold Start Setting = {cold_type}, seed:{seed}, batch size:{BATCH_SIZE}, '
         f'max epochs:{MAX_EPOCHS}, lr:{LR}, weight decay:{WEIGHT_DECAY}, '
         f'patience:{PATIENCE}, warm up epo:{WARM_UP_EPOCHS}, delta:{DELTA}'
    )

    p_list, s_list, rmse_list, r2_list = [], [], [], []

    # Outer 5 fold cold start setting loop
    for i, curr_cold_df in enumerate(cold_dfs):
        logger.info(f"Fold number {i+1}/5")

        train_id, test_id = get_cold_splits(curr_cold_df=curr_cold_df, cold_dfs=cold_dfs, all_df=all_df, fold=i)
        train_df, test_df = all_df.iloc[train_id].reset_index(drop=True), all_df.iloc[test_id].reset_index(drop=True)

        logger.info(f"(|train|, |test|) -> ({train_df.shape[0]}, {test_df.shape[0]})")

        train_df, test_df = all_df.iloc[train_id].reset_index(drop=True), all_df.iloc[test_id].reset_index(drop=True)

        train_ds = MolRNADataset(ds_=train_df, setting=train_setting, aug=aug, is_seq_emb=is_seq_emb, task="training")
        val_ds = MolRNADataset(ds_=test_df, setting=val_setting, aug=False, is_seq_emb=is_seq_emb, task='evaluating')

        g = torch.Generator().manual_seed(SEED) # determinism per fold
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, generator=g, shuffle=True, collate_fn=MolRNADataset.collate_fn, num_workers=NUM_WORKERS)
        test_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=MolRNADataset.collate_fn, num_workers=NUM_WORKERS)

        early_stopping = EarlyStopping(mode=MODE, patience=PATIENCE, warm_up_epochs=WARM_UP_EPOCHS, delta=DELTA) # early stopping object 
        
        model = choose_model_setting(option=model_setting).to(device)
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
                logger.info(f"[Fold {i+1}/{5}] Early stopping at epoch {epo+1}. "
                            f"Best rmse={early_stopping.best_score:.4f} @ epoch {early_stopping.best_epoch+1}")
                break

        ########################################################################
        # ---- plot train vs. validation losses over time (epoch) ---- # 
        if is_plot:
            # extract date & fold name in order to create experiment plot directory at -> results/plots/cold_type/date/plot_name/
            date, plot = plot_name.split('_') # format -> 'date_name' -> ['date', 'name']  
            plots_dir = os.path.join("results", "plots", cold_type, date, plot)
            os.makedirs(plots_dir, exist_ok=True)
            plot_train_val_loss(
                train_losses,
                val_losses,
                save_path=f"{plots_dir}/{plot_name}_fold_{i+1}.png",
                title="Loss Curve (train vs. val MSE)"
            )
        ########################################################################

        rmse_list.append(early_stopping.best_score)
        p_list.append(early_stopping.best_metrics[0])
        s_list.append(early_stopping.best_metrics[1])
        r2_list.append(early_stopping.best_metrics[2])

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