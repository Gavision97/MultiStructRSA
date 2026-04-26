import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.utils import *

from utils.tools import set_seed
from utils.utils import rmse_loss
from utils.plot_utils import plot_train_val_loss, plot_prediction_correlation

from utils.MolRNACDataset import MolRNADataset
from models.MultiStructRSA import *


import logging
logger = logging.getLogger(__name__)

# early stopping hyperparameters: 
PATIENCE = 30 # for early stopping
WARM_UP_EPOCHS = 30
MODE = 'min' # min for RMSE, max for pearson (PPC)
DELTA = 1e-6 # for early stopping
MIN_EPOCHS = 50
MAX_EPOCHS = 1000

# datasets paths
COLD_PATH = os.path.join("datasets", "blind_test", "processed")
RNA_COLD_PATH = os.path.join("datasets", "blind_test", "rna", "train_test_splits")
RSIM_PATH = os.path.join("datasets", "RSIM")


def choose_cold_type(cold_type= "cold_rna"):
     assert cold_type in ['cold_rna', 'cold_mol', 'cold_rm'] # senaty check for cold type input
     if cold_type == "cold_rna":
          return RNA_COLD_PATH, "rna"
     else:
          return None # TODO -> add cold_mol & cold_rm


def preform_cold_blind_eval(cold_type="cold_rna", model_setting="full", train_setting=4, val_setting=4, rna_aug=False, mol_aug=False, lr=5e-5, wd=1e-5, dropout=0.1, exclude_modalities=None, freeze_grnade=False, smiles_seq_setting='deep', smiles_expert_setting='proj', batch_size=1, num_workers=8,
            is_plot=False, is_seq_emb=False,  d_model=128, ffn_mult=4, bdcf_blocks=1, node_pool='attn', token_pool='mean', n=10, fam_name=None, plot_name="loss_curve", device="cuda", dataset_seed=2, seed=1):

    
    cold_path, cold_name = choose_cold_type(cold_type=cold_type)    
    logger.info(
         f'Performing Blind Test with Cold Start Setting = {cold_type}, seed:{seed}, batch size:{batch_size}, '
         f'max epochs:{MAX_EPOCHS}, lr:{lr}, weight decay:{wd}, '
         f'patience:{PATIENCE}, warm up epo:{WARM_UP_EPOCHS}, delta:{DELTA}'
    )

    p_list, s_list, rmse_list, r2_list = [], [], [], []

    # Outer 5 fold cold start setting loop
    for i in range(1, 6):
        set_seed(seed_=seed) # SEED = 2 as in DeepRSMA setting
        logger.info(f"Fold number {i}/5")
        train_df = pd.read_csv(os.path.join(cold_path, f"cold_{cold_name}_train_fold{i}.csv")).loc[:, ["SMILES", "Target_RNA_sequence", "pKd"]]
        test_df = pd.read_csv(os.path.join(cold_path, f"cold_{cold_name}_test_fold{i}.csv")).loc[:, ["SMILES", "Target_RNA_sequence", "pKd"]]
        logger.info(f"(|train|, |test|) -> ({train_df.shape[0]}, {test_df.shape[0]})")

        train_ds = MolRNADataset(train_df, setting=train_setting, rna_aug=rna_aug, mol_aug=mol_aug, task="training", is_seq_emb=is_seq_emb)
        val_ds = MolRNADataset(test_df, setting=val_setting, rna_aug=False, mol_aug=False, task="evaluating", is_seq_emb=is_seq_emb)

        g = torch.Generator().manual_seed(seed) # determinism per fold
        train_loader = DataLoader(train_ds, batch_size=batch_size, generator=g, shuffle=True, collate_fn=MolRNADataset.collate_fn, num_workers=num_workers)
        test_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=MolRNADataset.collate_fn, num_workers=num_workers)

        early_stopping = EarlyStopping(mode=MODE, patience=PATIENCE, warm_up_epochs=WARM_UP_EPOCHS, delta=DELTA) # early stopping object 
        
        model = choose_model_setting(option=model_setting, exclude_modalities=exclude_modalities, freeze_grnade=freeze_grnade, dropout=dropout, smiles_seq_setting=smiles_seq_setting, smiles_expert_setting=smiles_expert_setting, d_model=d_model, ffn_mult=ffn_mult, bdcf_blocks=bdcf_blocks, node_pool=node_pool, token_pool=token_pool).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
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

            early_stopping.check_early_stop(model, rmse, [p, s, r2], train_loss, epo, [y_true, y_pred])   
            if early_stopping.stop_training:
                logger.info(f"[Fold {i+1}/{5}] Early stopping at epoch {epo+1}. "
                            f"Best rmse={early_stopping.best_score:.4f} @ epoch {early_stopping.best_epoch+1}")
                break

        ########################################################################
        # ---- plot train vs. validation losses over time (epoch) ---- # 
        if is_plot:
            # extract date & fold name in order to create experiment plot directory at -> results/plots/date/plot_name/
            date, plot = plot_name.split('_') # format -> 'date_name' -> ['date', 'name']   
            if fam_name:
                plots_dir = os.path.join("results", "plots", date, plot, fam_name)
            else:
                plots_dir = os.path.join("results", "plots", date, plot)
            os.makedirs(plots_dir, exist_ok=True)
            plot_train_val_loss(train_losses, val_losses,
                                save_path=f"{plots_dir}/losses_{plot_name}_fold_{i}.png",
                                title="Loss Curve (train vs. val MSE)")
            plot_prediction_correlation(early_stopping.y_true, early_stopping.y_pred,
                                        save_path=f"{plots_dir}/corr_{plot_name}_fold_{i}.png",
                                        title="Prediction Correlation")
        #######################################################################
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