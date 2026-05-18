import os
#os.environ["PYTHONHASHSEED"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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

from utils.utils import *
from utils.tools import *
from utils.plot_utils import plot_train_val_loss, plot_prediction_correlation
#from utils.MolRNADataset import MolRNADataset # w/o RNA cross-attention fusion mechanism
from utils.MolRNACDataset import MolRNADataset # w RNA cross-attention fusion mechanism

from models.MultiStructRSA import *

import logging
logger = logging.getLogger(__name__)

# same seeds as in DeepRSMA
#SEED = 1
#SEED_DATASET = 2 

# early stopping hyperparameters: 
#PATIENCE = 20 # for early stopping
#WARM_UP_EPOCHS = 50
MODE = 'max'
#DELTA = 1e-6 # 0.0001 ;for early stopping (11/03, 1e-4 -> 5e-5 -> 1e-6)
#MIN_EPOCHS = 50

# MultiStructRSA training hyperparameters:
MAX_EPOCHS = 1000
NUM_WORKERS = 0

COLD_PATH = os.path.join("datasets", "blind_test", "processed")
RSIM_PATH = os.path.join("datasets", "RSIM")

ALL_SF_DATASET_PATH = os.path.join("datasets", "RSIM", "All_sf_dataset_v1.csv")


# custom RMSE loss function
def rmse_loss(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

def cv_eval(dataset=None, model_setting="full", train_setting=4, val_setting=4, rna_aug=False, mol_aug=False, patience=20, warm_up_epochs=10, delta=1e-5, lr=5e-5, wd=1e-5, dropout=0.1, exclude_modalities=None, freeze_grnade=False, smiles_seq_setting='deep', smiles_expert_setting='proj', batch_size=1, num_workers=8,
            is_plot=False, is_seq_emb=False,  d_model=128, ffn_mult=4, bdcf_blocks=1, node_pool='attn', token_pool='mean', n=10, fam_name=None, plot_name="loss_curve", device="cuda", dataset_seed=2, seed=1):
    """
    Run stratified k-fold cross-validation for a selected MultiStructRSA model setting.

    The function trains and evaluates the chosen model configuration on `n`
    stratified folds of the input dataset. For each fold, it trains the model,
    evaluates it after every epoch on the validation split, applies early
    stopping based on validation RMSE, saves a loss plot, and records the best
    validation metrics.

    Parameters
    ----------
    dataset : pandas.DataFrame, optional
        Full dataset used for cross-validation. Must be compatible with
        `MolRNADataset` and include the target column `pKd`.

    model_setting : str, default="full"
        Model variant passed to `choose_model_setting()`, such as:
        `"full"`, `"freezed"`, `"w/o seq"`, `"w/o 3D struct"`, or `"w/o fusion"`.

    n : int, default=10
        Number of cross-validation folds.

    plot_name : str, default="loss_curve"
        Plot naming string in the format `"date_name"`. Determines the directory
        structure for saved fold loss curves under `results/plots/`.

    device : str, default="cuda"
        Device used for training and evaluation.

    Returns
    -------
    dict
        Dictionary with mean/std metrics across folds and per-fold results:
        Pearson (`p`), Spearman (`s`), RMSE, and R^2.

    Notes
    -----
    Early stopping is based on validation RMSE. The best metrics from each fold
    are stored and aggregated at the end of the procedure.
    """
        
    logger.info(
        f'K folds:{n}, seed:{seed}, dataset seed:{dataset_seed} batch size:{batch_size}, '
        f'max epochs:{MAX_EPOCHS}, lr:{lr}, weight decay:{wd}, dropout:{dropout} num_workers:{num_workers}, '
        f'patience:{patience}, warm up epo:{warm_up_epochs}, delta:{delta}'
    )
    
    all_df = dataset
    Y = all_df["pKd"].to_numpy(dtype=np.float32)

    logging.info(f"--- Start statified {n}-fold cross-validation; pd.DataFrame shape -> {all_df.shape}, lr={lr} ---")
    
    # same initialization as in DeepRSMA and others
    skf = regressor_stratified_cv(n_splits=n, n_repeats=1, random_state=dataset_seed, group_count=5, strategy="uniform")
    p_list, s_list, rmse_list, r2_list = [], [], [], []
    for fold, (train_index, test_index) in enumerate(skf.split(np.zeros(len(Y)), Y)):
        set_seed(seed) # set seed (same as in DeepRSMA and others)
        train_df = all_df.iloc[train_index].reset_index(drop=True)
        test_df = all_df.iloc[test_index].reset_index(drop=True)

        train_ds = MolRNADataset(train_df, setting=train_setting, rna_aug=rna_aug, mol_aug=mol_aug, task="training", is_seq_emb=is_seq_emb)
        val_ds = MolRNADataset(test_df, setting=val_setting, rna_aug=False, mol_aug=False, task="evaluating", is_seq_emb=is_seq_emb)
        g = torch.Generator().manual_seed(seed) # determinism per fold

        logger.info(f"\nFold number {fold+1}/{n}")

        # initialize current fold train & test dataloaders
        train_loader = DataLoader(train_ds, batch_size=batch_size, generator=g, shuffle=True, collate_fn=MolRNADataset.collate_fn, num_workers=num_workers)
        test_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=MolRNADataset.collate_fn, num_workers=num_workers)
        logger.info(f'Train len -> {len(train_loader.dataset)}, Test len -> {len(test_loader.dataset)}')


        early_stopping = EarlyStopping(mode=MODE, patience=patience, warm_up_epochs=warm_up_epochs, delta=delta) # early stopping object 
        
        model = choose_model_setting(option=model_setting, exclude_modalities=exclude_modalities, freeze_grnade=freeze_grnade, dropout=dropout, smiles_seq_setting=smiles_seq_setting, smiles_expert_setting=smiles_expert_setting, d_model=d_model, ffn_mult=ffn_mult, bdcf_blocks=bdcf_blocks,
                                     node_pool=node_pool, token_pool=token_pool).to(device)
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
            es_metric = rmse if MODE == 'min' else p # rmse for mode=min, ppc for mode=max
            early_stopping.check_early_stop(model, es_metric, [p, s, r2, rmse], train_loss, epo, [y_true, y_pred])   
            if early_stopping.stop_training:
                logger.info(f"[Fold {fold+1}/{n}] Early stopping at epoch {epo+1}. "
                            f"Best score (rmse/ppc)={early_stopping.best_score:.4f} @ epoch {early_stopping.best_epoch+1}")
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
                                save_path=f"{plots_dir}/losses_{plot_name}_fold_{fold+1}.png",
                                title="Loss Curve (train vs. val MSE)")
            plot_prediction_correlation(early_stopping.y_true, early_stopping.y_pred,
                                        save_path=f"{plots_dir}/corr_{plot_name}_fold_{fold+1}.png",
                                        title="Prediction Correlation")
        ########################################################################

        rmse_list.append(early_stopping.best_score[-1])
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