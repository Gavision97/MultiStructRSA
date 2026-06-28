import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, GroupShuffleSplit
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
PATIENCE = 20 # for early stopping
WARM_UP_EPOCHS = 10
MODE = 'min' # min for RMSE, max for pearson (PPC)
DELTA = 1e-3 # for early stopping
MAX_EPOCHS = 200

SPLIT=True

# datasets paths
COLD_PATH = os.path.join("datasets", "blind_test", "processed")
RNA_COLD_PATH = os.path.join("datasets", "blind_test", "rna", "train_val_test_splits")
MOLE_COLD_PATH = os.path.join("datasets", "blind_test", "mole", "train_val_test_splits")
BOTH_COLD_PATH = os.path.join("datasets", "blind_test", "rm", "train_val_test_splits")
RSIM_PATH = os.path.join("datasets", "RSIM")


def choose_cold_type(cold_type= "cold_rna"):
     assert cold_type in ['cold_rna', 'cold_mole', 'cold_rm'] # senaty check for cold type input
     if cold_type == "cold_rna":
          return RNA_COLD_PATH, "rna"
     elif cold_type == 'cold_mole':
          return MOLE_COLD_PATH, "mole"
     else:
         return BOTH_COLD_PATH, 'rm'
     

def preform_cold_blind_eval(cold_type="cold_rna", model_setting="full", train_setting=4, val_setting=4,
                            rna_aug=False, mol_aug=False, patience=20, warm_up_epochs=10, delta=1e-5, lr=5e-5, wd=1e-5, dropout=0.1, exclude_modalities=None,
                            freeze_grnade=False, smiles_seq_setting='deep', smiles_expert_setting='proj', batch_size=1,
                            num_workers=8, is_plot=False, is_seq_emb=False, d_model=128, ffn_mult=4, bdcf_blocks=1,
                            node_pool='attn', token_pool='mean', n=10, fam_name=None, plot_name="loss_curve",
                            device="cuda", dataset_seed=2, seed=1):
    cold_column = 'Target_RNA_sequence' if cold_type == "cold_rna" else 'Molecule_name'
    cold_path, cold_name = choose_cold_type(cold_type=cold_type)

    logger.info(
        f'Performing Blind Test with Cold Start Setting = {cold_type}, seed:{seed}, batch size:{batch_size}, '
        f'max epochs:{MAX_EPOCHS}, lr:{lr}, weight decay:{wd}, '
        f'patience:{patience}, warm up epo:{warm_up_epochs}, delta:{delta}'
    )

    p_list, s_list, rmse_list, r2_list = [], [], [], []

    for i in range(1, 6):

        set_seed(seed_=seed) # set seed for reproducibility
        logger.info(f"Fold number {i}/5")

        
        # load current fold train, val & test splits
        # train was splitted into 85, 15 train & val
        train_df = pd.read_csv(os.path.join(cold_path, f"cold_{cold_name}_train_fold{i}.csv")).loc[:, ["SMILES", "Target_RNA_sequence", "Molecule_name", "pKd"]].copy()
        val_df = pd.read_csv(os.path.join(cold_path, f"cold_{cold_name}_val_fold{i}.csv")).loc[:, ["SMILES", "Target_RNA_sequence", "Molecule_name", "pKd"]].copy()
        test_df = pd.read_csv(os.path.join(cold_path, f"cold_{cold_name}_test_fold{i}.csv")).loc[:, ["SMILES", "Target_RNA_sequence", "Molecule_name", "pKd"]].copy()
        

        #splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=dataset_seed)
        #train_idx, val_idx = next(splitter.split(train_df, groups=train_df["Target_RNA_sequence"]))

        #val_df = train_df.iloc[val_idx].copy()
        #train_df = train_df.iloc[train_idx].copy()
        # sanity checks for cold split
        train_val_overlap = set(train_df[cold_column]) & set(val_df[cold_column])
        train_test_overlap = set(train_df[cold_column]) & set(test_df[cold_column])
        val_test_overlap = set(val_df[cold_column]) & set(test_df[cold_column])

        assert len(train_val_overlap) == 0
        assert len(train_test_overlap) == 0
        assert len(val_test_overlap) == 0

        logger.info(f"(|train|, |val|, |test|) -> ({train_df.shape[0]}, {val_df.shape[0]}, {test_df.shape[0]})")
        if cold_type == "cold_rna":
            logger.info(f"(unique RNA train, val, test) -> ({train_df['Target_RNA_sequence'].nunique()}, {val_df['Target_RNA_sequence'].nunique()}, {test_df['Target_RNA_sequence'].nunique()})")
        elif cold_type == 'cold_mole':
            logger.info(f"(unique molecule names train, val, test) -> ({train_df['Molecule_name'].nunique()}, {val_df['Molecule_name'].nunique()}, {test_df['Molecule_name'].nunique()})")
        else: # cold start both setting
            logger.info(f"(unique RNA train, val, test) -> ({train_df['Target_RNA_sequence'].nunique()}, {val_df['Target_RNA_sequence'].nunique()}, {test_df['Target_RNA_sequence'].nunique()})")
            logger.info(f"(unique molecule names train, val, test) -> ({train_df['Molecule_name'].nunique()}, {val_df['Molecule_name'].nunique()}, {test_df['Molecule_name'].nunique()})")

        train_ds = MolRNADataset(train_df, setting=train_setting, rna_aug=rna_aug, mol_aug=mol_aug, task="training", is_seq_emb=is_seq_emb)
        val_ds = MolRNADataset(val_df, setting=val_setting, rna_aug=False, mol_aug=False, task="evaluating", is_seq_emb=is_seq_emb)
        test_ds = MolRNADataset(test_df, setting=val_setting, rna_aug=False, mol_aug=False, task="evaluating", is_seq_emb=is_seq_emb)

        g = torch.Generator().manual_seed(seed)

        train_loader = DataLoader(train_ds, batch_size=batch_size, generator=g, shuffle=True, collate_fn=MolRNADataset.collate_fn, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=MolRNADataset.collate_fn, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=MolRNADataset.collate_fn, num_workers=num_workers)

        early_stopping = EarlyStopping(mode=MODE, patience=patience, warm_up_epochs=warm_up_epochs, delta=delta)

        model = choose_model_setting(option=model_setting, exclude_modalities=exclude_modalities, freeze_grnade=freeze_grnade, dropout=dropout, smiles_seq_setting=smiles_seq_setting, smiles_expert_setting=smiles_expert_setting, d_model=d_model, ffn_mult=ffn_mult, bdcf_blocks=bdcf_blocks, node_pool=node_pool, token_pool=token_pool).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = rmse_loss

        train_losses, val_losses = [], []

        for epo in range(MAX_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            train_losses.append(train_loss)

            y_true_val, y_pred_val = predict(model, val_loader, device)
            p_val, s_val, rmse_val, r2_val = safe_regression_metrics(y_true_val, y_pred_val)

            val_losses.append(rmse_val)
            es_metric = rmse_val if MODE == 'min' else p_val # rmse for mode=min, ppc for mode=max
            early_stopping.check_early_stop(model, es_metric, [p_val, s_val, r2_val, rmse_val], train_loss, epo, [y_true_val, y_pred_val])

            if early_stopping.stop_training:
                logger.info(
                    f"[Fold {i}/5] Early stopping at epoch {epo + 1}. "
                    f"Best val score (rmse/ppc)={early_stopping.best_score:.4f} "
                    f"@ epoch {early_stopping.best_epoch + 1}"
                )
                break

        ####################################################################
        # evaluate on test set using best validation model
        model.load_state_dict(early_stopping.best_model)

        y_true_test, y_pred_test = predict(model, test_loader, device)
        p_test, s_test, rmse_test, r2_test = safe_regression_metrics(y_true_test, y_pred_test)

        logger.info(f"[Fold {i}/5] TEST using best validation model: p={p_test:.4f}, s={s_test:.4f}, rmse={rmse_test:.4f}, r2={r2_test:.4f}")

        p_list.append(p_test)
        s_list.append(s_test)
        rmse_list.append(rmse_test)
        r2_list.append(r2_test)
        ####################################################################

        if is_plot:
            date, plot = plot_name.split('_')
            plots_dir = os.path.join("results", "plots", date, plot)
            os.makedirs(plots_dir, exist_ok=True)
            plot_train_val_loss(train_losses, val_losses, save_path=f"{plots_dir}/losses_{plot_name}_fold_{i}.png", title="Loss Curve (train vs. val RMSE)")
            plot_prediction_correlation(y_true_test, y_pred_test, save_path=f"{plots_dir}/corr_best_model_{plot_name}_fold_{i}.png", title="Prediction Correlation - Best Validation Model")

    logger.info("Results using best validation model:")
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

    logger.info("\nCold-start blind test results:")
    logger.info(res)

    return res