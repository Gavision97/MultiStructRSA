import argparse
import logging

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from evaluate.cv_eval import cv_eval
from evaluate.blind_cv_eval import preform_cold_blind_cv

from utils.generate_smiles_features import generate_smiles_features


import logging
logger = logging.getLogger(__name__)

RSIM_PATH = os.path.join("datasets", "RSIM")
ALL_SF_DATASET_PATH = os.path.join("datasets", "RSIM", "All_sf_dataset_v1.csv")

def load_dataset(return_concat=False):
    rna_families = {
        "aptamers":  pd.read_csv(os.path.join(RSIM_PATH, "Aptamers_dataset_v1.csv"), sep="\t"),
        "mirna":     pd.read_csv(os.path.join(RSIM_PATH, "miRNA_dataset_v1.csv"), sep="\t"),
        "repeats":   pd.read_csv(os.path.join(RSIM_PATH, "Repeats_dataset_v1.csv"), sep="\t"),
        "ribosomal": pd.read_csv(os.path.join(RSIM_PATH, "Ribosomal_dataset_v1.csv"), sep="\t"),
        "riboswitch":pd.read_csv(os.path.join(RSIM_PATH, "Riboswitch_dataset_v1.csv"), sep="\t"),
        "viralrna":  pd.read_csv(os.path.join(RSIM_PATH, "Viral_RNA_dataset_v1.csv"), sep="\t"),
    }
    if not return_concat:
        return rna_families
    
    # concate all RNA families datasets into one whole dataset
    concat_df = pd.concat(rna_families.values(), axis=0, ignore_index=True)

    logging.info(f"[Before deduplication] shape = {concat_df.shape}")
    concat_df = concat_df.drop_duplicates()
    logging.info(f"[After deduplication]  shape = {concat_df.shape}")

    concat_df = concat_df[concat_df["Target_RNA_sequence"].str.len() <= 5000].reset_index(drop=True)
    logging.info(f"[After removing RNA sequences with sequence length > 1000] shape = {concat_df.shape} ")
    return concat_df

def preform_k_fold_cv_all(exp_name = 'exp', res_file_name="cv10_all_results", n=10, plot_name='loss_plot' ,device="cuda"):
    all_df = pd.read_csv(ALL_SF_DATASET_PATH, sep="\t")

    logging.info(f"\n==================== ALL FAMILIES  ====================")

    res = cv_eval(all_df, n=n, plot_name=plot_name ,device=device)


    # summary (single row, same fields as the per-family version)
    summary_dict = {
        "exp": exp_name,
        "family": "all_families_concat",
        "n_samples": len(all_df),
        "rmse_mean": res.get("rmse_mean"),
        "p_mean": res.get("p_mean"),
        "s_mean": res.get("s_mean"),
        "r2_mean": res.get("r2_mean"),
    }

    summary_df = pd.DataFrame([summary_dict])

    logger.info(
        f"RNA family: all_families_concat, n_samples: {len(all_df)}, "
        f"rmse_mean: {res.get('rmse_mean')}, p_mean: {res.get('p_mean')}, "
        f"s_mean: {res.get('s_mean')}, r2_mean: {res.get('r2_mean')}"
    )

    out_dir = os.path.join("results")
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, f"{res_file_name}.csv")

    if os.path.exists(summary_path):
        # File exists, append new row
        existing_df = pd.read_csv(summary_path)
        updated_df = pd.concat([existing_df, summary_df], ignore_index=True)
    else:
        # File does not exist, create new one
        updated_df = summary_df

    updated_df.to_csv(summary_path, index=False)

    logging.info(f"\nSaved/updated summary to: {summary_path}")


def preform_k_fold_cv_all_rna_fam(exp_name='cv_rna_fam', res_file_name='cv10_all_results_by_family', n=10, plot_name='loss_plot', device='cuda'):
    
    # load dictionary that maps RNA family name into its correct dataset
    rna_families = load_dataset()
    summary_rows = []

    for fam_name, fam_df in rna_families.items():
        logging.info(f"\n==================== {fam_name.upper()} ====================")
        #res = cv_eval(fam_df, name=fam_name, device=device)  # make sure cv_eval has name param or ignore
        res = cv_eval(fam_df, name=fam_name, n=n, plot_name=plot_name, device=device)
        # ---- summary row ----
        summary_row = {
            "exp": exp_name,
            "family": fam_name,
            "n_samples": len(fam_df),
            "avg_mae": res.get("Average MAE"),
            "avg_mse": res.get("Average MSE"),
            "avg_r2": res.get("Average R2 score"),
            "avg_pearson": res.get("Average Pearson correlation (r)"),
        }
        summary_rows.append(summary_row)
        logging.info(f'RNA family: {fam_name}, n_samples: {len(fam_df)}, avg_mse: {res.get("Average MSE")}, avg_mae: {res.get("Average MAE")}, avg_r^2: {res.get("Average R2 score")}, avg_pearson (r): {res.get("Average Pearson correlation (r)")}')
    summary_df = pd.DataFrame(summary_rows).sort_values("family").reset_index(drop=True)


    # Save
    out_dir = os.path.join("results")
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, f"{res_file_name}.csv")

    if os.path.exists(summary_path):
        # File exists, append new row
        existing_df = pd.read_csv(summary_path)
        updated_df = pd.concat([existing_df, summary_df], ignore_index=True)
    else:
        # File does not exist, create new one
        updated_df = summary_df

    updated_df.to_csv(summary_path, index=False)

    logging.info(f"\nSaved/updated summary to: {summary_path}")

def preprocess_and_save_smiles_features():
    full_df = pd.read_csv(ALL_SF_DATASET_PATH, sep="\t")
    logger.info(f"Full R-SIM dataset shape -> {full_df.shape}")

    unique_smiles = full_df["SMILES"].unique()
    
    # generate & save SMILES embeddings
    generate_smiles_features(mol_set=unique_smiles, save=True)



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_msg", type=str, default='log message', help="First message to display in the log file; for experiment description")
    parser.add_argument("--res_file_name", type=str, default='res', help="name of the result; pandas.DataFrame() object")
    parser.add_argument("--exp_name", type=str, default='exp', help="exirement name; critical for saving results row in pd.DataFrame()")
    parser.add_argument("--eval_method", type=str, default='cv', choices=['cv', 'loo_cv', 'cold_blind_cv'], help="path to the smiles file")
    parser.add_argument("--cv_method", type=str, default='all', choices=['all', 'per_fam'], help="The split for 10-fold cross-validation - all data or per RNA subtype ")
    parser.add_argument("--cold_type", type=str, default='rna', choices=['rna', 'smiles', 'rna_and_smiles'], help="The type of cold start setting - deafult 'rna'")
    parser.add_argument("--plot_name", type=str, default='loss_curve', help="loss curve plot name; will be saved to results/plots/plot_name.png")
    parser.add_argument("--log_file_name", type=str, default="res", help="name of the logger file (e.g., result.log)")
    parser.add_argument("--k_fold", type=int,  default=5, help="Number of folds for stratified K-fold cross-validation (default: 10)")
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'], help="device to use - cuda/cpu")
    args = parser.parse_args()


    log_msg = args.log_msg
    res_file_name = args.res_file_name
    exp_name = args.exp_name
    log_file_name = args.log_file_name
    eval_method= args.eval_method
    plot_name = args.plot_name
    k_fold = args.k_fold
    device = args.device


    # constract log file in order to log the results
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_file_name}.log")
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(message)s')
    
    logger.info(log_msg)
    if eval_method == 'cv':
        if args.cv_method == "all":
            preform_k_fold_cv_all(exp_name=exp_name, res_file_name=res_file_name, n=k_fold, plot_name=plot_name ,device=device)
        else:
            preform_k_fold_cv_all_rna_fam(exp_name=exp_name, res_file_name=res_file_name, n=k_fold, plot_name=plot_name ,device=device)
    elif eval_method == 'cold_blind_cv':
        preform_cold_blind_cv(args.cold_type)


if __name__ == "__main__":
    main()


