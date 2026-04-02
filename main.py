import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import os
import json

from evaluate.cv_eval import cv_eval, cv_eval_, cv_eval__
from evaluate.cv_eval_ns import cv_eval_ns
from evaluate.cold_eval import preform_cold_blind_eval

from utils.generate_smiles_features import generate_smiles_features
from utils.parser import get_args

import logging
logger = logging.getLogger(__name__)

RSIM_PATH = os.path.join("datasets", "RSIM")
ALL_SF_DATASET_PATH = os.path.join("datasets", "RSIM", "All_sf_dataset_v1.csv")

def preform_cold_blind_eval_(exp_name='exp', res_file_name="cold_eval", cold_type="cold_rna",
                           model_setting="full", train_setting=4, val_setting=4,smoothing_factor=0.75, negative_factor=1,
                             aug=False, lr=5e-5, wd=1e-5, is_plot=False, is_seq_emb=False, plot_name="loss_curve",
                           device="cuda", seed=1):
    """
    Run cold-start blind evaluation and append results to a summary CSV file.

    The function:
    1. Executes cold-start evaluation using `preform_cold_blind_cv`.
    2. Collects mean/std metrics and per-fold results.
    3. Saves (or appends) a single-row summary to a CSV file.

    Parameters
    ----------
    exp_name : str, default="exp"
        Experiment name stored in the results table.

    res_file_name : str, default="cold_eval"
        Output CSV file name (saved under `results/table_results/`).

    cold_type : str, default="cold_rna"
        Cold-start setting ('cold_rna', 'cold_mol', or 'cold_rm').

    model_setting : str, default="full"
        Model configuration passed to the training pipeline.

    device : str, default="cuda"
        Device used for training and evaluation.

    seed : int, default=1
        Random seed for reproducibility.

    Returns
    -------
    None
        Results are appended to the specified CSV file.
    """
    res = preform_cold_blind_eval(cold_type=cold_type, model_setting=model_setting, train_setting=train_setting, val_setting=val_setting,
                                aug=aug, lr=lr, wd=wd, is_plot=is_plot, is_seq_emb=is_seq_emb, plot_name=plot_name,
                                device=device, seed=seed)
    # summary (single row, same fields as the per-family version)
    summary_dict = {"exp": exp_name, "cold_type": cold_type, **res}
    summary_df = pd.DataFrame([summary_dict])

    logger.info(
        f"Cold evaluation on {cold_type}"
        f"rmse_mean: {res.get('rmse_mean')}, p_mean: {res.get('p_mean')}, "
        f"s_mean: {res.get('s_mean')}, r2_mean: {res.get('r2_mean')}"
    )

    out_dir = os.path.join("results", "table_results")
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

def preform_k_fold_cv_all(ev_setting=1, exp_name = 'exp', res_file_name="cv10_all_results", model_setting = "full", train_setting=4, val_setting=4, 
                          aug=False, smoothing_factor=0.75, negative_factor=1, 
                          lr=5e-5, wd=1e-5, is_plot=False, is_seq_emb=False, n=10, plot_name='loss_plot', 
                          device="cuda", seed=1, dataset_seed=2):
    """
    Run k-fold cross-validation on the full dataset (all RNA families combined) and save a summary of the results.

    The function:
    1. Loads the full dataset from `ALL_SF_DATASET_PATH`.
    2. Runs cross-validation using `cv_eval`.
    3. Extracts mean metrics (RMSE, Pearson, Spearman, R²).
    4. Saves (or appends) a single-row summary to a CSV file.

    Parameters
    ----------
    exp_name : str, default="exp"
        Name of the experiment (saved in the results table).

    model_setting : str, default="full"
        Model variant passed to `cv_eval` via `choose_model_setting`.

    res_file_name : str, default="cv10_all_results"
        Output CSV file name (stored under `results/table_results/`).

    n : int, default=10
        Number of cross-validation folds.

    plot_name : str, default="loss_plot"
        Name used for saving fold loss plots.

    device : str, default="cuda"
        Device used for training and evaluation.

    Returns
    -------
    None
        Results are saved to disk. A summary row is appended (or created)
        in the CSV file.

    Notes
    -----
    - The dataset includes all RNA families concatenated.
    - Each run appends a new row to the results file, enabling experiment tracking.
    """
    logging.info(f"\n==================== ALL FAMILIES  ==================== (setting:{ev_setting})")

    # 1 - like in DeepRSMA, early stopping on the test set (test-leakage, overoptimistic)
    # 2 - early stopping on the validation loss & test best mode 
    # 3 - early stopping on the validation loss & then retrain new model with chosen N epochs using train+val data
    cv_setting2method = {1 : cv_eval, 2: cv_eval_, 3: cv_eval__, 4:cv_eval_ns} 
    selected_cv_eval  = cv_setting2method[ev_setting]

    all_df = pd.read_csv(ALL_SF_DATASET_PATH, sep="\t")
    res = selected_cv_eval (dataset=all_df, model_setting=model_setting,train_setting=train_setting, val_setting=val_setting,
                  aug=aug, smoothing_factor=smoothing_factor, negative_factor=negative_factor, lr=lr, wd=wd, is_plot=is_plot, is_seq_emb=is_seq_emb, n=n, plot_name=plot_name ,device=device, seed=seed, dataset_seed=dataset_seed)


    # summary (single row, same fields as the per-family version)
    summary_dict = {"exp": exp_name, "family": "all_families", "n_samples": len(all_df), **res}
    summary_df = pd.DataFrame([summary_dict])

    logger.info(
        f"RNA family: all_families_concat, n_samples: {len(all_df)}, "
        f"rmse_mean: {res.get('rmse_mean')}, p_mean: {res.get('p_mean')}, "
        f"s_mean: {res.get('s_mean')}, r2_mean: {res.get('r2_mean')}"
    )

    out_dir = os.path.join("results", "table_results")
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


def preform_k_fold_cv_all_rna_fam(model_setting = "full", train_setting=4, val_setting=4, aug=False, lr=5e-5, wd=1e-5, is_plot=False, is_seq_emb=False,
                                   exp_name='cv_rna_fam', res_file_name='cv10_all_results_by_family', n=10, plot_name='loss_plot',
                                   device='cuda', seed=1, dataset_seed=2):
    """
    Run k-fold cross-validation separately for each RNA family and save a summary table.

    The function:
    1. Loads predefined RNA family datasets (aptamers, miRNA, repeats, etc.).
    2. Runs `cv_eval` independently on each family.
    3. Collects mean metrics (RMSE, Pearson, Spearman, R²) per family.
    4. Saves (or appends) all family results into a single CSV file.

    Parameters
    ----------
    model_setting : str, default="full"
        Model variant passed to `cv_eval`.

    exp_name : str, default="cv_rna_fam"
        Experiment name stored in the results table.

    res_file_name : str, default="cv10_all_results_by_family"
        Output CSV file name (saved under `results/table_results/`).

    n : int, default=10
        Number of cross-validation folds per RNA family.

    plot_name : str, default="loss_plot"
        Name used for saving fold loss plots.

    device : str, default="cuda"
        Device used for training and evaluation.

    Returns
    -------
    None
        Results are saved to disk as a CSV table (one row per RNA family).

    Notes
    -----
    - Each RNA family is evaluated independently (no mixing between families).
    - The output file accumulates results across runs for experiment tracking.
    """

    summary_rows = []
    # load dictionary that maps RNA family name into its correct dataset
    rna_families = {
        "aptamers":  pd.read_csv(os.path.join(RSIM_PATH, "Aptamers_dataset_v1.csv"), sep="\t"),
        "mirna":     pd.read_csv(os.path.join(RSIM_PATH, "miRNA_dataset_v1.csv"), sep="\t"),
        "repeats":   pd.read_csv(os.path.join(RSIM_PATH, "Repeats_dataset_v1.csv"), sep="\t"),
        "ribosomal": pd.read_csv(os.path.join(RSIM_PATH, "Ribosomal_dataset_v1.csv"), sep="\t"),
        "riboswitch":pd.read_csv(os.path.join(RSIM_PATH, "Riboswitch_dataset_v1.csv"), sep="\t"),
        "viralrna":  pd.read_csv(os.path.join(RSIM_PATH, "Viral_RNA_dataset_v1.csv"), sep="\t"),
    }
    

    for fam_name, fam_df in rna_families.items():
        logging.info(f"\n==================== {fam_name.upper()} ====================")
        #res = cv_eval(fam_df, name=fam_name, device=device)  # make sure cv_eval has name param or ignore
        res = cv_eval(dataset=fam_df, model_setting=model_setting, train_setting=train_setting, val_setting=val_setting,
                      aug=aug, lr=lr, wd=wd, is_plot=is_plot, is_seq_emb=is_seq_emb, fam_name=fam_name, n=n, plot_name=plot_name, device=device, seed=seed, dataset_seed=dataset_seed)
        # ---- summary row ----
        summary_row = {"exp": exp_name, "family": fam_name, "n_samples": len(fam_df), **res}
        summary_rows.append(summary_row)
        
        logging.info(f'RNA family: {fam_name}, n_samples: {len(fam_df)}, avg_mse: {res.get("Average MSE")}, avg_mae: {res.get("Average MAE")}, avg_r^2: {res.get("Average R2 score")}, avg_pearson (r): {res.get("Average Pearson correlation (r)")}')
    summary_df = pd.DataFrame(summary_rows).sort_values("family").reset_index(drop=True)

    # Save
    out_dir = os.path.join("results", "table_results")
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
    args = get_args()

    log_msg = args.log_msg
    model_setting = args.model_setting
    res_file_name = args.res_file_name
    exp_name = args.exp_name
    log_file_name = args.log_file_name
    eval_method = args.eval_method
    cv_method = args.cv_method
    cold_type = args.cold_type
    plot_name = args.plot_name
    k_fold = args.k_fold

    train_setting = args.train_setting
    val_setting = args.val_setting
    aug = True if args.aug == "True" else False
    is_plot = True if args.is_plot == "True" else False
    is_seq_emb = True if args.is_seq_emb == "True" else False

    smoothing_factor = args.smoothing_factor
    negative_factor = args.negative_factor 

    lr = args.lr
    wd = args.wd
    ev_setting = args.ev_setting # [1, 2, 3]
    device = args.device
    seed = args.seed # controls model initialization, dataloader shuffle, etc.
    dataset_seed = args.dataset_seed # controls cross-validation fold creation (for statified K-fold CV)

    # ---- construct log file path ---- #
    # expected format: {eval_method}_{date}_{exp_num}
    log_sub_dir = log_file_name.split("_")[-2]
    log_dir = os.path.join("results", "logs", log_sub_dir)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{log_file_name}.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )

    logger.info(log_msg)

    # ---- run evaluation ---- #
    if eval_method == "cv":
        if cv_method == "all":
            preform_k_fold_cv_all(
                ev_setting = ev_setting,
                model_setting = model_setting,
                train_setting=train_setting,
                val_setting=val_setting,
                smoothing_factor=smoothing_factor,
                negative_factor=negative_factor,
                aug=aug,
                lr=lr,
                wd=wd,
                is_plot=is_plot,
                is_seq_emb=is_seq_emb,
                exp_name=exp_name,
                res_file_name=res_file_name,
                n=k_fold,
                plot_name=plot_name,
                device=device,
                seed=seed,
                dataset_seed=dataset_seed
            )
        else:
            preform_k_fold_cv_all_rna_fam(
                ev_setting = ev_setting,
                model_setting = model_setting,
                train_setting=train_setting,
                val_setting=val_setting,
                aug=aug,
                lr=lr,
                wd=wd,
                is_plot=is_plot,
                is_seq_emb=is_seq_emb,
                exp_name=exp_name,
                res_file_name=res_file_name,
                n=k_fold,
                plot_name=plot_name,
                device=device,
                seed=seed,
                dataset_seed=dataset_seed
            )

    elif eval_method == "cold":
        preform_cold_blind_eval_(exp_name=exp_name,
                               res_file_name=res_file_name,
                               cold_type=cold_type,
                               model_setting=model_setting,
                               train_setting=train_setting,
                               val_setting=val_setting,
                               aug=aug,
                               lr=lr,
                               wd=wd,
                               is_plot=is_plot,
                               is_seq_emb=is_seq_emb,
                               plot_name=plot_name,
                               device=device,
                               seed=seed)

    elif eval_method == "loo_cv":
        raise NotImplementedError("loo_cv is not implemented yet.")


if __name__ == "__main__":
    main()