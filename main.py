import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from pathlib import Path
import os
import pickle

from evaluate.cv_eval import cv_eval
from evaluate.cold_eval_ import preform_cold_blind_eval

from utils.generate_smiles_features import generate_smiles_features
from utils.parser import get_args

import logging
logger = logging.getLogger(__name__)

RSIM_PATH = os.path.join("datasets", "RSIM")
ALL_SF_DATASET_PATH = os.path.join("datasets", "RSIM", "All_sf_dataset_v1.csv")

def preform_cold_blind_eval_(exp_name='exp', res_file_name="cold_eval", cold_type="cold_rna",
                           model_setting="full", train_setting=4, val_setting=4,
                           rna_aug=False, mol_aug=False, patience=20, warm_up_epochs=10, delta=1e-5, lr=5e-5, wd=1e-5, dropout=0.1, exclude_modalities=None, freeze_grnade=False, smiles_seq_setting='deep', smiles_expert_setting='proj',
                            batch_size=1, num_workers=8, is_plot=False,
                            is_seq_emb=False, d_model=128, ffn_mult=4, bdcf_blocks=1, node_pool='attn', token_pool='mean', plot_name="loss_curve",
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
                                rna_aug=rna_aug, mol_aug=mol_aug, patience=patience, warm_up_epochs=warm_up_epochs, delta=delta,
                                lr=lr, wd=wd, dropout=dropout, exclude_modalities=exclude_modalities, freeze_grnade=freeze_grnade, smiles_seq_setting=smiles_seq_setting, smiles_expert_setting=smiles_expert_setting, batch_size=batch_size, is_plot=is_plot,
                                is_seq_emb=is_seq_emb, d_model=d_model, ffn_mult=ffn_mult, bdcf_blocks=bdcf_blocks, node_pool=node_pool, token_pool=token_pool,
                                plot_name=plot_name, device=device, seed=seed)
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

def preform_k_fold_cv_all(exp_name = 'exp', res_file_name="cv10_all_results", model_setting = "full", train_setting=4, val_setting=4, 
                          rna_aug=False, mol_aug=False, patience=20, warm_up_epochs=10, delta=1e-5,  lr=5e-5, wd=1e-5, dropout=0.1, exclude_modalities=None, freeze_grnade=False, smiles_seq_setting='deep', smiles_expert_setting='proj', batch_size=1, num_workers=8, is_plot=False, 
                          is_seq_emb=False, d_model=128, ffn_mult=4, bdcf_blocks=1, node_pool='attn', token_pool='mean', n=10, plot_name='loss_plot', device="cuda", seed=1, dataset_seed=2):
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
    logging.info(f"\n==================== ALL FAMILIES  ==================== )")

    all_df = pd.read_csv(ALL_SF_DATASET_PATH, sep="\t")
    res = cv_eval(dataset=all_df, model_setting=model_setting,train_setting=train_setting, val_setting=val_setting,
                  rna_aug=rna_aug, mol_aug=mol_aug, patience=patience, warm_up_epochs=warm_up_epochs, delta=delta, lr=lr, wd=wd, dropout=dropout, exclude_modalities=exclude_modalities, freeze_grnade=freeze_grnade, smiles_seq_setting=smiles_seq_setting, smiles_expert_setting=smiles_expert_setting, batch_size=batch_size, num_workers=num_workers, is_plot=is_plot,
                  is_seq_emb=is_seq_emb, d_model=d_model, ffn_mult=ffn_mult, bdcf_blocks=bdcf_blocks, node_pool=node_pool, token_pool=token_pool, n=n, plot_name=plot_name ,device=device, seed=seed, dataset_seed=dataset_seed)


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


def preform_k_fold_cv_all_rna_fam(model_setting = "full", train_setting=4, val_setting=4, rna_aug=False, mol_aug=False, patience=20, warm_up_epochs=10, delta=1e-5,
                                  lr=5e-5, wd=1e-5, dropout=0.1, exclude_modalities=None, freeze_grnade=False, smiles_seq_setting='deep', smiles_expert_setting='proj',
                                   batch_size=1, num_workers=8, is_plot=False, is_seq_emb=False, d_model=128, ffn_mult=4, bdcf_blocks=1, node_pool='attn', token_pool='mean',
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
                      rna_aug=rna_aug, mol_aug=mol_aug, patience=patience, warm_up_epochs=warm_up_epochs, delta=delta, lr=lr, wd=wd, dropout=dropout, exclude_modalities=exclude_modalities, freeze_grnade=freeze_grnade, smiles_seq_setting=smiles_seq_setting, smiles_expert_setting=smiles_expert_setting, batch_size=batch_size, num_workers=num_workers, is_plot=is_plot,
                      is_seq_emb=is_seq_emb, d_model=d_model, ffn_mult=ffn_mult, bdcf_blocks=bdcf_blocks, node_pool=node_pool, token_pool=token_pool, fam_name=fam_name, n=n, plot_name=plot_name, device=device, seed=seed, dataset_seed=dataset_seed)
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



def preprocess_smiles_emb(smiles_path, save_name="smiles_embeddings", features=['simson', 'chemberta', 'morgan_fp', 'desc'], device='cpu') -> None:
    """
    Load a pickled SMILES molecule set and generate selected SMILES embeddings.

    Parameters
    ----------
    smiles_path : str or Path
        Path to a pickle file containing the molecule set (mol_set).
    features : list of str, optional
        List of feature types to generate. Supported options include:
        ['simson', 'chemberta', 'morgan_fp', 'desc'].

    Raises
    ------
    FileNotFoundError
        If the provided smiles_path does not exist.
    """
    smiles_path = Path(smiles_path)

    if not smiles_path.exists():
        raise FileNotFoundError(f"SMILES file not found: {smiles_path}")

    with open(smiles_path, "rb") as f:
        mol_set = pickle.load(f)

    _ = generate_smiles_features(mol_set=mol_set, save_name=save_name, features=features, save=True, device=device)

def main():
    args = get_args()

    log_msg = args.log_msg
    res_file_name = args.res_file_name
    exp_name = args.exp_name
    log_file_name = args.log_file_name
    setting = args.setting

    plot_name = args.plot_name
    is_plot = True if args.is_plot == "True" else False

    # cv kwargs
    cv_method = args.cv_method
    k_fold = args.k_fold


    # cold kwargs
    cold_type = args.cold_type

    # data kwargs
    is_seq_emb = True if args.is_seq_emb == "True" else False
    train_setting = args.train_setting
    val_setting = args.val_setting
    rna_aug = True if args.rna_aug == "True" else False
    mol_aug = True if args.mol_aug == "True" else False
    

    dataset_kwards = {'is_seq_emb':is_seq_emb, 'train_setting':train_setting}
    model_setting = args.model_setting
    
    d_model = args.d_model
    ffn_mult = args.ffn_mult
    bdcf_blocks = args.bdcf_blocks
    node_pool = args.node_pool
    token_pool = args.token_pool
    freeze_grnade = True if args.freeze_grnade == "True" else False
    exclude_modalities = args.exclude_modalities
    smiles_seq_setting = args.smiles_seq_setting
    smiles_expert_setting = args.smiles_expert_setting
    model_kwargs = {'setting':model_setting, 'exclude_modalities':exclude_modalities, 'freeze_grnade': freeze_grnade,
                    'd_model': d_model, 'ffn_mult': ffn_mult, 'bdcf_blocks':bdcf_blocks, 'node_pool': node_pool,
                    'token_pool':token_pool, 'smiles_seq_setting':smiles_seq_setting, 'smiles_expert_setting':smiles_expert_setting}
    

    patience = args.patience
    warm_up_epochs = args.warm_up_epochs
    delta = args.delta

    # train kwargs
    lr = args.lr
    wd = args.wd
    dropout = args.dropout
    batch_size = args.batch_size
    num_workers = args.num_workers
    device = args.device
    seed = args.seed # controls model initialization, dataloader shuffle, etc.
    dataset_seed = args.dataset_seed # controls cross-validation fold creation (for statified K-fold CV)
    training_kwargs = {'lr':lr, 'wd':wd, 'dropout':dropout, 'batch_size':batch_size,
                       'num_workers':num_workers, 'device':device, 'seed':seed, 'seed_dataset':dataset_seed,
                       'is_plot':is_plot, 'plot_name':plot_name}
    
    # SMILES feature extraction kwargs
    smiles_path = args.smiles_path
    smiles_features_gen = args.smiles_features_gen
    save_name = args.save_name

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
    if setting == "cv":
        if cv_method == "all":
            preform_k_fold_cv_all(
                model_setting = model_setting,
                train_setting=train_setting,
                val_setting=val_setting,
                rna_aug=rna_aug,
                mol_aug=mol_aug,
                patience=patience,
                warm_up_epochs=warm_up_epochs,
                delta=delta,
                lr=lr,
                wd=wd,
                dropout=dropout,
                exclude_modalities=exclude_modalities,
                freeze_grnade=freeze_grnade,
                smiles_seq_setting=smiles_seq_setting,
                smiles_expert_setting=smiles_expert_setting,
                batch_size=batch_size,
                num_workers=num_workers,
                is_plot=is_plot,
                is_seq_emb=is_seq_emb,
                d_model=d_model,
                ffn_mult=ffn_mult,
                bdcf_blocks=bdcf_blocks,
                node_pool=node_pool,
                token_pool=token_pool,
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
                model_setting = model_setting,
                train_setting=train_setting,
                val_setting=val_setting,
                rna_aug=rna_aug,
                mol_aug=mol_aug,
                patience=patience,
                warm_up_epochs=warm_up_epochs,
                delta=delta,
                lr=lr,
                wd=wd,
                dropout=dropout,
                exclude_modalities=exclude_modalities,
                freeze_grnade=freeze_grnade,
                smiles_seq_setting=smiles_seq_setting,
                smiles_expert_setting=smiles_expert_setting,
                batch_size=batch_size,
                num_workers=num_workers,
                is_plot=is_plot,
                is_seq_emb=is_seq_emb,
                d_model=d_model,
                ffn_mult=ffn_mult,
                bdcf_blocks=bdcf_blocks,
                node_pool=node_pool,
                token_pool=token_pool,
                exp_name=exp_name,
                res_file_name=res_file_name,
                n=k_fold,
                plot_name=plot_name,
                device=device,
                seed=seed,
                dataset_seed=dataset_seed
            )

    elif setting == "cold":
        preform_cold_blind_eval_(exp_name=exp_name,
                               res_file_name=res_file_name,
                               cold_type=cold_type,
                               model_setting=model_setting,
                               train_setting=train_setting,
                               val_setting=val_setting,
                               rna_aug=rna_aug,
                               mol_aug=mol_aug,
                               patience=patience,
                               warm_up_epochs=warm_up_epochs,
                               delta=delta,
                               lr=lr,
                               wd=wd,
                               dropout=dropout,
                               exclude_modalities=exclude_modalities,
                               freeze_grnade=freeze_grnade,
                               smiles_seq_setting=smiles_seq_setting,
                               smiles_expert_setting=smiles_expert_setting,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               is_plot=is_plot,
                               is_seq_emb=is_seq_emb,
                               d_model=d_model,
                               ffn_mult=ffn_mult,
                               bdcf_blocks=bdcf_blocks,
                               node_pool=node_pool,
                               token_pool=token_pool,
                               plot_name=plot_name,
                               device=device,
                               seed=seed)
        
    # extract & save SMILES embeddings to desire path
    elif setting == "smi_emb_gen":
        preprocess_smiles_emb(smiles_path=smiles_path, save_name=save_name ,features=smiles_features_gen, device=device)


if __name__ == "__main__":
    main()