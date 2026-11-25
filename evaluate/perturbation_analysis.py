import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from utils.utils import *
from utils.tools import *
from utils.MolRNACDatasetPertubation import MolRNADatasetPertubation
from models.MultiStructRSA import *

import logging
logger = logging.getLogger(__name__)


CHECKPOINT_ROOT = os.path.join("saved_objects", "cv")
OUTPUT_DIR = os.path.join("results", "table_results", "perturbation_analysis_permutation")

METRICS = ["p", "s", "rmse", "r2"]


def get_metrics(y_true, y_pred):
    p, s, rmse, r2 = safe_regression_metrics(y_true, y_pred)
    return {"p": float(p), "s": float(s), "rmse": float(rmse), "r2": float(r2)}


def get_summary(fold_results):
    fold_df = pd.DataFrame(fold_results)
    summary = {
        "model_setting": fold_df["model_setting"].iloc[0],
        "n_folds": len(fold_df),
        "n_perturbations": int(fold_df["n_perturbations"].iloc[0]),
    }
    if "model_seed" in fold_df.columns:
        summary["model_seed"] = int(fold_df["model_seed"].iloc[0])

    for metric in METRICS:
        perturbation_values = fold_df[f"perturbation_{metric}_mean"].to_numpy(dtype=float)
        mean_prediction_values = fold_df[f"mean_prediction_{metric}"].to_numpy(dtype=float)

        summary[f"perturbation_{metric}_mean_across_folds"] = float(np.mean(perturbation_values))
        summary[f"perturbation_{metric}_std_across_folds"] = float(np.std(perturbation_values))
        summary[f"mean_prediction_{metric}_mean_across_folds"] = float(np.mean(mean_prediction_values))
        summary[f"mean_prediction_{metric}_std_across_folds"] = float(np.std(mean_prediction_values))

    return pd.DataFrame([summary])


def save_results(output_dir, repeat_results, fold_results, sample_results,
                 permutation_results):
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(repeat_results).to_csv(
        os.path.join(output_dir, "repeat_metrics.csv"), index=False)
    pd.DataFrame(fold_results).to_csv(
        os.path.join(output_dir, "fold_summary.csv"), index=False)
    pd.DataFrame(sample_results).to_csv(
        os.path.join(output_dir, "sample_predictions.csv"), index=False)
    pd.DataFrame(permutation_results).to_csv(
        os.path.join(output_dir, "structure_permutations.csv"), index=False)
    get_summary(fold_results).to_csv(
        os.path.join(output_dir, "summary.csv"), index=False)


def perturbation_analysis(dataset=None, model_setting="full", val_setting=4,
                          dropout=0.1, exclude_modalities=None,
                          freeze_grnade=False, smiles_seq_setting='deep',
                          smiles_expert_setting='proj', batch_size=1,
                          num_workers=0, is_seq_emb=False, d_model=128,
                          ffn_mult=4, bdcf_blocks=2, node_pool='mean',
                          token_pool='mean', n=5, n_perturbations=50,
                          device="cuda", dataset_seed=2, seed=1, folds=None,
                          checkpoint_root=CHECKPOINT_ROOT, output_dir=OUTPUT_DIR,
                          fold_datasets=None, checkpoint_dir=None,
                          checkpoint_pattern="model_fold{fold}.pt",
                          model_seed=None):
    """
    Evaluate fixed CV checkpoints using complete permutations of RNA structures.

    For every perturbation, the unique RNA structures in the held-out fold are
    permuted one-to-one. The mapping is fixed for the complete evaluation pass,
    has no unchanged RNA-to-structure pairs, and is reproducible from the
    perturbation seed. Only perturbed structures are evaluated.
    """

    logger.info(
        f'K folds:{n}, perturbations:{n_perturbations}, seed:{seed}, '
        f'dataset seed:{dataset_seed}, batch size:{batch_size}, '
        f'model setting:{model_setting}'
    )

    if n_perturbations < 1:
        raise ValueError("n_perturbations must be at least 1")

    selected_folds = folds if folds else list(range(1, n + 1))

    invalid_folds = [fold for fold in selected_folds if fold < 1 or fold > n]
    if invalid_folds:
        raise ValueError(f"Invalid folds: {invalid_folds}")

    if fold_datasets is None:
        if dataset is None:
            raise ValueError("dataset is required for stratified CV evaluation")
        all_df = dataset
        Y = all_df["pKd"].to_numpy(dtype=np.float32)
        skf = regressor_stratified_cv(
            n_splits=n, n_repeats=1, random_state=dataset_seed,
            group_count=5, strategy="uniform")
        cv_splits = list(skf.split(np.zeros(len(Y)), Y))
        evaluation_folds = [
            (fold, test_index, all_df.iloc[test_index].reset_index(drop=True))
            for fold, (_, test_index) in enumerate(cv_splits, start=1)
        ]
    else:
        missing_folds = set(range(1, n + 1)) - set(fold_datasets)
        if missing_folds:
            raise ValueError(
                f"Missing predefined fold datasets: {sorted(missing_folds)}")
        evaluation_folds = [
            (fold, np.arange(len(fold_datasets[fold])),
             fold_datasets[fold].reset_index(drop=True))
            for fold in range(1, n + 1)
        ]

    repeat_results = []
    fold_results = []
    sample_results = []
    permutation_results = []

    for fold, test_index, test_df in evaluation_folds:
        if fold not in selected_folds:
            continue

        set_seed(seed)

        test_ds = MolRNADatasetPertubation(
            test_df, setting=val_setting, shuffle_struct=False,
            rna_aug=False, mol_aug=False, task="perturbation analysis",
            is_seq_emb=is_seq_emb)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            collate_fn=MolRNADatasetPertubation.collate_fn,
            num_workers=num_workers)

        if checkpoint_dir is None:
            checkpoint_path = os.path.join(
                checkpoint_root, model_setting, f"model_fold{fold}.pt")
        else:
            checkpoint_name = checkpoint_pattern.format(
                fold=fold, model_seed=model_seed)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        model = choose_model_setting(
            option=model_setting, exclude_modalities=exclude_modalities,
            freeze_grnade=freeze_grnade, dropout=dropout,
            smiles_seq_setting=smiles_seq_setting,
            smiles_expert_setting=smiles_expert_setting, d_model=d_model,
            ffn_mult=ffn_mult, bdcf_blocks=bdcf_blocks,
            node_pool=node_pool, token_pool=token_pool).to(device)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        logger.info(
            f'Fold number {fold}/{n}; test len:{len(test_loader.dataset)}; '
            f'unique RNA structures:{len(test_ds.unique_rna_seqs)}; '
            f'checkpoint:{checkpoint_path}'
        )

        perturbation_predictions = np.empty(
            (n_perturbations, len(test_df)), dtype=np.float32)
        current_fold_repeat_results = []
        y_true = None

        test_ds.shuffle_struct = True
        for perturbation in range(1, n_perturbations + 1):
            # The same fold/repetition receives the same permutation for every
            # model setting, while different repetitions receive new mappings.
            permutation_seed = seed + perturbation
            struct_permutation = test_ds.permute_structures(seed=permutation_seed)

            if len(set(struct_permutation.values())) != len(struct_permutation):
                raise RuntimeError("The structure permutation is not one-to-one")
            if any(rna_seq == struct_seq for rna_seq, struct_seq in
                   struct_permutation.items()):
                raise RuntimeError("The structure permutation contains a fixed point")

            for rna_seq, perturbed_rna_seq in struct_permutation.items():
                permutation_row = {
                    "model_setting": model_setting,
                    "fold": fold,
                    "perturbation": perturbation,
                    "source_rna_sequence": rna_seq,
                    "perturbed_rna_sequence": perturbed_rna_seq,
                }
                if model_seed is not None:
                    permutation_row["model_seed"] = model_seed
                permutation_results.append(permutation_row)

            repeat_y_true, repeat_y_pred = predict(model, test_loader, device)
            if y_true is None:
                y_true = repeat_y_true
            elif not np.array_equal(repeat_y_true, y_true):
                raise RuntimeError(
                    f"Target order changed in fold {fold}, perturbation {perturbation}")

            perturbation_predictions[perturbation - 1] = repeat_y_pred
            perturbation_metrics = get_metrics(y_true, repeat_y_pred)
            repeat_row = {
                "model_setting": model_setting,
                "fold": fold,
                "perturbation": perturbation,
                **perturbation_metrics,
            }
            if model_seed is not None:
                repeat_row["model_seed"] = model_seed
            repeat_results.append(repeat_row)
            current_fold_repeat_results.append(repeat_row)

            logger.info(
                f'[Fold {fold}/{n}] completed perturbation '
                f'{perturbation}/{n_perturbations}')

        mean_predictions = np.mean(perturbation_predictions, axis=0)
        std_predictions = np.std(perturbation_predictions, axis=0)
        mean_prediction_metrics = get_metrics(y_true, mean_predictions)
        current_fold_repeat_df = pd.DataFrame(current_fold_repeat_results)

        fold_row = {
            "model_setting": model_setting,
            "fold": fold,
            "n_test_samples": len(test_df),
            "n_unique_rna_structures": len(test_ds.unique_rna_seqs),
            "n_perturbations": n_perturbations,
        }
        if model_seed is not None:
            fold_row["model_seed"] = model_seed
        for metric in METRICS:
            fold_row[f"perturbation_{metric}_mean"] = float(
                current_fold_repeat_df[metric].mean())
            fold_row[f"perturbation_{metric}_std"] = float(
                current_fold_repeat_df[metric].std(ddof=0))
            fold_row[f"mean_prediction_{metric}"] = mean_prediction_metrics[metric]
        fold_results.append(fold_row)

        for sample_idx, source_idx in enumerate(test_index):
            row = test_df.iloc[sample_idx]
            sample_row = {
                "model_setting": model_setting,
                "fold": fold,
                "source_index": int(source_idx),
                "SMILES": row["SMILES"],
                "Target_RNA_sequence": row["Target_RNA_sequence"],
                "ground_truth_pKd": float(y_true[sample_idx]),
                "perturbed_prediction_mean": float(mean_predictions[sample_idx]),
                "perturbed_prediction_std": float(std_predictions[sample_idx]),
            }
            if model_seed is not None:
                sample_row["model_seed"] = model_seed
            for perturbation in range(1, n_perturbations + 1):
                sample_row[f"perturbed_prediction_{perturbation}"] = float(
                    perturbation_predictions[perturbation - 1, sample_idx])
            sample_results.append(sample_row)

        save_results(output_dir, repeat_results, fold_results, sample_results,
                     permutation_results)

        if hasattr(test_ds, "rna_emb"):
            test_ds.rna_emb.close()
        del test_loader, test_ds, model
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    summary = get_summary(fold_results)
    logger.info(f"Completed perturbation analysis\n{summary.to_string(index=False)}")
    return summary.to_dict(orient="records")[0]
