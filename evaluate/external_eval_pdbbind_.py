import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.MultiStructRSA import choose_model_setting
from utils.MolRNACDatasetPDBBind import MolRNACDatasetPDBBind
from utils.utils import predict, safe_regression_metrics


logger = logging.getLogger(__name__)


N_MODELS = 5
PDBBIND_PATH = os.path.join(
    "datasets", "PDBBind2020", "pdbbindnl2020_preprocessed.csv"
)
MODEL_ROOT = os.path.join("saved_objects", "trained_cold")
RESULTS_ROOT = os.path.join(
    "results", "table_results", "external_validation"
)


def perform_pdbbind_external_eval(
    exp_id,
    date,
    model_save_dir=None,
    num_workers=0,
    device="cuda",
):
    """Evaluate five pretrained cold-RNA models on PDBBindNL2020."""

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    exp_num = f"exp{exp_id}"
    if model_save_dir is None:
        model_save_dir = os.path.join(
            MODEL_ROOT, str(date), f"{exp_num}_cold_rna"
        )

    output_dir = os.path.join(RESULTS_ROOT, str(date), exp_num)
    results_path = os.path.join(output_dir, "results.csv")

    ########################################################################
    # Validate all checkpoint paths before evaluation starts
    ########################################################################

    checkpoint_paths = [
        os.path.join(model_save_dir, f"model_fold_{i}.pt")
        for i in range(1, N_MODELS + 1)
    ]
    missing_paths = [path for path in checkpoint_paths if not os.path.isfile(path)]

    if missing_paths:
        raise FileNotFoundError(
            f"Missing pretrained cold-RNA model checkpoints: {missing_paths}"
        )

    model_configs = []
    excluded_modalities = []
    sequence_embedding_settings = []

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Checkpoint does not contain the expected metadata: {checkpoint_path}"
            )

        required_checkpoint_keys = {"model_state_dict", "model_config"}
        missing_checkpoint_keys = required_checkpoint_keys - set(checkpoint)
        if missing_checkpoint_keys:
            raise ValueError(
                f"Checkpoint {checkpoint_path} is missing keys: "
                f"{sorted(missing_checkpoint_keys)}"
            )

        training_config = checkpoint.get("training_config", {})
        model_config = checkpoint["model_config"]
        model_configs.append(model_config)
        excluded_modalities.append(set(model_config.get("exclude_modalities") or []))
        sequence_embedding_settings.append(
            bool(training_config.get("is_seq_emb", False))
        )
        del checkpoint

    excluded_by_all_models = set.intersection(*excluded_modalities)
    if len(set(sequence_embedding_settings)) != 1:
        raise ValueError(
            "All checkpoints must use the same is_seq_emb setting for external evaluation"
        )
    is_seq_emb = sequence_embedding_settings[0]

    ########################################################################
    # Load the fixed PDBBindNL2020 external-validation set once
    ########################################################################

    pdbbind_df = pd.read_csv(PDBBIND_PATH).copy()
    required_columns = {"SMILES", "Target_RNA_sequence", "pKd"}
    missing_columns = required_columns - set(pdbbind_df.columns)

    if missing_columns:
        raise ValueError(
            "PDBBind dataframe is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if pdbbind_df.empty:
        raise ValueError("PDBBindNL2020 external-validation dataset is empty")

    external_ds = MolRNACDatasetPDBBind(
        pdbbind_df,
        task="external evaluation",
        is_seq_emb=is_seq_emb,
        exclude_modalities=excluded_by_all_models,
    )
    external_loader = DataLoader(
        external_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=MolRNACDatasetPDBBind.collate_fn,
        num_workers=num_workers,
    )
    ordered_external_df = external_ds.get_df()

    logger.info(
        "PDBBindNL2020 external evaluation | "
        f"rows={len(ordered_external_df)}, models={N_MODELS}, "
        f"checkpoint_dir={model_save_dir}, is_seq_emb={is_seq_emb}, device={device}"
    )

    ########################################################################
    # Evaluate every pretrained checkpoint independently
    ########################################################################

    model_results = []
    prediction_paths = []

    for i, (checkpoint_path, model_config) in enumerate(
        zip(checkpoint_paths, model_configs), start=1
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model = choose_model_setting(**model_config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            y_true, y_pred = predict(model, external_loader, device)

        y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)

        if len(y_true) != len(ordered_external_df):
            raise ValueError(
                f"Model {i} produced {len(y_true)} targets for "
                f"{len(ordered_external_df)} PDBBindNL2020 samples"
            )
        if len(y_pred) != len(ordered_external_df):
            raise ValueError(
                f"Model {i} produced {len(y_pred)} predictions for "
                f"{len(ordered_external_df)} PDBBindNL2020 samples"
            )
        if not np.isfinite(y_true).all():
            raise ValueError(f"Model {i} evaluation contains non-numeric true pKd values")
        if not np.isfinite(y_pred).all():
            raise ValueError(f"Model {i} produced non-numeric predicted pKd values")

        pearson, spearman, rmse, r2 = safe_regression_metrics(y_true, y_pred)

        model_name = f"model_{i}"
        model_output_dir = os.path.join(output_dir, model_name)
        predictions_path = os.path.join(model_output_dir, "predictions.csv")
        os.makedirs(model_output_dir, exist_ok=True)

        predictions_df = ordered_external_df[
            ["SMILES", "Target_RNA_sequence"]
        ].copy()
        predictions_df["true_pkd"] = y_true
        predictions_df["predicted_pkd"] = y_pred
        predictions_df.to_csv(predictions_path, index=False)

        model_results.append(
            {
                "model": model_name,
                "rmse": float(rmse),
                "pearson": float(pearson),
                "spearman": float(spearman),
                "r2": float(r2),
            }
        )
        prediction_paths.append(predictions_path)

        logger.info(
            f"[{model_name}] PDBBindNL2020 | rmse={rmse:.4f}, "
            f"pearson={pearson:.4f}, spearman={spearman:.4f}, r2={r2:.4f}"
        )
        logger.info(f"[{model_name}] Predictions saved to: {predictions_path}")

        del model, checkpoint
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    ########################################################################
    # Save one row per independently evaluated model
    ########################################################################

    results_df = pd.DataFrame(model_results)
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(results_path, index=False)

    summary_lines = [
        "PDBBindNL2020 external evaluation complete",
        f"Dataset size: {len(ordered_external_df)}",
        f"Checkpoint directory: {model_save_dir}",
        f"Models evaluated: {len(model_results)}",
    ]
    for result, predictions_path in zip(model_results, prediction_paths):
        summary_lines.append(
            f"{result['model']}: rmse={result['rmse']:.4f}, "
            f"pearson={result['pearson']:.4f}, "
            f"spearman={result['spearman']:.4f}, r2={result['r2']:.4f} | "
            f"predictions={predictions_path}"
        )
    summary_lines.append(f"Statistics: {results_path}")
    summary = "\n".join(summary_lines)

    logger.info(summary)
    print(summary)

    return {
        "dataset_size": len(ordered_external_df),
        "checkpoint_dir": model_save_dir,
        "models_evaluated": len(model_results),
        "model_results": model_results,
        "prediction_paths": prediction_paths,
        "results_path": results_path,
    }
