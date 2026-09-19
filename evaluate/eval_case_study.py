import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.MultiStructRSA import choose_model_setting
from utils.MolRNACDataset_chemdiv import MolRNACDatasetChemDiv
from utils.layers import Dropout as GVPDropout


TARGET_RNA_SEQUENCE = "AUUUUUCCUCGAACUUGGCGGAACGCAGAAAAAU"
CHEMDIV_EMBEDDINGS_PATH = os.path.join(
    PROJECT_ROOT, "embeddings", "smiles", "smiles_emb_chemdiv.pkl"
)
MODEL_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_objects",
    "trained_scaffold",
    "case_study_model.pt",
)
OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "table_results",
    "case_study",
    "screening_results.csv",
)

# Exact architecture used by job_settings/job_setting3.sh to train the raw
# state_dict saved as case_study_model.pt.
MODEL_CONFIG = {
    "option": "full",
    "exclude_modalities": ["morgan", "ernie"],
    "freeze_grnade": False,
    "smiles_seq_setting": "deep",
    "smiles_expert_setting": "proj",
    "dropout": 0.1,
    "d_model": 128,
    "ffn_mult": 4,
    "bdcf_blocks": 2,
    "node_pool": "mean",
    "token_pool": "mean",
}

DROPOUT_MODULES = (
    torch.nn.Dropout
)


def load_chemdiv_embeddings():
    if not os.path.isfile(CHEMDIV_EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"ChemDiv embeddings pickle not found: {CHEMDIV_EMBEDDINGS_PATH}"
        )

    with open(CHEMDIV_EMBEDDINGS_PATH, "rb") as file:
        smiles_embeddings = pickle.load(file)

    if not isinstance(smiles_embeddings, dict):
        raise TypeError("ChemDiv embeddings pickle must contain a dictionary")

    expected_keys = {"mordred descriptors", "simson", "chemberta"}
    missing_keys = expected_keys - set(smiles_embeddings)
    if missing_keys:
        raise KeyError(
            "ChemDiv embeddings pickle is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    return smiles_embeddings


def load_model(device):
    if not os.path.isfile(MODEL_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Trained scaffold checkpoint not found: {MODEL_CHECKPOINT_PATH}"
        )

    model = choose_model_setting(**MODEL_CONFIG)
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location="cpu")

    # train_scaffold.py saves a raw state_dict. Supporting the standard wrapped
    # key keeps the error clear if the checkpoint storage format is later updated.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError("Model checkpoint does not contain a valid state_dict")

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    if model.training:
        raise RuntimeError("MultiStructRSA must be in evaluation mode")

    return model


def enable_mc_dropout(model):
    """Keep the model in eval mode while enabling its stochastic dropout paths."""
    model.eval()
    enabled_modules = []

    for name, module in model.named_modules():
        if isinstance(module, GVPDropout):
            # gRNAde uses a custom tuple dropout. Training this wrapper enables
            # both its scalar nn.Dropout and custom vector-channel dropout.
            module.train()
            enabled_modules.append(name)
        elif isinstance(module, DROPOUT_MODULES):
            module.train()
            enabled_modules.append(name)
        elif (
            isinstance(module, torch.nn.MultiheadAttention)
            and module.dropout > 0
        ):
            # MultiheadAttention applies dropout functionally based on its own
            # training flag. The case-study single-sample blocks currently use
            # the PyTorch default dropout=0, so they remain in evaluation mode.
            module.train()
            enabled_modules.append(name)

    if model.training:
        raise RuntimeError("The root MultiStructRSA model must remain in evaluation mode")
    if not enabled_modules:
        raise RuntimeError("No dropout modules were found for MC Dropout inference")
    if any(
        module.training
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ):
        raise RuntimeError("BatchNorm must remain in evaluation mode during MC Dropout")

    return enabled_modules


def validate_mc_dropout_is_stochastic(model, data_loader, device):
    try:
        inputs, _ = next(iter(data_loader))
    except StopIteration as error:
        raise ValueError("Cannot validate MC Dropout with an empty DataLoader") from error

    inputs = [value.to(device) for value in inputs]
    sanity_predictions = []

    with torch.inference_mode():
        if torch.is_grad_enabled():
            raise RuntimeError("Gradients must be disabled during MC Dropout inference")

        for _ in range(2):
            prediction = model(*inputs).reshape(-1)
            if prediction.requires_grad:
                raise RuntimeError("MC Dropout predictions must not require gradients")
            sanity_predictions.append(prediction.cpu())

    if torch.equal(sanity_predictions[0], sanity_predictions[1]):
        raise RuntimeError(
            "MC Dropout sanity check failed: two stochastic forward passes "
            "returned exactly identical predictions. Dropout may not be enabled."
        )


def run_mc_dropout_inference(model, data_loader, dataset, device, mc_samples):
    expected_smiles = dataset.get_df()["SMILES"].tolist()
    prediction_rows = []

    with torch.inference_mode():
        if torch.is_grad_enabled():
            raise RuntimeError("Gradients must be disabled during MC Dropout inference")

        for mc_iteration in range(mc_samples):
            pass_smiles = []
            pass_predictions = []

            for inputs, smiles in data_loader:
                inputs = [value.to(device) for value in inputs]
                batch_predictions = model(*inputs).reshape(-1)

                if batch_predictions.requires_grad:
                    raise RuntimeError("MC Dropout predictions must not require gradients")
                if batch_predictions.numel() != len(smiles):
                    raise ValueError(
                        "Prediction count does not match the current ChemDiv batch"
                    )

                pass_smiles.extend(smiles)
                pass_predictions.extend(batch_predictions.cpu().tolist())

            if len(pass_predictions) != len(dataset):
                raise ValueError(
                    f"MC pass {mc_iteration + 1} returned {len(pass_predictions)} "
                    f"predictions for {len(dataset)} evaluated ChemDiv molecules"
                )
            if pass_smiles != expected_smiles:
                raise ValueError(
                    f"SMILES order changed during MC pass {mc_iteration + 1}"
                )

            pass_predictions = np.asarray(pass_predictions, dtype=np.float64)
            if not np.isfinite(pass_predictions).all():
                raise ValueError(
                    f"MC pass {mc_iteration + 1} produced NaN or infinite predictions"
                )
            prediction_rows.append(pass_predictions)

    prediction_matrix = np.stack(prediction_rows, axis=0)
    expected_shape = (mc_samples, len(dataset))
    if prediction_matrix.shape != expected_shape:
        raise ValueError(
            f"MC prediction matrix has shape {prediction_matrix.shape}; "
            f"expected {expected_shape}"
        )

    return expected_smiles, prediction_matrix


def validate_model_parameters_unchanged(model, parameter_snapshot):
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("A model parameter accumulated gradients during screening")

    for name, parameter in model.named_parameters():
        if not torch.equal(parameter.detach().cpu(), parameter_snapshot[name]):
            raise RuntimeError(f"Model parameter '{name}' changed during screening")


def validate_and_save_results(dataset, predicted_smiles, prediction_matrix, mc_samples):
    if len(predicted_smiles) != len(dataset):
        raise ValueError(
            f"Received {len(predicted_smiles)} SMILES values for {len(dataset)} "
            "evaluated ChemDiv molecules"
        )

    expected_smiles = dataset.get_df()["SMILES"].tolist()
    if predicted_smiles != expected_smiles:
        raise ValueError("DataLoader prediction order does not match the dataset")

    expected_shape = (mc_samples, len(dataset))
    if prediction_matrix.shape != expected_shape:
        raise ValueError(
            f"MC prediction matrix has shape {prediction_matrix.shape}; "
            f"expected {expected_shape}"
        )
    if not np.isfinite(prediction_matrix).all():
        raise ValueError("Raw MC predictions contain NaN or infinite values")

    raw_predictions = prediction_matrix.T.tolist()
    if any(len(predictions) != mc_samples for predictions in raw_predictions):
        raise ValueError("A molecule does not have exactly mc_samples raw predictions")

    results_df = pd.DataFrame(
        {
            "RNA_Target_Name": "SLI",
            "SMILES": predicted_smiles,
            "mean_pkd": np.mean(prediction_matrix, axis=0),
            "std_pkd": np.std(prediction_matrix, axis=0, ddof=0),
            "pkd05": np.percentile(prediction_matrix, 5, axis=0),
            "pkd95": np.percentile(prediction_matrix, 95, axis=0),
            "raw_pred": [json.dumps(predictions) for predictions in raw_predictions],
        }
    )

    if results_df["SMILES"].duplicated().any():
        duplicates = results_df.loc[
            results_df["SMILES"].duplicated(keep=False), "SMILES"
        ].unique()[:3]
        raise ValueError(
            "Duplicate ChemDiv predictions were produced. Examples: "
            f"{list(duplicates)}"
        )
    if not results_df["RNA_Target_Name"].eq("SLI").all():
        raise ValueError("A result row contains the wrong RNA target name")

    statistic_columns = ["mean_pkd", "std_pkd", "pkd05", "pkd95"]
    results_df[statistic_columns] = results_df[statistic_columns].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(results_df[statistic_columns].to_numpy()).all():
        raise ValueError("MC Dropout statistics contain NaN or infinite values")
    if (results_df["std_pkd"] < 0).any():
        raise ValueError("MC Dropout standard deviations cannot be negative")
    if (results_df["pkd05"] > results_df["pkd95"]).any():
        raise ValueError("A 5th percentile is greater than its 95th percentile")

    # The sample mean is not mathematically guaranteed to lie within the
    # empirical 5th-95th percentile interval for a strongly skewed sample.

    results_df = results_df.sort_values(
        "mean_pkd", ascending=False
    ).reset_index(drop=True)

    if list(results_df.columns) != [
        "RNA_Target_Name",
        "SMILES",
        "mean_pkd",
        "std_pkd",
        "pkd05",
        "pkd95",
        "raw_pred",
    ]:
        raise ValueError("Screening result columns do not match the required schema")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    return results_df


def evaluate_case_study(device="cuda", num_workers=0, mc_samples=100):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if mc_samples <= 0:
        raise ValueError("mc_samples must be greater than zero")

    # Check the checkpoint before loading the much larger ChemDiv feature store.
    if not os.path.isfile(MODEL_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Trained scaffold checkpoint not found: {MODEL_CHECKPOINT_PATH}"
        )

    smiles_embeddings = load_chemdiv_embeddings()
    dataset = MolRNACDatasetChemDiv(
        smiles_embeddings,
        target_rna_sequence=TARGET_RNA_SEQUENCE,
    )
    del smiles_embeddings

    data_loader = dataset.make_loader(
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
    )
    model = load_model(device)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Loaded model unexpectedly contains parameter gradients")

    parameter_snapshot = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    enable_mc_dropout(model)
    validate_mc_dropout_is_stochastic(model, data_loader, device)

    predicted_smiles, prediction_matrix = run_mc_dropout_inference(
        model, data_loader, dataset, device, mc_samples
    )
    validate_model_parameters_unchanged(model, parameter_snapshot)
    results_df = validate_and_save_results(
        dataset, predicted_smiles, prediction_matrix, mc_samples
    )

    highest_ranked = results_df.iloc[0]
    summary_lines = [
        "ChemDiv virtual screening complete",
        "RNA target name: SLI",
        f"RNA sequence: {TARGET_RNA_SEQUENCE}",
        f"ChemDiv molecules found: {dataset.molecules_found}",
        f"Successfully evaluated: {len(results_df)}",
        f"Skipped: {dataset.molecules_skipped}",
    ]
    for reason, count in sorted(dataset.skip_reasons.items()):
        summary_lines.append(f"  {reason}: {count}")
    summary_lines.extend(
        [
            f"MC Dropout samples: {mc_samples}",
            f"Model checkpoint: {MODEL_CHECKPOINT_PATH}",
            f"Highest mean predicted pKd: {highest_ranked['mean_pkd']:.6f}",
            f"Std of highest-ranked molecule: {highest_ranked['std_pkd']:.6f}",
            "5th-95th percentile interval of highest-ranked molecule: "
            f"[{highest_ranked['pkd05']:.6f}, {highest_ranked['pkd95']:.6f}]",
            f"Highest-ranked SMILES: {highest_ranked['SMILES']}",
            f"Output CSV: {OUTPUT_PATH}",
        ]
    )
    print("\n".join(summary_lines))

    return results_df
