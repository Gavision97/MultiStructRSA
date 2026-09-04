import argparse
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


def run_inference(model, data_loader, device):
    predicted_smiles = []
    predictions = []

    with torch.inference_mode():
        if torch.is_grad_enabled():
            raise RuntimeError("Gradients must be disabled during case-study inference")

        for inputs, smiles in data_loader:
            inputs = [value.to(device) for value in inputs]
            batch_predictions = model(*inputs).reshape(-1)

            if batch_predictions.numel() != len(smiles):
                raise ValueError(
                    "Prediction count does not match the current ChemDiv batch"
                )

            predicted_smiles.extend(smiles)
            predictions.extend(batch_predictions.cpu().tolist())

    return predicted_smiles, predictions


def validate_and_save_results(dataset, predicted_smiles, predictions):
    if len(predictions) != len(dataset):
        raise ValueError(
            f"Received {len(predictions)} predictions for {len(dataset)} "
            "evaluated ChemDiv molecules"
        )
    if len(predicted_smiles) != len(dataset):
        raise ValueError(
            f"Received {len(predicted_smiles)} SMILES values for {len(dataset)} "
            "evaluated ChemDiv molecules"
        )

    expected_smiles = dataset.get_df()["SMILES"].tolist()
    if predicted_smiles != expected_smiles:
        raise ValueError("DataLoader prediction order does not match the dataset")

    results_df = pd.DataFrame(
        {
            "SMILES": predicted_smiles,
            "Target_RNA_sequence": TARGET_RNA_SEQUENCE,
            "pkd": predictions,
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
    if not results_df["Target_RNA_sequence"].eq(TARGET_RNA_SEQUENCE).all():
        raise ValueError("A result row contains the wrong target RNA sequence")

    results_df["pkd"] = pd.to_numeric(results_df["pkd"], errors="raise")
    if results_df["pkd"].isna().any():
        raise ValueError("MultiStructRSA produced NaN pKd predictions")
    if not np.isfinite(results_df["pkd"].to_numpy()).all():
        raise ValueError("MultiStructRSA produced non-finite pKd predictions")

    results_df = results_df.sort_values(
        "pkd", ascending=False
    ).reset_index(drop=True)

    if list(results_df.columns) != [
        "SMILES",
        "Target_RNA_sequence",
        "pkd",
    ]:
        raise ValueError("Screening result columns do not match the required schema")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    return results_df


def evaluate_case_study(device="cuda", num_workers=0):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

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

    predicted_smiles, predictions = run_inference(model, data_loader, device)
    results_df = validate_and_save_results(
        dataset, predicted_smiles, predictions
    )

    highest_ranked = results_df.iloc[0]
    summary_lines = [
        "ChemDiv virtual screening complete",
        f"ChemDiv molecules found: {dataset.molecules_found}",
        f"Successfully evaluated: {len(results_df)}",
        f"Skipped: {dataset.molecules_skipped}",
    ]
    for reason, count in sorted(dataset.skip_reasons.items()):
        summary_lines.append(f"  {reason}: {count}")
    summary_lines.extend(
        [
            f"SLI RNA sequence: {TARGET_RNA_SEQUENCE}",
            f"Model checkpoint: {MODEL_CHECKPOINT_PATH}",
            f"Highest predicted pKd: {highest_ranked['pkd']:.6f}",
            f"Highest-ranked SMILES: {highest_ranked['SMILES']}",
            f"Output CSV: {OUTPUT_PATH}",
        ]
    )
    print("\n".join(summary_lines))

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Screen the ChemDiv RNA-binding library against SLI"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Inference device, for example 'cuda', 'cuda:0', or 'cpu'",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers",
    )
    args = parser.parse_args()
    evaluate_case_study(device=args.device, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
