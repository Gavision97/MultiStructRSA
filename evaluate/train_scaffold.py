import logging
import os
import random

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import DataLoader

from models.MultiStructRSA import choose_model_setting
from utils.MolRNACDataset import MolRNADataset
from utils.tools import set_seed
from utils.utils import (
    EarlyStopping,
    predict,
    rmse_loss,
    safe_regression_metrics,
    train_one_epoch,
)


logger = logging.getLogger(__name__)

MODE = "min"
VALIDATION_FRACTION = 0.15
DATA_COLUMNS = ["SMILES", "Target_RNA_sequence", "Molecule_name", "pKd"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RSIM_DATASET_PATH = os.path.join(
    PROJECT_ROOT, "datasets", "RSIM", "All_sf_dataset_v1.csv"
)
MODEL_SAVE_DIR = os.path.join(
    PROJECT_ROOT, "saved_objects", "trained_scaffold"
)
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "case_study_model.pt")
STATS_SAVE_PATH = os.path.join(
    MODEL_SAVE_DIR, "case_study_training_stats.csv"
)


def get_bemis_murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=False
    )


def create_scaffold_split(dataset, validation_fraction=VALIDATION_FRACTION, seed=2):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if dataset.empty:
        raise ValueError("Cannot split an empty dataset")
    if "SMILES" not in dataset.columns:
        raise ValueError("R-SIM dataframe is missing the SMILES column")
    if not dataset.index.is_unique:
        raise ValueError("R-SIM dataframe indices must be unique")

    scaffold_groups = {}
    invalid_smiles_indices = []

    for index, smiles in dataset["SMILES"].items():
        scaffold = get_bemis_murcko_scaffold(smiles)
        if scaffold is None:
            invalid_smiles_indices.append(index)
            scaffold = "__RDKIT_UNPARSABLE_SMILES__"
        scaffold_groups.setdefault(scaffold, []).append(index)

    if invalid_smiles_indices:
        logger.warning(
            "RDKit could not parse %d R-SIM rows; keeping them together in "
            "one scaffold group",
            len(invalid_smiles_indices),
        )

    scaffold_keys = sorted(scaffold_groups)
    random.Random(seed).shuffle(scaffold_keys)

    target_validation_size = round(len(dataset) * validation_fraction)
    validation_scaffolds = set()
    validation_size = 0

    for scaffold in scaffold_keys:
        if validation_size >= target_validation_size:
            break
        validation_scaffolds.add(scaffold)
        validation_size += len(scaffold_groups[scaffold])

    train_scaffolds = set(scaffold_keys) - validation_scaffolds
    train_indices = [
        index
        for scaffold in train_scaffolds
        for index in scaffold_groups[scaffold]
    ]
    validation_indices = [
        index
        for scaffold in validation_scaffolds
        for index in scaffold_groups[scaffold]
    ]

    train_df = dataset.loc[sorted(train_indices)].copy()
    val_df = dataset.loc[sorted(validation_indices)].copy()

    scaffold_overlap = train_scaffolds.intersection(validation_scaffolds)
    if scaffold_overlap:
        raise ValueError(
            f"Found {len(scaffold_overlap)} scaffolds in both train and validation"
        )

    index_overlap = set(train_df.index).intersection(val_df.index)
    if index_overlap:
        raise ValueError(
            f"Found {len(index_overlap)} dataframe indices in both train and validation"
        )

    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Scaffold split contains an empty partition: "
            f"train={len(train_df)}, validation={len(val_df)}"
        )
    if len(train_df) + len(val_df) != len(dataset):
        raise ValueError("Scaffold split does not contain every R-SIM row exactly once")

    return train_df, val_df, train_scaffolds, validation_scaffolds


def train_scaffold_model(
    model_setting="full",
    train_setting=4,
    val_setting=4,
    rna_aug=False,
    mol_aug=False,
    patience=20,
    warm_up_epochs=10,
    delta=1e-5,
    lr=5e-5,
    wd=1e-5,
    dropout=0.1,
    exclude_modalities=None,
    freeze_grnade=False,
    smiles_seq_setting="deep",
    smiles_expert_setting="proj",
    batch_size=1,
    num_workers=0,
    is_seq_emb=False,
    d_model=128,
    ffn_mult=4,
    bdcf_blocks=1,
    node_pool="attn",
    token_pool="mean",
    max_epochs=200,
    device="cuda",
    seed=1,
    dataset_seed=2,
):
    """Train one MultiStructRSA model on a fixed R-SIM scaffold split."""

    if max_epochs < 1:
        raise ValueError("max_epochs must be at least 1")
    if patience < 1:
        raise ValueError("patience must be at least 1")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if not os.path.isfile(RSIM_DATASET_PATH):
        raise FileNotFoundError(f"R-SIM dataset not found: {RSIM_DATASET_PATH}")

    all_df = pd.read_csv(RSIM_DATASET_PATH, sep="\t")
    missing_columns = set(DATA_COLUMNS) - set(all_df.columns)
    if missing_columns:
        raise ValueError(
            f"R-SIM dataframe is missing required columns: {sorted(missing_columns)}"
        )

    all_df = all_df.loc[:, DATA_COLUMNS].copy()
    train_df, val_df, train_scaffolds, val_scaffolds = create_scaffold_split(
        all_df, validation_fraction=VALIDATION_FRACTION, seed=dataset_seed
    )

    scaffold_overlap = train_scaffolds.intersection(val_scaffolds)
    if scaffold_overlap:
        raise ValueError(
            f"Scaffold overlap must be empty, found: {sorted(scaffold_overlap)}"
        )
    if set(train_df.index).intersection(val_df.index):
        raise ValueError("Train and validation dataframe indices overlap")

    logger.info(
        f"Training one scaffold-split model | seed:{seed}, dataset seed:{dataset_seed}, "
        f"batch size:{batch_size}, max epochs:{max_epochs}, lr:{lr}, "
        f"weight decay:{wd}, dropout:{dropout}, num_workers:{num_workers}, "
        f"patience:{patience}, warm up epo:{warm_up_epochs}, delta:{delta}"
    )
    logger.info(
        f"R-SIM scaffold split | total:{len(all_df)}, train:{len(train_df)}, "
        f"validation:{len(val_df)}, train scaffolds:{len(train_scaffolds)}, "
        f"validation scaffolds:{len(val_scaffolds)}, overlap:{len(scaffold_overlap)}"
    )

    set_seed(seed_=seed)

    train_ds = MolRNADataset(
        train_df,
        setting=train_setting,
        rna_aug=rna_aug,
        mol_aug=mol_aug,
        task="training",
        is_seq_emb=is_seq_emb,
    )
    val_ds = MolRNADataset(
        val_df,
        setting=val_setting,
        rna_aug=False,
        mol_aug=False,
        task="validation",
        is_seq_emb=is_seq_emb,
    )

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        generator=generator,
        shuffle=True,
        collate_fn=MolRNADataset.collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=MolRNADataset.collate_fn,
        num_workers=num_workers,
    )

    early_stopping = EarlyStopping(
        mode=MODE,
        patience=patience,
        warm_up_epochs=warm_up_epochs,
        delta=delta,
    )
    model = choose_model_setting(
        option=model_setting,
        exclude_modalities=exclude_modalities,
        freeze_grnade=freeze_grnade,
        dropout=dropout,
        smiles_seq_setting=smiles_seq_setting,
        smiles_expert_setting=smiles_expert_setting,
        d_model=d_model,
        ffn_mult=ffn_mult,
        bdcf_blocks=bdcf_blocks,
        node_pool=node_pool,
        token_pool=token_pool,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = rmse_loss
    history = []

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        y_true_val, y_pred_val = predict(model, val_loader, device)
        p_val, s_val, rmse_val, r2_val = safe_regression_metrics(
            y_true_val, y_pred_val
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(rmse_val),
                "val_rmse": float(rmse_val),
                "val_pearson": float(p_val),
                "val_spearman": float(s_val),
                "val_r2": float(r2_val),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        early_stopping.check_early_stop(
            model,
            rmse_val,
            [p_val, s_val, r2_val, rmse_val],
            train_loss,
            epoch,
            [y_true_val, y_pred_val],
        )

        if early_stopping.stop_training:
            logger.info(
                f"Early stopping at epoch {epoch + 1}. Best validation RMSE="
                f"{early_stopping.best_score:.4f} at epoch "
                f"{early_stopping.best_epoch + 1}"
            )
            break

    if early_stopping.best_model is False:
        raise RuntimeError("No finite validation model was stored")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    best_model_state = {
        name: value.detach().cpu()
        for name, value in early_stopping.best_model.items()
    }
    torch.save(best_model_state, MODEL_SAVE_PATH)
    pd.DataFrame(history).to_csv(STATS_SAVE_PATH, index=False)

    p_best, s_best, _, rmse_best = early_stopping.best_metrics
    summary = (
        "\nScaffold training complete\n"
        f"R-SIM samples: {len(all_df)}\n"
        f"Train size: {len(train_df)}\n"
        f"Validation size: {len(val_df)}\n"
        f"Unique train scaffolds: {len(train_scaffolds)}\n"
        f"Unique validation scaffolds: {len(val_scaffolds)}\n"
        f"Scaffold overlap: {len(scaffold_overlap)} (verified zero)\n"
        f"Best epoch: {early_stopping.best_epoch + 1}\n"
        f"Best validation RMSE: {float(rmse_best):.6f}\n"
        f"Validation Pearson at best epoch: {float(p_best):.6f}\n"
        f"Validation Spearman at best epoch: {float(s_best):.6f}\n"
        f"Saved model: {MODEL_SAVE_PATH}\n"
        f"Saved CSV: {STATS_SAVE_PATH}"
    )
    print(summary)
    logger.info(summary)

    return {
        "model_path": MODEL_SAVE_PATH,
        "stats_path": STATS_SAVE_PATH,
        "best_epoch": early_stopping.best_epoch + 1,
        "best_validation_rmse": float(rmse_best),
    }
