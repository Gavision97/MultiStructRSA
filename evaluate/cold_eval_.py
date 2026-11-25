import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.MultiStructRSA import *
from utils.MolRNACDataset import MolRNADataset
from utils.plot_utils import plot_prediction_correlation, plot_train_val_loss
from utils.tools import set_seed
from utils.utils import *
from utils.utils import rmse_loss


logger = logging.getLogger(__name__)

# Early stopping hyperparameters.
PATIENCE = 20
WARM_UP_EPOCHS = 10
MODE = "min"
DELTA = 1e-3
MAX_EPOCHS = 200

SPLIT = True

# Dataset paths.
COLD_PATH = os.path.join("datasets", "blind_test", "processed")
RNA_COLD_PATH = os.path.join(
    "datasets", "blind_test", "rna", "train_val_test_splits"
)
MOLE_COLD_PATH = os.path.join(
    "datasets", "blind_test", "mole", "train_val_test_splits"
)
BOTH_COLD_PATH = os.path.join(
    "datasets", "blind_test", "rm", "train_val_test_splits"
)
RSIM_PATH = os.path.join("datasets", "RSIM")


def choose_cold_type(cold_type="cold_rna"):
    assert cold_type in ["cold_rna", "cold_mole", "cold_rm"]
    if cold_type == "cold_rna":
        return RNA_COLD_PATH, "rna"
    if cold_type == "cold_mole":
        return MOLE_COLD_PATH, "mole"
    return BOTH_COLD_PATH, "rm"


def _safe_path_component(value):
    """Return a stable, filesystem-safe directory name for model settings."""
    return str(value).strip().replace("/", "_").replace(" ", "_")


def _save_test_predictions(
    test_df,
    y_true,
    y_pred,
    cold_type,
    model_setting,
    model_seed,
    fold,
):
    """Save row-aligned test predictions and metadata for downstream statistics."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(test_df) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError(
            "Prediction export is not row-aligned: "
            f"test rows={len(test_df)}, targets={len(y_true)}, predictions={len(y_pred)}"
        )

    prediction_df = test_df.reset_index(drop=True).copy()
    prediction_df.insert(0, "test_row_index", np.arange(len(prediction_df)))
    prediction_df.insert(0, "fold", fold)
    prediction_df.insert(0, "model_seed", model_seed)
    prediction_df.insert(0, "model_setting", model_setting)
    prediction_df.insert(0, "cold_type", cold_type)
    prediction_df["y_true"] = y_true
    prediction_df["y_pred"] = y_pred
    prediction_df["residual"] = y_pred - y_true
    prediction_df["absolute_error"] = np.abs(prediction_df["residual"])
    prediction_df["squared_error"] = prediction_df["residual"] ** 2

    safe_model_setting = _safe_path_component(model_setting)
    output_dir = os.path.join(
        "results",
        "table_results",
        "cold",
        cold_type,
        safe_model_setting,
        str(model_seed),
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"fold{fold}.csv")
    prediction_df.to_csv(output_path, index=False)
    logger.info(f"[Fold {fold}/5] Saved raw test predictions to: {output_path}")
    return output_path


def preform_cold_blind_eval(
    cold_type="cold_rna",
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
    num_workers=8,
    is_plot=False,
    is_seq_emb=False,
    d_model=128,
    ffn_mult=4,
    bdcf_blocks=1,
    node_pool="attn",
    token_pool="mean",
    n=10,
    fam_name=None,
    plot_name="loss_curve",
    device="cuda",
    dataset_seed=2,
    seed=1,
    save_model=False,
):
    cold_column = (
        "Target_RNA_sequence" if cold_type == "cold_rna" else "Molecule_name"
    )
    cold_path, cold_name = choose_cold_type(cold_type=cold_type)

    logger.info(
        f"Performing Blind Test with Cold Start Setting = {cold_type}, "
        f"seed:{seed}, batch size:{batch_size}, max epochs:{MAX_EPOCHS}, "
        f"lr:{lr}, weight decay:{wd}, patience:{patience}, "
        f"warm up epo:{warm_up_epochs}, delta:{delta}"
    )

    p_list, s_list, rmse_list, r2_list = [], [], [], []

    for i in range(1, 6):
        set_seed(seed_=seed)
        logger.info(f"Fold number {i}/5")

        columns = ["SMILES", "Target_RNA_sequence", "Molecule_name", "pKd"]
        train_df = pd.read_csv(
            os.path.join(cold_path, f"cold_{cold_name}_train_fold{i}.csv")
        ).loc[:, columns].copy()
        val_df = pd.read_csv(
            os.path.join(cold_path, f"cold_{cold_name}_val_fold{i}.csv")
        ).loc[:, columns].copy()
        test_df = pd.read_csv(
            os.path.join(cold_path, f"cold_{cold_name}_test_fold{i}.csv")
        ).loc[:, columns].copy()

        train_val_overlap = set(train_df[cold_column]) & set(val_df[cold_column])
        train_test_overlap = set(train_df[cold_column]) & set(test_df[cold_column])
        val_test_overlap = set(val_df[cold_column]) & set(test_df[cold_column])

        assert len(train_val_overlap) == 0
        assert len(train_test_overlap) == 0
        assert len(val_test_overlap) == 0

        logger.info(
            f"(|train|, |val|, |test|) -> "
            f"({train_df.shape[0]}, {val_df.shape[0]}, {test_df.shape[0]})"
        )
        if cold_type == "cold_rna":
            logger.info(
                f"(unique RNA train, val, test) -> "
                f"({train_df['Target_RNA_sequence'].nunique()}, "
                f"{val_df['Target_RNA_sequence'].nunique()}, "
                f"{test_df['Target_RNA_sequence'].nunique()})"
            )
        elif cold_type == "cold_mole":
            logger.info(
                f"(unique molecule names train, val, test) -> "
                f"({train_df['Molecule_name'].nunique()}, "
                f"{val_df['Molecule_name'].nunique()}, "
                f"{test_df['Molecule_name'].nunique()})"
            )
        else:
            logger.info(
                f"(unique RNA train, val, test) -> "
                f"({train_df['Target_RNA_sequence'].nunique()}, "
                f"{val_df['Target_RNA_sequence'].nunique()}, "
                f"{test_df['Target_RNA_sequence'].nunique()})"
            )
            logger.info(
                f"(unique molecule names train, val, test) -> "
                f"({train_df['Molecule_name'].nunique()}, "
                f"{val_df['Molecule_name'].nunique()}, "
                f"{test_df['Molecule_name'].nunique()})"
            )

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
            task="evaluating",
            is_seq_emb=is_seq_emb,
        )
        test_ds = MolRNADataset(
            test_df,
            setting=val_setting,
            rna_aug=False,
            mol_aug=False,
            task="evaluating",
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
        test_loader = DataLoader(
            test_ds,
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
        train_losses, val_losses = [], []

        for epo in range(MAX_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            train_losses.append(train_loss)

            y_true_val, y_pred_val = predict(model, val_loader, device)
            p_val, s_val, rmse_val, r2_val = safe_regression_metrics(
                y_true_val, y_pred_val
            )
            val_losses.append(rmse_val)
            es_metric = rmse_val if MODE == "min" else p_val
            early_stopping.check_early_stop(
                model,
                es_metric,
                [p_val, s_val, r2_val, rmse_val],
                train_loss,
                epo,
                [y_true_val, y_pred_val],
            )

            if early_stopping.stop_training:
                logger.info(
                    f"[Fold {i}/5] Early stopping at epoch {epo + 1}. "
                    f"Best val score (rmse/ppc)={early_stopping.best_score:.4f} "
                    f"@ epoch {early_stopping.best_epoch + 1}"
                )
                break

        model.load_state_dict(early_stopping.best_model)

        if save_model:
            model_dir = os.path.join(
                "saved_objects", "cold", cold_name, model_setting
            )
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"model_fold{i}.pt")
            torch.save(early_stopping.best_model, model_path)
            logger.info(
                f"[Fold {i}/5] Saved best validation model to: {model_path}"
            )

        y_true_test, y_pred_test = predict(model, test_loader, device)
        p_test, s_test, rmse_test, r2_test = safe_regression_metrics(
            y_true_test, y_pred_test
        )

        _save_test_predictions(
            test_df=test_df,
            y_true=y_true_test,
            y_pred=y_pred_test,
            cold_type=cold_type,
            model_setting=model_setting,
            model_seed=seed,
            fold=i,
        )

        logger.info(
            f"[Fold {i}/5] TEST using best validation model: "
            f"p={p_test:.4f}, s={s_test:.4f}, "
            f"rmse={rmse_test:.4f}, r2={r2_test:.4f}"
        )

        p_list.append(p_test)
        s_list.append(s_test)
        rmse_list.append(rmse_test)
        r2_list.append(r2_test)

        if is_plot:
            date, plot = plot_name.split("_")
            plots_dir = os.path.join("results", "plots", date, plot)
            os.makedirs(plots_dir, exist_ok=True)
            plot_train_val_loss(
                train_losses,
                val_losses,
                save_path=f"{plots_dir}/losses_{plot_name}_fold_{i}.png",
                title="Loss Curve (train vs. val RMSE)",
            )
            plot_prediction_correlation(
                y_true_test,
                y_pred_test,
                save_path=(
                    f"{plots_dir}/corr_best_model_{plot_name}_fold_{i}.png"
                ),
                title="Prediction Correlation - Best Validation Model",
            )

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
        },
    }

    logger.info("\nCold-start blind test results:")
    logger.info(res)
    return res
