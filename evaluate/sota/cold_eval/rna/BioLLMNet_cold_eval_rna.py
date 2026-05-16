import os
import copy
import json
import random
import logging
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =========================================================
# paths / settings
# =========================================================
RNA_EMB_PATH = "/home/gavrilev/Thesis/Transfer Learning: PT Models/RiNALMo/rinalmo_embeddings_cls.pkl"
MOL_EMB_PATH = "/home/gavrilev/Thesis/Transfer Learning: PT Models/Mole-BERT/rsim_molebert_embeddings.pkl"

RNA_TYPE = "All_sf"
ALL_DATA_PATH = f"data/RSM_data/{RNA_TYPE}_dataset_v1.csv"
RNA_COLD_PATH = os.path.join("datasets", "blind_test", "rna", "train_val_test_splits")

OUTPUT_DIR = "results/biollmnet_cold_rna_rinalmo_molebert300_scaled"
SAVE_DIR = os.path.join(OUTPUT_DIR, "save")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")

RNA_COL = "Target_RNA_sequence"
MOL_COL = "SMILES"
LABEL_COL = "pKd"
ENTRY_COL = "Entry_ID"

TRAIN_PATTERN = "cold_rna_train_fold{fold}.csv"
VAL_PATTERN = "cold_rna_val_fold{fold}.csv"
TEST_PATTERN = "cold_rna_test_fold{fold}.csv"


# =========================================================
# hyperparameters
# =========================================================
BATCH_SIZE = 16               # BioLLMNet batch_size=16
MAX_EPOCHS = 1500             # same maximum training budget style as we used in DeepRSMA cold script
LR = 1e-3                     # BioLLMNet Adam lr=0.001
WEIGHT_DECAY = 1e-7

# same early-stopping style as MultiStructRSA cold script
PATIENCE = 15                 
WARM_UP_EPOCHS = 10
DELTA = 0.01

HIDDEN_DIM = 256              # BioLLMNet hidden_dim=256
SEED = 2
NUM_FOLDS = 5
CUDA_VISIBLE_DEVICES = "0"

# BioLLMNet-style label preprocessing:
# 1. clip labels above 10 (pkd > 10 -> 10)
# 2. train on labels scaled to [0, 1] by dividing by MAX_TARGET_VALUE
CLIP_LABEL_MAX = 10.0
MAX_TARGET_VALUE = 10.0       # we fixed to 10 to avoid test leakage and still matche BioLLMNet clipped affinity scale


# =========================================================
# setup
# =========================================================
def setup_device():
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=20)


def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"BioLLMNet_cold_rna_seed{SEED}.log")

    logger = logging.getLogger("BioLLMNetColdRNA")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger, log_path


# =========================================================
# loading / formatting embeddings
# =========================================================
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def to_1d_numpy(emb):
    """
    Convert embedding into one vector.

    Handles:
    - torch.Tensor
    - numpy array
    - list
    - shape (dim,)
    - shape (1, dim)
    - shape (seq_len, dim): takes first token, matching the BioLLMNet notebook behavior
      where RNA embeddings are indexed with rna_emb[0].
    """
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()

    arr = np.asarray(emb, dtype=np.float32)

    if arr.ndim == 1:
        return arr.astype(np.float32)

    if arr.ndim == 2:
        if arr.shape[0] == 1:
            return arr.squeeze(0).astype(np.float32)
        return arr[0].astype(np.float32)

    return arr.reshape(-1).astype(np.float32)


def infer_embedding_dim(emb_dict):
    if len(emb_dict) == 0:
        raise ValueError("Embedding dictionary is empty.")

    first_key = next(iter(emb_dict.keys()))
    return to_1d_numpy(emb_dict[first_key]).shape[0]


# =========================================================
# BioLLMNet dataset object
# =========================================================
class InteractionDataset(Dataset):
    def __init__(self, interaction_data, interaction_labels):
        self.interaction_data = interaction_data
        self.interaction_labels = interaction_labels

    def __len__(self):
        return len(self.interaction_labels)

    def __getitem__(self, idx):
        molecule_emb, rna_emb = self.interaction_data[idx]
        label = self.interaction_labels[idx]

        molecule_emb = torch.tensor(molecule_emb, dtype=torch.float32)
        rna_emb = torch.tensor(rna_emb, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return (molecule_emb, rna_emb), label


# =========================================================
# BioLLMNet model
# =========================================================
class GatedFeatureFusion(nn.Module):
    """Gates to control the contribution of each path."""
    def __init__(self, input_dim):
        super().__init__()
        self.gate = nn.Parameter(torch.rand(input_dim))

    def forward(self, x1, x2):
        gate_values = torch.sigmoid(self.gate)
        return x1 * gate_values + x2 * (1 - gate_values)


class DualPathNetworkRegression(nn.Module):
    def __init__(self, molecule_dim, rna_dim, hidden_dim):
        super().__init__()

        self.protein_path = nn.Sequential(
            nn.Linear(molecule_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.rna_path = nn.Sequential(
            nn.Linear(rna_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.fusion = GatedFeatureFusion(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, molecule_emb, rna_emb):
        molecule_features = self.protein_path(molecule_emb)
        rna_features = self.rna_path(rna_emb)
        combined_features = self.fusion(molecule_features, rna_features)
        return self.classifier(combined_features).squeeze(1)


# =========================================================
# data handling
# =========================================================
def read_all_data():
    # DeepRSMA uses delimiter="\t" for All_sf_dataset_v1.csv.
    return pd.read_csv(ALL_DATA_PATH, delimiter="\t")


def load_split_dataframe(split_path, all_df):
    split_df = pd.read_csv(split_path)

    needed_cols = {RNA_COL, MOL_COL, LABEL_COL}

    if needed_cols.issubset(set(split_df.columns)):
        return split_df.copy()

    if ENTRY_COL in split_df.columns and ENTRY_COL in all_df.columns:
        out_df = split_df[[ENTRY_COL]].merge(all_df, on=ENTRY_COL, how="left")

        if RNA_COL not in out_df.columns or MOL_COL not in out_df.columns or LABEL_COL not in out_df.columns:
            raise ValueError(f"Could not recover required columns from {split_path} using {ENTRY_COL}.")

        missing_rows = out_df[RNA_COL].isna().sum() + out_df[MOL_COL].isna().sum() + out_df[LABEL_COL].isna().sum()
        if missing_rows > 0:
            raise ValueError(f"Some rows from {split_path} were not matched to all_df using {ENTRY_COL}.")

        return out_df

    raise ValueError(
        f"Split file {split_path} must contain either {needed_cols} or {ENTRY_COL}."
    )


def parse_label(label_value):
    if pd.isna(label_value):
        return None

    # BioLLMNet skipped string labels that start with '-'.
    if isinstance(label_value, str) and label_value.strip().startswith("-"):
        return None

    try:
        label = float(label_value)
    except Exception:
        return None

    if CLIP_LABEL_MAX is not None and label > CLIP_LABEL_MAX:
        label = CLIP_LABEL_MAX

    return label


def make_dataset_from_df(df, rna_embeddings, molecule_embeddings, logger, split_name):
    interaction_data = []
    labels = []
    rows_for_predictions = []

    missing_rna = []
    missing_molecule = []
    skipped_bad_rows = 0

    for _, row in df.iterrows():
        rna_key = row[RNA_COL]
        molecule_key = row[MOL_COL]
        label_value = row[LABEL_COL]

        if pd.isna(rna_key) or pd.isna(molecule_key):
            skipped_bad_rows += 1
            continue

        rna_key = str(rna_key).strip()
        molecule_key = str(molecule_key).strip()

        label = parse_label(label_value)
        if label is None:
            skipped_bad_rows += 1
            continue

        if rna_key not in rna_embeddings:
            missing_rna.append(rna_key)
            continue

        if molecule_key not in molecule_embeddings:
            missing_molecule.append(molecule_key)
            continue

        rna_emb = to_1d_numpy(rna_embeddings[rna_key])
        molecule_emb = to_1d_numpy(molecule_embeddings[molecule_key])

        interaction_data.append((molecule_emb, rna_emb))
        labels.append(label)

        row_dict = row.to_dict()
        row_dict[RNA_COL] = rna_key
        row_dict[MOL_COL] = molecule_key
        row_dict[LABEL_COL] = label
        rows_for_predictions.append(row_dict)

    if missing_rna:
        examples = sorted(set(missing_rna))[:5]
        logger.warning(f"[{split_name}] Missing RNA embeddings: {len(missing_rna)} rows. Examples: {examples}")

    if missing_molecule:
        examples = sorted(set(missing_molecule))[:5]
        logger.warning(f"[{split_name}] Missing molecule embeddings: {len(missing_molecule)} rows. Examples: {examples}")

    if skipped_bad_rows > 0:
        logger.warning(f"[{split_name}] Skipped bad rows: {skipped_bad_rows}")

    if len(labels) == 0:
        raise ValueError(f"No usable rows were found for {split_name}.")

    dataset = InteractionDataset(interaction_data, labels)
    rows_df = pd.DataFrame(rows_for_predictions)

    return dataset, rows_df


# =========================================================
# metrics / evaluation
# =========================================================
def safe_corr(func, y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    try:
        return float(func(y_true, y_pred)[0])
    except Exception:
        return np.nan


def safe_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    p = safe_corr(pearsonr, y_true, y_pred)
    s = safe_corr(spearmanr, y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan

    return p, s, rmse, mae, r2


def evaluate(model, loader, loss_fct, device, max_target_value):
    model.eval()

    y_true = []
    y_pred = []
    total_loss = 0.0

    with torch.no_grad():
        for embeddings, targets in loader:
            molecule_emb, rna_emb = embeddings
            molecule_emb = molecule_emb.to(device)
            rna_emb = rna_emb.to(device)
            targets = targets.to(device)

            targets_scaled = targets / max_target_value
            predictions_scaled = model(molecule_emb, rna_emb)
            loss = loss_fct(predictions_scaled, targets_scaled)
            total_loss += loss.item()

            predictions = predictions_scaled * max_target_value

            y_true.extend(targets.detach().cpu().numpy().flatten().tolist())
            y_pred.extend(predictions.detach().cpu().numpy().flatten().tolist())

    p, s, rmse, mae, r2 = safe_metrics(y_true, y_pred)
    avg_loss = total_loss / max(len(loader), 1)

    return avg_loss, p, s, rmse, mae, r2, y_true, y_pred



def main():
    device = setup_device()
    set_seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    logger, log_path = setup_logger()

    logger.info("=" * 80)
    logger.info("BioLLMNet adapted to RNA cold-start evaluation")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"RNA embeddings: {RNA_EMB_PATH}")
    logger.info(f"Molecule embeddings: {MOL_EMB_PATH}")
    logger.info(f"Cold split path: {RNA_COLD_PATH}")
    logger.info(f"All data path: {ALL_DATA_PATH}")
    logger.info(f"BATCH_SIZE={BATCH_SIZE}, MAX_EPOCHS={MAX_EPOCHS}, LR={LR}, WEIGHT_DECAY={WEIGHT_DECAY}")
    logger.info(f"PATIENCE={PATIENCE}, WARM_UP_EPOCHS={WARM_UP_EPOCHS}, DELTA={DELTA}")
    logger.info(f"CLIP_LABEL_MAX={CLIP_LABEL_MAX}, MAX_TARGET_VALUE={MAX_TARGET_VALUE}")

    logger.info("Loading embeddings...")
    rna_embeddings = load_pickle(RNA_EMB_PATH)
    molecule_embeddings = load_pickle(MOL_EMB_PATH)

    rna_dim = infer_embedding_dim(rna_embeddings)
    molecule_dim = infer_embedding_dim(molecule_embeddings)

    logger.info(f"Number of RNA embeddings: {len(rna_embeddings)}")
    logger.info(f"Number of molecule embeddings: {len(molecule_embeddings)}")
    logger.info(f"RNA embedding dim: {rna_dim}")
    logger.info(f"Molecule embedding dim: {molecule_dim}")

    all_df = read_all_data()
    logger.info(f"Loaded all data: {ALL_DATA_PATH}, shape={all_df.shape}")

    fold_results = []

    for fold in range(1, NUM_FOLDS + 1):
        set_seed(SEED)

        logger.info("=" * 80)
        logger.info(f"Fold {fold}/{NUM_FOLDS}")
        logger.info("=" * 80)

        train_path = os.path.join(RNA_COLD_PATH, TRAIN_PATTERN.format(fold=fold))
        val_path = os.path.join(RNA_COLD_PATH, VAL_PATTERN.format(fold=fold))
        test_path = os.path.join(RNA_COLD_PATH, TEST_PATTERN.format(fold=fold))

        train_df = load_split_dataframe(train_path, all_df)
        val_df = load_split_dataframe(val_path, all_df)
        test_df = load_split_dataframe(test_path, all_df)

        train_dataset, train_rows = make_dataset_from_df(train_df, rna_embeddings, molecule_embeddings, logger, f"fold{fold}_train")
        val_dataset, val_rows = make_dataset_from_df(val_df, rna_embeddings, molecule_embeddings, logger, f"fold{fold}_val")
        test_dataset, test_rows = make_dataset_from_df(test_df, rna_embeddings, molecule_embeddings, logger, f"fold{fold}_test")

        logger.info(f"train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)

        model = DualPathNetworkRegression(
            molecule_dim=molecule_dim,
            rna_dim=rna_dim,
            hidden_dim=HIDDEN_DIM,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fct = nn.MSELoss()

        best_val_rmse = float("inf")
        best_model_state = None
        best_epoch = -1
        patience_counter = 0
        history_rows = []

        for epoch in range(MAX_EPOCHS):
            model.train()
            train_loss = 0.0

            for embeddings, targets in train_loader:
                molecule_emb, rna_emb = embeddings
                molecule_emb = molecule_emb.to(device)
                rna_emb = rna_emb.to(device)
                targets = targets.to(device)

                targets_scaled = targets / MAX_TARGET_VALUE

                optimizer.zero_grad()
                predictions_scaled = model(molecule_emb, rna_emb)
                loss = loss_fct(predictions_scaled, targets_scaled)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss = train_loss / max(len(train_loader), 1)

            val_loss, p_val, s_val, rmse_val, mae_val, r2_val, _, _ = evaluate(
                model, val_loader, loss_fct, device, MAX_TARGET_VALUE
            )

            history_rows.append({
                "epoch": epoch + 1,
                "train_scaled_mse_loss": train_loss,
                "val_scaled_mse_loss": val_loss,
                "val_p": p_val,
                "val_s": s_val,
                "val_rmse": rmse_val,
                "val_mae": mae_val,
                "val_r2": r2_val,
            })

            logger.info(
                f"epoch: {epoch + 1} | train_scaled_mse: {train_loss:.6f} | "
                f"val_scaled_mse: {val_loss:.6f} | val_rmse: {rmse_val:.4f} | "
                f"val_p: {p_val:.4f} | val_s: {s_val:.4f} | val_r2: {r2_val:.4f}"
            )

            improved = rmse_val < best_val_rmse - DELTA

            if improved:
                best_val_rmse = rmse_val
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_counter = 0

                save_path = os.path.join(SAVE_DIR, f"biollmnet_cold_rna_fold{fold}_seed{SEED}.pth")
                torch.save(best_model_state, save_path)

                logger.info(f"New best model saved: epoch={epoch + 1}, val_rmse={best_val_rmse:.4f}, path={save_path}")

            elif epoch + 1 >= WARM_UP_EPOCHS:
                patience_counter += 1

            if epoch + 1 >= WARM_UP_EPOCHS and patience_counter >= PATIENCE:
                logger.info(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"Best epoch: {best_epoch + 1}, best val RMSE: {best_val_rmse:.4f}"
                )
                break

        history_path = os.path.join(OUTPUT_DIR, f"fold{fold}_training_history.csv")
        pd.DataFrame(history_rows).to_csv(history_path, index=False)

        if best_model_state is None:
            raise RuntimeError(f"No best model was saved for fold {fold}.")

        model.load_state_dict(best_model_state)

        test_loss, p_test, s_test, rmse_test, mae_test, r2_test, y_true_test, y_pred_test = evaluate(
            model, test_loader, loss_fct, device, MAX_TARGET_VALUE
        )

        logger.info(
            f"[Fold {fold}/{NUM_FOLDS}] TEST using best validation model: "
            f"test_scaled_mse={test_loss:.6f}, p={p_test:.4f}, s={s_test:.4f}, "
            f"rmse={rmse_test:.4f}, mae={mae_test:.4f}, r2={r2_test:.4f}"
        )

        pred_df = test_rows.copy()
        pred_df["y_true"] = y_true_test
        pred_df["y_pred"] = y_pred_test
        pred_df["max_target_value"] = MAX_TARGET_VALUE
        pred_path = os.path.join(PRED_DIR, f"fold{fold}_test_predictions.csv")
        pred_df.to_csv(pred_path, index=False)

        fold_result = {
            "fold": fold,
            "best_epoch": best_epoch + 1,
            "best_val_rmse": best_val_rmse,
            "test_scaled_mse": test_loss,
            "p": p_test,
            "s": s_test,
            "rmse": rmse_test,
            "mae": mae_test,
            "r2": r2_test,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "prediction_path": pred_path,
        }
        fold_results.append(fold_result)

        fold_results_path = os.path.join(OUTPUT_DIR, "fold_results.csv")
        pd.DataFrame(fold_results).to_csv(fold_results_path, index=False)

        logger.info(f"Current mean p: {np.nanmean([x['p'] for x in fold_results])}")
        logger.info(f"Current mean s: {np.nanmean([x['s'] for x in fold_results])}")
        logger.info(f"Current mean rmse: {np.nanmean([x['rmse'] for x in fold_results])}")
        logger.info(f"Current mean mae: {np.nanmean([x['mae'] for x in fold_results])}")
        logger.info(f"Current mean r2: {np.nanmean([x['r2'] for x in fold_results])}")

    results_df = pd.DataFrame(fold_results)
    results_path = os.path.join(OUTPUT_DIR, "fold_results.csv")
    results_df.to_csv(results_path, index=False)

    summary = {
        "p_mean": float(np.nanmean(results_df["p"])),
        "p_std": float(np.nanstd(results_df["p"])),
        "s_mean": float(np.nanmean(results_df["s"])),
        "s_std": float(np.nanstd(results_df["s"])),
        "rmse_mean": float(np.nanmean(results_df["rmse"])),
        "rmse_std": float(np.nanstd(results_df["rmse"])),
        "mae_mean": float(np.nanmean(results_df["mae"])),
        "mae_std": float(np.nanstd(results_df["mae"])),
        "r2_mean": float(np.nanmean(results_df["r2"])),
        "r2_std": float(np.nanstd(results_df["r2"])),
        "per_fold": fold_results,
    }

    summary_path = os.path.join(OUTPUT_DIR, "summary_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    summary_csv_path = os.path.join(OUTPUT_DIR, "summary_mean_std.csv")
    pd.DataFrame([summary]).drop(columns=["per_fold"]).to_csv(summary_csv_path, index=False)

    logger.info("=" * 80)
    logger.info("Final results:")
    logger.info(f"p mean: {summary['p_mean']}, std: {summary['p_std']}")
    logger.info(f"s mean: {summary['s_mean']}, std: {summary['s_std']}")
    logger.info(f"rmse mean: {summary['rmse_mean']}, std: {summary['rmse_std']}")
    logger.info(f"mae mean: {summary['mae_mean']}, std: {summary['mae_std']}")
    logger.info(f"r2 mean: {summary['r2_mean']}, std: {summary['r2_std']}")
    logger.info(f"Saved fold results to: {results_path}")
    logger.info(f"Saved summary json to: {summary_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
