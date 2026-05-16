import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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
from sklearn.model_selection import RepeatedStratifiedKFold


# =========================================================
# paths / settings
# =========================================================
RNA_EMB_PATH = "/home/gavrilev/Thesis/Transfer Learning: PT Models/RiNALMo/rinalmo_embeddings_cls.pkl"
MOL_EMB_PATH = "/home/gavrilev/Thesis/Transfer Learning: PT Models/Mole-BERT/rsim_molebert_embeddings.pkl"
ALL_DATA_PATH = os.path.join("data", "RSM_data", "All_sf_dataset_v1.csv")

OUTPUT_DIR = "results/biollmnet_stratified_cv_rinalmo_molebert300_scaled"
SAVE_DIR = os.path.join(OUTPUT_DIR, "save")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")

RNA_COL = "Target_RNA_sequence"
MOL_COL = "SMILES"
LABEL_COL = "pKd"
ENTRY_COL = "Entry_ID"


# =========================================================
# hyperparameters
# =========================================================
N_SPLITS = 5 # for stritified 5-fold CV
DATASET_SEED = 2 # same across all SOTA and MultiStructRSA
SEED = 1 # same across all SOTA and MultiStructRSA

BATCH_SIZE = 16
MAX_EPOCHS = 1000
LR = 1e-3
WEIGHT_DECAY = 1e-7
NUM_WORKERS = 0

PATIENCE = 20
WARM_UP_EPOCHS = 50
DELTA = 1e-6
MODE = "min"

HIDDEN_DIM = 256
CUDA_VISIBLE_DEVICES = "0"

# BioLLMNet-style label preprocessing.
CLIP_LABEL_MAX = 10.0
MAX_TARGET_VALUE = 10.0

# Regression-stratified CV settings, matching your cv_eval logic.
GROUP_COUNT = 5
STRATEGY = "uniform"


# =========================================================
# determinism / setup
# =========================================================
def setup_device():
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def enable_determinism():
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
    log_path = os.path.join(LOG_DIR, f"BioLLMNet_stratified_cv_seed{SEED}_dataset_seed{DATASET_SEED}.log")

    logger = logging.getLogger("BioLLMNetStratifiedCV")
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
    - shape (seq_len, dim): takes first token, matching the BioLLMNet/RiNALMo CLS-style use.
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
    data_path = ALL_DATA_PATH

    # sep=None allows both comma-separated and tab-separated files.
    df = pd.read_csv(data_path, sep=None, engine="python")
    return df, data_path


def parse_label(label_value):
    if pd.isna(label_value):
        return None

    # BioLLMNet notebook skipped string labels that start with '-'.
    if isinstance(label_value, str) and label_value.strip().startswith("-"):
        return None

    try:
        label = float(label_value)
    except Exception:
        return None

    if CLIP_LABEL_MAX is not None and label > CLIP_LABEL_MAX:
        label = CLIP_LABEL_MAX

    return label


def clean_dataframe(df):
    required_cols = [RNA_COL, MOL_COL, LABEL_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}. Available columns: {list(df.columns)}")

    clean_rows = []

    for _, row in df.iterrows():
        rna_key = row[RNA_COL]
        molecule_key = row[MOL_COL]
        label = parse_label(row[LABEL_COL])

        if pd.isna(rna_key) or pd.isna(molecule_key) or label is None:
            continue

        row_dict = row.to_dict()
        row_dict[RNA_COL] = str(rna_key).strip()
        row_dict[MOL_COL] = str(molecule_key).strip()
        row_dict[LABEL_COL] = label
        clean_rows.append(row_dict)

    clean_df = pd.DataFrame(clean_rows)
    if clean_df.empty:
        raise ValueError("No usable rows after cleaning labels and keys.")

    return clean_df.reset_index(drop=True)


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
        label = parse_label(row[LABEL_COL])

        if pd.isna(rna_key) or pd.isna(molecule_key) or label is None:
            skipped_bad_rows += 1
            continue

        rna_key = str(rna_key).strip()
        molecule_key = str(molecule_key).strip()

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
# Regression-stratified CV
# =========================================================
def make_regression_strata(y, group_count=5, strategy="uniform"):
    y = np.asarray(y, dtype=np.float64)

    if strategy == "uniform":
        bins = np.linspace(np.min(y), np.max(y), group_count + 1)
        bins[0] -= 1e-8
        bins[-1] += 1e-8
        strata = np.digitize(y, bins[1:-1], right=False)

    elif strategy == "quantile":
        strata = pd.qcut(y, q=group_count, labels=False, duplicates="drop")
        strata = np.asarray(strata, dtype=int)

    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    return strata.astype(int)


def regression_stratified_cv_indices(y, n_splits=10, random_state=2, group_count=5, strategy="uniform"):
    strata = make_regression_strata(y, group_count=group_count, strategy=strategy)

    unique, counts = np.unique(strata, return_counts=True)
    min_count = int(np.min(counts))

    if min_count < n_splits:
        # Fallback to quantile bins with fewer groups if uniform bins are too sparse.
        safe_group_count = max(2, min(group_count, min_count))
        strata = make_regression_strata(y, group_count=safe_group_count, strategy="quantile")

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=1,
        random_state=random_state,
    )

    return splitter.split(np.zeros(len(y)), strata)


# =========================================================
# metrics / train / evaluate
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


def train_one_epoch(model, loader, optimizer, loss_fct, device, max_target_value):
    model.train()
    total_loss = 0.0

    for embeddings, targets in loader:
        molecule_emb, rna_emb = embeddings
        molecule_emb = molecule_emb.to(device)
        rna_emb = rna_emb.to(device)
        targets = targets.to(device)

        targets_scaled = targets / max_target_value

        optimizer.zero_grad()
        predictions_scaled = model(molecule_emb, rna_emb)
        loss = loss_fct(predictions_scaled, targets_scaled)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


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


# =========================================================
# BioLLMNet stratified k-fold CV
# =========================================================
def cv_eval_biollmnet(dataset=None):
    device = setup_device()
    enable_determinism()
    set_seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    logger, log_path = setup_logger()

    logger.info("=" * 80)
    logger.info("BioLLMNet adapted to stratified k-fold CV")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"RNA embeddings: {RNA_EMB_PATH}")
    logger.info(f"Molecule embeddings: {MOL_EMB_PATH}")
    logger.info(f"N_SPLITS={N_SPLITS}, SEED={SEED}, DATASET_SEED={DATASET_SEED}")
    logger.info(f"BATCH_SIZE={BATCH_SIZE}, MAX_EPOCHS={MAX_EPOCHS}, LR={LR}, WEIGHT_DECAY={WEIGHT_DECAY}")
    logger.info(f"PATIENCE={PATIENCE}, WARM_UP_EPOCHS={WARM_UP_EPOCHS}, DELTA={DELTA}")
    logger.info(f"CLIP_LABEL_MAX={CLIP_LABEL_MAX}, MAX_TARGET_VALUE={MAX_TARGET_VALUE}")
    logger.info(f"Regression strata: group_count={GROUP_COUNT}, strategy={STRATEGY}")

    logger.info("Loading embeddings...")
    rna_embeddings = load_pickle(RNA_EMB_PATH)
    molecule_embeddings = load_pickle(MOL_EMB_PATH)

    rna_dim = infer_embedding_dim(rna_embeddings)
    molecule_dim = infer_embedding_dim(molecule_embeddings)

    logger.info(f"Number of RNA embeddings: {len(rna_embeddings)}")
    logger.info(f"Number of molecule embeddings: {len(molecule_embeddings)}")
    logger.info(f"RNA embedding dim: {rna_dim}")
    logger.info(f"Molecule embedding dim: {molecule_dim}")

    if dataset is None:
        all_df, data_path = read_all_data()
        logger.info(f"Loaded all data: {data_path}, shape={all_df.shape}")
    else:
        all_df = dataset.copy()
        logger.info(f"Loaded dataset from function argument, shape={all_df.shape}")

    all_df = clean_dataframe(all_df)
    logger.info(f"Cleaned all data shape: {all_df.shape}")

    y = all_df[LABEL_COL].to_numpy(dtype=np.float32)
    cv_iterator = regression_stratified_cv_indices(
        y,
        n_splits=N_SPLITS,
        random_state=DATASET_SEED,
        group_count=GROUP_COUNT,
        strategy=STRATEGY,
    )

    fold_results = []

    for fold, (train_index, val_index) in enumerate(cv_iterator, start=1):
        set_seed(SEED)

        logger.info("=" * 80)
        logger.info(f"Fold {fold}/{N_SPLITS}")
        logger.info("=" * 80)

        train_df = all_df.iloc[train_index].reset_index(drop=True)
        val_df = all_df.iloc[val_index].reset_index(drop=True)

        train_dataset, train_rows = make_dataset_from_df(
            train_df, rna_embeddings, molecule_embeddings, logger, f"fold{fold}_train"
        )
        val_dataset, val_rows = make_dataset_from_df(
            val_df, rna_embeddings, molecule_embeddings, logger, f"fold{fold}_val"
        )

        logger.info(f"Train len -> {len(train_dataset)}, Val/Test len -> {len(val_dataset)}")

        generator = torch.Generator().manual_seed(SEED)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
            drop_last=False,
            num_workers=NUM_WORKERS,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=NUM_WORKERS,
        )

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
        best_metrics = None
        best_y_true = None
        best_y_pred = None
        patience_counter = 0
        history_rows = []

        for epoch in range(MAX_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fct, device, MAX_TARGET_VALUE)

            val_loss, p_val, s_val, rmse_val, mae_val, r2_val, y_true_val, y_pred_val = evaluate(
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
                best_metrics = [p_val, s_val, rmse_val, mae_val, r2_val]
                best_y_true = y_true_val
                best_y_pred = y_pred_val
                patience_counter = 0

                save_path = os.path.join(SAVE_DIR, f"biollmnet_cv_fold{fold}_seed{SEED}.pth")
                torch.save(best_model_state, save_path)

                logger.info(f"New best model saved: epoch={epoch + 1}, val_rmse={best_val_rmse:.4f}, path={save_path}")

            elif epoch + 1 >= WARM_UP_EPOCHS:
                patience_counter += 1

            if epoch + 1 >= WARM_UP_EPOCHS and patience_counter >= PATIENCE:
                logger.info(
                    f"[Fold {fold}/{N_SPLITS}] Early stopping at epoch {epoch + 1}. "
                    f"Best rmse={best_val_rmse:.4f} @ epoch {best_epoch + 1}"
                )
                break

        if best_model_state is None:
            raise RuntimeError(f"No best model was saved for fold {fold}.")

        history_path = os.path.join(OUTPUT_DIR, f"fold{fold}_training_history.csv")
        pd.DataFrame(history_rows).to_csv(history_path, index=False)

        # Reload best model and re-evaluate held-out fold, same metrics as stored by early stopping.
        model.load_state_dict(best_model_state)
        val_loss, p_val, s_val, rmse_val, mae_val, r2_val, y_true_val, y_pred_val = evaluate(
            model, val_loader, loss_fct, device, MAX_TARGET_VALUE
        )

        pred_df = val_rows.copy()
        pred_df["y_true"] = y_true_val
        pred_df["y_pred"] = y_pred_val
        pred_df["max_target_value"] = MAX_TARGET_VALUE
        pred_path = os.path.join(PRED_DIR, f"fold{fold}_val_predictions.csv")
        pred_df.to_csv(pred_path, index=False)

        fold_result = {
            "fold": fold,
            "best_epoch": best_epoch + 1,
            "best_val_rmse": best_val_rmse,
            "val_scaled_mse": val_loss,
            "p": p_val,
            "s": s_val,
            "rmse": rmse_val,
            "mae": mae_val,
            "r2": r2_val,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "prediction_path": pred_path,
        }
        fold_results.append(fold_result)

        fold_results_path = os.path.join(OUTPUT_DIR, "fold_results.csv")
        pd.DataFrame(fold_results).to_csv(fold_results_path, index=False)

        logger.info(
            f"[Fold {fold}/{N_SPLITS}] Best held-out metrics: "
            f"p={p_val:.4f}, s={s_val:.4f}, rmse={rmse_val:.4f}, mae={mae_val:.4f}, r2={r2_val:.4f}"
        )
        logger.info(f"Current mean p: {np.nanmean([x['p'] for x in fold_results])}")
        logger.info(f"Current mean s: {np.nanmean([x['s'] for x in fold_results])}")
        logger.info(f"Current mean rmse: {np.nanmean([x['rmse'] for x in fold_results])}")
        logger.info(f"Current mean mae: {np.nanmean([x['mae'] for x in fold_results])}")
        logger.info(f"Current mean r2: {np.nanmean([x['r2'] for x in fold_results])}")

    results_df = pd.DataFrame(fold_results)
    results_path = os.path.join(OUTPUT_DIR, "fold_results.csv")
    results_df.to_csv(results_path, index=False)

    summary = {
        "p_mean": round(float(np.nanmean(results_df["p"])), 5),
        "p_std": round(float(np.nanstd(results_df["p"])), 5),
        "s_mean": round(float(np.nanmean(results_df["s"])), 5),
        "s_std": round(float(np.nanstd(results_df["s"])), 5),
        "rmse_mean": round(float(np.nanmean(results_df["rmse"])), 5),
        "rmse_std": round(float(np.nanstd(results_df["rmse"])), 5),
        "mae_mean": round(float(np.nanmean(results_df["mae"])), 5),
        "mae_std": round(float(np.nanstd(results_df["mae"])), 5),
        "r2_mean": round(float(np.nanmean(results_df["r2"])), 5),
        "r2_std": round(float(np.nanstd(results_df["r2"])), 5),
        "per_fold": {
            "p": results_df["p"].tolist(),
            "s": results_df["s"].tolist(),
            "rmse": results_df["rmse"].tolist(),
            "mae": results_df["mae"].tolist(),
            "r2": results_df["r2"].tolist(),
        },
    }

    summary_path = os.path.join(OUTPUT_DIR, "summary_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    summary_csv_path = os.path.join(OUTPUT_DIR, "summary_mean_std.csv")
    pd.DataFrame([summary]).drop(columns=["per_fold"]).to_csv(summary_csv_path, index=False)

    logger.info("=" * 80)
    logger.info("Final stratified k-fold cross-validation results:")
    logger.info(f"p mean: {summary['p_mean']}, std: {summary['p_std']}")
    logger.info(f"s mean: {summary['s_mean']}, std: {summary['s_std']}")
    logger.info(f"rmse mean: {summary['rmse_mean']}, std: {summary['rmse_std']}")
    logger.info(f"mae mean: {summary['mae_mean']}, std: {summary['mae_std']}")
    logger.info(f"r2 mean: {summary['r2_mean']}, std: {summary['r2_std']}")
    logger.info(f"Saved fold results to: {results_path}")
    logger.info(f"Saved summary json to: {summary_path}")
    logger.info("=" * 80)

    return summary


def main():
    cv_eval_biollmnet(dataset=None)


if __name__ == "__main__":
    main()
