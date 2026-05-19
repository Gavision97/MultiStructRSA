import os
import sys
import pickle
import copy
import random
import logging
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score

try:
    from rna_dta_dataset import (
        smiles_to_pyg_graph,
        parse_st_file_enhanced,
        rna_struct_to_graph_fragment_enhanced,
        seq_to_local_eskmer_sequence,
        rna_dta_collate_fn,
        RNA_FM_EMBEDDING_DIR,
        RNA_FM_MAX_LEN,
        RNA_FM_PAD_VALUE,
        RNA_BASE_VOCAB,
        ST_FILE_DIR,
        LOCAL_ESKMER_K,
        LOCAL_ESKMER_WINDOW_SIZE,
        ESKMER_VOCAB_MAP,
    )
    from rna_dta_fm_cross_attention_model import RNADTA_FM_CrossAttentionModel
    from finetune_rna_dta_Regression_cv import ScriptArgs
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure this script is in the same folder as the DeepMIF source files.")
    sys.exit(1)


warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger("DeepMIFColdStart")


NUM_FOLDS = 5


RNA_FM_DICT_FILENAME = "rnafm_embeddings_dict.pkl"


def get_rnafm_dict_path():
    return os.path.join(RNA_FM_EMBEDDING_DIR, RNA_FM_DICT_FILENAME)


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logger(output_dir, cold_name, seed):
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"DeepMIF_cold_{cold_name}_seed{seed}.log")

    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    logger.info(f"Log file: {log_path}")
    return log_path


def choose_column(df, preferred_name, candidates, role):
    if preferred_name is not None and preferred_name in df.columns:
        return preferred_name

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find {role} column. Tried: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def resolve_columns(args, train_df):
    args.smiles_col = choose_column(
        train_df,
        getattr(args, "smiles_col", None),
        ["SMILES", "smiles", "canonical_smiles", "Molecule_SMILES"],
        "SMILES",
    )
    args.seq_col = choose_column(
        train_df,
        getattr(args, "seq_col", None),
        ["Target_RNA_sequence", "RNA_sequence", "Sequence", "sequence", "seq"],
        "RNA sequence",
    )
    args.affinity_col = choose_column(
        train_df,
        getattr(args, "affinity_col", None),
        ["pKd", "affinity", "label", "Y", "y", "Kd"],
        "affinity/label",
    )
    args.target_id_col = choose_column(
        train_df,
        getattr(args, "target_id_col", None),
        ["Entry_ID", "Target_ID", "Target_RNA_ID", "RNA_ID", "rna_id", "target_id"],
        "target RNA id",
    )

    logger.info(
        "Using columns: "
        f"smiles_col={args.smiles_col}, seq_col={args.seq_col}, "
        f"affinity_col={args.affinity_col}, target_id_col={args.target_id_col}"
    )


class LiveRNADataset(Dataset):
    _rna_fm_embeddings_by_seq_cache = None

    @classmethod
    def load_rna_fm_embeddings_by_seq(cls):
        """
        Load the RNA-FM dictionary once and reuse it across train/val/test datasets.

        Expected format:
            {
                original_RNA_sequence_as_in_dataset: np.ndarray with shape (L, 640)
            }

        The key is intentionally NOT normalized: no strip, no upper, no T->U.
        """
        if cls._rna_fm_embeddings_by_seq_cache is not None:
            return cls._rna_fm_embeddings_by_seq_cache

        dict_path = get_rnafm_dict_path()
        if not os.path.exists(dict_path):
            raise FileNotFoundError(
                f"Missing RNA-FM embeddings dictionary: {dict_path}. "
                f"Expected file name: {RNA_FM_DICT_FILENAME} inside RNA_FM_EMBEDDING_DIR={RNA_FM_EMBEDDING_DIR}"
            )

        with open(dict_path, "rb") as f:
            cls._rna_fm_embeddings_by_seq_cache = pickle.load(f)

        logger.info(
            f"Loaded RNA-FM embeddings dictionary from {dict_path} | "
            f"num_sequences={len(cls._rna_fm_embeddings_by_seq_cache)}"
        )
        return cls._rna_fm_embeddings_by_seq_cache

    def __init__(self, dataframe, args):
        self.args = args
        self.rna_fm_embeddings_by_seq = self.load_rna_fm_embeddings_by_seq()

        required_cols = [args.smiles_col, args.seq_col, args.affinity_col, args.target_id_col]

        df_clean = dataframe.dropna(subset=required_cols).copy()
        df_clean[args.affinity_col] = pd.to_numeric(df_clean[args.affinity_col], errors="coerce")
        self.df = df_clean.dropna(subset=[args.affinity_col]).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        smiles_str = row[self.args.smiles_col]
        affinity_val = row[self.args.affinity_col]
        target_rna_id_str = str(row[self.args.target_id_col])

        # Important: this key must stay exactly as it appears in the dataset.
        # Do not use strip(), upper(), replace("T", "U"), or any other normalization here.
        target_rna_seq_key = str(row[self.args.seq_col])

        drug_graph_obj = smiles_to_pyg_graph(smiles_str)
        if drug_graph_obj is None:
            return None

        seq_st, struct_db_st, modules, elements = parse_st_file_enhanced(target_rna_id_str, ST_FILE_DIR)
        if seq_st is None:
            return None

        raw_fm_emb = self.rna_fm_embeddings_by_seq.get(target_rna_seq_key)
        if raw_fm_emb is None:
            return None

        raw_fm_emb = np.asarray(raw_fm_emb, dtype=np.float32)
        if raw_fm_emb.ndim != 2 or raw_fm_emb.shape[0] == 0:
            return None

        # Align all RNA views to the same final length.
        # This is needed because long RNA-FM inputs were truncated during embedding extraction.
        len_final = min(len(seq_st), raw_fm_emb.shape[0], RNA_FM_MAX_LEN)
        if len_final <= 0:
            return None

        seq_st = seq_st[:len_final]
        struct_db_st = struct_db_st[:len_final]
        modules = modules[:len_final]

        fm_processed = raw_fm_emb[:len_final, :]
        padding_to_max = RNA_FM_MAX_LEN - len_final
        if padding_to_max < 0:
            return None

        fm_cnn_np = np.pad(
            fm_processed,
            ((0, padding_to_max), (0, 0)),
            "constant",
            constant_values=RNA_FM_PAD_VALUE,
        )
        fm_mask_np = np.concatenate([np.ones(len_final), np.zeros(padding_to_max)])

        node_feats_gnn = torch.from_numpy(fm_processed).float()
        rna_graph_obj = rna_struct_to_graph_fragment_enhanced(
            seq_st,
            struct_db_st,
            modules,
            elements,
            node_feats_gnn,
            RNA_BASE_VOCAB,
        )
        if rna_graph_obj is None:
            return None

        local_eskmer_feat_seq, _ = seq_to_local_eskmer_sequence(
            seq_st.upper().replace("T", "U"),
            LOCAL_ESKMER_K,
            LOCAL_ESKMER_WINDOW_SIZE,
            ESKMER_VOCAB_MAP,
            RNA_FM_MAX_LEN,
        )

        return {
            "drug_graph": drug_graph_obj,
            "original_smiles": smiles_str,
            "rna_fm_embeddings": torch.from_numpy(fm_cnn_np).float(),
            "rna_fm_mask": torch.from_numpy(fm_mask_np).long(),
            "rna_local_eskmer_sequence": local_eskmer_feat_seq,
            "rna_graph": rna_graph_obj,
            "affinity": torch.tensor([affinity_val], dtype=torch.float),
        }


def unwrap_model_output(model_output):
    """
    DeepMIF's model may return either:
    - a prediction tensor, or
    - a tuple/list where the first item is the prediction tensor and the rest are auxiliary outputs.
    """
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


def train_epoch(model, data_loader, optimizer, loss_criterion, grad_scaler=None):
    model.train()
    total_loss = 0.0
    valid_batches = 0

    progress_bar = tqdm(data_loader, desc="Training", leave=False, ncols=100, file=sys.stdout)

    for batch_data in progress_bar:
        if batch_data is None:
            continue

        drug_graph_batch = batch_data["drug_graph"].to(device)
        original_smiles_list = batch_data["original_smiles"]
        rna_fm_cnn_batch = batch_data["rna_fm_embeddings"].to(device)
        rna_fm_mask_cnn_batch = batch_data["rna_fm_mask"].to(device)
        rna_local_eskmer_sequence_batch = batch_data["rna_local_eskmer_sequence"].to(device)
        rna_graph_gnn_batch = batch_data["rna_graph"].to(device)
        true_affinity_values = batch_data["affinity"].to(device)

        optimizer.zero_grad()
        use_amp = grad_scaler is not None and device.type == "cuda"

        with torch.cuda.amp.autocast(enabled=use_amp):
            predictions = model(
                drug_graph_batch,
                rna_fm_cnn_batch,
                rna_fm_mask_cnn_batch,
                rna_local_eskmer_sequence_batch,
                rna_graph_gnn_batch,
                original_smiles_list,
            )
            predictions = unwrap_model_output(predictions)
            loss = loss_criterion(predictions.view(-1), true_affinity_values.view(-1))

        if torch.isnan(loss):
            logger.info("Warning: NaN loss. Skipping batch.")
            continue

        if use_amp:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        valid_batches += 1
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    if valid_batches == 0:
        return float("inf")

    return total_loss / valid_batches


def safe_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    pcc = 0.0
    scc = 0.0

    if len(y_true) > 1 and np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        pcc = pearsonr(y_true, y_pred)[0]
        scc = spearmanr(y_true, y_pred)[0]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else -1.0

    return pcc, scc, rmse, r2


def evaluate_epoch(model, data_loader, loss_criterion):
    model.eval()
    all_preds = []
    all_trues = []
    total_loss = 0.0
    valid_batches = 0

    with torch.no_grad():
        for batch_data in data_loader:
            if batch_data is None:
                continue

            drug_graph_batch = batch_data["drug_graph"].to(device)
            original_smiles_list = batch_data["original_smiles"]
            rna_fm_cnn_batch = batch_data["rna_fm_embeddings"].to(device)
            rna_fm_mask_cnn_batch = batch_data["rna_fm_mask"].to(device)
            rna_local_eskmer_sequence_batch = batch_data["rna_local_eskmer_sequence"].to(device)
            rna_graph_gnn_batch = batch_data["rna_graph"].to(device)
            true_affinity_values = batch_data["affinity"].to(device)

            predictions = model(
                drug_graph_batch,
                rna_fm_cnn_batch,
                rna_fm_mask_cnn_batch,
                rna_local_eskmer_sequence_batch,
                rna_graph_gnn_batch,
                original_smiles_list,
            )
            predictions = unwrap_model_output(predictions)

            loss = loss_criterion(predictions.view(-1), true_affinity_values.view(-1))
            total_loss += loss.item()
            valid_batches += 1

            all_preds.extend(predictions.detach().cpu().numpy().reshape(-1).tolist())
            all_trues.extend(true_affinity_values.detach().cpu().numpy().reshape(-1).tolist())

    if len(all_trues) == 0:
        return {
            "loss": float("inf"),
            "pcc": 0.0,
            "scc": 0.0,
            "rmse": float("inf"),
            "r2": -1.0,
            "y_true": [],
            "y_pred": [],
        }

    pcc, scc, rmse, r2 = safe_metrics(all_trues, all_preds)
    avg_loss = total_loss / max(valid_batches, 1)

    return {
        "loss": avg_loss,
        "pcc": pcc,
        "scc": scc,
        "rmse": rmse,
        "r2": r2,
        "y_true": all_trues,
        "y_pred": all_preds,
    }


def read_fold_files(split_dir, cold_name, fold):
    train_path = os.path.join(split_dir, f"cold_{cold_name}_train_fold{fold}.csv")
    val_path = os.path.join(split_dir, f"cold_{cold_name}_val_fold{fold}.csv")
    test_path = os.path.join(split_dir, f"cold_{cold_name}_test_fold{fold}.csv")

    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing split file: {path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df, train_path, val_path, test_path


def check_no_rna_overlap(train_df, val_df, test_df, seq_col):
    train_rnas = set(train_df[seq_col].dropna().astype(str))
    val_rnas = set(val_df[seq_col].dropna().astype(str))
    test_rnas = set(test_df[seq_col].dropna().astype(str))

    train_val_overlap = len(train_rnas & val_rnas)
    train_test_overlap = len(train_rnas & test_rnas)
    val_test_overlap = len(val_rnas & test_rnas)

    logger.info(
        "RNA sequence overlap check: "
        f"train-val={train_val_overlap}, train-test={train_test_overlap}, val-test={val_test_overlap}"
    )


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=rna_dta_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


def train_one_fold(args, fold):
    set_seed(args.seed)

    logger.info("=" * 80)
    logger.info(f"Fold {fold}/{NUM_FOLDS}")
    logger.info("=" * 80)

    train_df, val_df, test_df, train_path, val_path, test_path = read_fold_files(
        args.split_dir, args.cold_name, fold
    )

    resolve_columns(args, train_df)
    check_no_rna_overlap(train_df, val_df, test_df, args.seq_col)

    logger.info(f"Train file: {train_path}")
    logger.info(f"Val file:   {val_path}")
    logger.info(f"Test file:  {test_path}")
    logger.info(f"Raw sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_dataset = LiveRNADataset(train_df, args)
    val_dataset = LiveRNADataset(val_df, args)
    test_dataset = LiveRNADataset(test_df, args)

    logger.info(
        f"Dataset sizes after cleaning: "
        f"train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
    )

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers)

    model = RNADTA_FM_CrossAttentionModel(**vars(args)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")

    best_val_rmse = float("inf")
    best_model_state = None
    best_epoch = -1
    patience_counter = 0

    fold_model_dir = os.path.join(args.output_dir, "models")
    os.makedirs(fold_model_dir, exist_ok=True)
    model_save_path = os.path.join(
        fold_model_dir,
        f"DeepMIF_cold_{args.cold_name}_fold{fold}_seed{args.seed}.pth",
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler)
        val_metrics = evaluate_epoch(model, val_loader, loss_fn)

        logger.info(
            f"epoch: {epoch} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} | "
            f"val_rmse: {val_metrics['rmse']:.4f} | "
            f"val_p: {val_metrics['pcc']:.4f} | "
            f"val_s: {val_metrics['scc']:.4f} | "
            f"val_r2: {val_metrics['r2']:.4f}"
        )

        improved = val_metrics["rmse"] < best_val_rmse - args.delta

        if improved:
            best_val_rmse = val_metrics["rmse"]
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
            torch.save(best_model_state, model_save_path)
            logger.info(
                f"New best model saved: epoch={epoch}, "
                f"best_val_rmse={best_val_rmse:.4f}, path={model_save_path}"
            )
        elif epoch >= args.warm_up_epochs:
            patience_counter += 1

        if epoch >= args.warm_up_epochs and patience_counter >= args.early_stopping_patience:
            logger.info(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch}, best val RMSE: {best_val_rmse:.4f}"
            )
            break

    if best_model_state is None:
        raise RuntimeError("No best model was saved. Check whether the train/val data is valid.")

    model.load_state_dict(best_model_state)
    test_metrics = evaluate_epoch(model, test_loader, loss_fn)

    logger.info(
        f"[Fold {fold}] TEST using best validation model: "
        f"test_loss={test_metrics['loss']:.4f}, "
        f"p={test_metrics['pcc']:.4f}, "
        f"s={test_metrics['scc']:.4f}, "
        f"rmse={test_metrics['rmse']:.4f}, "
        f"r2={test_metrics['r2']:.4f}"
    )

    if args.save_predictions:
        pred_dir = os.path.join(args.output_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        pred_path = os.path.join(
            pred_dir,
            f"DeepMIF_cold_{args.cold_name}_fold{fold}_seed{args.seed}_predictions.csv",
        )
        pred_df = pd.DataFrame(
            {
                "y_true": test_metrics["y_true"],
                "y_pred": test_metrics["y_pred"],
            }
        )
        pred_df.to_csv(pred_path, index=False)
        logger.info(f"Saved predictions: {pred_path}")

    return {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "test_p": test_metrics["pcc"],
        "test_s": test_metrics["scc"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_loss": test_metrics["loss"],
    }


def get_folds_to_run(fold_arg):
    if fold_arg is None:
        return list(range(1, NUM_FOLDS + 1))

    if 1 <= fold_arg <= NUM_FOLDS:
        return [fold_arg]

    if 0 <= fold_arg < NUM_FOLDS:
        return [fold_arg + 1]

    raise ValueError("--fold must be either 1-5. 0-4 is also accepted and converted to 1-5.")


def apply_cli_to_script_args(script_args, cli_args):
    script_args.split_dir = cli_args.split_dir
    script_args.cold_name = cli_args.cold_name
    script_args.output_dir = cli_args.output_dir
    script_args.seed = cli_args.seed
    script_args.epochs = cli_args.epochs
    script_args.early_stopping_patience = cli_args.patience
    script_args.warm_up_epochs = cli_args.warm_up_epochs
    script_args.delta = cli_args.delta
    script_args.num_workers = cli_args.num_workers
    script_args.save_predictions = cli_args.save_predictions

    if cli_args.batch_size is not None:
        script_args.batch_size = cli_args.batch_size
    if cli_args.lr is not None:
        script_args.lr = cli_args.lr
    if cli_args.weight_decay is not None:
        script_args.weight_decay = cli_args.weight_decay

    if cli_args.smiles_col is not None:
        script_args.smiles_col = cli_args.smiles_col
    if cli_args.seq_col is not None:
        script_args.seq_col = cli_args.seq_col
    if cli_args.affinity_col is not None:
        script_args.affinity_col = cli_args.affinity_col
    if cli_args.target_id_col is not None:
        script_args.target_id_col = cli_args.target_id_col

    if not hasattr(script_args, "use_amp"):
        script_args.use_amp = False

    # Keep the same settings you manually used in the previous DeepMIF blind script.
    script_args.freeze_drug_encoder = False
    script_args.cross_attention_dropout = 0.3

    return script_args


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run DeepMIF on predefined cold-start train/val/test splits."
    )

    parser.add_argument(
        "--split_dir",
        type=str,
        default=os.path.join("datasets", "blind_test", "rna", "train_val_test_splits"),
        help="Folder containing cold_<cold_name>_train/val/test_foldX.csv files.",
    )
    parser.add_argument("--cold_name", type=str, default="rna")
    parser.add_argument("--fold", type=int, default=None, help="Run one fold only. Use 1-5. Omit to run all folds.")
    parser.add_argument("--output_dir", type=str, default=os.path.join("results", "DeepMIF_cold_rna"))

    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--warm_up_epochs", type=int, default=50)
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)

    parser.add_argument("--smiles_col", type=str, default=None)
    parser.add_argument("--seq_col", type=str, default=None)
    parser.add_argument("--affinity_col", type=str, default=None)
    parser.add_argument("--target_id_col", type=str, default=None)

    parser.add_argument("--save_predictions", action="store_true")

    return parser


def main():
    parser = build_parser()
    cli_args = parser.parse_args()

    args = ScriptArgs()
    args = apply_cli_to_script_args(args, cli_args)

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(args.output_dir, args.cold_name, args.seed)

    logger.info(f"Device: {device}")
    logger.info(f"RNA-FM embedding dict path: {get_rnafm_dict_path()}")
    logger.info(f"Split dir: {args.split_dir}")
    logger.info(f"Cold name: {args.cold_name}")
    logger.info(
        f"epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, "
        f"weight_decay={args.weight_decay}, patience={args.early_stopping_patience}, "
        f"warm_up_epochs={args.warm_up_epochs}, delta={args.delta}, seed={args.seed}"
    )

    folds_to_run = get_folds_to_run(cli_args.fold)
    results = []

    for fold in folds_to_run:
        fold_result = train_one_fold(args, fold)
        results.append(fold_result)

        logger.info(f"Current mean p: {np.mean([r['test_p'] for r in results])}")
        logger.info(f"Current mean s: {np.mean([r['test_s'] for r in results])}")
        logger.info(f"Current mean rmse: {np.mean([r['test_rmse'] for r in results])}")
        logger.info(f"Current mean r2: {np.mean([r['test_r2'] for r in results])}")

    results_df = pd.DataFrame(results)
    per_fold_path = os.path.join(args.output_dir, f"DeepMIF_cold_{args.cold_name}_per_fold_results.csv")
    results_df.to_csv(per_fold_path, index=False)
    logger.info(f"Saved per-fold results: {per_fold_path}")

    logger.info("=" * 80)
    logger.info("Final results:")
    logger.info(f"p mean: {results_df['test_p'].mean()}, std: {results_df['test_p'].std(ddof=0)}")
    logger.info(f"s mean: {results_df['test_s'].mean()}, std: {results_df['test_s'].std(ddof=0)}")
    logger.info(f"rmse mean: {results_df['test_rmse'].mean()}, std: {results_df['test_rmse'].std(ddof=0)}")
    logger.info(f"r2 mean: {results_df['test_r2'].mean()}, std: {results_df['test_r2'].std(ddof=0)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
