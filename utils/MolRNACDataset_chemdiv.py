import logging
import os

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from utils.tools import scale_nonzero


logger = logging.getLogger(__name__)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNA_EMB_PATH = os.path.join(
    PROJECT_ROOT, "embeddings", "rna", "rna_embeddings_clean.h5"
)
SEQ_TO_G_PATH = os.path.join(PROJECT_ROOT, "saved_objects", "seq_to_g4.pt")

EXPECTED_SMILES_MODALITIES = (
    "mordred descriptors",
    "simson",
    "chemberta",
)
EXPECTED_FEATURE_DIMS = {
    "mordred descriptors": 1205,
    "simson": 512,
    "chemberta": 384,
}


class MolRNACDatasetChemDiv(Dataset):
    """Dataset for one fixed RNA target paired with many ChemDiv molecules."""

    def __init__(
        self,
        smiles_features_dict,
        target_rna_sequence,
        task="ChemDiv virtual screening",
    ):
        logger.info(
            f"MolRNACDatasetChemDiv for {task} | "
            f"target RNA length={len(target_rna_sequence)}"
        )

        self.task = task
        self.target_rna_sequence = self.normalize_sequence(target_rna_sequence)

        missing_keys = set(EXPECTED_SMILES_MODALITIES) - set(smiles_features_dict)
        if missing_keys:
            raise KeyError(
                "ChemDiv embedding dictionary is missing required keys: "
                f"{sorted(missing_keys)}"
            )

        feature_frames = {}
        present_smiles = {}
        invalid_smiles = {}
        invalid_identifier_rows = {}

        for modality in EXPECTED_SMILES_MODALITIES:
            (
                feature_frames[modality],
                present_smiles[modality],
                invalid_smiles[modality],
                invalid_identifier_rows[modality],
            ) = self.prepare_feature_frame(
                smiles_features_dict[modality], modality
            )

        all_smiles = set().union(*present_smiles.values())
        valid_smiles = set.intersection(
            *(set(frame.index) for frame in feature_frames.values())
        )

        # Mordred is the reference only for deterministic output order. Feature
        # matching itself is always done through the explicit SMILES index.
        ordered_smiles = [
            smiles
            for smiles in feature_frames["mordred descriptors"].index
            if smiles in valid_smiles
        ]

        if not ordered_smiles:
            raise ValueError(
                "No ChemDiv molecules have valid Mordred, SimSon, and "
                "ChemBERTa features"
            )

        self.molecules_found = len(all_smiles)
        self.molecules_skipped = self.molecules_found - len(ordered_smiles)
        self.skip_reasons = {}

        for modality in EXPECTED_SMILES_MODALITIES:
            missing = all_smiles - present_smiles[modality]
            invalid = invalid_smiles[modality]
            if missing:
                self.skip_reasons[f"missing {modality}"] = len(missing)
            if invalid:
                self.skip_reasons[f"invalid {modality}"] = len(invalid)
            if invalid_identifier_rows[modality]:
                self.skip_reasons[
                    f"blank SMILES rows in {modality}"
                ] = invalid_identifier_rows[modality]

        self.data = pd.DataFrame(
            {
                "SMILES": ordered_smiles,
                "Target_RNA_sequence": self.target_rna_sequence,
            }
        )

        mordred_frame = scale_nonzero(
            feature_frames["mordred descriptors"].reset_index()
        ).set_index("SMILES")

        self.mordred = self.frame_to_tensor(mordred_frame, ordered_smiles)
        self.simson = self.frame_to_tensor(
            feature_frames["simson"], ordered_smiles
        )
        self.chemberta = self.frame_to_tensor(
            feature_frames["chemberta"], ordered_smiles
        )

        self.graph, self.rinalmo, self.seq_len = self.load_rna_features()

        # These modalities were excluded when case_study_model.pt was trained.
        # Zero placeholders preserve MultiStructRSA's established input order.
        self.ernie = torch.zeros((self.seq_len, 768), dtype=torch.float32)
        self.morgan = torch.zeros(1024, dtype=torch.float32)

        logger.info(
            f"ChemDiv alignment complete | found={self.molecules_found}, "
            f"valid={len(self.data)}, skipped={self.molecules_skipped}"
        )

    @staticmethod
    def normalize_sequence(sequence):
        return (
            str(sequence)
            .strip()
            .upper()
            .replace("T", "U")
            .replace("Y", "U")
            .replace("X", "U")
        )

    @staticmethod
    def prepare_feature_frame(frame, modality):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"ChemDiv '{modality}' features must be a pandas DataFrame"
            )
        if frame.shape[1] < 2:
            raise ValueError(
                f"ChemDiv '{modality}' DataFrame must contain a SMILES "
                "column followed by feature columns"
            )

        expected_dim = EXPECTED_FEATURE_DIMS[modality]
        actual_dim = frame.shape[1] - 1
        if actual_dim != expected_dim:
            raise ValueError(
                f"ChemDiv '{modality}' feature dimension is {actual_dim}; "
                f"expected {expected_dim}"
            )

        prepared = frame.copy()
        prepared = prepared.rename(columns={prepared.columns[0]: "SMILES"})

        non_null_identifier = prepared["SMILES"].notna()
        smiles = prepared["SMILES"].astype(str).str.strip()
        valid_identifier = non_null_identifier & smiles.ne("")
        invalid_identifier_rows = int((~valid_identifier).sum())

        duplicate_mask = smiles[valid_identifier].duplicated(keep=False)
        if duplicate_mask.any():
            examples = smiles[valid_identifier][duplicate_mask].unique()[:3]
            raise ValueError(
                f"ChemDiv '{modality}' contains duplicate SMILES, so its "
                f"features cannot be aligned unambiguously. Examples: {list(examples)}"
            )

        feature_columns = list(prepared.columns[1:])
        numeric_features = prepared[feature_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        finite_features = np.isfinite(
            numeric_features.to_numpy(dtype=np.float32)
        ).all(axis=1)

        invalid_feature_mask = valid_identifier & ~finite_features
        invalid_smiles = set(smiles[invalid_feature_mask])
        valid_rows = valid_identifier & finite_features

        cleaned = numeric_features.loc[valid_rows].copy()
        cleaned.insert(0, "SMILES", smiles.loc[valid_rows].values)
        cleaned = cleaned.set_index("SMILES")

        return (
            cleaned,
            set(smiles[valid_identifier]),
            invalid_smiles,
            invalid_identifier_rows,
        )

    @staticmethod
    def frame_to_tensor(frame, ordered_smiles):
        aligned = frame.loc[ordered_smiles]
        values = aligned.to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Aligned ChemDiv features contain NaN or infinity")
        return torch.from_numpy(values)

    def load_rna_features(self):
        if not os.path.isfile(SEQ_TO_G_PATH):
            raise FileNotFoundError(f"RNA graph store not found: {SEQ_TO_G_PATH}")
        if not os.path.isfile(RNA_EMB_PATH):
            raise FileNotFoundError(f"RNA embedding store not found: {RNA_EMB_PATH}")

        seq_to_g = torch.load(SEQ_TO_G_PATH, map_location="cpu")
        if self.target_rna_sequence not in seq_to_g:
            raise KeyError(
                "SLI RNA sequence is missing from the setting-4 gRNAde graph store"
            )
        graph = seq_to_g[self.target_rna_sequence].clone()
        del seq_to_g

        with h5py.File(RNA_EMB_PATH, "r") as rna_embeddings:
            if self.target_rna_sequence not in rna_embeddings:
                raise KeyError(
                    "SLI RNA sequence is missing from the RNA embedding store"
                )

            current_embedding = rna_embeddings[self.target_rna_sequence]
            seq_len = int(current_embedding["seq_len"][()])
            rinalmo = torch.from_numpy(
                current_embedding["rinalmo"][:]
            ).float()

        if seq_len != len(self.target_rna_sequence):
            raise ValueError(
                f"SLI sequence length is {len(self.target_rna_sequence)}, but "
                f"the stored RNA length is {seq_len}"
            )
        if rinalmo.shape != (seq_len, 1280):
            raise ValueError(
                f"Expected SLI RiNALMo shape ({seq_len}, 1280), received "
                f"{tuple(rinalmo.shape)}"
            )

        return graph, rinalmo, seq_len

    @staticmethod
    def collate_fn(batch):
        if len(batch) != 1:
            raise ValueError(
                "The checkpoint uses the single-sample MultiStructRSAFull "
                "implementation; ChemDiv inference requires batch_size=1"
            )

        inputs, smiles = zip(*batch)
        graphs, ernie, rinalmo, seq_len, morgan, mordred, chemberta, simson = zip(
            *inputs
        )

        inputs = [
            Batch.from_data_list(list(graphs)),
            torch.stack(ernie, dim=0),
            torch.stack(rinalmo, dim=0),
            torch.tensor(seq_len),
            torch.stack(morgan, dim=0),
            torch.stack(mordred, dim=0),
            torch.stack(chemberta, dim=0),
            torch.stack(simson, dim=0),
        ]
        return inputs, list(smiles)

    def make_loader(self, num_workers=0, pin_memory=True):
        return DataLoader(
            self,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=MolRNACDatasetChemDiv.collate_fn,
        )

    def get_df(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = (
            self.graph,
            self.ernie,
            self.rinalmo,
            self.seq_len,
            self.morgan,
            self.mordred[idx],
            self.chemberta[idx],
            self.simson[idx],
        )
        return inputs, self.data.iloc[idx]["SMILES"]
