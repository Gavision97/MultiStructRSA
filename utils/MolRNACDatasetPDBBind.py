import logging
import pickle
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

from utils.tools import scale_nonzero


SEQ_TO_G_PATH = "/home/gavrilev/Thesis/src/saved_objects/seq_to_g_pdbbind.pt"
SMILES_EMB_PATH = "/home/gavrilev/Thesis/src/embeddings/smiles/smiles_emb_pdbbind_aligned.pkl"
RINALMO_EMB_PATH = "/home/gavrilev/Thesis/src/embeddings/rna/rinalmo_embeddings_pdbbind.pkl"


logger = logging.getLogger(__name__)


class MolRNACDatasetPDBBind(Dataset):

    def __init__(self, ds_, task="external evaluation", is_seq_emb=False):
        logger.info(f"MolRNACDatasetPDBBind for {task} | is_seq_emb={is_seq_emb}")

        self.task = task
        self.is_seq_emb = is_seq_emb
        self.data = ds_.reset_index(drop=True).copy()

        self.data["Target_RNA_sequence"] = self.data["Target_RNA_sequence"].apply(self.normalize_sequence)
        self.data["SMILES"] = self.data["SMILES"].astype(str).str.strip()

        self.seq_to_g = torch.load(SEQ_TO_G_PATH, map_location="cpu")
        self.seq_to_g = {self.normalize_sequence(sequence): graph for sequence, graph in self.seq_to_g.items()}

        self.rinalmo_embeddings = self.open_pkl(RINALMO_EMB_PATH)
        self.rinalmo_embeddings = {self.normalize_sequence(sequence): embedding for sequence, embedding in self.rinalmo_embeddings.items()}

        smiles_features_dict = self.open_pkl(SMILES_EMB_PATH)

        self.smiles_morgan_features = smiles_features_dict["morgan fingerprints"]
        self.mordred_descriptors_features = scale_nonzero(smiles_features_dict["mordred descriptors"])
        self.smiles_chemberta_features = smiles_features_dict["chemberta"]
        self.smiles_simson_features = smiles_features_dict["simson"]

        self.morgan_dict = self.df_to_tensor_dict(self.smiles_morgan_features)
        self.mordred_dict = self.df_to_tensor_dict(self.mordred_descriptors_features)
        self.chemberta_dict = self.df_to_tensor_dict(self.smiles_chemberta_features)
        self.simson_dict = self.df_to_tensor_dict(self.smiles_simson_features)

        self.check_missing_features()

    @staticmethod
    def normalize_sequence(sequence):
        return str(sequence).strip().upper().replace("T", "U").replace("Y", "U").replace("X", "U")

    @staticmethod
    def open_pkl(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def df_to_tensor_dict(df):
        feature_columns = [column for column in df.columns if column != "SMILES"]
        return {str(row["SMILES"]).strip(): torch.tensor(row[feature_columns].values.astype(np.float32)) for _, row in df.iterrows()}

    @staticmethod
    def to_float_tensor(value):
        if isinstance(value, dict):
            if "rinalmo" not in value:
                raise KeyError("RiNALMo embedding dictionary value does not contain a 'rinalmo' key.")
            value = value["rinalmo"]

        if isinstance(value, torch.Tensor):
            tensor = value.clone().detach().float()
        else:
            tensor = torch.tensor(np.asarray(value), dtype=torch.float32)

        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)

        if tensor.ndim != 2:
            raise ValueError(f"Expected a 2D RiNALMo matrix, but received shape {tuple(tensor.shape)}.")

        return tensor

    def check_missing_features(self):
        sequences = set(self.data["Target_RNA_sequence"])
        smiles_values = set(self.data["SMILES"])

        missing_graphs = sequences - set(self.seq_to_g)
        missing_rinalmo = sequences - set(self.rinalmo_embeddings)
        missing_morgan = smiles_values - set(self.morgan_dict)
        missing_mordred = smiles_values - set(self.mordred_dict)
        missing_chemberta = smiles_values - set(self.chemberta_dict)
        missing_simson = smiles_values - set(self.simson_dict)

        if missing_graphs:
            raise KeyError(f"Missing gRNAde graphs for {len(missing_graphs)} sequences. Examples: {list(missing_graphs)[:3]}")

        if missing_rinalmo:
            raise KeyError(f"Missing RiNALMo embeddings for {len(missing_rinalmo)} sequences. Examples: {list(missing_rinalmo)[:3]}")

        if missing_morgan:
            raise KeyError(f"Missing Morgan fingerprints for {len(missing_morgan)} SMILES. Examples: {list(missing_morgan)[:3]}")

        if missing_mordred:
            raise KeyError(f"Missing Mordred descriptors for {len(missing_mordred)} SMILES. Examples: {list(missing_mordred)[:3]}")

        if missing_chemberta:
            raise KeyError(f"Missing ChemBERTa embeddings for {len(missing_chemberta)} SMILES. Examples: {list(missing_chemberta)[:3]}")

        if missing_simson:
            raise KeyError(f"Missing SimSon embeddings for {len(missing_simson)} SMILES. Examples: {list(missing_simson)[:3]}")

    @staticmethod
    def collate_fn(batch):
        inputs, y = zip(*batch)
        graphs, ernie, rinalmo, seq_len, morgan, mordred, chemberta, simson = zip(*inputs)

        graphs = Batch.from_data_list(list(graphs))
        ernie = torch.stack(ernie, dim=0)
        rinalmo = torch.stack(rinalmo, dim=0)
        seq_len = torch.tensor(seq_len)
        morgan = torch.stack(morgan, dim=0)
        mordred = torch.stack(mordred, dim=0)
        chemberta = torch.stack(chemberta, dim=0)
        simson = torch.stack(simson, dim=0)
        y = torch.stack(y, dim=0)

        inputs = [graphs, ernie, rinalmo, seq_len, morgan, mordred, chemberta, simson]

        return inputs, y

    def make_loader(self, batch_size=1, shuffle=False, num_workers=0, pin_memory=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=MolRNACDatasetPDBBind.collate_fn)

    def get_df(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        smiles = row["SMILES"]
        rna_seq = row["Target_RNA_sequence"]

        g = self.seq_to_g[rna_seq].clone()

        rinalmo = self.to_float_tensor(self.rinalmo_embeddings[rna_seq])
        seq_len = len(rna_seq)

        if rinalmo.shape[0] != seq_len:
            raise ValueError(f"RiNALMo length mismatch for sequence {rna_seq[:20]}...: sequence length={seq_len}, embedding length={rinalmo.shape[0]}")

        # ERNIE-RNA is not used, but the zero matrix keeps the input interface unchanged.
        ernie = torch.zeros((seq_len, 768), dtype=torch.float32)

        if self.is_seq_emb:
            ernie = ernie.mean(dim=0)
            rinalmo = rinalmo.mean(dim=0)

        morgan_fingerprint = self.morgan_dict[smiles]
        mordred_descriptors = self.mordred_dict[smiles]
        chemberta = self.chemberta_dict[smiles]
        simson = self.simson_dict[smiles]

        y = torch.tensor(row["pKd"], dtype=torch.float32)

        return (g, ernie, rinalmo, seq_len, morgan_fingerprint, mordred_descriptors, chemberta, simson), y