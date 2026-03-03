from utils.generate_smiles_features import generate_smiles_features

'''
# in order to execute local main() function, comment the 
# imports above and uncomment the import below

# from generate_smiles_features import generate_smiles_features
'''
#from generate_smiles_features import generate_smiles_features
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch_geometric.data import Batch, Data

import os
import pandas as pd
import numpy as np

import torch

ERNIE_RNA_EMB_PATH = "embeddings/rna/ernie_rna_full_emb.parquet"
SEQ_TO_G_PATH = "saved_objects/seq_to_g3.pt"

def subgraph_node_range(g: Data, start: int, end: int) -> Data:
    """
    Extract induced subgraph containing nodes [start:end) (assumes node order follows sequence order).
    Keeps only edges with BOTH endpoints inside the window, then reindexes nodes to [0..n_sub-1].
    """
    assert 0 <= start < end <= g.num_nodes
    device = g.edge_index.device

    node_ids = torch.arange(start, end, device=device)
    node_mask = torch.zeros(g.num_nodes, dtype=torch.bool, device=device)
    node_mask[node_ids] = True

    ei = g.edge_index
    e_mask = node_mask[ei[0]] & node_mask[ei[1]]
    ei_sub = ei[:, e_mask]

    # reindex nodes
    idx_map = torch.full((g.num_nodes,), -1, dtype=torch.long, device=device)
    idx_map[node_ids] = torch.arange(node_ids.numel(), device=device)
    ei_sub = idx_map[ei_sub]

    out = Data()
    out.edge_index = ei_sub

    # --- node-like fields (slice on dim0 == num_nodes) ---
    # Explicitly slice the fields you use in the encoder:
    out.node_s = g.node_s[start:end].clone()
    out.node_v = g.node_v[start:end].clone()
    out.mask_confs = g.mask_confs[start:end].clone()

    if hasattr(g, "seq"):
        out.seq = g.seq[start:end].clone()
    if hasattr(g, "mask_seq"):
        out.mask_seq = g.mask_seq[start:end].clone()
    if hasattr(g, "mask_coords"):
        out.mask_coords = g.mask_coords[start:end].clone()

    # --- edge-like fields (slice on dim0 == num_edges) ---
    out.edge_s = g.edge_s[e_mask].clone()
    out.edge_v = g.edge_v[e_mask].clone()

    return out


def chunk_graph(g: Data, chunk_size: int = 512, overlap: int = 64):
    """
    Return list of chunked graphs. For short RNAs returns [g].
    """
    n = int(g.num_nodes)
    if n <= chunk_size:
        return [g]

    step = max(1, chunk_size - overlap)
    chunks = []
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(subgraph_node_range(g, start, end))
        if end == n:
            break
        start += step
    return chunks


class MolRNADataset(Dataset):
    """
    Dataset that returns:
      (gRNAde_graph, morgan_fp, rdkit_desc, ernie_rna_emb, label)

    Notes
    -----
    * self.seq_to_g is a dict: {Target_RNA_sequence (str) -> torch_geometric.data.Data}
    * We clone() the cached Data to avoid accidental in-place modification across samples.
    * Features are returned as torch.FloatTensor so collate_fn can stack them cleanly.
    * node_counts is provided for gRNAde-like BatchSampler (optional).
    """
    def __init__(self, ds, chunk_size=512, chunk_overlap=64, max_chunks=None):
        """
        ds columns:
        SMILES | Target_RNA_sequence | ... | pKd
        """
        self.data = ds.reset_index(drop=True)
        # replace all invalid nucleotides ('T', 'X' and 'Y') -> 'U'
        self.data["Target_RNA_sequence"] = (
             self.data["Target_RNA_sequence"].str.replace("T", "U").str.replace("Y", "U").str.replace("X", "U")
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks = max_chunks  # optional safety cap

        # dictionary that maps 'rna_seq': graph object (pyG.Data()). Inspired by gRNAde which
        # is a 3D RNA inverse design by CK Joshi et al.
        self.seq_to_g = torch.load(SEQ_TO_G_PATH, map_location="cpu") 

        # rna features
        ernie_rna_emb_ = pd.read_parquet(ERNIE_RNA_EMB_PATH)
        ernie_rna_emb_ = ernie_rna_emb_.rename(columns={"sequence": "Target_RNA_sequence"})

        ernie_rna_emb_["Target_RNA_sequence"] = (
            ernie_rna_emb_["Target_RNA_sequence"].str.replace("T", "U").str.replace("Y", "U").str.replace("X", "U")
        )

        ernie_rna_emb_ = (
            ernie_rna_emb_
            .drop_duplicates(subset="Target_RNA_sequence", keep="first")
            .reset_index(drop=True)
        )

        self.ernie_rna_emb = self.data.merge(
            ernie_rna_emb_,
            how="left",
            on="Target_RNA_sequence",
            sort=False,
            validate="many_to_one" 
        )
        rna_feature_cols = [c for c in self.ernie_rna_emb.columns if c.startswith("emb_")]
        
        self.ernie_rna_emb = self.ernie_rna_emb[rna_feature_cols].reset_index(drop=True)

        # smiles features
        data_smiles = self.data["SMILES"].tolist()
        smiles_features_dict = generate_smiles_features(data_smiles)

        self.smiles_chemberta_features = smiles_features_dict["chemberta"]
        self.smiles_morgan_features = smiles_features_dict["morgan fingerprints"]
        self.smiles_rdkit_features = smiles_features_dict["chemical descriptors"]

        # ---- for gRNAde-like batching (optional) ----
        self.node_counts = [
            int(self.seq_to_g[seq].num_nodes)
            for seq in self.data["Target_RNA_sequence"].tolist()
        ]

    @staticmethod
    def collate_fn(batch):
        """
        Flattens variable #chunks per sample into one big Batch of chunks,
        and returns chunk_to_sample mapping so model can aggregate chunks back to samples.
        """
        graphs_lists, morgan, rdkit, ernie, chemberta, y = zip(*batch)

        flat_graphs = []
        chunk_to_sample = []

        for i, glist in enumerate(graphs_lists):
            for g in glist:
                flat_graphs.append(g)
                chunk_to_sample.append(i)

        graphs_batch = Batch.from_data_list(flat_graphs)
        chunk_to_sample = torch.tensor(chunk_to_sample, dtype=torch.long)  # [n_chunks_total]

        morgan = torch.stack(morgan, dim=0)
        rdkit  = torch.stack(rdkit, dim=0)
        ernie  = torch.stack(ernie, dim=0)
        chemberta = torch.stack(chemberta, dim=0)
        y      = torch.stack([yi.view(1) for yi in y], dim=0).view(-1)

        return graphs_batch, chunk_to_sample, ernie, morgan, rdkit, chemberta, y

    def make_loader(self, batch_size=8, shuffle=True, num_workers=0, pin_memory=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=MolRNADataset.collate_fn,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rna_seq = self.data.loc[idx, "Target_RNA_sequence"]
        print(f'seq len -> {len(rna_seq)}')
        base_g = self.seq_to_g[rna_seq]
        g = base_g.clone()

        graphs_list = chunk_graph(g, chunk_size=self.chunk_size, overlap=self.chunk_overlap)

        # optional: cap number of chunks to avoid extreme cases
        if self.max_chunks is not None and len(graphs_list) > self.max_chunks:
            graphs_list = graphs_list[: self.max_chunks]

        ernie = torch.from_numpy(self.ernie_rna_emb.iloc[idx].values.astype(np.float32))
        morgan = torch.from_numpy(self.smiles_morgan_features.iloc[idx, 1:].values.astype(np.float32))
        rdkit  = torch.from_numpy(self.smiles_rdkit_features.iloc[idx, 1:].values.astype(np.float32))
        chemberta = torch.from_numpy(self.smiles_chemberta_features.iloc[idx, 1:].values.astype(np.float32))
        y = torch.tensor(self.data.iloc[idx]["pKd"], dtype=torch.float32)

        # NOTE: first element is *list* of Data
        return (graphs_list, morgan, rdkit, ernie, chemberta, y)

def main():
    # dataset object check: load RSIM RNA patamerts dataset and create MolRNADataset object from it
    aptamers_dataset_path = os.path.join('..' ,'datasets', 'RSIM', 'Aptamers_dataset_v1.csv')
    aptamers_df = pd.read_csv(aptamers_dataset_path, sep='\t')
    print(aptamers_df.head())

    molRNAds = MolRNADataset(aptamers_df)
    print(len(molRNAds))
    print(molRNAds[10])

if __name__ == "__main__":
     main()

    