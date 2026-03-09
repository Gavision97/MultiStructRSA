from utils.generate_smiles_features import generate_smiles_features

'''
# in order to execute local main() function, comment the 
# imports above and uncomment the import below

# from generate_smiles_features import generate_smiles_features
'''
#from generate_smiles_features import generate_smiles_features
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_scatter import scatter_mean
import os
import h5py # in order to load RNA nucleotide level features (ERNIE-RNA & RiNALMo)
import pandas as pd
import numpy as np

import pickle
import random
import torch

from utils.tools import scale_nonzero

RNA_EMB_PATH = "embeddings/rna/rna_embeddings.h5"
SMILES_EMB_PATH = "embeddings/smiles/smiles_embeddings.pkl"
ERNIE_RNA_EMB_PATH = "embeddings/rna/ernie_rna_full_emb.parquet"
SEQ_TO_G_PATH = "saved_objects/seq_to_g4.pt"  # k=3 if len(seq) > 1000, k=8 if len(seq) < 1000 and len(seq) > 100, else k=32
# SEQ_TO_G_PATH = "saved_objects/seq_to_g3.pt" # k=3 if len(seq) > 1000, else k=32


# k = 32
SEQ_TO_G_PATH_1 = "saved_objects/seq_to_g1.pt" # no aug & no multi-conformer - just 'model_0'
SEQ_TO_G_PATH_2 = "saved_objects/seq_to_g2.pt" # multi-conformer @ conf = 5
SEQ_TO_G_PATH_3 = "saved_objects/seq_to_g3.pt" # data-augmentation; rna -> |list|=5 conformations

# k=3 if len(seq) > 1000, k=8 if len(seq) < 1000 and len(seq) > 100, else k=32
SEQ_TO_G_PATH_4 = "saved_objects/seq_to_g4.pt" # no aug & no multi-conformer - just 'model_0'
SEQ_TO_G_PATH_5 = "saved_objects/seq_to_g5.pt" # multi-conformer @ conf = 5
SEQ_TO_G_PATH_6 = "saved_objects/seq_to_g6.pt" # data-augmentation; rna -> |list|=5 conformations

import logging
logger = logging.getLogger(__name__)

def _get_seq_to_g_setting(setting=1):
    if setting == 1:
        return SEQ_TO_G_PATH_1
    elif setting == 2:
        return SEQ_TO_G_PATH_2
    elif setting == 3:
        return SEQ_TO_G_PATH_3
    elif setting == 4:
        return SEQ_TO_G_PATH_4  
    elif setting == 5:
        return SEQ_TO_G_PATH_5
    else:
        return SEQ_TO_G_PATH_6

# with pt ft gRNAde & RNA cross attention fusion module
class MolRNADataset_(Dataset):
    """
    Dataset that returns:
      (gRNAde_graph, morgan_fp, rdkit_desc, ernie_rna_emb, label)

    Notes
    -----
    * self.seq_to_g is a dict: {Target_RNA_sequence (str) -> torch_geometric.data.Data}
    * We clone() the cached Data to avoid accidental in-place modification across samples.
    * Features are returned as torch.FloatTensor so collate_fn can stack them cleanly.
    """
    def __init__(self, ds_):
        """
        ds_ columns:
        SMILES | Target_RNA_sequence | ... | pKd
        """
        
        self.data = ds_.reset_index(drop=True)
        # replace all invalid nucleotides ('T', 'X' and 'Y') -> 'U'
        self.data["Target_RNA_sequence"] = (
             self.data["Target_RNA_sequence"].str.replace("T", "U").str.replace("Y", "U").str.replace("X", "U")
        )
        # dictionary that maps 'rna_seq': graph object (pyG.Data()). Inspired by gRNAde which
        # is a 3D RNA inverse design by CK Joshi et al.
        self.seq_to_g = torch.load(SEQ_TO_G_PATH, map_location="cpu") 

        self.rna_emb = h5py.File(RNA_EMB_PATH, "r")

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
        #data_smiles = self.data["SMILES"].unique()
        #smiles_features_dict = generate_smiles_features(data_smiles)

        with open(SMILES_EMB_PATH, "rb") as f:
            smiles_features_dict =  pickle.load(f)

        self.smiles_chemberta_features = smiles_features_dict["chemberta"]
        self.smiles_morgan_features = smiles_features_dict["morgan fingerprints"]
        # we standardize  descriptors (rdkit & Mordred) for stability & ignore invalid molecules by leaving
        # their embedding vector with zero values.
        self.smiles_rdkit_features = scale_nonzero(smiles_features_dict["chemical descriptors"])
        self.mordred_descriptors_features = scale_nonzero(smiles_features_dict["mordred descriptors"])

    @staticmethod
    def collate_fn(batch):
        graphs, ernie, seq_len, morgan, rdkit, chemberta, y = zip(*batch)

        graphs = Batch.from_data_list(list(graphs))
        morgan = torch.stack(morgan, dim=0)
        rdkit  = torch.stack(rdkit, dim=0)
        chemberta = torch.stack(chemberta, dim=0)
        ernie  = torch.stack(ernie, dim=0) # (B, max_seq_len, 768)
        seq_len = torch.tensor(seq_len)
        y = torch.tensor(y)

        # List of RNA & small molecule representations
        inputs = [graphs, ernie, seq_len, morgan, rdkit, chemberta]
        return inputs, y

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
        smiles = self.data.loc[idx, "SMILES"]
        rna_seq = self.data.loc[idx, "Target_RNA_sequence"]
        #print(f'seq len -> {len(rna_seq)}')
        base_g = self.seq_to_g[rna_seq]

        # RNA features & graph
        g = base_g.clone()

        curr_rna_emb = self.rna_emb[rna_seq] # get current RNA nucleotide level embedding group (ERNIE-RNA & RiNALMo)
        ernie = torch.from_numpy(curr_rna_emb["ernie"][:]).float()        # (max_seq_len, 768) -> (2922, 768)
        rinalmo = torch.from_numpy(curr_rna_emb["rinalmo"][:]).float()    # (max_seq_len, 1280) -> (2922, 1280)
        seq_len = int(curr_rna_emb["seq_len"][()]) # RNA sequence length, for attention nucleotide level masking


        #ernie = torch.from_numpy(self.ernie_rna_emb.iloc[idx].values.astype(np.float32))

        # Small molecule features
        chemberta = torch.from_numpy(self.smiles_chemberta_features.loc[self.smiles_chemberta_features['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32))
        morgan_fingerprint = torch.from_numpy(self.smiles_morgan_features.loc[self.smiles_morgan_features['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32))
        chemical_descriptors = torch.from_numpy(self.smiles_rdkit_features.loc[self.smiles_rdkit_features['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32))
        y = torch.tensor(self.data.iloc[idx]["pKd"], dtype=torch.float32)

        return (
            g,
            ernie,
            seq_len,
            morgan_fingerprint,
            chemical_descriptors,
            chemberta,
            y
        )

# with pt ft gRNAde, without rna cross-attention fusion module    
class MolRNADataset(Dataset):
    """
    Dataset that returns:
      (gRNAde_graph, morgan_fp, rdkit_desc, ernie_rna_emb, label)

    Notes
    -----
    * self.seq_to_g is a dict: {Target_RNA_sequence (str) -> torch_geometric.data.Data}
    * We clone() the cached Data to avoid accidental in-place modification across samples.
    * Features are returned as torch.FloatTensor so collate_fn can stack them cleanly.
    """
    def __init__(self, ds_, setting=1, aug=False, task='training'):
        """
        ds_ columns:
        SMILES | Target_RNA_sequence | ... | pKd
        """
        logger.info(f'Initializing MolRNADataset with for {task} setting={setting} and aug={aug}')
        self.data = ds_.reset_index(drop=True)
        self.aug = aug

        # replace all invalid nucleotides ('T', 'X' and 'Y') -> 'U'
        self.data["Target_RNA_sequence"] = (
             self.data["Target_RNA_sequence"].str.replace("T", "U").str.replace("Y", "U").str.replace("X", "U")
        )

        # dictionary that maps 'rna_seq': graph object (pyG.Data()). Inspired by gRNAde which
        # is a 3D RNA inverse design by CK Joshi et al.
        seq_to_g_setting = _get_seq_to_g_setting(setting=setting)
        self.seq_to_g = torch.load(seq_to_g_setting, map_location="cpu")

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
        #data_smiles = self.data["SMILES"].unique()
        #smiles_features_dict = generate_smiles_features(data_smiles)
        with open(SMILES_EMB_PATH, "rb") as f:
            smiles_features_dict =  pickle.load(f)

        self.smiles_chemberta_features = smiles_features_dict["chemberta"]
        self.smiles_morgan_features = smiles_features_dict["morgan fingerprints"]
        # we standardize  descriptors (rdkit & Mordred) for stability & ignore invalid molecules by leaving
        # their embedding vector with zero values.
        self.smiles_rdkit_features = scale_nonzero(smiles_features_dict["chemical descriptors"])
        self.mordred_descriptors_features = scale_nonzero(smiles_features_dict["mordred descriptors"])

    def set_mode(self, mode: str):
        assert mode in ("train", "eval")
        self.mode = mode

    @staticmethod
    def collate_fn(batch):
        graphs, ernie, morgan, rdkit, chemberta, y = zip(*batch)

        graphs = Batch.from_data_list(list(graphs))
        morgan = torch.stack(morgan, dim=0)
        rdkit  = torch.stack(rdkit, dim=0)
        chemberta = torch.stack(chemberta, dim=0)
        ernie  = torch.stack(ernie, dim=0) # (B, 768)
        y = torch.tensor(y)

        # List of RNA & small molecule representations
        inputs = [graphs, ernie, morgan, rdkit, chemberta]
        return inputs, y

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
        smiles = self.data.loc[idx, "SMILES"]
        rna_seq = self.data.loc[idx, "Target_RNA_sequence"]
        #print(f'seq len -> {len(rna_seq)}')

        _g = self.seq_to_g[rna_seq]
        if self.aug:
            # settings (3)/(6): seq_to_g[seq] is a list[Data] (5 conformers)
            _g = random.choice(_g)
        g = _g.clone()

        ernie = torch.from_numpy(self.ernie_rna_emb.iloc[idx].values.astype(np.float32))

        # Small molecule features
        chemberta = torch.from_numpy(self.smiles_chemberta_features.loc[self.smiles_chemberta_features['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32))
        morgan_fingerprint = torch.from_numpy(self.smiles_morgan_features.loc[self.smiles_morgan_features['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32))
        #chemical_descriptors = torch.from_numpy(self.smiles_rdkit_features.loc[self.smiles_rdkit_features['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32))
        mordred_descriptors = torch.from_numpy(self.mordred_descriptors_features.loc[self.mordred_descriptors_features['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32))
        y = torch.tensor(self.data.iloc[idx]["pKd"], dtype=torch.float32)

        return (
            g,
            ernie,
            morgan_fingerprint,
            mordred_descriptors,
            chemberta,
            y
        )



def main():
    
    # dataset object check: load RSIM RNA patamerts dataset and create MolRNADataset object from it
    aptamers_dataset_path = os.path.join('..' ,'datasets', 'RSIM', 'Aptamers_dataset_v1.csv')
    aptamers_df = pd.read_csv(aptamers_dataset_path, sep='\t')
    print(aptamers_df.head())

    molRNAds = MolRNADataset(aptamers_df)
    print(len(molRNAds))
    print(molRNAds[10])
    

    '''
    rna_emb = h5py.File(RNA_EMB_PATH, "r")
    rna_seq = 'GGGAGAGGGUUUAAUUACGAAAGUAAUUGGAUCCGCAAGG'
    seq_len = len(rna_seq)
    max_seq_len = 2922
    grp = rna_emb[rna_seq]
    print(f'seq ernie embedding shape test -> {grp["ernie"].shape}') # (max_len, 768)
    print(f'seq rinalmo embedding shape test -> {grp["rinalmo"].shape}') # (max_len, 1280)

    # senaty checks
    curr_ernie =  torch.from_numpy(np.array(grp["ernie"]))
    curr_rinalmo =  torch.from_numpy(np.array(grp["rinalmo"]))
    
    assert not torch.equal(curr_ernie[len(rna_seq)-1], torch.zeros(768))
    assert not torch.equal(curr_rinalmo[len(rna_seq)-1], torch.zeros(1280))
    if len(rna_seq) != max_seq_len:
        assert torch.equal(curr_ernie[len(rna_seq)], torch.zeros(768))
        assert torch.equal(curr_rinalmo[len(rna_seq)], torch.zeros(1280))
    '''

if __name__ == "__main__":
     main()

    