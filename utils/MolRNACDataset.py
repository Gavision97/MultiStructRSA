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

# RNA & Small molecule pre-computed embeddings files paths
RNA_EMB_PATH = "embeddings/rna/rna_embeddings_clean.h5" # w/o padding
#RNA_EMB_PATH = "embeddings/rna/rna_embeddings_padded.h5" # w padding

# small molecule pre-computer embeddings (for R-SIM dataset)
#SMILES_EMB_PATH = "embeddings/smiles/smiles_embeddings.pkl"
SMILES_EMB_PATH = "embeddings/smiles/smiles_emb.pkl"

SMILES_AUG_EMB_PATH = "embeddings/smiles/smiles_seq_aug_emb.pkl"
SMILES2AUG_PATH = "saved_objects/smiles2aug.pkl"

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

class MolRNADataset(Dataset):
    """
    Dataset that returns:
      (gRNAde_graph, ernie_emb, rinalmo_emb, morgan_fp, mordred_desc, chemberta, simson), label

    Notes
    -----
    * RNA augmentation stays on-the-fly: one conformer is sampled in __getitem__.
    * If mol_aug=True, the dataset is expanded in __init__ using smiles2aug:
      each original row is duplicated for all augmented SMILES of that molecule.
    * Morgan/Mordred are kept on the canonical/original SMILES because they are
      molecule-level invariant features.
    """
    def __init__(self, ds_, setting=1, shuffle_struct=False, rna_aug=False, mol_aug=False, task='training', is_seq_emb=False):
        logger.info(
            f'W/O PADDING -> MolRNACDataset for {task} setting={setting} '
            f'rna aug={rna_aug} mol aug={mol_aug} shuffle_sturct={shuffle_struct} - for RNA cross-attention fusion'
        )

        self.is_seq_emb = is_seq_emb
        self.rna_aug = rna_aug
        self.mol_aug = mol_aug
        self.task = task

        self.data = ds_.reset_index(drop=True).copy()

        self.data["Target_RNA_sequence"] = (
            self.data["Target_RNA_sequence"]
            .str.replace("T", "U")
            .str.replace("Y", "U")
            .str.replace("X", "U")
        )

        self.shuffle_struct = shuffle_struct # whether shuffle and choose randomly RNA 3D structure (for ablation study experimet)
        self.unique_rna_seqs = self.data["Target_RNA_sequence"].unique().tolist()

        seq_to_g_setting = _get_seq_to_g_setting(setting=setting)
        self.seq_to_g = torch.load(seq_to_g_setting, map_location="cpu")

        # load RNA pre-computed embeddings
        self.rna_emb = h5py.File(RNA_EMB_PATH, "r")

        # load canonical SMILES features
        smiles_features_dict = self.open_pkl(SMILES_EMB_PATH)
        self.smiles_morgan_features = smiles_features_dict["morgan fingerprints"]
        self.mordred_descriptors_features = scale_nonzero(smiles_features_dict["mordred descriptors"])

        # canonical sequence-based SMILES embeddings
        self.smiles_chemberta_features = smiles_features_dict["chemberta"]
        self.smiles_simson_features = smiles_features_dict["simson"]

        # lookup dictionaries for faster lookup
        self.morgan_dict = self.df_to_tensor_dict(self.smiles_morgan_features)
        self.mordred_dict = self.df_to_tensor_dict(self.mordred_descriptors_features)

        if self.mol_aug:
            self.smiles2aug = self.open_pkl(SMILES2AUG_PATH)  # canonical SMILES -> list of 10 augmented SMILES

            smiles_seq_aug_emb = self.open_pkl(SMILES_AUG_EMB_PATH)
            self.smiles_chemberta_features = smiles_seq_aug_emb["chemberta"]
            self.smiles_simson_features = smiles_seq_aug_emb["simson"]

            # expand dataset: each row gets duplicated for every augmented SMILES
            self.data = self.expand_with_augmented_smiles(self.data, self.smiles2aug)
        else:
            # no augmentation: use canonical SMILES as the sequence-SMILES used by the row
            self.data["SMILES_aug"] = self.data["SMILES"]

        # lookup dicts for sequence-based encoders
        self.chemberta_dict = self.df_to_tensor_dict(self.smiles_chemberta_features)
        self.simson_dict = self.df_to_tensor_dict(self.smiles_simson_features)

    @staticmethod
    def collate_fn(batch):
        inputs, y = zip(*batch)
        graphs, ernie, rinalmo, seq_len, morgan, mordred, chemberta, simson = zip(*inputs)

        graphs = Batch.from_data_list(list(graphs))
        mordred = torch.stack(mordred, dim=0)
        morgan = torch.stack(morgan, dim=0)
        chemberta = torch.stack(chemberta, dim=0)
        simson = torch.stack(simson, dim=0)
        y = torch.stack(y, dim=0)

        ernie = torch.stack(ernie, dim=0)
        rinalmo = torch.stack(rinalmo, dim=0)
        seq_len = torch.tensor(seq_len)

        inputs = [graphs, ernie, rinalmo, seq_len, morgan, mordred, chemberta, simson]
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
    
    def open_pkl(self, path):
        with open(path, "rb") as f:
            result = pickle.load(f)
        return result

    @staticmethod
    def df_to_tensor_dict(df):
        return {
            row["SMILES"]: torch.tensor(row.iloc[1:].values.astype(np.float32))
            for _, row in df.iterrows()
        }

    @staticmethod
    def expand_with_augmented_smiles(df, smiles2aug, aug_n=3):
        expanded_rows = []

        for _, row in df.iterrows():
            canonical_smiles = row["SMILES"]
            aug_smiles_list = smiles2aug.get(canonical_smiles, [canonical_smiles])
            
            for i, aug_smiles in enumerate(aug_smiles_list):
                new_row = row.copy()
                new_row["SMILES_aug"] = aug_smiles
                expanded_rows.append(new_row)

                if i+1 == aug_n:
                    break

        expanded_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
        return expanded_df

    def get_df(self):
        return self.data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        smiles = row["SMILES"]              # canonical/original SMILES
        smiles_aug = row["SMILES_aug"]      # actual SMILES string used for sequence-based encoders
        rna_seq = row["Target_RNA_sequence"]

        if self.shuffle_struct:
            candidates_rnas = [seq for seq in self.unique_rna_seqs if seq != rna_seq]
            rna_seq_g = random.choice(candidates_rnas)
        # RNA graph
        g_obj = self.seq_to_g[rna_seq_g]
    
        if self.rna_aug:
            # settings (3)/(6): seq_to_g[seq] is a list[Data] of conformers
            g_obj = random.choice(g_obj)
        g = g_obj.clone()

        # RNA embeddings
        curr_rna_emb = self.rna_emb[rna_seq]
        seq_len = int(curr_rna_emb["seq_len"][()])

        ernie = torch.from_numpy(curr_rna_emb["ernie"][:]).float()
        rinalmo = torch.from_numpy(curr_rna_emb["rinalmo"][:]).float()

        if self.is_seq_emb:
            ernie = ernie.mean(dim=0)
            rinalmo = rinalmo.mean(dim=0)

        # small molecule features
        # use augmented SMILES only for sequence-based encoders
        chemberta = self.chemberta_dict[smiles_aug]
        simson = self.simson_dict[smiles_aug]

        # keep molecule-invariant expert features on canonical SMILES
        morgan_fingerprint = self.morgan_dict[smiles]
        mordred_descriptors = self.mordred_dict[smiles]

        y = torch.tensor(row["pKd"], dtype=torch.float32)

        return (g, ernie, rinalmo, seq_len, morgan_fingerprint, mordred_descriptors, chemberta, simson), y




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

    