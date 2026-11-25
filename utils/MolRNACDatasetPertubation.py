import random
import torch

from utils.MolRNACDataset import MolRNADataset


class MolRNADatasetPertubation(MolRNADataset):
    """
    MolRNADataset for the 3D structure perturbation analysis.

    Instead of choosing a random RNA structure independently in __getitem__,
    one complete permutation of the unique RNA structures is used for the
    entire evaluation pass. The permutation is a derangement, so every RNA is
    paired with a different structure and every structure is used exactly once.
    """

    def __init__(self, ds_, setting=1, shuffle_struct=False, rna_aug=False,
                 mol_aug=False, task='perturbation analysis', is_seq_emb=False):
        super().__init__(ds_, setting=setting, shuffle_struct=shuffle_struct,
                         rna_aug=rna_aug, mol_aug=mol_aug, task=task,
                         is_seq_emb=is_seq_emb)

        self.struct_permutation = {}
        if self.shuffle_struct:
            self.permute_structures()

    def permute_structures(self, seed=None):
        """Create one seeded, one-to-one permutation of the RNA structures."""
        if len(self.unique_rna_seqs) < 2:
            raise ValueError("At least two unique RNA structures are required")

        rng = random.Random(seed) if seed is not None else random
        permuted_rna_seqs = self.unique_rna_seqs.copy()

        while True:
            rng.shuffle(permuted_rna_seqs)
            if all(rna_seq != permuted_seq for rna_seq, permuted_seq in
                   zip(self.unique_rna_seqs, permuted_rna_seqs)):
                break

        self.struct_permutation = dict(zip(self.unique_rna_seqs, permuted_rna_seqs))
        return self.struct_permutation

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        smiles = row["SMILES"]
        smiles_aug = row["SMILES_aug"]
        rna_seq = row["Target_RNA_sequence"]

        if self.shuffle_struct:
            if not self.struct_permutation:
                raise RuntimeError("Call permute_structures before evaluation")
            rna_seq_g = self.struct_permutation[rna_seq]
            g_obj = self.seq_to_g[rna_seq_g]
        else:
            g_obj = self.seq_to_g[rna_seq]

        if self.rna_aug:
            g_obj = random.choice(g_obj)
        g = g_obj.clone()

        curr_rna_emb = self.rna_emb[rna_seq]
        seq_len = int(curr_rna_emb["seq_len"][()])

        ernie = torch.from_numpy(curr_rna_emb["ernie"][:]).float()
        rinalmo = torch.from_numpy(curr_rna_emb["rinalmo"][:]).float()

        if self.is_seq_emb:
            ernie = ernie.mean(dim=0)
            rinalmo = rinalmo.mean(dim=0)

        chemberta = self.chemberta_dict[smiles_aug]
        simson = self.simson_dict[smiles_aug]
        morgan_fingerprint = self.morgan_dict[smiles]
        mordred_descriptors = self.mordred_dict[smiles]

        y = torch.tensor(row["pKd"], dtype=torch.float32)

        return (g, ernie, rinalmo, seq_len, morgan_fingerprint,
                mordred_descriptors, chemberta, simson), y
