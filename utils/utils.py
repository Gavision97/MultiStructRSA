import torch
import torch.nn as nn
from rdkit import Chem
import numpy as np
from torch_geometric.utils import softmax

##################################################################################################
#### Gated feature fusion which adapted from BioLMMNet by Abit et al. #### 
#### Github - https://github.com/abrarrahmanabir/BioLLMNet/blob/main/rna_molecule.ipynb ####
class GatedFeatureFusion(nn.Module):
    """Gates to control the contribution of each path"""
    def __init__(self, input_dim):
        super().__init__()
        self.gate = nn.Parameter(torch.rand(input_dim))

    def forward(self, x1, x2):
        # print(x1.shape, x2.shape)
        gate_values = torch.sigmoid(self.gate)
        return x1 * gate_values + x2 * (1 - gate_values)

##################################################################################################
class AttnPool(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

    def forward(self, x, batch):
        # x: (N, dim)
        a = self.score(x).squeeze(-1)          # (N,)
        a = softmax(a, batch)                  # normalize within each graph
        out = torch.zeros(batch.max()+1, x.size(1), device=x.device)
        out.index_add_(0, batch, x * a.unsqueeze(-1))
        return out 
    
##################################################################################################
# custom RMSE loss function
def rmse_loss(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))
##################################################################################################

"""
SMILES randomization adapted from:

Bjerrum, E. J. (2017).
"SMILES enumeration as data augmentation for neural network modeling of molecules."
arXiv:1703.07076

Official implementation:
https://github.com/Ebjerrum/SMILES-enumeration
"""
def randomize_smiles(smiles, canonical=False, isomericSmiles=True):
    """Randomize SMILES atom ordering for data augmentation"""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None

    atom_order = list(range(m.GetNumAtoms()))
    np.random.shuffle(atom_order)

    m = Chem.RenumberAtoms(m, atom_order)
    return Chem.MolToSmiles(
        m,
        canonical=canonical,
        isomericSmiles=isomericSmiles
    )

def enumerate_unique(smiles, n=10):
    """
    Generate up to n unique randomized SMILES for the same molecule
    using atom-order randomization.
    """
    seen = set([smiles])
    while len(seen) < n:
        s = randomize_smiles(smiles)
        if s:
            seen.add(s)
    return list(seen)

#########################################################################################