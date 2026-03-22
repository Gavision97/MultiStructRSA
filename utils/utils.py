import torch
import torch.nn as nn
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