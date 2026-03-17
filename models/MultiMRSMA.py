import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool

from torch_scatter import scatter_mean
from models.gRNAde import *
from utils.utils import *

import logging
logger = logging.getLogger(__name__)

gRNAde_WEIGHT_PATH = "saved_objects/gRNAde_checkpoints/gRNAde_drop3d@0.75_maxlen@500.h5"

BDCF_DROPOUT = 0.1 # BiDirectionalCrossFusion dropout
DROPOUT = 0.1 # all MLP dropout values
LST_DROPOUT = 0.1 # last MLP (affinity head) dropout values

# without gRNAde & without RNA cross-attention fusion module
class MultiMRSMA_(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
            
        logger.info(f'--- w/o 3D stuct (gRNAde encoder) ---')       
        self.gated_fusion = GatedFeatureFusion(128)
        self.pt_language_gated_fusion = GatedFeatureFusion(128)

        self.chemberta_mlp = nn.Sequential(
            nn.Linear(384, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

        # cat -> fuse; got best results from 03/03 exp3
        self.expert_mlp = nn.Sequential(
            nn.Linear(1024 + 1205, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

        self.mol_mlp = nn.Sequential(
            nn.Linear(2*128, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

        self.rna_mlp = nn.Sequential(
            nn.Linear(768, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

        self.rna_proj = nn.Sequential(
            nn.Linear(768 + 1280, 128, bias=False),
            nn.LayerNorm(128)
        )
      
        # ---- binding affinity mlp ----
        self.head = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(LST_DROPOUT),
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(LST_DROPOUT),
            nn.Linear(32, 1, bias=True)
        )

    def forward(self, graph, ernie_emb, rinalmo_emb, morgan_fp, mordred_desc, chemberta_emb):
        """
        graphs: torch_geometric.data.Batch (from Batch.from_data_list)
        ernie_emb: [B, 768]
        morgan_fp: [B, 1024]
        chemical_desc: [B, 194]
        """

        # ----- RNA branch -----
        rna_emb = self.rna_proj(torch.cat([ernie_emb, rinalmo_emb], dim=1)) # linear -> layer norm
        #rna_emb = self.rna_mlp(ernie_emb)
        # ----- SMILES branch -----
        expert_smi_emb = torch.cat([morgan_fp, mordred_desc], dim=1)
        expert_smi_emb = self.expert_mlp(expert_smi_emb)
        chemberta_mol_emb = self.chemberta_mlp(chemberta_emb) # [B, 128]
        mol_emb = self.mol_mlp(torch.cat([expert_smi_emb, chemberta_mol_emb], dim=1))
        
        # fuse RNA & ligand embeddings and then predict pKd through MLP layer
        #x = torch.cat([mol_emb, rna_emb], dim=1)         # [B, 256]
        x = self.gated_fusion(mol_emb, rna_emb)
        pkd = self.head(x) # [B, 1] MLP layer for binding affinity prediciton
        return pkd


class MultiMRSMA_(nn.Module):
    def __init__(self):
        super().__init__()
        logger.info(f'--- POC - w/o sequence (rinalmo & ernie-rna)---') 

        self.gated_fusion = GatedFeatureFusion(128)
        self.gated_rna_fusion = GatedFeatureFusion(128)
        self.pt_language_gated_fusion = GatedFeatureFusion(128)

        #################################################################################3
        # adapted from gRNAde framework - https://github.com/chaitjo/geometric-rna-design.git
        # gRNAde encoder layers (for fine-tuning)

        pretrained = gRNAde()  # MUST match training defaults (num_layers=4 etc.)
        state_dict = torch.load(gRNAde_WEIGHT_PATH, map_location="cpu")
        pretrained.load_state_dict(state_dict, strict=True)

        self.W_v = pretrained.W_v
        self.W_e = pretrained.W_e
        self.encoder_layers = pretrained.encoder_layers
    
        # for fine-tuning - unfreeze gRNAde encoder layers (p.requires_grad = True)
        for p in self.W_v.parameters(): p.requires_grad = True
        for p in self.W_e.parameters(): p.requires_grad = True
        for p in self.encoder_layers.parameters(): p.requires_grad = True        
        #################################################################################

        # learnable global attention pooling for gRNAde GNN encoder output 
        self.attn_pooling = AttnPool(128)

        self.struct_proj = nn.Sequential(
            nn.Linear(176, 128, bias=False),
            nn.LayerNorm(128)
        )

        self.chemberta_mlp = nn.Sequential(
            nn.Linear(384, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

        # cat -> fuse; got best results from 03/03 exp3
        self.expert_mlp = nn.Sequential(
            nn.Linear(1024 + 1205, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

        self.mol_mlp = nn.Sequential(
            nn.Linear(2*256, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

        # ---- binding affinity mlp layer ----
        self.head = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(LST_DROPOUT),
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(LST_DROPOUT),
            nn.Linear(32, 1, bias=True)
        )
    
    # adapted from gRNAde framework - https://github.com/chaitjo/geometric-rna-design.git
    def pool_multi_conf(self, h_V, h_E, mask_confs, edge_index):
        # paste your gRNAde pool_multi_conf here exactly (same as you already have)
        if mask_confs.size(1) == 1:
            return (h_V[0][:, 0], h_V[1][:, 0]), (h_E[0][:, 0], h_E[1][:, 0])

        n_conf_true = mask_confs.sum(1, keepdim=True)

        mask = mask_confs.unsqueeze(2)
        h_V0 = h_V[0] * mask
        h_E0 = h_E[0] * mask[edge_index[0]]

        mask = mask.unsqueeze(3)
        h_V1 = h_V[1] * mask
        h_E1 = h_E[1] * mask[edge_index[0]]

        h_V = (
            h_V0.sum(dim=1) / n_conf_true,
            h_V1.sum(dim=1) / n_conf_true.unsqueeze(2),
        )
        h_E = (
            h_E0.sum(dim=1) / n_conf_true[edge_index[0]],
            h_E1.sum(dim=1) / n_conf_true[edge_index[0]].unsqueeze(2),
        )
        return h_V, h_E

    def forward(self, graph, ernie_emb, rinalmo_emb, morgan_fp, mordred_desc, chemberta_emb):
        """
        graphs: torch_geometric.data.Batch (from Batch.from_data_list)
        ernie_emb: [B, 768]
        chemBERTa: [B, 368]
        """
        
        # ----- gRNAde encoder; adapted from https://github.com/chaitjo/geometric-rna-design.git ----- #
        # outputs: node_scalar (N_total_nodes, 128)
        # emb(node)=|128| embedding for every node in graph G in batch B
        h_V = (graph.node_s, graph.node_v)
        h_E = (graph.edge_s, graph.edge_v)
        edge_index = graph.edge_index

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)

        # pool over conformers -> per-node
        h_V, h_E = self.pool_multi_conf(h_V, h_E, graph.mask_confs, edge_index)
        
        #node_scalar, node_vec = h_V[0], h_V[1].reshape(h_V[1].size(0), -1)  # (N, 128), (N, 16*3=48)
        node_scalar = self.struct_proj(torch.cat([h_V[0], h_V[1].reshape(h_V[1].size(0), -1)], dim=1)) # (N, 128+48)
        graph_emb = self.attn_pooling(node_scalar, graph.batch)  # shape -> (B, node_h_dim[0]=128)
        rna_emb = graph_emb

        # ----- SMILES branch -----
        expert_smi_emb = torch.cat([morgan_fp, mordred_desc], dim=1)
        expert_smi_emb = self.expert_mlp(expert_smi_emb)
        chemberta_mol_emb = self.chemberta_mlp(chemberta_emb) # [B, 128]
        mol_emb = self.mol_mlp(torch.cat([expert_smi_emb, chemberta_mol_emb], dim=1))

        # fuse RNA & ligand embeddings and then predict pKd through MLP layer
        #x = torch.cat([mol_emb, rna_emb], dim=1)         # [B, 768]
        x = self.gated_fusion(mol_emb, rna_emb)
        pkd = self.head(x) # [B, 1] MLP layer for binding affinity prediciton
        return pkd

class MultiMRSMA(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        logger.info(f'--- POC - w/o fusion (BDCF; just cat(mlm, gnn))---') 

        self.gated_fusion = GatedFeatureFusion(256)
        self.gated_rna_fusion = GatedFeatureFusion(128)
        self.pt_language_gated_fusion = GatedFeatureFusion(128)

        #################################################################################3
        # adapted from gRNAde framework - https://github.com/chaitjo/geometric-rna-design.git
        # gRNAde encoder layers (for fine-tuning)

        pretrained = gRNAde()  # MUST match training defaults (num_layers=4 etc.)
        state_dict = torch.load(gRNAde_WEIGHT_PATH, map_location="cpu")
        pretrained.load_state_dict(state_dict, strict=True)

        self.W_v = pretrained.W_v
        self.W_e = pretrained.W_e
        self.encoder_layers = pretrained.encoder_layers
    
        # for fine-tuning - unfreeze gRNAde encoder layers (p.requires_grad = True)
        for p in self.W_v.parameters(): p.requires_grad = True
        for p in self.W_e.parameters(): p.requires_grad = True
        for p in self.encoder_layers.parameters(): p.requires_grad = True        
        #################################################################################

        # learnable global attention pooling for gRNAde GNN encoder output 
        self.attn_pooling = AttnPool(128)

        self.struct_proj = nn.Sequential(
            nn.Linear(176, 128, bias=False),
            nn.LayerNorm(128)
        )

        self.rna_proj = nn.Sequential(
            nn.Linear(768 + 1280, 128, bias=False),
            nn.LayerNorm(128)
        )

        self.chemberta_mlp = nn.Sequential(
            nn.Linear(384, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

        # cat -> fuse; got best results from 03/03 exp3
        self.expert_mlp = nn.Sequential(
            nn.Linear(1024 + 1205, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

        self.mol_mlp = nn.Sequential(
            nn.Linear(2*256, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

        # ---- binding affinity mlp layer ----
        self.head = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(LST_DROPOUT),
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(LST_DROPOUT),
            nn.Linear(64, 1, bias=True)
        )
    
    # adapted from gRNAde framework - https://github.com/chaitjo/geometric-rna-design.git
    def pool_multi_conf(self, h_V, h_E, mask_confs, edge_index):
        # paste your gRNAde pool_multi_conf here exactly (same as you already have)
        if mask_confs.size(1) == 1:
            return (h_V[0][:, 0], h_V[1][:, 0]), (h_E[0][:, 0], h_E[1][:, 0])

        n_conf_true = mask_confs.sum(1, keepdim=True)

        mask = mask_confs.unsqueeze(2)
        h_V0 = h_V[0] * mask
        h_E0 = h_E[0] * mask[edge_index[0]]

        mask = mask.unsqueeze(3)
        h_V1 = h_V[1] * mask
        h_E1 = h_E[1] * mask[edge_index[0]]

        h_V = (
            h_V0.sum(dim=1) / n_conf_true,
            h_V1.sum(dim=1) / n_conf_true.unsqueeze(2),
        )
        h_E = (
            h_E0.sum(dim=1) / n_conf_true[edge_index[0]],
            h_E1.sum(dim=1) / n_conf_true[edge_index[0]].unsqueeze(2),
        )
        return h_V, h_E

    def forward(self, graph, ernie_emb, rinalmo_emb, morgan_fp, mordred_desc, chemberta_emb):
        """
        graphs: torch_geometric.data.Batch (from Batch.from_data_list)
        ernie_emb: [B, 768]
        chemBERTa: [B, 368]
        """
        
        # ----- gRNAde encoder; adapted from https://github.com/chaitjo/geometric-rna-design.git ----- #
        # outputs: node_scalar (N_total_nodes, 128)
        # emb(node)=|128| embedding for every node in graph G in batch B
        h_V = (graph.node_s, graph.node_v)
        h_E = (graph.edge_s, graph.edge_v)
        edge_index = graph.edge_index

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)

        # pool over conformers -> per-node
        h_V, _ = self.pool_multi_conf(h_V, h_E, graph.mask_confs, edge_index)
        #node_scalar, node_vec = h_V[0], h_V[1].reshape(h_V[1].size(0), -1)  # (N, 128), (N, 16*3=48)
        node_scalar = self.struct_proj(torch.cat([h_V[0], h_V[1].reshape(h_V[1].size(0), -1)], dim=1)) # (N, 128+48)
        graph_emb = self.attn_pooling(node_scalar, graph.batch)  # shape -> (B, node_h_dim[0]=128)

        rna_mlm_emb = self.rna_proj(torch.cat([ernie_emb, rinalmo_emb], dim=1)) # (1, 128)
            
        # ----- RNA feature fusion ------ #
        rna_emb = torch.cat([rna_mlm_emb, graph_emb], dim=1)  # (B, 128 + 128)
        #rna_emb = self.gated_rna_fusion(rna_mlm_emb, graph_emb)
        
        # ----- SMILES branch -----
        expert_smi_emb = torch.cat([morgan_fp, mordred_desc], dim=1)
        expert_smi_emb = self.expert_mlp(expert_smi_emb)
        chemberta_mol_emb = self.chemberta_mlp(chemberta_emb) # [B, 128]
        mol_emb = self.mol_mlp(torch.cat([expert_smi_emb, chemberta_mol_emb], dim=1))

        # fuse RNA & ligand embeddings and then predict pKd through MLP layer
        #x = torch.cat([mol_emb, rna_emb], dim=1)         # [B, 768]
        x = self.gated_fusion(mol_emb, rna_emb)
        pkd = self.head(x) # [B, 1] MLP layer for binding affinity prediciton
        return pkd

# without gRNAde & without RNA cross-attention fusion module
class MultiMRSMA_(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
            
        logger.info(f'--- POC - w/o gRNAde encoder  (BioLLMNet style  26/02) ---')       
        self.gated_fusion = GatedFeatureFusion(256)

        self.chemberta_mlp = nn.Sequential(
            nn.Linear(384, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

        # cat -> fuse; got best results from 03/03 exp3
        self.expert_mlp = nn.Sequential(
            nn.Linear(1024 + 1205, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )


        self.mol_mlp = nn.Sequential(
            nn.Linear(2*128, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.ernie_mlp = nn.Sequential(
            nn.Linear(768, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )
      
        # ---- binding affinity mlp ----
        self.head = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(LST_DROPOUT),
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(LST_DROPOUT),
            nn.Linear(64, 1, bias=True)
        )

    def forward(self, graphs, ernie_emb, morgan_fp, chemical_desc, chemberta_emb):
        """
        graphs: torch_geometric.data.Batch (from Batch.from_data_list)
        ernie_emb: [B, 768]
        morgan_fp: [B, 1024]
        chemical_desc: [B, 194]
        """

        # ----- SMILES branch -----
        mol_emb = self.chemberta_mlp(chemberta_emb) # [B, 256]
        
        # ----- RNA branch -----
        rna_emb = self.ernie_mlp(ernie_emb)# [B, 256]

        # fuse RNA & ligand embeddings and then predict pKd through MLP layer
        #x = torch.cat([mol_emb, rna_emb], dim=1)         # [B, 256]
        x = self.gated_fusion(mol_emb, rna_emb)
        pkd = self.head(x) # [B, 1] MLP layer for binding affinity prediciton
        return pkd


class MultiMRSMA_(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        logger.info(f'--- POC - model w ft gRNAde (BioLLMNet style  26/02) gate & gate ---') 



        self.gated_fusion = GatedFeatureFusion(128)
        self.gated_rna_fusion = GatedFeatureFusion(128)

        #################################################################################3
        # adapted from gRNAde framework - https://github.com/chaitjo/geometric-rna-design.git
        # gRNAde encoder layers (for fine-tuning)

        pretrained = gRNAde()  # MUST match training defaults (num_layers=4 etc.)
        state_dict = torch.load(gRNAde_WEIGHT_PATH, map_location="cpu")
        pretrained.load_state_dict(state_dict, strict=True)

        self.W_v = pretrained.W_v
        self.W_e = pretrained.W_e
        self.encoder_layers = pretrained.encoder_layers
    
        # for fine-tuning - unfreeze gRNAde encoder layers (p.requires_grad = True)
        for p in self.W_v.parameters(): p.requires_grad = True
        for p in self.W_e.parameters(): p.requires_grad = True
        for p in self.encoder_layers.parameters(): p.requires_grad = True        
        #################################################################################

        # learnable global attention pooling for gRNAde GNN encoder output 
        self.attn_pooling = AttnPool(128)

        self.chemberta_mlp = nn.Sequential(
            nn.Linear(384, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        
        self.ernie_mlp = nn.Sequential(
            nn.Linear(768, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # cat -> fuse; got best results from 03/03 exp3
        self.expert_mlp = nn.Sequential(
            nn.Linear(1024 + 1205, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )


        self.mol_mlp = nn.Sequential(
            nn.Linear(2*128, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # ---- binding affinity mlp layer ----
        self.head = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1, bias=True)
        )
    
    # adapted from gRNAde framework - https://github.com/chaitjo/geometric-rna-design.git
    def pool_multi_conf(self, h_V, h_E, mask_confs, edge_index):
        # paste your gRNAde pool_multi_conf here exactly (same as you already have)
        if mask_confs.size(1) == 1:
            return (h_V[0][:, 0], h_V[1][:, 0]), (h_E[0][:, 0], h_E[1][:, 0])

        n_conf_true = mask_confs.sum(1, keepdim=True)

        mask = mask_confs.unsqueeze(2)
        h_V0 = h_V[0] * mask
        h_E0 = h_E[0] * mask[edge_index[0]]

        mask = mask.unsqueeze(3)
        h_V1 = h_V[1] * mask
        h_E1 = h_E[1] * mask[edge_index[0]]

        h_V = (
            h_V0.sum(dim=1) / n_conf_true,
            h_V1.sum(dim=1) / n_conf_true.unsqueeze(2),
        )
        h_E = (
            h_E0.sum(dim=1) / n_conf_true[edge_index[0]],
            h_E1.sum(dim=1) / n_conf_true[edge_index[0]].unsqueeze(2),
        )
        return h_V, h_E

    def forward(self, graphs, ernie_emb, morgan_fp, mordred_desc, chemberta_emb):
        """
        graphs: torch_geometric.data.Batch (from Batch.from_data_list)
        ernie_emb: [B, 768]
        chemBERTa: [B, 368]
        """
        
        # ----- gRNAde encoder; adapted from https://github.com/chaitjo/geometric-rna-design.git ----- #
        # outputs: node_scalar (N_total_nodes, 128)
        # emb(node)=|128| embedding for every node in graph G in batch B
        h_V = (graphs.node_s, graphs.node_v)
        h_E = (graphs.edge_s, graphs.edge_v)
        edge_index = graphs.edge_index

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)

        # pool over conformers -> per-node
        h_V, h_E = self.pool_multi_conf(h_V, h_E, graphs.mask_confs, edge_index)
        node_scalar = h_V[0]  # shape ->  (N_total_nodes, 128)
        graph_emb = self.attn_pooling(node_scalar, graphs.batch)  # shape -> (B, node_h_dim[0]=128)


        ernie_rna_emb = self.ernie_mlp(ernie_emb)                     # [B, 128]
        
        # ----- RNA feature fusion ------ #
        #rna_emb = torch.cat([ernie_rna_emb, graph_emb], dim=1)  # (B, 128 + 128)
        rna_emb = self.gated_rna_fusion(ernie_rna_emb, graph_emb)
        
        # ----- SMILES branch -----
        expert_smi_emb = torch.cat([morgan_fp, mordred_desc], dim=1)
        expert_smi_emb = self.expert_mlp(expert_smi_emb)
        chemberta_mol_emb = self.chemberta_mlp(chemberta_emb) # [B, 128]
        mol_emb = self.mol_mlp(torch.cat([expert_smi_emb, chemberta_mol_emb], dim=1))

        # fuse RNA & ligand embeddings and then predict pKd through MLP layer
        #x = torch.cat([mol_emb, rna_emb], dim=1)         # [B, 768]
        x = self.gated_fusion(mol_emb, rna_emb)
        pkd = self.head(x) # [B, 1] MLP layer for binding affinity prediciton
        return pkd
