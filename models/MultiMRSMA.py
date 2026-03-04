import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool

from torch_scatter import scatter_mean
from models.gRNAde import *
#from models.feature_fusion import *
from models.model_utils import *
import logging
logger = logging.getLogger(__name__)

gRNAde_WEIGHT_PATH = "saved_objects/gRNAde_checkpoints/gRNAde_drop3d@0.75_maxlen@500.h5"


# without gRNAde & without RNA cross-attention fusion module
class MultiMRSMA_(nn.Module):
    def __init__(
        self,
        dropout=0.1,
    ):
        super().__init__()
        logger.info(f'--- model w ft gRNAde & w\o RNA cross-attention fusion module ---') 

        pretrained = gRNAde()  # MUST match training defaults (num_layers=4 etc.)
        state_dict = torch.load(gRNAde_WEIGHT_PATH, map_location="cpu")
        pretrained.load_state_dict(state_dict, strict=True)

        self.W_v = pretrained.W_v
        self.W_e = pretrained.W_e
        self.encoder_layers = pretrained.encoder_layers

        # For fine-tuning - unfreeze gRNAde encoder layers (p.requires_grad = True)
        for p in self.W_v.parameters(): p.requires_grad = True
        for p in self.W_e.parameters(): p.requires_grad = True
        for p in self.encoder_layers.parameters(): p.requires_grad = True        
        
        # norm sub-modality embeddings (e.g., smiles -> chemBERTa)
        self.norm_branch = nn.LayerNorm(128)
        self.attn_pool = AttnPool(128)
        
        # ---- SMILES branch ----
        self.morgan_mlp = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256, bias=True)
        )

        self.mordred_mlp = nn.Sequential(
            nn.Linear(1205, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256, bias=True)
        )
        # morgan fingerprints MLP output & chemical descriptors MLP output
        self.expert_mlp = nn.Sequential(
            nn.Linear(256 + 256, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128, bias=True)  
        )

        self.chemberta_mlp = nn.Sequential(
            nn.Linear(384, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128)
        )

        # ---- ERNIE branch ----
        self.ernie_mlp = nn.Sequential(
            nn.Linear(768, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64, bias=True)
        )

        self.smiles_fuse_mlp = nn.Sequential(
            nn.Linear(128 + 128, 128, bias=False), 
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64, bias=True)
        )

        # ---- fusion head ----
        # note - without gRNAde substract 128
        self.affinity_mlp = nn.Sequential(
            nn.Linear(64 + 64, 64, bias=False),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1, bias=True)
        )


    def forward(self, graphs, ernie_emb, morgan_fp, mordred_desc, chemberta_emb):
        """
        graphs: torch_geometric.data.Batch (from Batch.from_data_list)
        ernie_emb: [B, 768]
        morgan_fp: [B, 1024]
        mordred_desc: [B, 1205]
        chemBERTa: [B, 384]
        """
        B = ernie_emb.size(0) # extract batch_size
        L = ernie_emb.size(1) # extract max_seq_len (which is 2922 by default)

        erine_rna_emb = self.ernie_mlp(ernie_emb)                     # [B, 128]
        rna_vec = erine_rna_emb

        # ----- SMILES branch ---- #
        descriptors_emb = self.mordred_mlp(mordred_desc)
        morgan_emb = self.morgan_mlp(morgan_fp)

        expert_emb = torch.cat([descriptors_emb, morgan_emb], dim=1)
        expert_emb = self.expert_mlp(expert_emb) # fuse expert features (i.e., forgan & chemical descriptros)

        chemberta_emb = self.chemberta_mlp(chemberta_emb) # [B, 128]

        # fuse smiles features
        mol_vec = torch.cat([expert_emb, chemberta_emb], dim=1)
        mol_vec = self.smiles_fuse_mlp(mol_vec)

        # fuse RNA & ligand embeddings and then predict pKd through MLP layer
        x = torch.cat([mol_vec, rna_vec], dim=1)         # [B, 768]
        pkd = self.affinity_mlp(x) # [B, 1] MLP layer for binding affinity prediciton
        return pkd

class MultiMRSMA_(nn.Module):
    def __init__(
        self,
        dropout=0.1,
    ):
        super().__init__()
        logger.info(f'--- model w ft gRNAde & w\o RNA cross-attention fusion module ---') 

        pretrained = gRNAde()  # MUST match training defaults (num_layers=4 etc.)
        state_dict = torch.load(gRNAde_WEIGHT_PATH, map_location="cpu")
        pretrained.load_state_dict(state_dict, strict=True)

        self.W_v = pretrained.W_v
        self.W_e = pretrained.W_e
        self.encoder_layers = pretrained.encoder_layers

        # For fine-tuning - unfreeze gRNAde encoder layers (p.requires_grad = True)
        for p in self.W_v.parameters(): p.requires_grad = True
        for p in self.W_e.parameters(): p.requires_grad = True
        for p in self.encoder_layers.parameters(): p.requires_grad = True        
        
        # norm sub-modality embeddings (e.g., smiles -> chemBERTa)
        self.norm_branch = nn.LayerNorm(128)
        self.attn_pool = AttnPool(128)
        # ---- SMILES branch ----
        self.morgan_mlp = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256, bias=True)
        )

        self.mordred_mlp = nn.Sequential(
            nn.Linear(1205, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256, bias=True)
        )
        # morgan fingerprints MLP output & chemical descriptors MLP output
        self.expert_mlp = nn.Sequential(
            nn.Linear(256 + 256, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128, bias=True)  
        )

        self.chemberta_mlp = nn.Sequential(
            nn.Linear(384, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128)
        )

        # ---- ERNIE branch ----
        self.ernie_mlp = nn.Sequential(
            nn.Linear(768, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128, bias=True)
        )

        self.smiles_fuse_mlp = nn.Sequential(
            nn.Linear(128 + 128, 128, bias=False), 
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64, bias=True)
        )

        self.rna_fuse_mlp = nn.Sequential(
            nn.Linear(128 + 128, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64, bias=True)
        )
      
        # ---- fusion head ----
        # note - without gRNAde substract 128
        self.affinity_mlp = nn.Sequential(
            nn.Linear(64 + 64, 64, bias=False),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1, bias=True)
        )

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
        morgan_fp: [B, 1024]
        mordred_desc: [B, 1205]
        chemBERTa: [B, 384]
        """
        B = ernie_emb.size(0) # extract batch_size
        L = ernie_emb.size(1) # extract max_seq_len (which is 2922 by default)


        # ----- gRNAde encoder -----
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

        graph_emb_mean = self.attn_pool(node_scalar, graphs.batch)  # shape -> (B, node_h_dim[0]=128)
        #graph_emb_max = global_max_pool(node_scalar, graphs.batch) # shape -> (B, node_h_dim[0]=128)

        erine_rna_emb = self.ernie_mlp(ernie_emb)                     # [B, 128]
        

        # fuse RNA features
        rna_vec = torch.cat([erine_rna_emb, graph_emb_mean], dim=1)  # (B, 128 + 128 + 128)
        rna_vec = self.rna_fuse_mlp(rna_vec)

        # ----- SMILES branch ---- #
        descriptors_emb = self.mordred_mlp(mordred_desc)
        morgan_emb = self.morgan_mlp(morgan_fp)

        expert_emb = torch.cat([descriptors_emb, morgan_emb], dim=1)
        expert_emb = self.expert_mlp(expert_emb) # fuse expert features (i.e., forgan & chemical descriptros)

        chemberta_emb = self.chemberta_mlp(chemberta_emb) # [B, 128]

        # fuse smiles features
        mol_vec = torch.cat([expert_emb, chemberta_emb], dim=1)
        mol_vec = self.smiles_fuse_mlp(mol_vec)

        # fuse RNA & ligand embeddings and then predict pKd through MLP layer
        x = torch.cat([mol_vec, rna_vec], dim=1)         # [B, 768]
        pkd = self.affinity_mlp(x) # [B, 1] MLP layer for binding affinity prediciton
        return pkd


class MultiMRSMA_(nn.Module):
    def __init__(
        self,
        dropout=0.1,
    ):
        super().__init__()
        
        pretrained = gRNAde()  # MUST match training defaults (num_layers=4 etc.)
        state_dict = torch.load(gRNAde_WEIGHT_PATH, map_location="cpu")
        pretrained.load_state_dict(state_dict, strict=True)

        self.W_v = pretrained.W_v
        self.W_e = pretrained.W_e
        self.encoder_layers = pretrained.encoder_layers

        for p in self.W_v.parameters(): p.requires_grad = True
        for p in self.W_e.parameters(): p.requires_grad = True
        for p in self.encoder_layers.parameters(): p.requires_grad = True        
        
        self.rna_fusion = BiDirectionalCrossFusion(a_in_dim=768, b_in_dim=128, d_model=128, n_heads=4, dropout=0.1)

        #logger.info(f'---Without using gRNAde (dropout={dropout}) ---')       
        # ---- SMILES branch ----
        self.mfp_cd_mlp = nn.Sequential(
            nn.Linear(1024 + 194, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128, bias=True)  
        )

        self.chemberta_mlp = nn.Sequential(
            nn.Linear(384, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128, bias=True)
        )

        # ---- ERNIE branch ----
        self.ernie_mlp = nn.Sequential(
            nn.Linear(768, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128, bias=True),
        )      
        # ---- fusion head ----
        # note - without gRNAde substract 128
        self.head = nn.Sequential(
            nn.Linear(128 + 128 + 128 + 128, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64, bias=False),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

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

    def forward(self, graphs, ernie_emb, seq_lens, morgan_fp, chemical_desc, chemberta_emb):
        """
        graphs: torch_geometric.data.Batch (from Batch.from_data_list)
        ernie_emb: [B, 768]
        morgan_fp: [B, 1024]
        chemical_desc: [B, 194]
        """
        B = ernie_emb.size(0) # extract batch_size
        L = ernie_emb.size(1) # extract max_seq_len (which is 2922 by default)


        # ----- gRNAde encoder -----
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

        # ----- RNA feature fusion ------ #

        ernie_pad_mask = torch.arange(L, device=ernie_emb.device).unsqueeze(0) >= seq_lens.unsqueeze(1)
        nodes_padded, node_pad_mask, _ = pad_nodes_by_graph(node_scalar, graphs.batch, B)

        # ernie fused -> torch.Size([16, 2922, 128]), nodes_fused -> torch.Size([16, N_total_nodes, 128])
        ernie_fused, nodes_fused = self.rna_fusion(A=ernie_emb, B=nodes_padded, A_pad_mask=ernie_pad_mask, B_pad_mask=node_pad_mask)

        #print(f'ernie fused -> {ernie_fused.shape}, nodes_fused -> {nodes_fused.shape}')

        ernie_vec = masked_mean(ernie_fused, ernie_pad_mask) # shape -> from (B, max_seq_len=2922, 128) to (B, 128)
        nodes_vec = masked_mean(nodes_fused, node_pad_mask) # shape -> from (B, N_total_nodes, 128) to (B, 128)
        
        rna_vec = torch.cat([ernie_vec, nodes_vec], dim=1)  # (B, 128 + 128)

        # ----- SMILES branch -----
        mol = torch.cat([morgan_fp, chemical_desc], dim=1)  # [B, 1218]
        mol = self.mfp_cd_mlp(mol) # [B, 128]
        chemberta_emb = self.chemberta_mlp(chemberta_emb) # [B, 128]
        mol_vec = torch.cat([mol, chemberta_emb], dim=1) # B[B, 128 + 128]
        

        # fuse RNA & ligand embeddings and then predict pKd through MLP layer
        x = torch.cat([mol_vec, rna_vec], dim=1)         # [B, 768]
        pkd = self.head(x) # [B, 1] MLP layer for binding affinity prediciton
        return pkd
    

# without gRNAde & without RNA cross-attention fusion module
class MultiMRSMA_(nn.Module):
    def __init__(
        self,
        dropout=0.1,
    ):
        super().__init__()
            
        logger.info(f'--- POC - w/o gRNAde encoder  (BioLLMNet style  26/02) ---')       
        self.gated_fusion = GatedFeatureFusion(256)

        self.chemberta_mlp = nn.Sequential(
            nn.Linear(384, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.ernie_mlp = nn.Sequential(
            nn.Linear(768, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
      
        # ---- binding affinity mlp ----
        self.head = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
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


class MultiMRSMA(nn.Module):
    def __init__(
        self,
        dropout=0.1,
    ):
        super().__init__()
        logger.info(f'--- POC - model w ft gRNAde (BioLLMNet style  26/02) gate & gate ---') 

        pretrained = gRNAde()  # MUST match training defaults (num_layers=4 etc.)
        state_dict = torch.load(gRNAde_WEIGHT_PATH, map_location="cpu")
        pretrained.load_state_dict(state_dict, strict=True)

        self.gated_fusion = GatedFeatureFusion(128)
        self.gated_rna_fusion = GatedFeatureFusion(128)

        # gRNAde encoder layers (for fine-tuning)
        self.W_v = pretrained.W_v
        self.W_e = pretrained.W_e
        self.encoder_layers = pretrained.encoder_layers

    
        # For fine-tuning - unfreeze gRNAde encoder layers (p.requires_grad = True)
        for p in self.W_v.parameters(): p.requires_grad = True
        for p in self.W_e.parameters(): p.requires_grad = True
        for p in self.encoder_layers.parameters(): p.requires_grad = True        
        
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

        self.mordred_mlp = nn.Sequential(
            nn.Linear(1205, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.fp_mlp = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

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
        #fp_smi_emb = self.fp_mlp(morgan_fp)
        #desc_smi_emb = self.mordred_mlp(mordred_desc)

        expert_smi_emb = torch.cat([morgan_fp, mordred_desc], dim=1)
        expert_smi_emb = self.expert_mlp(expert_smi_emb)
        chemberta_mol_emb = self.chemberta_mlp(chemberta_emb) # [B, 128]
        mol_emb = self.mol_mlp(torch.cat([expert_smi_emb, chemberta_mol_emb], dim=1))

        # fuse RNA & ligand embeddings and then predict pKd through MLP layer
        #x = torch.cat([mol_emb, rna_emb], dim=1)         # [B, 768]
        x = self.gated_fusion(mol_emb, rna_emb)
        pkd = self.head(x) # [B, 1] MLP layer for binding affinity prediciton
        return pkd


'''
class MultiMRSMA(nn.Module):
    def __init__(
        self,
        dropout=0.1,
    ):
        super().__init__()
        logger.info(f'--- POC - model w ft gRNAde (BioLLMNet style  26/02) gate & gate ---') 

        pretrained = gRNAde()  # MUST match training defaults (num_layers=4 etc.)
        state_dict = torch.load(gRNAde_WEIGHT_PATH, map_location="cpu")
        pretrained.load_state_dict(state_dict, strict=True)

        self.gated_fusion = GatedFeatureFusion(128)
        self.gated_rna_fusion = GatedFeatureFusion(128)

        # gRNAde encoder layers (for fine-tuning)
        self.W_v = pretrained.W_v
        self.W_e = pretrained.W_e
        self.encoder_layers = pretrained.encoder_layers

    
        # For fine-tuning - unfreeze gRNAde encoder layers (p.requires_grad = True)
        for p in self.W_v.parameters(): p.requires_grad = True
        for p in self.W_e.parameters(): p.requires_grad = True
        for p in self.encoder_layers.parameters(): p.requires_grad = True        
        
        # learnable global attention pooling for gRNAde GNN encoder output 
        self.attn_pooling = AttnPool(128)

        self.chemberta_mlp = nn.Sequential(
            nn.Linear(384, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        
        self.ernie_mlp = nn.Sequential(
            nn.Linear(768, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.mordred_mlp = nn.Sequential(
            nn.Linear(1205, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.fp_mlp = nn.Sequential(
            nn.Sequential(1024, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128, bias=False),
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
        fp_smi_emb = self.fp_mlp(morgan_fp)
        desc_smi_emb = self.mordred_mlp(mordred_desc)
        expert_smi_emb = torch.cat([fp_smi_emb, desc_smi_emb], dim=1)
        self.
        chemberta_mol_emb = self.chemberta_mlp(chemberta_emb) # [B, 256]
        

        # fuse RNA & ligand embeddings and then predict pKd through MLP layer
        #x = torch.cat([mol_emb, rna_emb], dim=1)         # [B, 768]
        x = self.gated_fusion(mol_emb, rna_emb)
        pkd = self.head(x) # [B, 1] MLP layer for binding affinity prediciton
        return pkd


'''