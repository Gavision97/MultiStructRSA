import torch
import torch.nn as nn

from models.gRNAde import *
from models.model_utils import *
from models.BiDirectionalCrossFusion import BiDirectionalCrossFusionSingle
import logging
logger = logging.getLogger(__name__)

gRNAde_WEIGHT_PATH = "saved_objects/gRNAde_checkpoints/gRNAde_drop3d@0.75_maxlen@500.h5"

BDCF_DROPOUT = 0.1 # BiDirectionalCrossFusion dropout
DROPOUT = 0.1 # all MLP dropout values
LST_DROPOUT = 0.1 # last MLP (affinity head) dropout values

# WITHOUT PADDING
class MultiMRSMA(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        logging.info('MultiStructRSA (frozen) - w 3D struct, w seq & with cross-attention fusion; without padding')
        logger.info(f'hyperparameters -> BiDirectionalCrossFusion dropout:{BDCF_DROPOUT}, Encoder MLPs dropout:{DROPOUT}; Last MLP dropout:{LST_DROPOUT}')
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
        for p in self.W_v.parameters(): p.requires_grad = False
        for p in self.W_e.parameters(): p.requires_grad = False
        for p in self.encoder_layers.parameters(): p.requires_grad = False        
        #################################################################################     
        
        # RiNALMo -> 1280, ERNIE-RNA -> 768; gRNAde -> 128 + 3*16
        self.rna_fusion = BiDirectionalCrossFusionSingle(a_in_dim=768 + 1280, b_in_dim=128 + 48, d_model=128,
                                                         n_heads=4, dropout=BDCF_DROPOUT)
        self.gated_fusion = GatedFeatureFusion(256)
        self.pt_language_gated_fusion = GatedFeatureFusion(128)
        self.bidcf_fated_fuse = GatedFeatureFusion(128)
        self.attn_pooling = AttnPool(128)

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

        # ---- fusion head ----
        # note - without gRNAde substract 128
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

    #########################################################################################
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
    
    #########################################################################################
    def forward(self, graph, ernie_emb, rinalmo_emb, morgan_fp, mordred_desc, chemberta_emb):
        
        # --------------- RNA module --------------- #
        # gRNAde GNN encoder 
        h_V, h_E = (graph.node_s, graph.node_v), (graph.edge_s, graph.edge_v)
        h_V, h_E = self.W_v(h_V), self.W_e(h_E)
        edge_index = graph.edge_index

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)

        h_V, _ = self.pool_multi_conf(h_V, h_E, graph.mask_confs, edge_index)
        encoder_emb = torch.cat([h_V[0], h_V[1].reshape(h_V[1].size(0), -1)], dim=1) # (N, 128+48)
        #encoder_emb = h_V[0]
        # option 1 -> cat(rinalmo, ernie) -> BiDirectionalCrossFusion(..) #
        ernie_emb, rinalmo_emb = ernie_emb.squeeze(0), rinalmo_emb.squeeze(0) # (1, L, |f|) -> (L, |f|)
        rna_emb = torch.cat([rinalmo_emb, ernie_emb], dim=1) # (L, 768 + 1280)
        
        
        # bidirectional RNA fusion 
        rna_emb, nodes_fused = self.rna_fusion(
            A=rna_emb,      # (L, 768) / (L, 1280) or (L, 768 + 1280) if fused
            B=encoder_emb     # (N, 128) gRNAde GNN encoder output
        )

        #rna_vec = self.bidcf_fated_fuse(rna_emb, nodes_fused).mean(dim=0) # (L, 128)
        
        rna_emb = rna_emb.mean(dim=0)   # (L, 128) -> (128,)
        nodes_vec = self.attn_pooling(nodes_fused, graph.batch).squeeze(0)  # (L, 128) -> (128,)
        rna_vec = torch.cat([rna_emb, nodes_vec], dim=0).unsqueeze(0) # (1, 256)

        # --------------- small molecule module ---------------
        expert_smi_emb = torch.cat([morgan_fp, mordred_desc], dim=1) # (1, 1024 + 1205)
        expert_smi_emb = self.expert_mlp(expert_smi_emb) # (1, 128)
        chemberta_mol_emb = self.chemberta_mlp(chemberta_emb) # (1, 128)
        mol_vec = self.mol_mlp(torch.cat([expert_smi_emb, chemberta_mol_emb], dim=1))


        # --------------- gated fusion + prediction module ---------------
        fused_emb = self.gated_fusion(mol_vec, rna_vec)   # (256,)
        pkd = self.head(fused_emb)  # (1,)

        return pkd


# WITH PADDING & BATCH TRAINING
class MultiMRSMABatched(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        
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
        
        logger.info(f'--- Uses gRNAde encoder with finetune the layers & Cross-Attention Fustion (RNA) ---')       

        self.rna_fusion = BiDirectionalCrossFusion(a_in_dim=768, b_in_dim=128, d_model=128, n_heads=4, dropout=0.1)
        self.gated_fusion = GatedFeatureFusion(256)


        self.chemberta_mlp = nn.Sequential(
            nn.Linear(384, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )


        # ---- fusion head ----
        # note - without gRNAde substract 128
        self.head = nn.Sequential(
            nn.Linear(128 + 128, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(LST_DROPOUT),
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(LST_DROPOUT),
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
        mol_vec = self.chemberta_mlp(chemberta_emb) # [B, 128]
 

        # fuse RNA & ligand embeddings and then predict pKd through MLP layer
        #x = torch.cat([mol_vec, rna_vec], dim=1)         # [B, 768]
        fused_emb = self.gated_fusion(mol_vec, rna_vec)
        pkd = self.head(fused_emb) # [B, 1] MLP layer for binding affinity prediciton
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
        
        logger.info(f'--- Uses gRNAde encoder with finetune the layers & Cross-Attention Fustion (RNA) ---')       

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
    
