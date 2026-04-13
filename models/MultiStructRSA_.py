
import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

#######################################################################################
# adapted from gRNAde framework - https://github.com/chaitjo/geometric-rna-design.git #
from models.gRNAde import * 
#######################################################################################

from utils.utils import *
from models.BiDirectionalCrossFusion import *


BDCF_DROPOUT = 0.1 # BiDirectionalCrossFusion dropout
DROPOUT = 0.1 # all MLP dropout values
LST_DROPOUT = 0.1 # last MLP (affinity head) dropout values

gRNAde_WEIGHT_PATH = "saved_objects/gRNAde_checkpoints/gRNAde_drop3d@0.75_maxlen@500.h5"
 

def choose_model_setting(option: str = "full"):
    assert option in ["full", "freezed", "w/o seq", "w/o 3D struct", "w/o fusion"]

    model_map = {
        "full": MultiStructRSAFull,
        "freezed": MultiStructRSAFreezed,
        "w/o seq": MultiStructRSAWithoutSeq,
        "w/o 3D struct": MultiStructRSAWithout3D,
        "w/o fusion": MultiStructRSAWithoutFusion,
    }

    model = model_map[option]()
    logger.info(f"Initialized model setting: {option}")
    return model


class MultiStructRSA(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @staticmethod
    def pool_multi_conf(h_V, h_E, mask_confs, edge_index):
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

    def _load_grnade_encoder(self, freeze: bool = False):
        pretrained = gRNAde()
        state_dict = torch.load(gRNAde_WEIGHT_PATH, map_location="cpu")
        pretrained.load_state_dict(state_dict, strict=True)

        self.W_v = pretrained.W_v
        self.W_e = pretrained.W_e
        self.encoder_layers = pretrained.encoder_layers

        requires_grad = not freeze
        for p in self.W_v.parameters():
            p.requires_grad = requires_grad
        for p in self.W_e.parameters():
            p.requires_grad = requires_grad
        for p in self.encoder_layers.parameters():
            p.requires_grad = requires_grad

    def _encode_3d_structure(self, graph):
        h_V, h_E = (graph.node_s, graph.node_v), (graph.edge_s, graph.edge_v)
        edge_index = graph.edge_index

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)

        h_V, h_E = self.pool_multi_conf(h_V, h_E, graph.mask_confs, edge_index)
        return h_V, h_E, edge_index

    @abstractmethod
    def forward(self, graph, ernie_emb, rinalmo_emb, morgan_fp, mordred_desc, chemberta_emb):
        pass


# =========================================================
# FULL model
# =========================================================
class MultiStructRSAFull(MultiStructRSA):
    def __init__(self):
        super().__init__()
        logging.info("MultiStructRSA - full: w 3D struct, w seq, w cross-attention fusion")
        logger.info(
            f"hyperparameters -> BiDirectionalCrossFusion dropout:{BDCF_DROPOUT}, "
            f"Encoder MLPs dropout:{DROPOUT}; Last MLP dropout:{LST_DROPOUT}"
        )

        self._load_grnade_encoder(freeze=False)

        self.rna_fusion = BiDirectionalCrossFusionSingle(
            a_in_dim=768 + 1280, # |ERNIE-RNA| & |RiNALMo|
            b_in_dim=128 + 48, # gRNAde GNN |node features| & |node vectors|
            d_model=128,
            n_heads=4,
            dropout=BDCF_DROPOUT
        )
        self.gated_fusion = GatedFeatureFusion(256)
        self.attn_pooling = AttnPool(128)

        self.chemberta_mlp = nn.Sequential(
            nn.Linear(384, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

        self.smiles_seq_mlp = nn.Sequential(
            nn.Linear(384 + 512, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 512, bias=False),
        )

        self.expert_mlp = nn.Sequential(
            nn.Linear(1024 + 1205, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 256, bias=True),
        )

        self.mol_mlp = nn.Sequential(
            nn.Linear(2 * 256, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

        self.head = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(LST_DROPOUT),
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

    def forward(self, graph, ernie_emb, rinalmo_emb, seq_len, morgan_fp, mordred_desc, chemberta_emb, simson_emb):
        h_V, h_E, edge_index = self._encode_3d_structure(graph)

        encoder_emb = torch.cat(
            [h_V[0], h_V[1].reshape(h_V[1].size(0), -1)],
            dim=1
        )  # (N, 128+48)

        ernie_emb = ernie_emb.squeeze(0)
        rinalmo_emb = rinalmo_emb.squeeze(0)
        rna_seq_emb = torch.cat([rinalmo_emb, ernie_emb], dim=1)  # (L, 2048)

        rna_emb, nodes_fused = self.rna_fusion(A=rna_seq_emb, B=encoder_emb)

        rna_emb = rna_emb.mean(dim=0)
        nodes_vec = self.attn_pooling(nodes_fused, graph.batch).squeeze(0)
        rna_vec = torch.cat([rna_emb, nodes_vec], dim=0).unsqueeze(0)  # (1, 256)

        expert_smi_emb = self.expert_mlp(torch.cat([morgan_fp, mordred_desc], dim=1))
        chemberta_mol_emb = self.chemberta_mlp(chemberta_emb)
        mol_vec = self.mol_mlp(torch.cat([expert_smi_emb, chemberta_mol_emb], dim=1))

        fused_emb = self.gated_fusion(mol_vec, rna_vec)
        pkd = self.head(fused_emb)
        return pkd


# =========================================================
# FREEZED model
# =========================================================
class MultiStructRSAFreezed(MultiStructRSA):
    def __init__(self):
        super().__init__()
        logging.info("MultiStructRSA - freezed gRNAde encoder, w seq, w cross-attention fusion")

        self._load_grnade_encoder(freeze=True)

        self.rna_fusion = BiDirectionalCrossFusionSingle(
            a_in_dim=768 + 1280,
            b_in_dim=128 + 48,
            d_model=128,
            n_heads=4,
            dropout=BDCF_DROPOUT
        )
        self.gated_fusion = GatedFeatureFusion(256)
        self.attn_pooling = AttnPool(128)

        self.chemberta_mlp = nn.Sequential(
            nn.Linear(384, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

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
            nn.Linear(2 * 256, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

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

    def forward(self, graph, ernie_emb, rinalmo_emb, seq_len, morgan_fp, mordred_desc, chemberta_emb, simson_emb):
        h_V, _, _ = self._encode_3d_structure(graph)
        encoder_emb = torch.cat([h_V[0], h_V[1].reshape(h_V[1].size(0), -1)],dim=1)

        ernie_emb = ernie_emb.squeeze(0)
        rinalmo_emb = rinalmo_emb.squeeze(0)
        rna_seq_emb = torch.cat([rinalmo_emb, ernie_emb], dim=1)

        rna_emb, nodes_fused = self.rna_fusion(A=rna_seq_emb, B=encoder_emb)

        rna_emb = rna_emb.mean(dim=0)
        nodes_vec = self.attn_pooling(nodes_fused, graph.batch).squeeze(0)
        rna_vec = torch.cat([rna_emb, nodes_vec], dim=0).unsqueeze(0)

        expert_smi_emb = self.expert_mlp(torch.cat([morgan_fp, mordred_desc], dim=1))
        chemberta_mol_emb = self.chemberta_mlp(chemberta_emb)
        mol_vec = self.mol_mlp(torch.cat([expert_smi_emb, chemberta_mol_emb], dim=1))

        fused_emb = self.gated_fusion(mol_vec, rna_vec)
        pkd = self.head(fused_emb)
        return pkd


# =========================================================
# WITHOUT 3D STRUCTURE
# =========================================================
class MultiStructRSAWithout3D(MultiStructRSA):
    def __init__(self):
        super().__init__()
        logger.info("--- MultiStructRSA w/o 3D structure ---")

        self.gated_fusion = GatedFeatureFusion(128)

        self.chemberta_mlp = nn.Sequential(
            nn.Linear(384, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

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
            nn.Linear(2 * 128, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

        self.rna_proj = nn.Sequential(
            nn.Linear(768 + 1280, 128, bias=False),
            nn.LayerNorm(128)
        )

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

    def forward(self, graph, ernie_emb, rinalmo_emb, seq_len, morgan_fp, mordred_desc, chemberta_emb, simson_emb):
        rna_emb = self.rna_proj(torch.cat([ernie_emb, rinalmo_emb], dim=1))

        expert_smi_emb = self.expert_mlp(torch.cat([morgan_fp, mordred_desc], dim=1))
        chemberta_mol_emb = self.chemberta_mlp(chemberta_emb)
        mol_emb = self.mol_mlp(torch.cat([expert_smi_emb, chemberta_mol_emb], dim=1))

        x = self.gated_fusion(mol_emb, rna_emb)
        pkd = self.head(x)
        return pkd


# =========================================================
# WITHOUT SEQUENCE
# =========================================================
class MultiStructRSAWithoutSeq(MultiStructRSA):
    def __init__(self):
        super().__init__()
        logger.info("--- MultiStructRSA w/o sequence (frozen gRNAde) ---")

        self._load_grnade_encoder(freeze=True)

        self.gated_fusion = GatedFeatureFusion(128)
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
            nn.Linear(2 * 256, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

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

    def forward(self, graph, ernie_emb, rinalmo_emb, seq_len, morgan_fp, mordred_desc, chemberta_emb, simson_emb):
        h_V, h_E, edge_index = self._encode_3d_structure(graph)

        node_scalar = self.struct_proj(
            torch.cat([h_V[0], h_V[1].reshape(h_V[1].size(0), -1)], dim=1)
        )
        rna_emb = self.attn_pooling(node_scalar, graph.batch)

        expert_smi_emb = self.expert_mlp(torch.cat([morgan_fp, mordred_desc], dim=1))
        chemberta_mol_emb = self.chemberta_mlp(chemberta_emb)
        mol_emb = self.mol_mlp(torch.cat([expert_smi_emb, chemberta_mol_emb], dim=1))

        x = self.gated_fusion(mol_emb, rna_emb)
        pkd = self.head(x)
        return pkd


# =========================================================
# WITHOUT FUSION
# =========================================================
class MultiStructRSAWithoutFusion(MultiStructRSA):
    def __init__(self):
        super().__init__()
        logger.info("--- MultiStructRSA w/o fusion (no BDCF; just concat MLM + GNN) (frozen) ---")

        self._load_grnade_encoder(freeze=True)

        self.gated_fusion = GatedFeatureFusion(256)
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

        self.smiles_seq_mlp = nn.Sequential(
            nn.Linear(384 + 512, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 256, bias=True)
        )


        self.expert_mlp = nn.Sequential(
            nn.Linear(1024 + 1205, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 256, bias=True)
        )

        self.mol_mlp = nn.Sequential(
            nn.Linear(2 * 256, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 256, bias=True)
        )

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

    def forward(self, graph, ernie_emb, rinalmo_emb, seq_len, morgan_fp, mordred_desc, chemberta_emb, simson_emb):
        h_V, h_E, edge_index = self._encode_3d_structure(graph)

        node_scalar = self.struct_proj(
            torch.cat([h_V[0], h_V[1].reshape(h_V[1].size(0), -1)], dim=1)
        )
        graph_emb = self.attn_pooling(node_scalar, graph.batch)

        rna_mlm_emb = self.rna_proj(torch.cat([ernie_emb, rinalmo_emb], dim=1))
        rna_emb = torch.cat([rna_mlm_emb, graph_emb], dim=1)  # (B, 256)

        expert_smi_emb = self.expert_mlp(torch.cat([morgan_fp, mordred_desc], dim=1))
        #chemberta_mol_emb = self.chemberta_mlp(chemberta_emb)
        mol_vec = self.smiles_seq_mlp(torch.cat([chemberta_emb, simson_emb], dim=1))
        mol_emb = self.mol_mlp(torch.cat([expert_smi_emb, mol_vec], dim=1))

        x = self.gated_fusion(mol_emb, rna_emb)
        pkd = self.head(x)
        return pkd

####################################################################################################3
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