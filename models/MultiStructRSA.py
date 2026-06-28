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

gRNAde_WEIGHT_PATH = "saved_objects/gRNAde_checkpoints/gRNAde_drop3d@0.75_maxlen@500.h5"
 
# SMILES features dimensions (SimSon, chemBERTa, Mordred, Morgan Fingerprints)
SIMSON_DIM = 512
CHEMBERTA_DIM = 384
MORDRED_DIM = 1205
MORGAN_FINGERPRINTS_DIM = 1024

# RNA features dimension (RiNALMo, ERNIE-RNA and gRNAde GNN encoder)
RINALMO_DIM = 1280
ERNIE_RNA_DIM = 768
GRNADE_DIM = 128 + 48


def choose_model_setting(
    option: str = "full",
    exclude_modalities=None,
    freeze_grnade=False,
    smiles_seq_setting: str = 'deep',
    smiles_expert_setting: str = 'proj',
    dropout: float = 0.1,
    d_model: int = 128,
    ffn_mult: int = 4,
    bdcf_blocks: int = 1,
    node_pool: str = 'attn',
    token_pool: str = 'mean'
):
    if exclude_modalities is None:
        exclude_modalities = []

    exclude_modalities = set(exclude_modalities)

    valid_modalities = {
        "simson", "chemberta", "mordred", "morgan",
        "rinalmo", "ernie", "grnade"
    }
    invalid = exclude_modalities - valid_modalities
    if invalid:
        raise ValueError(f"Invalid excluded modalities: {sorted(invalid)}")

    assert option in ["full", "freezed", "w/o seq", "w/o 3D struct", "w/o fusion", "full batched"]
    assert smiles_expert_setting in ['proj', 'shallow', 'deep']
    assert smiles_seq_setting in ['proj', 'shallow', 'deep']
    if option == 'freezed':
      freeze_grnade = True
    # automatic routing for special sensitivity cases
    if option == 'full':
        if "grnade" in exclude_modalities:
            option = "w/o 3D struct"
            logger.info("exclude_modalities contains 'grnade' -> switching model setting to 'w/o 3D struct'")
        elif {"ernie", "rinalmo"}.issubset(exclude_modalities):
            option = "w/o seq"
            logger.info("exclude_modalities contains both 'ernie' and 'rinalmo' -> switching model setting to 'w/o seq'")
            
    model_map = {
        "full": MultiStructRSAFull,
        "freezed": MultiStructRSAFull,
        "w/o seq": MultiStructRSAWithoutSeq,
        "w/o 3D struct": MultiStructRSAWithout3D,
        "w/o fusion": MultiStructRSAWithoutFusion,
        "full batched": MultiStructRSABatched
    }

    if option in ["full", "freezed"]:
        model = model_map[option](
            exclude_modalities=exclude_modalities,
            freeze_grnade=freeze_grnade,
            smiles_seq_setting=smiles_seq_setting,
            smiles_expert_setting=smiles_expert_setting,
            dropout=dropout,
            d_model=d_model,
            ffn_mult=ffn_mult,
            bdcf_blocks=bdcf_blocks,
            node_pool=node_pool,
            token_pool=token_pool
        )
        logger.info(
            f"Initialized model setting: {option}, exclude_modalities: {sorted(exclude_modalities)}, "
            f"dropout: {dropout}, d_model: {d_model}, ffn_mult: {ffn_mult}, "
            f"bdcf_blocks: {bdcf_blocks}, node_pool: {node_pool}, token_pool: {token_pool}"
        )
    else:
        model = model_map[option](
            exclude_modalities=exclude_modalities,
            freeze_grnade=freeze_grnade,
            smiles_seq_setting=smiles_seq_setting,
            smiles_expert_setting=smiles_expert_setting,
            dropout=dropout,
            d_model=d_model,
            node_pool=node_pool
        )
        logger.info(
            f"Initialized model setting: {option}, dropout: {dropout}, "
            f"exclude_modalities ignored by this model: {sorted(exclude_modalities)}"
        )

    return model


class MultiStructRSA(nn.Module, ABC):
    def __init__(self, dropout=0.1, exclude_modalities=None):
        super().__init__()

        self.dropout = dropout

        if exclude_modalities is None:
            exclude_modalities = []

        self.exclude_modalities = set(exclude_modalities)

        valid_modalities = {
            "simson", "chemberta", "mordred", "morgan",
            "rinalmo", "ernie", "grnade"
        }
        invalid = self.exclude_modalities - valid_modalities
        if invalid:
            raise ValueError(f"Invalid excluded modalities: {sorted(invalid)}")

    #######################################################################################
    # adapted from gRNAde framework - https://github.com/chaitjo/geometric-rna-design.git #

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
    ####################################################################################

    def _get_mlp(self, input_dim, setting):
        assert setting in ['proj', 'shallow', 'deep']

        if setting == 'proj':
            return nn.Sequential(
                nn.Linear(input_dim, 256, bias=False),
                nn.LayerNorm(256)
            )
        elif setting == 'shallow':
            return nn.Sequential(
                nn.Linear(input_dim, 512, bias=False),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(512, 256, bias=True)
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dim, 512, bias=False),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(512, 512, bias=False),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(512, 256, bias=True)
            )

    @abstractmethod
    def forward(self, graph, ernie_emb, rinalmo_emb, seq_len, morgan_fp, mordred_desc, chemberta_emb, simson_emb):
        pass


class MultiStructRSAFull(MultiStructRSA):
    def __init__(
        self,
        exclude_modalities=None,
        freeze_grnade=False,
        smiles_seq_setting='deep',
        smiles_expert_setting='proj',
        dropout=0.1,
        d_model=128,
        ffn_mult=4,
        bdcf_blocks=1,
        node_pool='attn',
        token_pool='mean'
    ):
        super().__init__(dropout=dropout, exclude_modalities=exclude_modalities)

        self.freeze_grnade = freeze_grnade
        self.smiles_seq_setting = smiles_seq_setting
        self.smiles_expert_setting = smiles_expert_setting

        self.use_simson = "simson" not in self.exclude_modalities
        self.use_chemberta = "chemberta" not in self.exclude_modalities
        self.use_mordred = "mordred" not in self.exclude_modalities
        self.use_morgan = "morgan" not in self.exclude_modalities
        self.use_rinalmo = "rinalmo" not in self.exclude_modalities
        self.use_ernie = "ernie" not in self.exclude_modalities
        self.use_grnade = "grnade" not in self.exclude_modalities

        self.smiles_seq_dim = (SIMSON_DIM if self.use_simson else 0) + (CHEMBERTA_DIM if self.use_chemberta else 0)
        self.smiles_expert_dim = (MORDRED_DIM if self.use_mordred else 0) + (MORGAN_FINGERPRINTS_DIM if self.use_morgan else 0)
        self.rna_seq_dim = (RINALMO_DIM if self.use_rinalmo else 0) + (ERNIE_RNA_DIM if self.use_ernie else 0)
        self.grnade_dim = GRNADE_DIM if self.use_grnade else 0

        if self.smiles_seq_dim == 0:
            raise ValueError("At least one SMILES sequence modality must remain active: 'simson' or 'chemberta'")
        if self.smiles_expert_dim == 0:
            raise ValueError("At least one expert modality must remain active: 'mordred' or 'morgan'")
        if self.rna_seq_dim == 0:
            raise ValueError("At least one RNA sequence modality must remain active: 'rinalmo' or 'ernie'")
        if self.grnade_dim == 0:
            raise ValueError("gRNAde is excluded; choose_model_setting should route this case to 'w/o 3D struct'")

        self.d_model = d_model
        self.ffn_mult = ffn_mult
        self.bdcf_blocks = bdcf_blocks
        self.node_pool = node_pool
        self.token_pool = token_pool
        self.n_heads = 4

        logger.info(
            f"MultiStructRSA - full | exclude_modalities={sorted(self.exclude_modalities)} | "
            f"freeze_grnade={self.freeze_grnade} | "
            f"smiles_seq_setting={smiles_seq_setting}, smiles_expert_setting={smiles_expert_setting}, "
            f"dropout={self.dropout}, d_model={self.d_model}, ffn_mult={self.ffn_mult}, "
            f"bdcf_blocks={self.bdcf_blocks}, node_pool={node_pool}, token_pool={token_pool}"
        )

        self._load_grnade_encoder(freeze=self.freeze_grnade)

        self.rna_fusion = BiDirectionalCrossFusionSingle(
            a_in_dim=self.rna_seq_dim,
            b_in_dim=self.grnade_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout=self.dropout,
            ffn_mult=self.ffn_mult,
            bdcf_blocks=self.bdcf_blocks,
            a_pool=self.token_pool,
            b_pool=self.node_pool
        )

        self.gated_fusion = GatedFeatureFusion(2 * self.d_model)

        self.smiles_seq_encoder = self._get_mlp(
            input_dim=self.smiles_seq_dim,
            setting=self.smiles_seq_setting
        )
        self.smiles_expert_encoder = self._get_mlp(
            input_dim=self.smiles_expert_dim,
            setting=self.smiles_expert_setting
        )

        self.mol_mlp = nn.Sequential(
            nn.Linear(2 * 256, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 2 * self.d_model, bias=True)
        )

        self.head = nn.Sequential(
            nn.Linear(2 * self.d_model, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1, bias=True)
        )

    def forward(self, graph, ernie_emb, rinalmo_emb, _, morgan_fp, mordred_desc, chemberta_emb, simson_emb):
        # RNA module
        h_V, _, _ = self._encode_3d_structure(graph)
        encoder_emb = torch.cat([h_V[0], h_V[1].reshape(h_V[1].size(0), -1)], dim=1)

        seq_parts = []
        if self.use_rinalmo:
            seq_parts.append(rinalmo_emb.squeeze(0))
        if self.use_ernie:
            seq_parts.append(ernie_emb.squeeze(0))
        rna_seq_emb = torch.cat(seq_parts, dim=1)

        rna_fused_emb = self.rna_fusion(A=rna_seq_emb, B=encoder_emb)

        # Small molecule module
        expert_parts = []
        if self.use_morgan:
            expert_parts.append(morgan_fp)
        if self.use_mordred:
            expert_parts.append(mordred_desc)
        smile_expert_emb = self.smiles_expert_encoder(torch.cat(expert_parts, dim=1))

        seq_smiles_parts = []
        if self.use_chemberta:
            seq_smiles_parts.append(chemberta_emb)
        if self.use_simson:
            seq_smiles_parts.append(simson_emb)
        smiles_seq_emb = self.smiles_seq_encoder(torch.cat(seq_smiles_parts, dim=1))

        mol_vec = self.mol_mlp(torch.cat([smile_expert_emb, smiles_seq_emb], dim=1))

        fused_emb = self.gated_fusion(mol_vec, rna_fused_emb)
        pkd = self.head(fused_emb)
        return pkd

# =========================================================
# W/O 3D STRUCTURE
# =========================================================
class MultiStructRSAWithout3D(MultiStructRSA):
    def __init__(
        self,
        exclude_modalities=None,
        freeze_grnade=False,
        smiles_seq_setting='deep',
        smiles_expert_setting='proj',
        dropout=0.1,
        d_model=128,
        node_pool=None,
    ):
        super().__init__(dropout=dropout, exclude_modalities=exclude_modalities)

        self.smiles_seq_setting = smiles_seq_setting
        self.smiles_expert_setting = smiles_expert_setting
        self.d_model = d_model

        self.use_simson = "simson" not in self.exclude_modalities
        self.use_chemberta = "chemberta" not in self.exclude_modalities
        self.use_mordred = "mordred" not in self.exclude_modalities
        self.use_morgan = "morgan" not in self.exclude_modalities
        self.use_rinalmo = "rinalmo" not in self.exclude_modalities
        self.use_ernie = "ernie" not in self.exclude_modalities

        self.smiles_seq_dim = (SIMSON_DIM if self.use_simson else 0) + (CHEMBERTA_DIM if self.use_chemberta else 0)
        self.smiles_expert_dim = (MORDRED_DIM if self.use_mordred else 0) + (MORGAN_FINGERPRINTS_DIM if self.use_morgan else 0)
        self.rna_seq_dim = (RINALMO_DIM if self.use_rinalmo else 0) + (ERNIE_RNA_DIM if self.use_ernie else 0)

        if self.smiles_seq_dim == 0:
            raise ValueError("At least one SMILES sequence modality must remain active")
        if self.smiles_expert_dim == 0:
            raise ValueError("At least one SMILES expert modality must remain active")
        if self.rna_seq_dim == 0:
            raise ValueError("At least one RNA sequence modality must remain active")

        logger.info(
            f"--- MultiStructRSA w/o 3D structure | exclude_modalities={sorted(self.exclude_modalities)} | "
            f"smiles_seq_setting={smiles_seq_setting}, smiles_expert_setting={smiles_expert_setting}, "
            f"d_model={d_model}, dropout={dropout} ---"
        )

        self.gated_fusion = GatedFeatureFusion(self.d_model)

        self.smiles_seq_encoder = self._get_mlp(
            input_dim=self.smiles_seq_dim,
            setting=self.smiles_seq_setting
        )
        self.smiles_expert_encoder = self._get_mlp(
            input_dim=self.smiles_expert_dim,
            setting=self.smiles_expert_setting
        )

        self.mol_mlp = nn.Sequential(
            nn.Linear(2 * 256, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, self.d_model, bias=True)
        )

        self.rna_proj = nn.Sequential(
            nn.Linear(self.rna_seq_dim, self.d_model, bias=False),
            nn.LayerNorm(self.d_model)
        )

        self.head = nn.Sequential(
            nn.Linear(self.d_model, 64, bias=False),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 1, bias=True)
        )

    def forward(self, _, ernie_emb, rinalmo_emb, __, morgan_fp, mordred_desc, chemberta_emb, simson_emb):
        rna_parts = []
        if self.use_ernie:
            rna_parts.append(ernie_emb)
        if self.use_rinalmo:
            rna_parts.append(rinalmo_emb)
        rna_emb = self.rna_proj(torch.cat(rna_parts, dim=1))

        expert_parts = []
        if self.use_morgan:
            expert_parts.append(morgan_fp)
        if self.use_mordred:
            expert_parts.append(mordred_desc)
        expert_smi_emb = self.smiles_expert_encoder(torch.cat(expert_parts, dim=1))

        seq_parts = []
        if self.use_chemberta:
            seq_parts.append(chemberta_emb)
        if self.use_simson:
            seq_parts.append(simson_emb)
        seq_emb = self.smiles_seq_encoder(torch.cat(seq_parts, dim=1))

        mol_emb = self.mol_mlp(torch.cat([expert_smi_emb, seq_emb], dim=1))

        x = self.gated_fusion(mol_emb, rna_emb)
        pkd = self.head(x)
        return pkd


# =========================================================
# W/O SEQUENCE
# =========================================================
class MultiStructRSAWithoutSeq(MultiStructRSA):
    def __init__(
        self,
        exclude_modalities=None,
        freeze_grnade=False,
        smiles_seq_setting='deep',
        smiles_expert_setting='proj',
        dropout=0.1,
        d_model=128,
        node_pool='attn',
    ):
        super().__init__(dropout=dropout, exclude_modalities=exclude_modalities)
        self.freeze_grnade = freeze_grnade
        self.smiles_seq_setting = smiles_seq_setting
        self.smiles_expert_setting = smiles_expert_setting
        self.d_model = d_model
        self.node_pool = node_pool

        self.use_simson = "simson" not in self.exclude_modalities
        self.use_chemberta = "chemberta" not in self.exclude_modalities
        self.use_mordred = "mordred" not in self.exclude_modalities
        self.use_morgan = "morgan" not in self.exclude_modalities

        self.smiles_seq_dim = (SIMSON_DIM if self.use_simson else 0) + (CHEMBERTA_DIM if self.use_chemberta else 0)
        self.smiles_expert_dim = (MORDRED_DIM if self.use_mordred else 0) + (MORGAN_FINGERPRINTS_DIM if self.use_morgan else 0)

        if self.smiles_seq_dim == 0:
            raise ValueError("At least one SMILES sequence modality must remain active")
        if self.smiles_expert_dim == 0:
            raise ValueError("At least one SMILES expert modality must remain active")

        logger.info(
            f"--- MultiStructRSA w/o sequence (frozen={self.freeze_grnade}) | exclude_modalities={sorted(self.exclude_modalities)} | "
            f"smiles_seq_setting={self.smiles_seq_setting}, smiles_expert_setting={self.smiles_expert_setting}, "
            f"d_model={self.d_model}, node_pool={self.node_pool}, dropout={self.dropout} ---"
        )

        self._load_grnade_encoder(freeze=self.freeze_grnade)

        self.gated_fusion = GatedFeatureFusion(self.d_model)
        self.struct_proj = nn.Sequential(
            nn.Linear(GRNADE_DIM, self.d_model, bias=False),
            nn.LayerNorm(self.d_model)
        )
        self.node_attn_pooling = AttnPoolSingle(self.d_model)

        self.smiles_seq_encoder = self._get_mlp(
            input_dim=self.smiles_seq_dim,
            setting=self.smiles_seq_setting
        )
        self.smiles_expert_encoder = self._get_mlp(
            input_dim=self.smiles_expert_dim,
            setting=self.smiles_expert_setting
        )

        self.mol_mlp = nn.Sequential(
            nn.Linear(2 * 256, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, self.d_model, bias=True)
        )

        self.head = nn.Sequential(
            nn.Linear(self.d_model, 64, bias=False),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 1, bias=True)
        )

    def forward(self, graph, _, __, ___, morgan_fp, mordred_desc, chemberta_emb, simson_emb):
        h_V, _, _ = self._encode_3d_structure(graph)
        node_scalar = self.struct_proj(torch.cat([h_V[0], h_V[1].reshape(h_V[1].size(0), -1)], dim=1))

        rna_emb = self.node_attn_pooling(node_scalar) if self.node_pool == 'attn' else node_scalar.mean(dim=0, keepdim=True)

        expert_parts = []
        if self.use_morgan:
            expert_parts.append(morgan_fp)
        if self.use_mordred:
            expert_parts.append(mordred_desc)
        expert_smi_emb = self.smiles_expert_encoder(torch.cat(expert_parts, dim=1))

        seq_parts = []
        if self.use_chemberta:
            seq_parts.append(chemberta_emb)
        if self.use_simson:
            seq_parts.append(simson_emb)
        seq_emb = self.smiles_seq_encoder(torch.cat(seq_parts, dim=1))

        mol_emb = self.mol_mlp(torch.cat([expert_smi_emb, seq_emb], dim=1))

        x = self.gated_fusion(mol_emb, rna_emb)
        pkd = self.head(x)
        return pkd


# =========================================================
# W/O FUSION
# =========================================================
class MultiStructRSAWithoutFusion(MultiStructRSA):
    def __init__(
        self,
        exclude_modalities=None,
        freeze_grnade=False,
        smiles_seq_setting='deep',
        smiles_expert_setting='proj',
        dropout=0.1,
        d_model=128,
        node_pool='attn',
    ):
        super().__init__(dropout=dropout, exclude_modalities=exclude_modalities)
        self.freeze_grnade = freeze_grnade
        self.smiles_seq_setting = smiles_seq_setting
        self.smiles_expert_setting = smiles_expert_setting
        self.d_model = d_model
        self.node_pool = node_pool

        self.use_simson = "simson" not in self.exclude_modalities
        self.use_chemberta = "chemberta" not in self.exclude_modalities
        self.use_mordred = "mordred" not in self.exclude_modalities
        self.use_morgan = "morgan" not in self.exclude_modalities
        self.use_rinalmo = "rinalmo" not in self.exclude_modalities
        self.use_ernie = "ernie" not in self.exclude_modalities

        self.smiles_seq_dim = (SIMSON_DIM if self.use_simson else 0) + (CHEMBERTA_DIM if self.use_chemberta else 0)
        self.smiles_expert_dim = (MORDRED_DIM if self.use_mordred else 0) + (MORGAN_FINGERPRINTS_DIM if self.use_morgan else 0)
        self.rna_seq_dim = (RINALMO_DIM if self.use_rinalmo else 0) + (ERNIE_RNA_DIM if self.use_ernie else 0)

        if self.smiles_seq_dim == 0:
            raise ValueError("At least one SMILES sequence modality must remain active")
        if self.smiles_expert_dim == 0:
            raise ValueError("At least one SMILES expert modality must remain active")
        if self.rna_seq_dim == 0:
            raise ValueError("At least one RNA sequence modality must remain active")

        logger.info(
            f"--- MultiStructRSA w/o fusion (frozen={self.freeze_grnade}) | exclude_modalities={sorted(self.exclude_modalities)} | "
            f"smiles_seq_setting={smiles_seq_setting}, smiles_expert_setting={smiles_expert_setting}, "
            f"d_model={d_model}, node_pool={node_pool}, dropout={dropout} ---"
        )

        self._load_grnade_encoder(freeze=freeze_grnade)

        self.gated_fusion = GatedFeatureFusion(2 * self.d_model)

        self.struct_proj = nn.Sequential(
            nn.Linear(GRNADE_DIM, self.d_model, bias=False),
            nn.LayerNorm(self.d_model)
        )
        self.node_attn_pooling = AttnPoolSingle(self.d_model)

        self.rna_proj = nn.Sequential(
            nn.Linear(self.rna_seq_dim, self.d_model, bias=False),
            nn.LayerNorm(self.d_model)
        )

        self.smiles_seq_encoder = self._get_mlp(
            input_dim=self.smiles_seq_dim,
            setting=self.smiles_seq_setting
        )
        self.smiles_expert_encoder = self._get_mlp(
            input_dim=self.smiles_expert_dim,
            setting=self.smiles_expert_setting
        )

        self.mol_mlp = nn.Sequential(
            nn.Linear(2 * 256, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 2 * self.d_model, bias=True)
        )

        self.head = nn.Sequential(
            nn.Linear(2 * self.d_model, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1, bias=True)
        )

    def forward(self, graph, ernie_emb, rinalmo_emb, _, morgan_fp, mordred_desc, chemberta_emb, simson_emb):
        h_V, _, _ = self._encode_3d_structure(graph)
        node_scalar = self.struct_proj(torch.cat([h_V[0], h_V[1].reshape(h_V[1].size(0), -1)], dim=1))
        graph_emb = self.node_attn_pooling(node_scalar) if self.node_pool == 'attn' else node_scalar.mean(dim=0, keepdim=True)

        rna_parts = []
        if self.use_ernie:
            rna_parts.append(ernie_emb)
        if self.use_rinalmo:
            rna_parts.append(rinalmo_emb)
        rna_mlm_emb = self.rna_proj(torch.cat(rna_parts, dim=1))

        rna_emb = torch.cat([rna_mlm_emb, graph_emb], dim=1)

        expert_parts = []
        if self.use_morgan:
            expert_parts.append(morgan_fp)
        if self.use_mordred:
            expert_parts.append(mordred_desc)
        expert_smi_emb = self.smiles_expert_encoder(torch.cat(expert_parts, dim=1))

        seq_parts = []
        if self.use_chemberta:
            seq_parts.append(chemberta_emb)
        if self.use_simson:
            seq_parts.append(simson_emb)
        seq_emb = self.smiles_seq_encoder(torch.cat(seq_parts, dim=1))

        mol_emb = self.mol_mlp(torch.cat([expert_smi_emb, seq_emb], dim=1))

        x = self.gated_fusion(mol_emb, rna_emb)
        pkd = self.head(x)
        return pkd

#########################################################################################
class MultiStructRSABatched(MultiStructRSA):
    def __init__(self):
        super().__init__()

        logger.info("--- MultiStructRSA batched: padded sequence + batched graph training ---")

        self._load_grnade_encoder(freeze=False)

        # batched BiDirectionalCrossFusion method
        self.rna_fusion = BiDirectionalCrossFusion(a_in_dim=768 + 1280, # |ERNIE-RNA| + |RiNALMo|
                                                   b_in_dim=128 + 48, # gRNAde GNN |node features| & |node vectors|
                                                   d_model=128,
                                                   n_heads=4,
                                                   dropout=BDCF_DROPOUT)

        self.gated_fusion = GatedFeatureFusion(256)

        self.smiles_seq_mlp = nn.Sequential(
            nn.Linear(384 + 512, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256, bias=True)
        )

        self.expert_mlp = nn.Sequential(
            nn.Linear(1024 + 1205, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 256, bias=False),
            nn.LayerNorm(256)
        )

        self.mol_mlp = nn.Sequential(
            nn.Linear(2 * 256, 256, bias=False),
            nn.LayerNorm(256)
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

    def forward(self, graphs, ernie_emb, rinalmo_emb, seq_lens, morgan_fp, mordred_desc, chemberta_emb, simson_emb):
        """
        graphs: torch_geometric.data.Batch
        ernie_emb: [B, L, 768]
        seq_lens: [B]
        chemberta_emb: [B, 384]
        """

        B = ernie_emb.size(0) # batch_size
        L = ernie_emb.size(1) # max_seq_len

        h_V, _, _ = self._encode_3d_structure(graph=graphs)
        encoder_emb = torch.cat([h_V[0], h_V[1].reshape(h_V[1].size(0), -1)], dim=1) # (N, 128+48)
        nodes_padded, node_pad_mask, _ = pad_nodes_by_graph(encoder_emb, graphs.batch, B)

        rna_seq_emb = torch.cat([rinalmo_emb, ernie_emb], dim=-1)  # (L, 2048)
        rna_pad_mask = torch.arange(L, device=rna_seq_emb.device).unsqueeze(0) >= seq_lens.unsqueeze(1)

        seq_fused, nodes_fused = self.rna_fusion(
            A=rna_seq_emb,
            B=nodes_padded,
            A_pad_mask=rna_pad_mask,
            B_pad_mask=node_pad_mask
        )

        seq_vec = masked_mean(seq_fused, rna_pad_mask)   # (B, 128)
        nodes_vec = masked_mean(nodes_fused, node_pad_mask)    # (B, 128)
        rna_vec = torch.cat([seq_vec, nodes_vec], dim=1)     # (B, 256)

        expert_smi_emb = self.expert_mlp(torch.cat([morgan_fp, mordred_desc], dim=1))
        #chemberta_mol_emb = self.chemberta_proj(chemberta_emb)
        seq_emb = self.smiles_seq_proj(torch.cat([chemberta_emb, simson_emb], dim=1))
        mol_vec = self.mol_mlp(torch.cat([expert_smi_emb, seq_emb], dim=1))

        fused_emb = self.gated_fusion(mol_vec, rna_vec)
        pkd = self.head(fused_emb)
        return pkd