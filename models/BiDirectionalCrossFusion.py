import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from utils.utils import GatedFeatureFusion

import logging
logger = logging.getLogger(__name__)

from utils.utils import *

##################################################################################################
# BiDirectionalCrossFusion when batch_size = 1;
# In that implementation, no need for padding graphs/sequences matrices, because every batch
# contains the same RNA, and therefore the same number of graph nodes to sequence nucleotides
# (|batch_i_graph_nodes| = |batch_i_sequence_nucleotides|)
# Simpler and more robust version for small scale dataset (e.g., R-SIM)
##################################################################################################


class CrossAttentionBlockSingle(nn.Module):
    """
    Single-sample cross-attention:
      q  : (Lq, q_in_dim)
      kv : (Lk, kv_in_dim)

    Returns:
      (Lq, d_model)
    """
    def __init__(self, q_in_dim, kv_in_dim, d_model=128, n_heads=4, dropout=0.1, ffn_mult=4):
        super().__init__()

        #self.q_proj = nn.Linear(q_in_dim, d_model, bias=False) if q_in_dim != d_model else nn.Identity()
        #self.kv_proj = nn.Linear(kv_in_dim, d_model, bias=False) if kv_in_dim != d_model else nn.Identity()

        self.q_proj = nn.Linear(q_in_dim, d_model, bias=True) if q_in_dim != d_model else nn.Linear(d_model, d_model, bias=True)
        self.kv_proj = nn.Linear(kv_in_dim, d_model, bias=True) if kv_in_dim != d_model else nn.Linear(d_model, d_model, bias=True)
        
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            #dropout=0,
            batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            #nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, q, kv):
        """
        q:  (Lq, q_in_dim)
        kv: (Lk, kv_in_dim)
        """
        q = self.q_proj(q)      # (Lq, d_model)
        kv = self.kv_proj(kv)   # (Lk, d_model)

        qn = self.norm_q(q).unsqueeze(0)
        kvn = self.norm_kv(kv).unsqueeze(0)

        out, _ = self.attn(
            query=qn,
            key=kvn,
            value=kvn,
            need_weights=False
        )

        out = out.squeeze(0)

        q = q + self.attn_dropout(out)
        f = self.ffn(self.ffn_norm(q))
        q = q + self.ffn_dropout(f)

        return q


class BiDirectionalCrossFusionSingle(nn.Module):
    """
    Single-sample bidirectional cross-attention:
      A <- attends to B
      B <- attends to A

    A: (LA, a_in_dim)
    B: (LB, b_in_dim)
    """
    def __init__(self, a_in_dim, b_in_dim, bdcf_blocks=1, d_model=128, n_heads=4,
                 dropout=0.1, ffn_mult=4, a_pool='mean', b_pool='mean'):
        super().__init__()

        if bdcf_blocks not in [1, 2]:
            raise ValueError(f"bdcf_blocks must be 1 or 2, got {bdcf_blocks}")
        
        logger.info(f'BiDirectionalCrossFusion @  hyperparam -> bdcf_blocks:{bdcf_blocks}, d_model:{d_model}, n_heads:{n_heads}, dropout:{dropout}, ffn_mult:{ffn_mult}, '
                    f'a_pool:{a_pool}, b_pool:{b_pool}')
        
        self.a_pool = a_pool
        self.b_pool = b_pool
        self.bdcf_blocks = bdcf_blocks
        
        self.a_from_b_1 = CrossAttentionBlockSingle(q_in_dim=a_in_dim, kv_in_dim=b_in_dim, d_model=d_model, n_heads=n_heads, dropout=dropout, ffn_mult=ffn_mult)
        self.a_from_b_2 = CrossAttentionBlockSingle(q_in_dim=d_model, kv_in_dim=d_model, d_model=d_model, n_heads=n_heads, dropout=dropout, ffn_mult=ffn_mult)

        self.b_from_a_1 = CrossAttentionBlockSingle(q_in_dim=b_in_dim, kv_in_dim=a_in_dim, d_model=d_model, n_heads=n_heads, dropout=dropout, ffn_mult=ffn_mult)
        self.b_from_a_2 = CrossAttentionBlockSingle(q_in_dim=d_model, kv_in_dim=d_model, d_model=d_model, n_heads=n_heads, dropout=dropout, ffn_mult=ffn_mult)
        
        self.a_attn_pool = AttnPoolSingle(d_model)
        self.b_attn_pool = AttnPoolSingle(d_model)
    
    def forward(self, A, B):
        """
        A: (LA, a_in_dim)
        B: (LB, b_in_dim)

        returns:
          A_fused: (LA, d_model)
          B_fused: (LB, d_model)
        """

        A_fused_1 = self.a_from_b_1(A, B)
        B_fused_1 = self.b_from_a_1(B, A)

        if self.bdcf_blocks == 2:
            A_fused_2 = self.a_from_b_2(A_fused_1, B_fused_1)
            B_fused_2 = self.b_from_a_2(B_fused_1, A_fused_1)
            
            # pool fused matrices
            A_fused_2 = self.a_attn_pool(A_fused_2) if self.a_pool == 'attn' else A_fused_2.mean(dim=0) # (, 128)
            B_fused_2 = self.b_attn_pool(B_fused_2) if self.b_pool == 'attn' else B_fused_2.mean(dim=0) # (, 128)
            fused_rna_vec = torch.cat([A_fused_2, B_fused_2], dim=0).unsqueeze(0) # unsqueenze to get (1, 2*128)
            return fused_rna_vec

        # pool fused matrices
        A_fused_1 = self.a_attn_pool(A_fused_1) if self.a_pool == 'attn' else A_fused_1.mean(dim=0) # (, 128)
        B_fused_1 = self.b_attn_pool(B_fused_1) if self.b_pool == 'attn' else B_fused_1.mean(dim=0) # (, 128)
        fused_rna_vec = torch.cat([A_fused_1, B_fused_1], dim=0).unsqueeze(0) # unsqueeze to get (1, 2*128)
        return fused_rna_vec
    


##################################################################################################
# BiDirectionalCrossFusion when batch_size > 1;
# Different implementation, need to use struture embeddings which padded
# to max sequence length & use padding helper method in order to pad batched
# graphs to the graph with the max nodes in that batch.
# Moreover, custom 'masked mean' method in order to pool only on "real" nodes/nucleotides
# and ignore padded nodes/nucleotides which padded to zero in order to use batched
# BiDirectionalCrossFusion and make all matrices have the same number of rows
# (i.e., same number of nodes/nucleotides for graph/sequence representations)
##################################################################################################


def pad_nodes_by_graph(node_scalar, batch_idx, B):
    """
    Method that pad graphs G in batch B with True values according to the biggest graph
    in batch B (graph with max nodes in the batch)
    node_scalar: (N_total, D)
    batch_idx:   (N_total,) values in [0..B-1]
    returns:
      kv_padded: (B, Nmax, D)
      kv_pad_mask: (B, Nmax) True at PAD positions
      lengths: (B,) node counts
    """
    chunks = []
    lengths = torch.zeros(B, dtype=torch.long, device=node_scalar.device)

    for i in range(B):
        nodes_i = node_scalar[batch_idx == i]  # (Ni, D)
        lengths[i] = nodes_i.size(0)
        chunks.append(nodes_i)

    kv_padded = pad_sequence(chunks, batch_first=True)  # (B, Nmax, D)

    Nmax = kv_padded.size(1)
    arange = torch.arange(Nmax, device=node_scalar.device).unsqueeze(0)  # (1, Nmax)
    #print(f'arange:\n{arange}')
    #print(f'length.unsqueeze(1):\n{lengths.unsqueeze(1)}')
    kv_pad_mask = arange >= lengths.unsqueeze(1)  # (B, Nmax) True=PAD

    return kv_padded, kv_pad_mask, lengths

def masked_mean(x, pad_mask):
    # x: (B, L, D)
    # pad_mask: (B, L) True=PAD
    keep = (~pad_mask).unsqueeze(-1)          # (B, L, 1) True=real
    x = x * keep
    denom = keep.sum(dim=1).clamp_min(1.0)    # (B, 1)
    return x.sum(dim=1) / denom               # (B, D)


class CrossAttentionBlock(nn.Module):
    """
    Updates the query sequence using cross-attention over kv sequence:
      out = Attn(Q=query, K=kv, V=kv) + residual + FFN

    Works with variable lengths via boolean masks.
    """
    def __init__(self, q_in_dim, kv_in_dim, d_model=128, n_heads=4, dropout=0.1, ffn_mult=4):
        super().__init__()

        self.q_proj  = nn.Linear(q_in_dim,  d_model) if q_in_dim  != d_model else nn.Identity()
        self.kv_proj = nn.Linear(kv_in_dim, d_model) if kv_in_dim != d_model else nn.Identity()

        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, q, kv, kv_key_padding_mask=None, q_padding_mask=None):
        """
        q:  (B, Lq, q_in_dim)
        kv: (B, Lk, kv_in_dim)

        kv_key_padding_mask: (B, Lk) boolean, True where PAD in keys/values
        q_padding_mask:      (B, Lq) boolean, True where PAD in queries

        returns: (B, Lq, d_model)
        """
        q = self.q_proj(q)         # (B, Lq, d_model)
        kv = self.kv_proj(kv)      # (B, Lk, d_model)

        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)

        out, _ = self.attn(
            query=qn,
            key=kvn,
            value=kvn,
            key_padding_mask=kv_key_padding_mask,  # ignores PAD keys/values
            need_weights=False
        )
        q = q + self.attn_dropout(out)

        f = self.ffn(self.ffn_norm(q))
        q = q + self.ffn_dropout(f)

        # IMPORTANT: if q has PAD rows, zero them out so they don't become "real" vectors
        #print(f'q ({q.shape}):\n{q}')
        #print(f'q_padded_mask ({q_padding_mask.shape}):\n{q_padding_mask}')
        if q_padding_mask is not None:
            q = q.masked_fill(q_padding_mask.unsqueeze(-1), 0.0)

        return q


class BiDirectionalCrossFusion(nn.Module):
    """
    Does:
      A <- cross-attend to B
      B <- cross-attend to A
    """
    def __init__(self, a_in_dim, b_in_dim, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.a_from_b = CrossAttentionBlock(a_in_dim, b_in_dim, d_model, n_heads, dropout)
        self.b_from_a = CrossAttentionBlock(b_in_dim, a_in_dim, d_model, n_heads, dropout)

    def forward(self, A, B, A_pad_mask=None, B_pad_mask=None):
        """
        A: (B, LA, a_in_dim)
        B: (B, LB, b_in_dim)
        *_pad_mask: boolean, True where PAD

        returns:
          A_fused: (B, LA, d_model)
          B_fused: (B, LB, d_model)
        """
        A_fused = self.a_from_b(A, B, kv_key_padding_mask=B_pad_mask, q_padding_mask=A_pad_mask)
        B_fused = self.b_from_a(B, A, kv_key_padding_mask=A_pad_mask, q_padding_mask=B_pad_mask)
        return A_fused, B_fused
    


def main():
    device = 'cpu'
    node_scalar_t = torch.randn((100, 128)).to(device=device)
    # number of nodes per graph (must sum to 100)
    lengths = torch.tensor([5, 5, 5, 5, 10, 10, 10, 50], device=device)  # example

    batch_index_t = torch.repeat_interleave(
        torch.arange(len(lengths), device=device),
        lengths
    )
    print(batch_index_t)
    kv_padded, kv_pad_mask, lengths = pad_nodes_by_graph(node_scalar=node_scalar_t,
                                                            batch_idx=batch_index_t,
                                                            B=8)
    print(f'kv_padded shape -> {kv_padded.shape}') # (8, 50, 128), padded to the graph with the max number of nodes with (0x128) vectors to achive (50,128) tensors
    print(f'first graph with padded 50-5=45 padded nodes:\n{kv_padded[0, :, :]}')
    print(f'kv_pad_mask shape -> {kv_pad_mask.shape}') # (8, 50), tells what nodes should be padded in each graph
    print(f'first graph with padded 50-5=45 padded nodes:\n{kv_pad_mask[0, :]}') # 5 False values & 45 True values for torch.nn.MultiheadAttention key_padding_mask value
    

if __name__ == '__main__':
    main()