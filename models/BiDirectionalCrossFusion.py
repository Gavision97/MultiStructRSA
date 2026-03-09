import torch.nn as nn

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

        self.q_proj = nn.Linear(q_in_dim, d_model) if q_in_dim != d_model else nn.Identity()
        self.kv_proj = nn.Linear(kv_in_dim, d_model) if kv_in_dim != d_model else nn.Identity()

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
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
    def __init__(self, a_in_dim, b_in_dim, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.a_from_b = CrossAttentionBlockSingle(
            q_in_dim=a_in_dim,
            kv_in_dim=b_in_dim,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        self.b_from_a = CrossAttentionBlockSingle(
            q_in_dim=b_in_dim,
            kv_in_dim=a_in_dim,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )

    def forward(self, A, B):
        """
        A: (LA, a_in_dim)
        B: (LB, b_in_dim)

        returns:
          A_fused: (LA, d_model)
          B_fused: (LB, d_model)
        """
        A_fused = self.a_from_b(A, B)
        B_fused = self.b_from_a(B, A)
        return A_fused, B_fused