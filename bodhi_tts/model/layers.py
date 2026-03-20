import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding: [B] -> [B, dim]"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class AdaLN(nn.Module):
    """Adaptive Layer Norm: modulates norm(x) with learned scale/shift from conditioning."""

    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, d_model * 2)

    def forward(self, x, cond):
        """
        Args:
            x: [B, C, T] (channel-first for conv blocks)
            cond: [B, cond_dim]
        """
        # x is [B, C, T], norm over C
        x_t = x.transpose(1, 2)  # [B, T, C]
        x_normed = self.norm(x_t).transpose(1, 2)  # [B, C, T]

        scale_shift = self.proj(cond)  # [B, 2*C]
        scale, shift = scale_shift.chunk(2, dim=-1)  # each [B, C]
        scale = scale.unsqueeze(-1)  # [B, C, 1]
        shift = shift.unsqueeze(-1)  # [B, C, 1]
        return x_normed * (1 + scale) + shift


class ResBlock1D(nn.Module):
    """Residual block with AdaLN conditioning for 1D convolutions."""

    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.1):
        super().__init__()
        self.adaln1 = AdaLN(in_ch, cond_dim)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.adaln2 = AdaLN(out_ch, cond_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        """x: [B, C, T], cond: [B, cond_dim]"""
        h = self.adaln1(x, cond)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.adaln2(h, cond)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class CrossAttention1D(nn.Module):
    """Cross-attention over 1D sequences (channel-first I/O)."""

    def __init__(self, d_model, text_dim, n_heads):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.kv_proj = nn.Linear(text_dim, d_model) if text_dim != d_model else nn.Identity()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x, context, context_mask=None):
        """
        Args:
            x: [B, C, T]
            context: [B, T_ctx, text_dim]
            context_mask: [B, T_ctx] bool, True = ignore
        """
        # To [B, T, C]
        x_t = x.transpose(1, 2)
        residual = x_t
        x_normed = self.norm(x_t)
        kv = self.kv_proj(context)
        out, _ = self.attn(x_normed, kv, kv, key_padding_mask=context_mask)
        return (residual + out).transpose(1, 2)  # [B, C, T]


class Downsample1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv1d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))
