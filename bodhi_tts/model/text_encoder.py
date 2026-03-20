import torch
import torch.nn as nn


class TextEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class TextEncoder(nn.Module):
    """6-layer pre-norm transformer text encoder."""

    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=6,
                 d_ff=2048, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            TextEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_ids, text_lengths):
        """
        Args:
            text_ids: [B, T_text]
            text_lengths: [B]
        Returns:
            h_text: [B, T_text, d_model]
        """
        B, T = text_ids.shape
        positions = torch.arange(T, device=text_ids.device).unsqueeze(0)
        x = self.char_embed(text_ids) + self.pos_embed(positions)
        x = self.dropout(x)

        # Padding mask: True where padded
        key_padding_mask = torch.arange(T, device=text_ids.device).unsqueeze(0) >= text_lengths.unsqueeze(1)

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        return self.norm(x)
