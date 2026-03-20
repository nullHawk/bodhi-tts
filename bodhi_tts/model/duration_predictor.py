import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numba import njit, prange


class DurationPredictor(nn.Module):
    """Conv-based duration predictor conditioned on description embedding."""

    def __init__(self, d_model=256, hidden_dim=256, n_layers=2, kernel_size=3, dropout=0.5):
        super().__init__()
        self.desc_proj = nn.Linear(d_model, d_model)

        layers = []
        for i in range(n_layers):
            in_dim = d_model if i == 0 else hidden_dim
            layers.extend([
                nn.Conv1d(in_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            ])
        self.convs = nn.ModuleList(layers)
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, text_enc, desc_embed, text_lengths):
        """
        Args:
            text_enc: [B, T_text, d_model]
            desc_embed: [B, d_model] (already projected by DescriptionEncoder)
            text_lengths: [B]
        Returns:
            log_durations: [B, T_text]
        """
        # Add description conditioning
        desc_cond = self.desc_proj(desc_embed).unsqueeze(1)  # [B, 1, d_model]
        h = (text_enc + desc_cond).transpose(1, 2)  # [B, d_model, T]

        for i in range(0, len(self.convs), 4):
            h = self.convs[i](h)      # Conv1d
            h = self.convs[i + 1](h)  # ReLU
            # LayerNorm expects [B, T, C]
            h = self.convs[i + 2](h.transpose(1, 2)).transpose(1, 2)
            h = self.convs[i + 3](h)  # Dropout

        log_dur = self.out_proj(h.transpose(1, 2)).squeeze(-1)  # [B, T_text]
        return log_dur


@njit()
def _mas_one(log_p, t_len, m_len):
    """MAS DP for a single sample — numba JIT compiled."""
    Q = np.full((t_len, m_len), -1e9, dtype=np.float64)
    Q[0, 0] = log_p[0, 0]
    for j in range(1, m_len):
        Q[0, j] = Q[0, j - 1] + log_p[0, j]

    for i in range(1, t_len):
        for j in range(i, m_len):
            prev = Q[i - 1, j - 1]
            curr = Q[i, j - 1]
            Q[i, j] = log_p[i, j] + (prev if prev > curr else curr)

    # Backtrack
    path = np.zeros(m_len, dtype=np.int64)
    path[m_len - 1] = t_len - 1
    for j in range(m_len - 2, -1, -1):
        i = path[j + 1]
        if i > 0 and Q[i - 1, j] > Q[i, j]:
            path[j] = i - 1
        else:
            path[j] = i

    # Convert path to durations
    durations = np.zeros(t_len, dtype=np.int64)
    for j in range(m_len):
        durations[path[j]] += 1
    return durations


@njit(parallel=True)
def _mas_batch(attn_np, t_lens, m_lens, out_durations):
    """Batch MAS with parallel processing via numba."""
    B = attn_np.shape[0]
    for b in prange(B):
        t_len = t_lens[b]
        m_len = m_lens[b]
        log_p = attn_np[b, :t_len, :m_len]
        durs = _mas_one(log_p, t_len, m_len)
        out_durations[b, :t_len] = durs


def monotonic_alignment_search(attn_logits, text_lengths, mel_lengths):
    """MAS from Glow-TTS: find monotonic alignment via dynamic programming.

    Uses numba JIT with parallel batch processing for speed.

    Args:
        attn_logits: [B, T_text, T_mel] log-probability of alignment
        text_lengths: [B]
        mel_lengths: [B]
    Returns:
        durations: [B, T_text] integer durations
    """
    B = attn_logits.shape[0]
    T_text = attn_logits.shape[1]

    attn_np = attn_logits.detach().cpu().to(torch.float64).numpy()
    t_lens = text_lengths.cpu().numpy().astype(np.int64)
    m_lens = mel_lengths.cpu().numpy().astype(np.int64)

    out_durations = np.zeros((B, T_text), dtype=np.int64)
    _mas_batch(attn_np, t_lens, m_lens, out_durations)

    return torch.from_numpy(out_durations).to(attn_logits.device)


def length_regulate(text_enc, durations, mel_lengths):
    """Expand text encoding by durations using repeat_interleave.

    Args:
        text_enc: [B, T_text, d_model]
        durations: [B, T_text]
        mel_lengths: [B]
    Returns:
        expanded: [B, T_mel_max, d_model]
    """
    B = text_enc.shape[0]
    max_mel = mel_lengths.max().item()
    d_model = text_enc.shape[2]
    expanded = torch.zeros(B, max_mel, d_model, device=text_enc.device, dtype=text_enc.dtype)

    for b in range(B):
        dur = durations[b]
        exp = torch.repeat_interleave(text_enc[b], dur, dim=0)
        length = min(exp.shape[0], max_mel)
        expanded[b, :length] = exp[:length]

    return expanded
