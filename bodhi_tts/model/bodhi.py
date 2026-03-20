import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_encoder import TextEncoder
from .description_encoder import DescriptionEncoder
from .duration_predictor import DurationPredictor, monotonic_alignment_search, length_regulate
from .decoder import FlowDecoder


class BodhiTTS(nn.Module):
    """Top-level Bodhi-TTS model: description-conditioned Matcha-TTS."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        te = config.text_encoder
        de = config.description_encoder
        dp = config.duration_predictor
        dec = config.decoder
        mel = config.mel

        self.text_encoder = TextEncoder(
            vocab_size=config.vocab_size,
            d_model=te.d_model,
            n_heads=te.n_heads,
            n_layers=te.n_layers,
            d_ff=te.d_ff,
            dropout=te.dropout,
            max_seq_len=te.max_seq_len,
        )
        self.desc_encoder = DescriptionEncoder(
            minilm_dim=de.minilm_dim,
            proj_dim=de.proj_dim,
        )
        self.duration_predictor = DurationPredictor(
            d_model=te.d_model,
            hidden_dim=dp.hidden_dim,
            n_layers=dp.n_layers,
            kernel_size=dp.kernel_size,
            dropout=dp.dropout,
        )
        self.decoder = FlowDecoder(
            in_channels=mel.n_mels,
            d_model=dec.d_model,
            channels_mult=dec.channels_mult,
            n_res_blocks=dec.n_res_blocks,
            n_heads=dec.n_heads,
        )

        # Mel projection for computing MAS attention logits
        self.mel_proj = nn.Linear(mel.n_mels, te.d_model)

    def forward(self, text_ids, text_lengths, mel, mel_lengths, desc_embed, t, x_t):
        """Training forward pass.

        Args:
            text_ids: [B, T_text]
            text_lengths: [B]
            mel: [B, 80, T_mel] ground truth mel
            mel_lengths: [B]
            desc_embed: [B, 384] precomputed MiniLM embedding
            t: [B] flow timestep
            x_t: [B, 80, T_mel] noisy mel
        Returns:
            v_pred: [B, 80, T_mel] velocity prediction
            dur_loss: scalar duration prediction loss
            durations: [B, T_text] ground truth durations from MAS
        """
        # 1. Text encoder
        h_text = self.text_encoder(text_ids, text_lengths)  # [B, T_text, 256]

        # 2. Description encoder
        e_desc = self.desc_encoder(desc_embed)  # [B, 256]

        # 3. MAS to get ground truth durations
        mel_t = mel.transpose(1, 2)  # [B, T_mel, 80]
        mel_proj = self.mel_proj(mel_t)  # [B, T_mel, 256]
        # Attention logits: [B, T_text, T_mel]
        attn_logits = torch.bmm(h_text, mel_proj.transpose(1, 2))
        # Mask invalid positions
        B, T_text, T_mel = attn_logits.shape
        text_mask = torch.arange(T_text, device=attn_logits.device).unsqueeze(0) >= text_lengths.unsqueeze(1)
        mel_mask = torch.arange(T_mel, device=attn_logits.device).unsqueeze(0) >= mel_lengths.unsqueeze(1)
        attn_logits.masked_fill_(text_mask.unsqueeze(2), -1e9)
        attn_logits.masked_fill_(mel_mask.unsqueeze(1), -1e9)
        attn_logits = F.log_softmax(attn_logits, dim=1)

        with torch.no_grad():
            durations = monotonic_alignment_search(attn_logits, text_lengths, mel_lengths)

        # 4. Duration loss
        log_dur_pred = self.duration_predictor(h_text, e_desc, text_lengths)
        log_dur_gt = torch.log(durations.float().clamp(min=1))
        dur_mask = torch.arange(T_text, device=text_ids.device).unsqueeze(0) < text_lengths.unsqueeze(1)
        dur_loss = F.mse_loss(log_dur_pred * dur_mask, log_dur_gt * dur_mask, reduction="sum")
        dur_loss = dur_loss / dur_mask.sum()

        # 5. Length regulate
        h_expanded = length_regulate(h_text, durations, mel_lengths)  # [B, T_mel, 256]

        # 6. Decoder
        v_pred = self.decoder(x_t, t, e_desc, h_expanded, mel_lengths)  # [B, 80, T_mel]

        return v_pred, dur_loss, durations

    @torch.no_grad()
    def synthesize(self, text_ids, text_lengths, desc_embed, n_steps=10):
        """Inference: predict durations, ODE solve from noise.

        Args:
            text_ids: [B, T_text]
            text_lengths: [B]
            desc_embed: [B, 384]
            n_steps: number of Euler steps
        Returns:
            mel: [B, 80, T_mel] synthesized mel spectrogram
        """
        from ..flow.ode_solver import euler_solve

        h_text = self.text_encoder(text_ids, text_lengths)
        e_desc = self.desc_encoder(desc_embed)

        # Predict durations
        log_dur = self.duration_predictor(h_text, e_desc, text_lengths)
        durations = torch.clamp(torch.round(torch.exp(log_dur)), min=1).long()
        # Zero out padding
        text_mask = torch.arange(text_ids.shape[1], device=text_ids.device).unsqueeze(0) < text_lengths.unsqueeze(1)
        durations = durations * text_mask

        mel_lengths = durations.sum(dim=1)  # [B]
        h_expanded = length_regulate(h_text, durations, mel_lengths)  # [B, T_mel, 256]

        max_mel = mel_lengths.max().item()
        noise = torch.randn(text_ids.shape[0], 80, max_mel, device=text_ids.device)

        mel = euler_solve(self, noise, n_steps, e_desc, h_expanded, mel_lengths)
        return mel, mel_lengths

    def decode_step(self, x_t, t, desc_embed, text_enc_expanded, mel_lengths):
        """Single decoder step for ODE solver."""
        return self.decoder(x_t, t, desc_embed, text_enc_expanded, mel_lengths)
