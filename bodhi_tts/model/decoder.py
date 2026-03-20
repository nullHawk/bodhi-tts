import torch
import torch.nn as nn
from .layers import SinusoidalPosEmb, ResBlock1D, CrossAttention1D, Downsample1D, Upsample1D


class ConditioningModule(nn.Module):
    """Combines timestep and description embeddings into a single conditioning vector."""

    def __init__(self, d_model=256):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t, desc_embed):
        """
        Args:
            t: [B] timestep
            desc_embed: [B, d_model] projected description
        Returns:
            cond: [B, d_model]
        """
        return self.time_embed(t) + desc_embed


class UNetLevel(nn.Module):
    """One level of the U-Net: n_res_blocks of (ResBlock + CrossAttention)."""

    def __init__(self, channels, cond_dim, text_dim, n_res_blocks, n_heads):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.cross_attns = nn.ModuleList()
        for _ in range(n_res_blocks):
            self.res_blocks.append(ResBlock1D(channels, channels, cond_dim))
            self.cross_attns.append(CrossAttention1D(channels, text_dim, n_heads))

    def forward(self, x, cond, context, context_mask=None):
        for res, attn in zip(self.res_blocks, self.cross_attns):
            x = res(x, cond)
            x = attn(x, context, context_mask)
        return x


class FlowDecoder(nn.Module):
    """1D U-Net flow matching decoder (~55M params).

    Input: cat(x_t, text_enc_expanded) -> [B, 80+256, T_mel]
    Output: velocity prediction [B, 80, T_mel]
    """

    def __init__(self, in_channels=80, d_model=256, channels_mult=None,
                 n_res_blocks=2, n_heads=4):
        super().__init__()
        if channels_mult is None:
            channels_mult = [1, 2, 4]

        self.conditioning = ConditioningModule(d_model)
        cond_dim = d_model

        # Input projection: cat(x_t[80], text_expanded[256]) -> d_model
        input_ch = in_channels + d_model
        self.input_proj = nn.Conv1d(input_ch, d_model, 1)

        channels = [d_model * m for m in channels_mult]  # [256, 512, 1024]

        # Encoder (downsampling)
        self.encoder_levels = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.encoder_text_projs = nn.ModuleList()
        prev_ch = d_model
        for i, ch in enumerate(channels):
            # Project input channels to level channels if needed
            in_conv = nn.Conv1d(prev_ch, ch, 1) if prev_ch != ch else nn.Identity()
            self.encoder_levels.append(nn.ModuleList([
                in_conv,
                UNetLevel(ch, cond_dim, ch, n_res_blocks, n_heads),
            ]))
            self.encoder_text_projs.append(
                nn.Linear(d_model, ch) if d_model != ch else nn.Identity()
            )
            if i < len(channels) - 1:
                self.downsamplers.append(Downsample1D(ch))
            prev_ch = ch

        # Bottleneck
        self.bottleneck = UNetLevel(channels[-1], cond_dim, channels[-1], n_res_blocks, n_heads)
        self.bottleneck_text_proj = nn.Linear(d_model, channels[-1]) if d_model != channels[-1] else nn.Identity()

        # Decoder (upsampling)
        self.decoder_levels = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.decoder_text_projs = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        for i, ch in enumerate(reversed_channels):
            # Skip connection doubles channels
            skip_ch = ch
            in_ch = ch + skip_ch if i > 0 else ch
            merge_conv = nn.Conv1d(in_ch, ch, 1) if in_ch != ch else nn.Identity()
            self.decoder_levels.append(nn.ModuleList([
                merge_conv,
                UNetLevel(ch, cond_dim, ch, n_res_blocks, n_heads),
            ]))
            self.decoder_text_projs.append(
                nn.Linear(d_model, ch) if d_model != ch else nn.Identity()
            )
            if i < len(reversed_channels) - 1:
                self.upsamplers.append(Upsample1D(ch, reversed_channels[i + 1]))

        # Output projection
        self.output_proj = nn.Conv1d(channels[0], in_channels, 1)

    def forward(self, x_t, t, desc_embed, text_enc_expanded, mel_lengths):
        """
        Args:
            x_t: [B, 80, T_mel] noisy mel
            t: [B] timestep
            desc_embed: [B, d_model] projected description
            text_enc_expanded: [B, T_mel, d_model] length-regulated text
            mel_lengths: [B]
        Returns:
            v_pred: [B, 80, T_mel] velocity prediction
        """
        # Conditioning
        cond = self.conditioning(t, desc_embed)  # [B, d_model]

        # Pad input to multiple of 4 for clean downsample/upsample
        T_orig = x_t.shape[2]
        n_levels = len(self.downsamplers)
        factor = 2 ** n_levels
        if T_orig % factor != 0:
            pad_len = factor - (T_orig % factor)
            x_t = torch.nn.functional.pad(x_t, (0, pad_len))
            text_enc_expanded = torch.nn.functional.pad(text_enc_expanded, (0, 0, 0, pad_len))

        # Input
        text_ctx = text_enc_expanded  # [B, T_mel_padded, d_model]
        x = torch.cat([x_t, text_enc_expanded.transpose(1, 2)], dim=1)  # [B, 336, T_mel]
        x = self.input_proj(x)  # [B, d_model, T_mel]

        # Mel mask for cross-attention: True where padded
        B, T = x.shape[0], x.shape[2]
        mel_mask = torch.arange(T, device=x.device).unsqueeze(0) >= mel_lengths.unsqueeze(1)

        # Encoder
        skips = []
        for i, (level_modules, text_proj) in enumerate(zip(self.encoder_levels, self.encoder_text_projs)):
            in_conv, level = level_modules
            x = in_conv(x)
            ctx = text_proj(text_ctx)
            x = level(x, cond, ctx, mel_mask)
            skips.append(x)
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)
                # Downsample context and mask
                text_ctx = text_ctx[:, ::2, :]
                mel_mask = mel_mask[:, ::2]

        # Bottleneck
        ctx = self.bottleneck_text_proj(text_ctx)
        x = self.bottleneck(x, cond, ctx, mel_mask)

        # Decoder
        reversed_skips = list(reversed(skips))
        for i, (level_modules, text_proj) in enumerate(zip(self.decoder_levels, self.decoder_text_projs)):
            merge_conv, level = level_modules
            if i > 0:
                skip = reversed_skips[i]
                # Align lengths (upsample may differ by 1 due to odd sizes)
                min_len = min(x.shape[2], skip.shape[2])
                x = x[:, :, :min_len]
                skip = skip[:, :, :min_len]
                x = torch.cat([x, skip], dim=1)
            x = merge_conv(x)
            ctx = text_proj(text_ctx)
            x = level(x, cond, ctx, mel_mask)
            if i < len(self.upsamplers):
                x = self.upsamplers[i](x)
                # Upsample context and mask back, then trim to match x
                text_ctx = text_ctx.repeat_interleave(2, dim=1)[:, :x.shape[2], :]
                mel_mask = mel_mask.repeat_interleave(2, dim=1)[:, :x.shape[2]]

        out = self.output_proj(x)
        return out[:, :, :T_orig]
