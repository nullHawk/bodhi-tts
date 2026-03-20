import torch


def euler_solve(model, noise, n_steps, desc_embed, text_enc_expanded, mel_lengths):
    """Euler ODE solver for flow matching inference.

    Integrates from t=0 (noise) to t=1 (mel).

    Args:
        model: BodhiTTS model
        noise: [B, 80, T_mel] initial noise
        n_steps: number of Euler steps
        desc_embed: [B, d_model] projected description embedding
        text_enc_expanded: [B, T_mel, d_model] length-regulated text
        mel_lengths: [B]
    Returns:
        x: [B, 80, T_mel] generated mel spectrogram
    """
    dt = 1.0 / n_steps
    x = noise

    for i in range(n_steps):
        t_val = i * dt
        t = torch.full((noise.shape[0],), t_val, device=noise.device)
        v = model.decode_step(x, t, desc_embed, text_enc_expanded, mel_lengths)
        x = x + dt * v

    return x
