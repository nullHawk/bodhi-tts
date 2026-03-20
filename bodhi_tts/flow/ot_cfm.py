import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def compute_ot_plan(x_0, x_1):
    """Mini-batch optimal transport: find permutation that minimizes transport cost.

    Args:
        x_0: [B, C, T] source (noise)
        x_1: [B, C, T] target (mel)
    Returns:
        permuted x_0: [B, C, T]
    """
    B = x_0.shape[0]
    if B <= 1:
        return x_0

    # Flatten for cost matrix
    x0_flat = x_0.reshape(B, -1)  # [B, C*T]
    x1_flat = x_1.reshape(B, -1)  # [B, C*T]

    # Pairwise L2 cost: [B, B]
    cost = torch.cdist(x0_flat, x1_flat, p=2).detach().cpu().numpy()

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost)

    # Permute x_0
    return x_0[col_ind]


def sample_and_compute_loss(model, batch, dur_loss_weight=0.1):
    """OT-CFM training step.

    Args:
        model: BodhiTTS
        batch: dict with text_ids, mel, desc_embed, text_lengths, mel_lengths
        dur_loss_weight: weight for duration prediction loss
    Returns:
        total_loss, flow_loss, dur_loss
    """
    text_ids = batch["text_ids"]
    mel = batch["mel"]           # [B, 80, T_mel]
    desc_embed = batch["desc_embed"]
    text_lengths = batch["text_lengths"]
    mel_lengths = batch["mel_lengths"]

    B = mel.shape[0]
    device = mel.device

    # 1. Sample t ~ U(0, 1)
    t = torch.rand(B, device=device)

    # 2. Sample noise x_0 ~ N(0, I)
    x_0 = torch.randn_like(mel)

    # 3. Mini-batch OT: permute x_0 to minimize transport cost
    x_0 = compute_ot_plan(x_0, mel)

    # 4. Interpolate: x_t = (1-t)*x_0 + t*mel
    t_expand = t[:, None, None]  # [B, 1, 1]
    x_t = (1 - t_expand) * x_0 + t_expand * mel

    # 5. Target velocity: v = mel - x_0
    v_target = mel - x_0

    # 6. Forward pass
    v_pred, dur_loss, _ = model(text_ids, text_lengths, mel, mel_lengths, desc_embed, t, x_t)

    # 7. Masked MSE loss
    T_mel = mel.shape[2]
    mel_mask = (torch.arange(T_mel, device=device).unsqueeze(0) < mel_lengths.unsqueeze(1)).unsqueeze(1).float()
    # mel_mask: [B, 1, T_mel]

    flow_loss = ((v_pred - v_target) ** 2 * mel_mask).sum() / mel_mask.sum() / 80

    total_loss = flow_loss + dur_loss_weight * dur_loss
    return total_loss, flow_loss, dur_loss
