import torch.nn as nn


class DescriptionEncoder(nn.Module):
    """Projects precomputed MiniLM embeddings to model dimension.

    MiniLM is frozen and precomputed during preprocessing.
    Only the 2-layer projection is trainable (~164K params).
    """

    def __init__(self, minilm_dim=384, proj_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(minilm_dim, proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, desc_embed):
        """
        Args:
            desc_embed: [B, minilm_dim] precomputed MiniLM embedding
        Returns:
            [B, proj_dim]
        """
        return self.proj(desc_embed)
