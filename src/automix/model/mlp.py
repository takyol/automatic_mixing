import torch
import torch.nn as nn
import torch.nn.functional as F


class PostProcessorMLP(nn.Module):
    """Predicts (gain, pan) per track from concat(track embedding, context).

    Input: (B, N, 256) = concat(track_embedding[128], context[128])
    Output: (B, N, 2) = (gain, theta), gain >= 0, theta in [0, pi/2]
    """

    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        input_dim = embedding_dim * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, track_embedding: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = torch.cat([track_embedding, context], dim=-1)
        raw = self.net(x)
        gain = F.softplus(raw[..., 0])
        theta = torch.sigmoid(raw[..., 1]) * (torch.pi / 2)
        return torch.stack([gain, theta], dim=-1)
