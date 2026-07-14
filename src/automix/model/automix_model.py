import torch
import torch.nn as nn

from automix.model.context import masked_mean_context
from automix.model.mixer import apply_gain_pan
from automix.model.mlp import PostProcessorMLP
from automix.model.vggish import VGGishEncoder


class AutomixModel(nn.Module):
    """End-to-end: frozen VGGish -> masked-mean context -> MLP -> mixer.

    Only `self.mlp` has trainable parameters.
    """

    def __init__(self, native_sample_rate: int, vggish_backbone: nn.Module = None):
        super().__init__()
        self.encoder = VGGishEncoder(native_sample_rate, backbone=vggish_backbone)
        self.mlp = PostProcessorMLP()

    def forward(self, stems: torch.Tensor, mask: torch.Tensor,
                anchor_theta: torch.Tensor = None) -> torch.Tensor:
        """
        stems: (B, N, T) raw audio, native sample rate
        mask: (B, N) bool, True = real track
        anchor_theta: optional (B, N) fixed pan angles in radians; NaN
            entries keep the model-predicted pan. Anchored tracks keep
            their learned gain.

        Returns: (B, 2, T) predicted stereo mix.
        """
        with torch.no_grad():
            embeddings = self.encoder(stems)  # (B, N, 128)

        context = masked_mean_context(embeddings, mask)  # (B, 128)
        context_broadcast = context.unsqueeze(1).expand(-1, embeddings.shape[1], -1)

        gain_theta = self.mlp(embeddings, context_broadcast)  # (B, N, 2)
        gain = gain_theta[..., 0]
        theta = gain_theta[..., 1]

        if anchor_theta is not None:
            theta = torch.where(torch.isnan(anchor_theta), theta, anchor_theta)

        return apply_gain_pan(stems, gain, theta, mask)
