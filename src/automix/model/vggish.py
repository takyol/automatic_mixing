import torch
import torch.nn as nn
import torchaudio
from torchaudio.prototype.pipelines import VGGISH

VGGISH_SAMPLE_RATE = 16000


class VGGishEncoder(nn.Module):
    """Frozen VGGish encoder producing one 128-d embedding per track.

    Uses torchaudio's native, pure-PyTorch VGGish port.
    All B*N tracks in a batch are framed into fixed-size examples and
    run through the backbone in a single forward call, then the
    per-track frame embeddings are split back out and time-averaged.
    """

    def __init__(self, native_sample_rate: int, backbone: nn.Module = None,
                 input_processor=None, resampler: nn.Module = None):
        super().__init__()
        if backbone is None:
            backbone = VGGISH.get_model()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.backbone.eval()
        self.input_processor = input_processor if input_processor is not None else VGGISH.get_input_processor()
        if resampler is None:
            resampler = torchaudio.transforms.Resample(native_sample_rate, VGGISH_SAMPLE_RATE)
        self.resampler = resampler

    def train(self, mode: bool = True):
        # Backbone must stay frozen regardless of the outer model's mode.
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, stems: torch.Tensor) -> torch.Tensor:
        """
        stems: (B, N, T) raw audio at native_sample_rate

        Returns: (B, N, 128) one embedding per track.
        """
        b, n, t = stems.shape
        flat = stems.reshape(b * n, t)
        resampled = self.resampler(flat)

        examples_per_track = [self.input_processor(resampled[i]) for i in range(resampled.shape[0])]
        counts = [examples.shape[0] for examples in examples_per_track]

        all_examples = torch.cat(examples_per_track, dim=0)
        all_embeddings = self.backbone(all_examples)  # one batched forward call for every track

        embeddings = [frame_embeddings.mean(dim=0)
                      for frame_embeddings in torch.split(all_embeddings, counts)]

        stacked = torch.stack(embeddings, dim=0)
        return stacked.reshape(b, n, -1)
