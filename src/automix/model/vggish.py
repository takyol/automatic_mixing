import torch
import torch.nn as nn
import torchaudio
from torchaudio.prototype.pipelines import VGGISH

VGGISH_SAMPLE_RATE = 16000


class VGGishEncoder(nn.Module):
    """Frozen VGGish encoder producing one 128-d embedding per track.

    Uses torchaudio's native, pure-PyTorch VGGish port.
    All B*N tracks in a batch are framed into fixed-size examples and
    run through the backbone batched together, then the per-track frame
    embeddings are split back out and time-averaged. Backbone calls are
    capped at `max_examples_per_batch` examples: a full-length song at
    inference time frames into hundreds of examples per track, and
    pushing them through the CNN in one call runs GPUs out of memory.
    """

    def __init__(self, native_sample_rate: int, backbone: nn.Module = None,
                 input_processor=None, resampler: nn.Module = None,
                 max_examples_per_batch: int = 512):
        super().__init__()
        self.max_examples_per_batch = max_examples_per_batch
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
        chunks = torch.split(all_examples, self.max_examples_per_batch)
        all_embeddings = torch.cat([self.backbone(chunk) for chunk in chunks], dim=0)

        embeddings = [frame_embeddings.mean(dim=0)
                      for frame_embeddings in torch.split(all_embeddings, counts)]

        stacked = torch.stack(embeddings, dim=0)
        return stacked.reshape(b, n, -1)
