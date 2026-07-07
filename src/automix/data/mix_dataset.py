import random
import torch
import torchaudio
from torch.utils.data import Dataset


class MixDataset(Dataset):
    """Draws random 5-second clips, then a random offset within it.

    Each item: (stems: Tensor(N, T), target: Tensor(2, T)) at the
    corpus's sample rate.

    The (song, offset) draws are precomputed and stored in `_draws` so
    that `__getitem__` is a pure index lookup.
    """

    def __init__(self, entries: list, sample_rate: int, clip_seconds: float = 5.0,
                 clips_per_epoch: int = 1000, seed: int = None):
        self.clip_frames = int(clip_seconds * sample_rate)
        self.entries = [e for e in entries if e.num_frames >= self.clip_frames]
        if not self.entries:
            raise ValueError("No songs long enough for the requested clip length")
        self.sample_rate = sample_rate
        self.clips_per_epoch = clips_per_epoch
        self._rng = random.Random(seed)
        self._draws = []
        self.resample()

    def resample(self):
        self._draws = []
        for _ in range(self.clips_per_epoch):
            entry = self._rng.choice(self.entries)
            max_start = entry.num_frames - self.clip_frames
            start = self._rng.randint(0, max_start)
            self._draws.append((entry, start))

    def __len__(self):
        return self.clips_per_epoch

    def __getitem__(self, index):
        entry, start = self._draws[index]
        stems = []
        
        for stem_path in entry.stem_paths:
            waveform, _ = torchaudio.load(str(stem_path), frame_offset=start, num_frames=self.clip_frames)
            stems.append(waveform.mean(dim=0))
        
        stems_tensor = torch.stack(stems, dim=0)
        target, _ = torchaudio.load(str(entry.target_path), frame_offset=start, num_frames=self.clip_frames)

        return stems_tensor, target
