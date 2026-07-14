import random

import torch
from torch.utils.data import Dataset

from automix.anchors import anchor_thetas_for
from automix.audio_io import load_wav


class MixDataset(Dataset):
    """Draws `clips_per_epoch` random 5-second clips, one song chosen
    uniformly at random per draw, then a random offset within it.

    Each item: (stems: Tensor(N, T), target: Tensor(2, T)) at the
    corpus's canonical sample rate.

    The (song, offset) draws are precomputed and stored in `_draws` so
    that `__getitem__` is a pure index lookup — this keeps behavior
    correct and reproducible under multi-worker DataLoader, where a
    stateful RNG called inside `__getitem__` would give inconsistent
    results across worker processes. Call `resample()` to redraw the
    clip set (e.g. once per training epoch); leave untouched for a
    fixed validation set.
    """

    def __init__(self, entries: list, sample_rate: int, clip_seconds: float = 5.0,
                 clips_per_epoch: int = 1000, seed: int = None, anchor_patterns: dict = None):
        self.clip_frames = int(clip_seconds * sample_rate)
        self.entries = [e for e in entries if e.num_frames >= self.clip_frames]
        if not self.entries:
            raise ValueError("No songs long enough for the requested clip length")
        self.sample_rate = sample_rate
        self.clips_per_epoch = clips_per_epoch
        self._anchors = {e.song_id: anchor_thetas_for(e.stem_paths, anchor_patterns)
                         for e in self.entries}
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
            waveform, _ = load_wav(stem_path, frame_offset=start, num_frames=self.clip_frames)
            stems.append(waveform.mean(dim=0))
        stems_tensor = torch.stack(stems, dim=0)

        target, _ = load_wav(entry.target_path, frame_offset=start, num_frames=self.clip_frames)

        return stems_tensor, self._anchors[entry.song_id], target
