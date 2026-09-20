import math
import random
from fnmatch import fnmatchcase

import torch
import yaml
from torch.utils.data import Dataset

from automix.anchors import anchor_thetas_for
from automix.audio_io import load_wav


def _draw_weights(entries: list, sampling_fractions: dict):
    """Per-entry draw weights so songs matching each glob in
    `sampling_fractions` ({song_id glob: share of draws}) together get that
    share, split evenly within the group; unmatched songs share the rest.
    An entry belongs to the first pattern it matches. Returns None when no
    fractions are configured (plain uniform choice)."""
    if not sampling_fractions:
        return None
    groups = {pattern: [] for pattern in sampling_fractions}
    rest = []
    for i, entry in enumerate(entries):
        pattern = next((p for p in sampling_fractions if fnmatchcase(entry.song_id, p)), None)
        (groups[pattern] if pattern is not None else rest).append(i)

    claimed = sum(sampling_fractions[p] for p, members in groups.items() if members)
    if claimed > 1 or (rest and claimed >= 1):
        raise ValueError(f"sampling_fractions {sampling_fractions} leave no share for the "
                         f"{len(rest)} unmatched songs (must sum to < 1)")
    weights = [0.0] * len(entries)
    for pattern, members in groups.items():
        for i in members:
            weights[i] = sampling_fractions[pattern] / len(members)
    for i in rest:
        weights[i] = (1 - claimed) / len(rest)
    return weights


class MixDataset(Dataset):
    """Draws `clips_per_epoch` random 5-second clips: a song per draw
    (uniformly, or by `sampling_fractions`), then a random offset within it.

    Each item: (stems: Tensor(N, T), anchor_theta: Tensor(N), target:
    Tensor(2, T)) at the corpus's canonical sample rate.

    `augment` ({patterns, level_db, stem_dropout}) applies to songs whose
    id matches one of `patterns`:
      - level_db: each stem is scaled by a random gain in +-level_db before
        the model sees it, target unchanged - the correct gain shifts by
        the opposite amount, so the model must read levels from the audio.
      - stem_dropout: each non-anchored stem is removed with this
        probability and its exact contribution subtracted from the target,
        using the song's mix_params.yaml (Synthmix ground truth, or a fit
        from scripts/fit_mix_params.py). Yields new valid mixes from a
        few real ones.

    The (song, offset, augmentation) draws are precomputed and stored in
    `_draws` so that `__getitem__` is a pure index lookup — this keeps
    behavior correct and reproducible under multi-worker DataLoader, where
    a stateful RNG called inside `__getitem__` would give inconsistent
    results across worker processes. Call `resample()` to redraw the clip
    set (e.g. once per training epoch); leave untouched for a fixed
    validation set.
    """

    def __init__(self, entries: list, sample_rate: int, clip_seconds: float = 5.0,
                 clips_per_epoch: int = 1000, seed: int = None, anchor_patterns: dict = None,
                 sampling_fractions: dict = None, augment: dict = None):
        self.clip_frames = int(clip_seconds * sample_rate)
        self.entries = [e for e in entries if e.num_frames >= self.clip_frames]
        if not self.entries:
            raise ValueError("No songs long enough for the requested clip length")
        self.sample_rate = sample_rate
        self.clips_per_epoch = clips_per_epoch
        self._anchors = {e.song_id: anchor_thetas_for(e.stem_paths, anchor_patterns)
                         for e in self.entries}
        self._weights = _draw_weights(self.entries, sampling_fractions)

        augment = augment or {}
        self._level_db = float(augment.get("level_db", 0.0))
        self._stem_dropout = float(augment.get("stem_dropout", 0.0))
        patterns = augment.get("patterns", [])
        self._augmented = {e.song_id for e in self.entries
                           if any(fnmatchcase(e.song_id, p) for p in patterns)}
        self._params = {}  # song_id -> {stem name: (gain, theta)} for dropout
        if self._stem_dropout > 0:
            for entry in self.entries:
                if entry.song_id in self._augmented:
                    self._params[entry.song_id] = self._load_spot_params(entry)

        self._rng = random.Random(seed)
        self._draws = []
        self.resample()

    @staticmethod
    def _load_spot_params(entry):
        path = entry.target_path.parent / "mix_params.yaml"
        if not path.exists():
            raise ValueError(f"stem_dropout needs {path} - run scripts/fit_mix_params.py "
                             f"for real mixes (Synthmix songs have it already)")
        tracks = yaml.safe_load(path.read_text())["tracks"]
        return {name: (v["gain"], math.radians(v["pan_degrees"])) for name, v in tracks.items()}

    def _draw_augmentation(self, entry):
        if entry.song_id not in self._augmented:
            return None
        n = len(entry.stem_paths)
        levels = None
        if self._level_db > 0:
            levels = [10 ** (self._rng.uniform(-self._level_db, self._level_db) / 20)
                      for _ in range(n)]
        keep = list(range(n))
        if self._stem_dropout > 0:
            params = self._params[entry.song_id]
            anchors = self._anchors[entry.song_id]
            keep = [i for i, path in enumerate(entry.stem_paths)
                    if not (torch.isnan(anchors[i]) and path.stem in params
                            and self._rng.random() < self._stem_dropout)]
            if not keep:  # a song of few spots can lose all of them; keep one
                keep = [self._rng.randrange(n)]
        return levels, keep

    def resample(self):
        self._draws = []
        for _ in range(self.clips_per_epoch):
            if self._weights is None:
                entry = self._rng.choice(self.entries)
            else:
                entry = self._rng.choices(self.entries, weights=self._weights)[0]
            max_start = entry.num_frames - self.clip_frames
            start = self._rng.randint(0, max_start)
            self._draws.append((entry, start, self._draw_augmentation(entry)))

    def __len__(self):
        return self.clips_per_epoch

    def __getitem__(self, index):
        entry, start, augmentation = self._draws[index]

        stems = []
        for stem_path in entry.stem_paths:
            waveform, _ = load_wav(stem_path, frame_offset=start, num_frames=self.clip_frames)
            stems.append(waveform.mean(dim=0))
        anchor_theta = self._anchors[entry.song_id]

        target, _ = load_wav(entry.target_path, frame_offset=start, num_frames=self.clip_frames)

        if augmentation is not None:
            levels, keep = augmentation
            kept = set(keep)
            params = self._params.get(entry.song_id, {})
            for i, path in enumerate(entry.stem_paths):
                if i not in kept:  # remove the dropped stem's exact share of the mix
                    gain, theta = params[path.stem]
                    target[0] -= gain * math.cos(theta) * stems[i]
                    target[1] -= gain * math.sin(theta) * stems[i]
            if levels is not None:
                stems = [s * level for s, level in zip(stems, levels)]
            stems = [stems[i] for i in keep]
            anchor_theta = anchor_theta[keep]

        return torch.stack(stems, dim=0), anchor_theta, target
