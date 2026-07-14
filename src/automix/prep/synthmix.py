import math
import os
import random
import shutil
from fnmatch import fnmatchcase
from pathlib import Path

import torch
import yaml

from automix.audio_io import load_wav, save_wav


def _matched_degrees(name: str, patterns: dict):
    for pattern, degrees in patterns.items():
        if fnmatchcase(name, pattern):
            return float(degrees)
    return None


def generate_synthetic_mixes(input_root: Path, output_root: Path, anchors: dict,
                             seating: dict, pan_jitter_degrees: float,
                             spot_gain_db_range, anchor_gain_range, seed: int = 0):
    """Renders synthetic ground-truth targets from already-prepped songs.

    Each output song reuses the input song's stems but replaces target.wav
    with a mix rendered by this project's own gain+pan model class, so the
    target is exactly representable and the sampled parameters are a known
    correct answer for evaluating training:

      target = anchor_gain * (anchored stems at their fixed pans)
             + sum(spot_gain_i * spot_i at seating-chart pan + jitter)

    Spot pans come from `seating` ({instrument-name glob -> degrees,
    0=L 45=C 90=R}); unmatched stems default to center. Pans are
    per-instrument (learnable from content), only gains and jitter are
    random. The sampled parameters are written to mix_params.yaml per
    song. Reproducible given `seed`; each song derives its own RNG so
    the corpus can grow without reshuffling existing songs.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)

    for song_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        stems_dir = song_dir / "stems"
        stem_paths = sorted(stems_dir.glob("*.wav")) if stems_dir.is_dir() else []
        if not stem_paths:
            continue

        rng = random.Random(f"{seed}:{song_dir.name}")
        anchor_gain = rng.uniform(*anchor_gain_range)

        out_stems = output_root / song_dir.name / "stems"
        out_stems.mkdir(parents=True, exist_ok=True)

        target = None
        params = {"seed": seed, "anchor_gain": anchor_gain, "tracks": {}}
        for path in stem_paths:
            waveform, sample_rate = load_wav(path)
            mono = waveform.mean(dim=0)

            anchor_degrees = _matched_degrees(path.stem, anchors or {})
            if anchor_degrees is not None:
                gain = anchor_gain
                pan_degrees = anchor_degrees
            else:
                gain = 10 ** (rng.uniform(*spot_gain_db_range) / 20)
                base = _matched_degrees(path.stem, seating or {})
                if base is None:
                    base = 45.0
                pan_degrees = base + rng.uniform(-pan_jitter_degrees, pan_jitter_degrees)
                pan_degrees = min(max(pan_degrees, 0.0), 90.0)
                params["tracks"][path.stem] = {"gain": gain, "pan_degrees": pan_degrees}

            theta = math.radians(pan_degrees)
            if target is None:
                target = torch.zeros(2, mono.shape[0])
            left = gain * math.cos(theta)
            right = gain * math.sin(theta)
            target[0] += left * mono
            target[1] += right * mono

            out_path = out_stems / path.name
            if not out_path.exists():
                try:
                    os.link(path, out_path)
                except OSError:
                    shutil.copy2(path, out_path)

        save_wav(output_root / song_dir.name / "target.wav", target, sample_rate,
                 subtype="FLOAT")
        with open(output_root / song_dir.name / "mix_params.yaml", "w") as f:
            yaml.safe_dump(params, f)
