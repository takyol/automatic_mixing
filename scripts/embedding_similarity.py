#!/usr/bin/env python
"""CLI: measures how distinguishable a song's stems are to the VGGish
encoder, via pairwise cosine similarity of their embeddings. High mean
similarity (near 1.0) means the mixing MLP cannot tell the tracks apart
and will collapse to one shared gain/pan for all of them.

Usage: python scripts/embedding_similarity.py --stems-dir data_processed/song_1/stems [--offset-seconds 60] [--clip-seconds 5]
"""
import argparse
from pathlib import Path

import torch

from automix.audio_io import load_wav
from automix.model.vggish import VGGishEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stems-dir", type=Path, required=True)
    parser.add_argument("--offset-seconds", type=float, default=60.0)
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    args = parser.parse_args()

    stem_paths = sorted(p for p in args.stems_dir.iterdir()
                        if p.suffix.lower() in (".wav", ".flac"))
    if len(stem_paths) < 2:
        raise ValueError(f"Need at least 2 stems in {args.stems_dir}")

    waveforms = []
    sample_rate = None
    for path in stem_paths:
        probe, sr = load_wav(path, num_frames=1)
        sample_rate = sr
        offset = int(args.offset_seconds * sr)
        num_frames = int(args.clip_seconds * sr)
        waveform, _ = load_wav(path, frame_offset=offset, num_frames=num_frames)
        waveforms.append(waveform.mean(dim=0))

    stems = torch.stack(waveforms).unsqueeze(0)

    encoder = VGGishEncoder(native_sample_rate=sample_rate)
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder(stems)[0]  # (N, 128)

    normalized = torch.nn.functional.normalize(embeddings, dim=1)
    similarity = normalized @ normalized.T
    n = similarity.shape[0]

    print(f"{n} stems from {args.stems_dir}")
    print(f"clip: {args.clip_seconds}s at offset {args.offset_seconds}s\n")
    off_diag = similarity[~torch.eye(n, dtype=torch.bool)]
    print(f"pairwise cosine similarity: min={off_diag.min():.4f} "
          f"mean={off_diag.mean():.4f} max={off_diag.max():.4f}\n")
    for i, path in enumerate(stem_paths):
        others = torch.cat([similarity[i, :i], similarity[i, i + 1:]])
        print(f"  {path.name:28s} mean sim to others: {others.mean():.4f}")


if __name__ == "__main__":
    main()
