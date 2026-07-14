#!/usr/bin/env python
"""CLI: prints the gain and pan angle a trained checkpoint assigns to
each stem in a folder. The key diagnostic is the *spread* of theta: if
every track gets nearly the same angle, the model is not differentiating
tracks and the mix will collapse toward one side of the stereo field.

Usage: python scripts/dump_gain_pan.py --stems-dir data_processed/song_1/stems --checkpoint checkpoints/best.pt [--offset-seconds 60] [--clip-seconds 5]
"""
import argparse
import math
from pathlib import Path

import torch

from automix.audio_io import load_wav
from automix.model.automix_model import AutomixModel
from automix.model.context import masked_mean_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stems-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--offset-seconds", type=float, default=60.0)
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    args = parser.parse_args()

    stem_paths = sorted(p for p in args.stems_dir.iterdir()
                        if p.suffix.lower() in (".wav", ".flac"))
    if not stem_paths:
        raise ValueError(f"No stems found in {args.stems_dir}")

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
    mask = torch.ones(1, len(waveforms), dtype=torch.bool)

    model = AutomixModel(native_sample_rate=sample_rate)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.mlp.load_state_dict(checkpoint["mlp_state_dict"])
    model.eval()

    print(f"checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch')}, "
          f"val_loss {checkpoint.get('val_loss'):.4f})")
    print(f"clip: {args.clip_seconds}s at offset {args.offset_seconds}s\n")

    with torch.no_grad():
        embeddings = model.encoder(stems)
        context = masked_mean_context(embeddings, mask)
        context_broadcast = context.unsqueeze(1).expand(-1, embeddings.shape[1], -1)
        gain_theta = model.mlp(embeddings, context_broadcast)[0]

    thetas = []
    for path, (gain, theta) in zip(stem_paths, gain_theta):
        degrees = math.degrees(theta.item())
        thetas.append(degrees)
        print(f"  {path.name:28s} gain={gain.item():8.4f} theta={degrees:7.2f} deg  (0=L, 45=C, 90=R)")

    thetas = torch.tensor(thetas)
    print(f"\ntheta spread: min={thetas.min():.1f} max={thetas.max():.1f} "
          f"std={thetas.std():.2f} deg")


if __name__ == "__main__":
    main()
