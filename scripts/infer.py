#!/usr/bin/env python
"""CLI: renders a stereo mix from a folder of stem WAV files using a
trained checkpoint.

Usage: python scripts/infer.py --stems-dir path/to/stems --checkpoint checkpoints/best.pt --output mix.wav
"""
import argparse
from pathlib import Path
import torch
import torchaudio
from automix.inference import render_mix


def main():
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--stems-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=default_device)
    args = parser.parse_args()

    mix, sample_rate = render_mix(args.stems_dir, args.checkpoint, device=args.device)
    torchaudio.save(str(args.output), mix, sample_rate)


if __name__ == "__main__":
    main()
