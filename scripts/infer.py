#!/usr/bin/env python
"""CLI: renders a stereo mix from a folder of stem WAV files using a
trained checkpoint.

Usage: python scripts/infer.py --stems-dir path/to/stems --checkpoint checkpoints_spheres/best.pt --output mix.wav [--config configs/spheres.yaml]

Pass the training config via --config so the same anchor patterns
(fixed-pan tracks) used in training are applied at inference.
"""
import argparse
from pathlib import Path

import yaml

from automix.audio_io import save_wav
from automix.device import resolve_device
from automix.inference import render_mix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stems-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", type=Path, default=None,
                        help="training config; its `anchors` patterns are applied at inference")
    args = parser.parse_args()

    anchor_patterns = None
    if args.config is not None:
        with open(args.config) as f:
            anchor_patterns = yaml.safe_load(f).get("anchors")

    mix, sample_rate = render_mix(args.stems_dir, args.checkpoint,
                                  device=resolve_device(args.device),
                                  anchor_patterns=anchor_patterns)
    save_wav(args.output, mix, sample_rate)


if __name__ == "__main__":
    main()
