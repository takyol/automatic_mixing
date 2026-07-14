#!/usr/bin/env python
"""CLI: prints the gain and pan angle a trained checkpoint assigns to
each stem in a folder. The key diagnostic is the *spread* of theta: if
every track gets nearly the same angle, the model is not differentiating
tracks and the mix will collapse toward one side of the stereo field.

Usage: python scripts/dump_gain_pan.py --stems-dir data_processed/song_1/stems --checkpoint checkpoints/best.pt [--config configs/default.yaml] [--offset-seconds 60] [--clip-seconds 5]

Pass the training config via --config to mark anchored (fixed-pan)
tracks and report the spread over the learned pans only.
"""
import argparse
import math
from pathlib import Path

import torch
import yaml

from automix.anchors import anchor_thetas_for
from automix.audio_io import load_wav
from automix.model.automix_model import AutomixModel
from automix.model.context import masked_mean_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stems-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None,
                        help="training config; its `anchors` patterns mark fixed-pan tracks")
    parser.add_argument("--offset-seconds", type=float, default=60.0)
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    args = parser.parse_args()

    anchor_patterns = None
    if args.config is not None:
        with open(args.config) as f:
            anchor_patterns = yaml.safe_load(f).get("anchors")

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

    anchors = anchor_thetas_for(stem_paths, anchor_patterns)

    learned = []
    for path, (gain, theta), anchor in zip(stem_paths, gain_theta, anchors):
        if torch.isnan(anchor):
            degrees = math.degrees(theta.item())
            learned.append(degrees)
            note = ""
        else:
            degrees = math.degrees(anchor.item())
            note = "  [ANCHORED]"
        print(f"  {path.name:28s} gain={gain.item():8.4f} theta={degrees:7.2f} deg  (0=L, 45=C, 90=R){note}")

    if learned:
        learned = torch.tensor(learned)
        print(f"\nlearned-pan spread: min={learned.min():.1f} max={learned.max():.1f} "
              f"std={learned.std():.2f} deg")


if __name__ == "__main__":
    main()
