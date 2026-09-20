#!/usr/bin/env python
"""CLI: is a candidate mix a target worth training on?

Fits gain and pan per track DIRECTLY to the target with Adam on the training
loss. The result is the loss floor - the best any model could reach with this
parameterization - plus the share of mix energy the non-anchored tracks end up
carrying, i.e. how much there would be to learn about them at all.

Usage: python scripts/target_feasibility.py --tracks-dir "path/to/session" \
    --target SUMME.wav --lag-ms 12.1 --config configs/kaggle.yaml \
    --rename "A=Main_L,B=Main_R,C=Main_C,OUT A=Out_L,OUT B=Out_R" \
    --exclude "TC,SUR A,SUR B,SUMME" --starts 1500,2040,5070,6510

Reference points (reports/evaluation.md): a corpus whose target IS a static
gain/pan sum reaches ~0.01, the trained model's validation loss is ~1.4, and a
target the parameterization cannot express stays above 5. A floor near zero
combined with a ~0% spot share is the other failure mode: the anchored main
mics already explain the target and the optimum mutes everything else.

`--target` is a stereo track inside the session folder; `--lag-ms` shifts it
against the multitrack (console latency - check it with a cross-correlation
first, an uncompensated offset makes the fit meaningless).
"""
import argparse
import math
from pathlib import Path

import torch
import yaml

from automix.anchors import anchor_thetas_for
from automix.audio_io import load_wav
from automix.losses.mrstft import MultiResolutionSTFTLoss


def load_case(args, start):
    sample_rate = load_wav(args.target, num_frames=1)[1]
    frames = int(args.seconds * sample_rate)
    offset = int(start * sample_rate)
    target, _ = load_wav(args.target, frame_offset=offset + int(args.lag_ms / 1000 * sample_rate),
                         num_frames=frames)

    rename = dict(pair.split("=", 1) for pair in args.rename.split(",")) if args.rename else {}
    exclude = set(args.exclude.split(",")) if args.exclude else set()
    names, tracks = [], []
    for path in sorted(args.tracks_dir.glob("*.wav")):
        if path.name.startswith("._") or path.stem in exclude or path == args.target:
            continue
        waveform, _ = load_wav(path, frame_offset=offset, num_frames=frames)
        waveform = waveform.mean(dim=0)
        if 20 * math.log10(float(waveform.pow(2).mean().sqrt()) + 1e-12) < args.min_dbfs:
            continue  # silent in this excerpt
        names.append(rename.get(path.stem, path.stem.replace(" ", "_")))
        tracks.append(waveform)
    anchor_patterns = yaml.safe_load(args.config.read_text()).get("anchors")
    anchors = anchor_thetas_for([Path(n + ".wav") for n in names], anchor_patterns)
    return names, torch.stack(tracks), target, anchors


def loss_floor(tracks, target, anchor_theta, steps, lr=0.05):
    """Returns (start loss, best loss, gain, theta). Gain and pan use the same
    softplus/sigmoid parameterization as the MLP's output layer."""
    loss_fn = MultiResolutionSTFTLoss()
    raw_gain = torch.zeros(len(tracks), requires_grad=True)
    raw_theta = torch.zeros(len(tracks), requires_grad=True)
    learned = torch.isnan(anchor_theta)
    optimizer = torch.optim.Adam([raw_gain, raw_theta], lr=lr)
    tracks, target = tracks.unsqueeze(0), target.unsqueeze(0)

    first = None
    for _ in range(steps):
        gain = torch.nn.functional.softplus(raw_gain)
        theta = torch.where(learned, math.pi / 2 * torch.sigmoid(raw_theta), anchor_theta)
        mix = torch.stack([(tracks * (gain * torch.cos(theta)).unsqueeze(-1)).sum(1),
                           (tracks * (gain * torch.sin(theta)).unsqueeze(-1)).sum(1)], dim=1)
        loss = loss_fn(mix, target)
        first = loss.item() if first is None else first
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        gain = torch.nn.functional.softplus(raw_gain)
        theta = torch.where(learned, math.pi / 2 * torch.sigmoid(raw_theta), anchor_theta)
    return first, loss.item(), gain, theta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True,
                        help="stereo mix to test as a target (file name inside --tracks-dir)")
    parser.add_argument("--config", type=Path, required=True, help="training config, for `anchors`")
    parser.add_argument("--starts", default="0", help="comma-separated excerpt starts in seconds")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--lag-ms", type=float, default=0.0,
                        help="how far the target lags the multitrack")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--rename", default="")
    parser.add_argument("--exclude", default="")
    parser.add_argument("--min-dbfs", type=float, default=-70.0)
    parser.add_argument("--drop-anchored", action="store_true",
                        help="remove the anchored tracks from the input - tests whether the "
                             "spots alone could reach the target")
    args = parser.parse_args()
    if not args.target.is_absolute():
        args.target = args.tracks_dir / args.target

    for start in (float(s) for s in args.starts.split(",")):
        names, tracks, target, anchors = load_case(args, start)
        if args.drop_anchored:
            keep = torch.isnan(anchors)
            names = [n for n, k in zip(names, keep) if k]
            tracks, anchors = tracks[keep], anchors[keep]
        first, best, gain, theta = loss_floor(tracks, target, anchors, args.steps)
        energy = gain ** 2 * torch.stack([t.pow(2).mean() for t in tracks])
        share = float(energy[torch.isnan(anchors)].sum() / (energy.sum() + 1e-12))
        print(f"start {start:7.0f}s  {len(names):3d} tracks  loss {first:5.2f} -> {best:5.2f}   "
              f"non-anchored share of mix energy {share:5.1%}")


if __name__ == "__main__":
    main()
