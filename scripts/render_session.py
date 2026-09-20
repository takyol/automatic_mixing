#!/usr/bin/env python
"""CLI: mixes an excerpt of a RAW multitrack session (a folder of
same-length track WAVs, as a recorder writes them) with a trained
checkpoint - no data prep needed.

Usage: python scripts/render_session.py --tracks-dir "/Volumes/T7 2/.../Wuppertal-Multitrack" \
    --start 1500 --seconds 90 --checkpoint checkpoints/<run>/last.pt --config configs/kaggle.yaml \
    --rename "A=Main_L,B=Main_R,C=Main_C,OUT A=Out_L,OUT B=Out_R" \
    --exclude "TC,SUR A,SUR B,SUMME" --out renders/wuppertal_1

`--rename` maps recorder track names to the naming the `anchors`
patterns expect, so the main array keeps its fixed pan (see README);
`--exclude` drops timecode, surround and sum tracks. Tracks that are
silent in the excerpt are dropped automatically - a model input of pure
noise floor would otherwise get a meaningless gain and pan.

Writes `model_mix.wav`, `main_array_only.wav` (the anchored tracks at
unity gain, as a reference to listen against), a console figure and
`gain_pan.json` with the predicted values per track.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from automix.anchors import anchor_thetas_for
from automix.audio_io import load_wav, save_wav
from automix.device import resolve_device
from automix.model.automix_model import AutomixModel
from automix.model.context import masked_mean_context
from automix.model.mixer import apply_gain_pan


def parse_map(text):
    return dict(pair.split("=", 1) for pair in text.split(",")) if text else {}


def load_excerpt(tracks_dir, start, seconds, rename, exclude, min_dbfs):
    names, waveforms, dropped = [], [], []
    sample_rate = None
    for path in sorted(tracks_dir.glob("*.wav")):
        if path.name.startswith("._") or path.stem in exclude:  # ._* = macOS resource forks
            continue
        probe, sr = load_wav(path, num_frames=1)
        waveform, _ = load_wav(path, frame_offset=int(start * sr), num_frames=int(seconds * sr))
        waveform = waveform.mean(dim=0)
        sample_rate = sr
        name = rename.get(path.stem, path.stem.replace(" ", "_"))
        level_db = 20 * math.log10(float(waveform.pow(2).mean().sqrt()) + 1e-12)
        if level_db < min_dbfs:
            dropped.append(f"{name} ({level_db:.0f} dBFS)")
            continue
        names.append(name)
        waveforms.append(waveform)
    if not names:
        raise ValueError(f"No usable tracks in {tracks_dir}")
    print(f"{len(names)} tracks, dropped silent: {', '.join(dropped) or 'none'}")
    return names, torch.stack(waveforms), sample_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="training config; its `anchors` patterns are applied here too")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.0, help="excerpt start in seconds")
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--rename", default="", help="'RecorderName=Stem_Name,...'")
    parser.add_argument("--exclude", default="", help="recorder track names to skip")
    parser.add_argument("--min-dbfs", type=float, default=-70.0)
    parser.add_argument("--title", default=None, help="console figure title")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    names, stems, sample_rate = load_excerpt(
        args.tracks_dir, args.start, args.seconds, parse_map(args.rename),
        set(args.exclude.split(",")) if args.exclude else set(), args.min_dbfs)

    anchor_patterns = yaml.safe_load(args.config.read_text()).get("anchors")
    anchor_theta = anchor_thetas_for([Path(n + ".wav") for n in names], anchor_patterns).unsqueeze(0)

    device = resolve_device(args.device)
    model = AutomixModel(native_sample_rate=sample_rate).to(device)
    model.mlp.load_state_dict(torch.load(args.checkpoint, map_location=device)["mlp_state_dict"])
    model.eval()

    stems = stems.unsqueeze(0).to(device)
    mask = torch.ones(1, len(names), dtype=torch.bool, device=device)
    anchor_theta = anchor_theta.to(device)
    with torch.no_grad():
        embeddings = model.encoder(stems)
        context = masked_mean_context(embeddings, mask).unsqueeze(1).expand(-1, embeddings.shape[1], -1)
        gain, theta = model.mlp(embeddings, context).unbind(-1)
        theta = torch.where(torch.isnan(anchor_theta), theta, anchor_theta)
        mix = apply_gain_pan(stems, gain, theta, mask)[0].cpu()
        anchored = ~torch.isnan(anchor_theta)
        main_only = apply_gain_pan(stems, torch.ones_like(gain), theta, anchored)[0].cpu()

    for path, audio in ((args.out / "model_mix.wav", mix), (args.out / "main_array_only.wav", main_only)):
        save_wav(path, audio / (audio.abs().max() + 1e-9) * 0.89, sample_rate)  # peak-normalized for listening

    rows = [{"track": n, "gain_db": 20 * math.log10(float(gain[0, i]) + 1e-9),
             "pan_deg": math.degrees(float(theta[0, i])), "anchored": bool(anchored[0, i])}
            for i, n in enumerate(names)]
    (args.out / "gain_pan.json").write_text(json.dumps(rows, indent=2))

    import plot_console  # noqa: E402  (sibling script, only needed for the figure)
    plot_console.draw_console(
        [r["track"].replace("_", " ") + ("*" if r["anchored"] else "") for r in rows],
        [10 ** (r["gain_db"] / 20) for r in rows], [math.radians(r["pan_deg"]) for r in rows],
        (args.title or f"{args.tracks_dir.name}, from {args.start:.0f}s") + "   (* = pan anchored, gain learned)",
        args.out / "console.png")

    for row in sorted(rows, key=lambda r: r["pan_deg"]):
        side = "L" if row["pan_deg"] < 44.5 else ("R" if row["pan_deg"] > 45.5 else "C")
        print(f"  {row['track']:<10} {row['gain_db']:+6.1f} dB  {row['pan_deg']:5.1f} deg {side}"
              f"{'  [anchor]' if row['anchored'] else ''}")
    left, right = (mix[0].pow(2).mean().sqrt().item(), mix[1].pow(2).mean().sqrt().item())
    print(f"L-R balance: {20 * math.log10(left / (right + 1e-12)):+.2f} dB -> {args.out}")


if __name__ == "__main__":
    main()
