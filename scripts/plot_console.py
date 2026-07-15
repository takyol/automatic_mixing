#!/usr/bin/env python
"""CLI: draws the gain/pan a trained checkpoint chooses for a folder of
stems, as a mixing-console figure (one channel strip per stem: pan
indicator on top, gain fader below).

Usage: python scripts/plot_console.py --stems-dir data_processed/<song>/stems \
           --checkpoint checkpoints/<run>/best.pt --output reports/console.png \
           [--title "Song name"]

Requires matplotlib (not a core dependency: pip install matplotlib).
"""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch
import torch.nn.functional as F

from automix.audio_io import load_wav
from automix.device import resolve_device
from automix.model.automix_model import AutomixModel
from automix.model.context import masked_mean_context

# palette shared with the report charts
BLUE = "#2a78d6"
INK, MUTED, GRID, BASE, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def model_params(stems_dir: Path, checkpoint_path: Path, device: str):
    """Returns (stem names, linear gains, pan angles theta) the checkpoint
    assigns to the stems. Mirrors inference.render_mix's loading."""
    stem_paths = sorted(Path(stems_dir).glob("*.wav"))
    if not stem_paths:
        raise ValueError(f"No .wav files found in {stems_dir}")

    waveforms, sample_rate = [], None
    for path in stem_paths:
        waveform, sr = load_wav(path)
        sample_rate = sample_rate or sr
        waveforms.append(waveform.mean(dim=0))
    max_len = max(w.shape[0] for w in waveforms)
    stems = torch.stack([F.pad(w, (0, max_len - w.shape[0])) for w in waveforms]
                        ).unsqueeze(0).to(device)
    mask = torch.ones(1, stems.shape[1], dtype=torch.bool, device=device)

    model = AutomixModel(native_sample_rate=sample_rate).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.mlp.load_state_dict(checkpoint["mlp_state_dict"])
    model.eval()

    with torch.no_grad():
        emb = model.encoder(stems)
        ctx = masked_mean_context(emb, mask)
        ctx = ctx.unsqueeze(1).expand(-1, emb.shape[1], -1)
        gain_theta = model.mlp(emb, ctx)[0].cpu()  # (N, 2)

    names = [p.stem for p in stem_paths]
    return names, gain_theta[:, 0].tolist(), gain_theta[:, 1].tolist()


def draw_console(names, gains, thetas, title, out_path):
    n = len(names)
    gains_db = [20 * math.log10(max(g, 1e-6)) for g in gains]
    pans = [(t / (math.pi / 2)) * 2 - 1 for t in thetas]  # -1 = L, +1 = R

    db_lo = min(-24, math.floor(min(gains_db) / 6) * 6)
    db_hi = max(6, math.ceil(max(gains_db) / 6) * 6)

    fig_w = max(6.5, 0.95 * n + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, 6.2), dpi=200)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    ax.set_xlim(-1.25, n - 0.1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    PAN_Y, PAN_W = 0.90, 0.30          # pan slider centre line
    FAD_TOP, FAD_BOT = 0.76, 0.20      # fader track extent
    NAME_Y, DB_Y, PANLBL_Y = 0.075, 0.135, 0.965

    def db_to_y(db):
        return FAD_BOT + (db - db_lo) / (db_hi - db_lo) * (FAD_TOP - FAD_BOT)

    for db in range(int(db_lo), int(db_hi) + 1, 6):
        y = db_to_y(db)
        ax.plot([-0.45, n - 0.55], [y, y], color=GRID, lw=0.75, zorder=1)
        ax.text(-0.55, y, f"{db:+d}" if db else "0", ha="right", va="center",
                color=MUTED, fontsize=7.5)
    ax.text(-1.05, (FAD_TOP + FAD_BOT) / 2, "gain (dB)", ha="center", va="center",
            color=MUTED, fontsize=8.5, rotation=90)

    for i, (name, db, pan) in enumerate(zip(names, gains_db, pans)):
        if i:
            ax.plot([i - 0.5, i - 0.5], [0.04, 0.985], color=GRID, lw=0.6, zorder=1)

        # pan slider
        ax.plot([i - PAN_W, i + PAN_W], [PAN_Y, PAN_Y], color=BASE, lw=1.4,
                solid_capstyle="round", zorder=2)
        ax.plot([i, i], [PAN_Y - 0.012, PAN_Y + 0.012], color=BASE, lw=1.0, zorder=2)
        ax.scatter([i + pan * PAN_W], [PAN_Y], s=52, color=BLUE, zorder=4,
                   edgecolor=SURF, linewidth=1.2)
        ax.text(i - PAN_W, PAN_Y - 0.032, "L", ha="center", color=MUTED, fontsize=7)
        ax.text(i + PAN_W, PAN_Y - 0.032, "R", ha="center", color=MUTED, fontsize=7)
        pan_pct = round(abs(pan) * 100)
        pan_lbl = "C" if pan_pct < 2 else (f"L{pan_pct}" if pan < 0 else f"R{pan_pct}")
        ax.text(i, PANLBL_Y, pan_lbl, ha="center", va="center", color=INK, fontsize=8.5)

        # gain fader
        ax.plot([i, i], [FAD_BOT, FAD_TOP], color=BASE, lw=2.4,
                solid_capstyle="round", zorder=2)
        y = db_to_y(db)
        cap = FancyBboxPatch((i - 0.13, y - 0.016), 0.26, 0.032,
                             boxstyle="round,pad=0.004,rounding_size=0.012",
                             linewidth=1.2, edgecolor=SURF, facecolor=BLUE, zorder=4)
        ax.add_patch(cap)
        ax.plot([i - 0.09, i + 0.09], [y, y], color=SURF, lw=0.9, zorder=5)

        ax.text(i, DB_Y, f"{db:+.1f}", ha="center", va="center", color=INK, fontsize=8.5)
        label = name.split("_", 1)[-1] if "_" in name and n > 8 else name
        ax.text(i, NAME_Y, label, ha="center", va="center", color=MUTED, fontsize=8,
                rotation=0 if len(label) <= 9 else 30)

    ax.set_title(title, color=INK, fontsize=11.5, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURF)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stems-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default=None,
                        help="figure title (default: derived from paths)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    names, gains, thetas = model_params(args.stems_dir, args.checkpoint, device)

    for nm, g, t in zip(names, gains, thetas):
        pan = (t / (math.pi / 2)) * 2 - 1
        print(f"{nm:24s} gain={20*math.log10(max(g,1e-6)):+6.1f} dB  pan={pan:+.2f}")

    title = args.title or (f"{args.stems_dir.parent.name}\n"
                           f"model gain & pan per stem — {args.checkpoint.parent.name}")
    draw_console(names, gains, thetas, title, args.output)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
