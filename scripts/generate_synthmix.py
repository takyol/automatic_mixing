#!/usr/bin/env python
"""CLI: renders synthetic ground-truth mixes from prepped SynthSOD songs.

Usage: python scripts/generate_synthmix.py --config configs/synthmix_gen.yaml

Run scripts/prepare_synthsod.py first; this reads its output and writes
a parallel corpus whose target.wav is a known gain+pan render (stems are
hard-linked, not copied, so disk cost is just the new targets). Each
song's sampled parameters land in mix_params.yaml for evaluation.
"""
import argparse
from pathlib import Path

import yaml

from automix.prep.synthmix import generate_synthetic_mixes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    generate_synthetic_mixes(
        input_root=Path(config["input_root"]),
        output_root=Path(config["output_root"]),
        anchors=config["anchors"],
        seating=config["seating"],
        pan_jitter_degrees=config["pan_jitter_degrees"],
        spot_gain_db_range=tuple(config["spot_gain_db_range"]),
        anchor_gain_range=tuple(config["anchor_gain_range"]),
        seed=config["seed"],
    )


if __name__ == "__main__":
    main()
