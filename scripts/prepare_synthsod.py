#!/usr/bin/env python
"""CLI: preprocesses SynthSOD into the common data_processed layout.

Usage: python scripts/prepare_synthsod.py --config configs/synthsod_prep.yaml
"""
import argparse
from pathlib import Path
import yaml
from automix.prep.synthsod import prepare_synthsod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    prepare_synthsod(
        raw_root=Path(config["raw_root"]),
        output_root=Path(config["output_root"]),
        target_sample_rate=config["target_sample_rate"],
        close_mic_glob=config["close_mic_glob"],
        tree_mic_files=config["tree_mic_files"],
    )


if __name__ == "__main__":
    main()
