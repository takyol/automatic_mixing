#!/usr/bin/env python
"""CLI: preprocesses Spheres into the common data_processed layout.

Usage: python scripts/prepare_spheres.py --config configs/spheres_prep.yaml
"""
import argparse
from pathlib import Path
import yaml
from automix.prep.spheres import prepare_spheres


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    prepare_spheres(
        raw_root=Path(config["raw_root"]),
        output_root=Path(config["output_root"]),
        target_sample_rate=config["target_sample_rate"],
        stems_subdir=config["stems_subdir"],
        mix_filename=config["mix_filename"],
    )


if __name__ == "__main__":
    main()
