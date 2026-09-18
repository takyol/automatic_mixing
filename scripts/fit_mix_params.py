#!/usr/bin/env python
"""CLI: recovers the gain/pan per stem that a real mix was built with and
writes it as mix_params.yaml next to each song's target.wav.

Usage: python scripts/fit_mix_params.py --root data_processed --pattern "Song*" --config configs/kaggle.yaml

Works for mixes that are a static gain/pan sum of the stems - which the
Spheres mixes are (fit R^2 = 1.000). The written file has the same shape
as Synthmix's, so it serves both evaluation (scripts/eval_synthmix.py)
and training-time stem dropout (the `augment` config section).
"""
import argparse
from pathlib import Path

import yaml

from automix.prep.fit_params import write_fitted_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="processed corpus root")
    parser.add_argument("--pattern", default="*", help="song folder glob, e.g. 'Song*' for Spheres")
    parser.add_argument("--config", type=Path, default=None,
                        help="training config; its `anchors` decide which stems count as anchored")
    args = parser.parse_args()

    anchor_patterns = None
    if args.config is not None:
        with open(args.config) as f:
            anchor_patterns = yaml.safe_load(f).get("anchors")

    results = write_fitted_params(args.root, args.pattern, anchor_patterns)
    if not results:
        raise SystemExit(f"no song folders under {args.root} match {args.pattern!r}")


if __name__ == "__main__":
    main()
