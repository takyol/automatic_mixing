#!/usr/bin/env python
"""CLI: turns a raw multitrack session with an engineer's stereo mix into
corpus songs (one per time window), so a real concert can be trained on.

Usage: python scripts/prepare_session.py --config configs/wuppertal_prep.yaml [--dry-run]

Each window is fitted to the mix track (see prep/session.py) and only kept
when the fit is good enough, because a passage our static mixer cannot
express would otherwise be trained on as if it were reachable. `--dry-run`
scans and reports without writing any audio.
"""
import argparse
from pathlib import Path

import yaml

from automix.prep.session import prepare_session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="scan and report, write nothing")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    anchor_patterns = config.get("anchors")
    if isinstance(anchor_patterns, str):  # path to a training config
        anchor_patterns = yaml.safe_load(Path(anchor_patterns).read_text()).get("anchors")

    results, lag = prepare_session(
        raw_dir=config["raw_dir"],
        output_root=config["output_root"],
        mix_name=config.get("mix_track", "SUMME.wav"),
        rename=config.get("rename"),
        exclude=config.get("exclude", ()),
        window_seconds=config.get("window_seconds", 30.0),
        stride_seconds=config.get("stride_seconds", 60.0),
        min_r2=config.get("min_r2", 0.95),
        min_dbfs=config.get("min_dbfs", -55.0),
        lag_seconds=config.get("lag_seconds"),
        anchor_patterns=anchor_patterns,
        song_prefix=config.get("song_prefix", "session"),
        max_windows=config.get("max_windows"),
        target_level_db=config.get("target_level_db"),
        dry_run=args.dry_run,
    )

    kept = [r for r in results if r["kept"]]
    quiet = [r for r in results if r.get("reason") == "quiet"]
    rejected = [r for r in results if r.get("reason") == "r2"]
    print(f"\nmix track lags the multitrack by {lag * 1000:.1f} ms")
    print(f"{len(results)} windows scanned: {len(kept)} kept, {len(rejected)} below min_r2, "
          f"{len(quiet)} too quiet")
    if kept:
        span = config.get("window_seconds", 30.0) * len(kept) / 60
        print(f"kept material: {span:.1f} min, fit R^2 "
              f"{min(r['r2'] for r in kept):.3f}-{max(r['r2'] for r in kept):.3f}")
    for r in results:
        if "r2" in r:
            mark = "keep" if r["kept"] else f"drop ({r['reason']})"
            print(f"  {r['start']:7.0f}s  R^2 {r['r2']:.3f}  {r['n_tracks']:2d} tracks  "
                  f"{r['level_db']:6.1f} dBFS  {mark}")


if __name__ == "__main__":
    main()
