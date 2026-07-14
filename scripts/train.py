#!/usr/bin/env python
"""CLI: trains the automixer MLP.

Usage: python scripts/train.py --config configs/spheres.yaml
"""
import argparse
from pathlib import Path

import yaml

from automix.data.manifest import build_manifest, split_train_val
from automix.data.mix_dataset import MixDataset
from automix.device import resolve_device
from automix.model.automix_model import AutomixModel
from automix.train_loop import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    entries = build_manifest(Path(config["data_processed_root"]))
    train_entries, val_entries = split_train_val(
        entries, val_fraction=config["val_fraction"], seed=config["split_seed"])

    sample_rate = config["sample_rate"]
    anchor_patterns = config.get("anchors")
    train_dataset = MixDataset(train_entries, sample_rate=sample_rate,
                                clip_seconds=config["clip_seconds"],
                                clips_per_epoch=config["clips_per_epoch"],
                                seed=config["split_seed"],
                                anchor_patterns=anchor_patterns)
    val_dataset = MixDataset(val_entries, sample_rate=sample_rate,
                              clip_seconds=config["clip_seconds"],
                              clips_per_epoch=config["val_clips"],
                              seed=config["split_seed"] + 1,
                              anchor_patterns=anchor_patterns)

    model = AutomixModel(native_sample_rate=sample_rate)

    device = resolve_device(config.get("device", "auto"))

    train(
        model, train_dataset, val_dataset,
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        checkpoint_dir=Path(config["checkpoint_dir"]),
        log_dir=Path(config["log_dir"]),
        device=device,
        checkpoint_every=config.get("checkpoint_every", 1),
    )


if __name__ == "__main__":
    main()
