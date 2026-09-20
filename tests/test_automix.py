"""Tests for the parts where a silent bug would be expensive: the pan law,
the anchor mapping, the parameter recovery, and the dataset's target
correction under stem dropout (a wrong sign there poisons training without
ever raising).

Run with ./venv/bin/pytest. VGGish and the training loop are not covered -
`python scripts/train.py --config configs/smoke_test.yaml` exercises those
end to end in about a minute.
"""
import math

import numpy as np
import torch
import yaml

from automix.anchors import anchor_thetas_for
from automix.audio_io import save_wav
from automix.data.manifest import build_manifest
from automix.data.mix_dataset import MixDataset, _draw_weights
from automix.model.mixer import apply_gain_pan
from automix.prep.fit_params import fit_mix_params

SR = 8000


def test_pan_law_is_constant_power():
    stem = torch.randn(1, 1, 1000)
    mask = torch.ones(1, 1, dtype=torch.bool)
    powers = []
    for degrees in (0, 30, 45, 60, 90):
        theta = torch.tensor([[math.radians(degrees)]])
        mix = apply_gain_pan(stem, torch.ones(1, 1), theta, mask)
        powers.append(float(mix.pow(2).sum()))
    assert max(powers) - min(powers) < 1e-3 * max(powers)


def test_pan_extremes_and_center():
    stem = torch.randn(1, 1, 1000)
    mask = torch.ones(1, 1, dtype=torch.bool)
    left = apply_gain_pan(stem, torch.ones(1, 1), torch.zeros(1, 1), mask)
    right = apply_gain_pan(stem, torch.ones(1, 1), torch.full((1, 1), math.pi / 2), mask)
    assert float(left[0, 1].abs().max()) < 1e-6      # theta=0 -> nothing on the right
    assert float(right[0, 0].abs().max()) < 1e-6     # theta=pi/2 -> nothing on the left


def test_masked_tracks_do_not_reach_the_mix():
    stems = torch.stack([torch.ones(1, 100), torch.full((1, 100), 9.0)], dim=1).squeeze(2)
    mask = torch.tensor([[True, False]])
    mix = apply_gain_pan(stems, torch.ones(1, 2), torch.full((1, 2), math.pi / 4), mask)
    assert float(mix.max()) < 1.0  # the padded track's 9.0 never shows up


def test_anchor_patterns_map_to_radians_and_nan():
    thetas = anchor_thetas_for(
        ["a/Main_L.wav", "a/Main_C.wav", "a/Main_R.wav", "a/Violin_1.wav"],
        {"*_L": 0, "*_C": 45, "*_R": 90})
    assert torch.allclose(thetas[:3], torch.tensor([0.0, math.pi / 4, math.pi / 2]))
    assert torch.isnan(thetas[3])  # unmatched stems stay learned


def test_draw_weights_split_the_configured_share():
    class E:
        def __init__(self, song_id):
            self.song_id = song_id
    entries = [E("Song1"), E("Song2"), E("other_a"), E("other_b"), E("other_c")]
    weights = _draw_weights(entries, {"Song*": 0.4})
    assert math.isclose(sum(weights[:2]), 0.4)
    assert math.isclose(sum(weights[2:]), 0.6)
    assert math.isclose(weights[0], weights[1]) and math.isclose(weights[2], weights[4])


def _write_song(root, gains, pans, seconds=2.0, seed=0):
    """Builds a song whose target is exactly sum(gain * pan(stem))."""
    rng = np.random.default_rng(seed)
    stems_dir = root / "stems"
    stems_dir.mkdir(parents=True)
    left = np.zeros(int(seconds * SR), dtype=np.float32)
    right = np.zeros_like(left)
    for i, (gain, pan) in enumerate(zip(gains, pans)):
        stem = rng.standard_normal(len(left)).astype(np.float32) * 0.1
        save_wav(stems_dir / f"stem_{i}.wav", torch.from_numpy(stem).unsqueeze(0), SR)
        theta = math.radians(pan)
        left += gain * math.cos(theta) * stem
        right += gain * math.sin(theta) * stem
    save_wav(root / "target.wav", torch.from_numpy(np.stack([left, right])), SR)


def test_fit_recovers_known_gain_and_pan(tmp_path):
    gains, pans = [0.8, 0.4, 1.2], [10.0, 45.0, 80.0]
    _write_song(tmp_path / "song", gains, pans)
    params = fit_mix_params(tmp_path / "song", stride=1)
    assert min(params["r2"]) > 0.999
    for i, (gain, pan) in enumerate(zip(gains, pans)):
        got = params["tracks"][f"stem_{i}"]
        assert abs(got["gain"] - gain) < 0.01
        assert abs(got["pan_degrees"] - pan) < 0.5


def test_stem_dropout_subtracts_the_dropped_share_from_the_target(tmp_path):
    """With dropout 1.0 exactly one stem survives (the dataset never empties a
    clip), so the corrected target must be that stem's contribution alone -
    the strongest check that the subtraction uses the right gain and angle."""
    gains, pans = [0.8, 0.4, 1.2], [10.0, 45.0, 80.0]
    song = tmp_path / "song"
    _write_song(song, gains, pans)
    params = fit_mix_params(song, stride=1)
    (song / "mix_params.yaml").write_text(yaml.safe_dump(params))

    entries = build_manifest(tmp_path)
    dataset = MixDataset(entries, sample_rate=SR, clip_seconds=1.0, clips_per_epoch=4, seed=0,
                         augment={"patterns": ["song"], "stem_dropout": 1.0})
    stems, _, target = dataset[0]
    assert stems.shape[0] == 1, "dropout must never empty a clip"
    kept = stems[0]
    # the corrected target must equal the surviving stem's own contribution,
    # i.e. match one of the known (gain, pan) pairs
    def explains(gain, pan):
        theta = math.radians(pan)
        expected = torch.stack([gain * math.cos(theta) * kept, gain * math.sin(theta) * kept])
        return float((target - expected).abs().max()) < 1e-3 * float(expected.abs().max())

    assert any(explains(g, p) for g, p in zip(gains, pans))

    keep_all = MixDataset(entries, sample_rate=SR, clip_seconds=1.0, clips_per_epoch=4, seed=0,
                          augment={"patterns": ["song"], "stem_dropout": 0.0})
    _, _, full_target = keep_all[0]
    assert float(full_target.abs().max()) > 1e-2  # the uncorrected target is not silent
