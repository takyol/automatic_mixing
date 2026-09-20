"""Turns a raw multitrack session (one WAV per recorder track, plus the
engineer's stereo mix as another track) into corpus songs.

Unlike the other prep modules there is no per-song structure to read: the
session is one long recording, so it is cut into fixed-length windows and
each window becomes a "song". Two things make the result trustworthy:

- The mix track usually lags the multitrack by the console's latency. That
  offset is estimated by cross-correlation and compensated; without it, the
  target does not line up with the stems at all.
- Not every passage is a static gain/pan sum of its tracks - fader moves,
  reverb and processing are not expressible by our mixer. Each window is
  therefore fitted (NNLS, as in fit_params) and kept only if the fit is good
  enough, so training never sees a target it could not reach.
"""
import math
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
import torch
import yaml

from automix.audio_io import frame_count, load_wav, save_wav
from automix.prep.fit_params import _nnls


def estimate_lag_seconds(mix_path: Path, reference_path: Path, start: float,
                         seconds: float = 20.0, max_lag_seconds: float = 2.0) -> float:
    """How far the mix track lags `reference_path` (a main mic), by
    cross-correlation. Positive = the mix is later."""
    sample_rate = load_wav(mix_path, num_frames=1)[1]
    mix, _ = load_wav(mix_path, frame_offset=int(start * sample_rate),
                      num_frames=int(seconds * sample_rate))
    reference, _ = load_wav(reference_path, frame_offset=int(start * sample_rate),
                            num_frames=int(seconds * sample_rate))
    a = mix.mean(dim=0).double().numpy()
    b = reference.mean(dim=0).double().numpy()
    size = 1 << int(math.ceil(math.log2(len(a) + len(b))))
    correlation = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    max_lag = int(max_lag_seconds * sample_rate)
    window = np.concatenate([correlation[-max_lag:], correlation[:max_lag + 1]])
    return (int(np.argmax(np.abs(window))) - max_lag) / sample_rate


def _track_paths(raw_dir: Path, mix_name: str, exclude: set):
    return [p for p in sorted(raw_dir.glob("*.wav"))
            if not p.name.startswith("._") and p.stem != Path(mix_name).stem
            and p.stem not in exclude]


def _read_window(path: Path, start_frame: int, frames: int) -> torch.Tensor:
    waveform, _ = load_wav(path, frame_offset=start_frame, num_frames=frames)
    return waveform.mean(dim=0)


def fit_window(tracks: list, target: torch.Tensor, stride: int = 4):
    """NNLS fit of the window's target from its tracks. Returns
    (r2, gains, pans in degrees) with one entry per track."""
    a = np.stack([t[::stride].double().numpy() for t in tracks], axis=1)
    coefficients, r2 = [], []
    for channel in range(2):
        y = target[channel, ::stride].double().numpy()
        x = _nnls(a, y)
        residual = y - a @ x
        coefficients.append(x)
        r2.append(float(1 - (residual @ residual) / (y @ y + 1e-20)))
    gains = [math.hypot(l, r) for l, r in zip(*coefficients)]
    pans = [math.degrees(math.atan2(r, l)) for l, r in zip(*coefficients)]
    return min(r2), gains, pans


def prepare_session(raw_dir, output_root, mix_name="SUMME.wav", rename=None, exclude=(),
                    window_seconds=30.0, stride_seconds=60.0, min_r2=0.95, min_dbfs=-55.0,
                    lag_seconds=None, anchor_patterns=None, song_prefix="wup", fit_stride=4,
                    max_windows=None, dry_run=False, target_level_db=None, peak_ceiling=0.97):
    """Scans the session window by window and writes the ones worth training
    on as `<output_root>/<prefix>_<start>s/{stems/*.wav, target.wav,
    mix_params.yaml}`. Returns one result dict per scanned window, so a
    caller can report how much of the session survived the gate."""
    raw_dir, output_root = Path(raw_dir), Path(output_root)
    rename = rename or {}
    exclude = set(exclude)
    mix_path = raw_dir / mix_name
    track_paths = _track_paths(raw_dir, mix_name, exclude)
    if not track_paths:
        raise ValueError(f"no usable tracks in {raw_dir}")

    sample_rate = load_wav(mix_path, num_frames=1)[1]
    total_frames = min(frame_count(p) for p in [mix_path] + track_paths)
    if lag_seconds is None:
        lag_seconds = estimate_lag_seconds(mix_path, track_paths[0], start=total_frames / sample_rate * 0.2)
    lag_frames = int(round(lag_seconds * sample_rate))
    frames = int(window_seconds * sample_rate)
    step = int(stride_seconds * sample_rate)

    results, written = [], 0
    for start_frame in range(0, total_frames - frames - lag_frames, step):
        target, _ = load_wav(mix_path, frame_offset=start_frame + lag_frames, num_frames=frames)
        level_db = 20 * math.log10(float(target.pow(2).mean().sqrt()) + 1e-12)
        start_seconds = start_frame / sample_rate
        if level_db < min_dbfs:  # applause, silence between movements
            results.append({"start": start_seconds, "level_db": level_db, "kept": False,
                            "reason": "quiet"})
            continue

        names, tracks = [], []
        for path in track_paths:
            waveform = _read_window(path, start_frame, frames)
            if 20 * math.log10(float(waveform.pow(2).mean().sqrt()) + 1e-12) < -70:
                continue  # track unused in this passage
            names.append(rename.get(path.stem, path.stem.replace(" ", "_")))
            tracks.append(waveform)

        r2, gains, pans = fit_window(tracks, target, stride=fit_stride)
        keep = r2 >= min_r2 and (max_windows is None or written < max_windows)
        results.append({"start": start_seconds, "level_db": level_db, "r2": r2,
                        "n_tracks": len(names), "kept": keep,
                        "reason": None if keep else ("r2" if r2 < min_r2 else "limit")})
        if not keep or dry_run:
            continue

        # A concert's raw level is far below the other corpora's (here ~20 dB),
        # which would let these windows dominate a mixed corpus' loss. Scaling
        # tracks AND target by one factor fixes that and leaves the fitted
        # gains valid, because the mix equation is linear in both sides.
        scale = 1.0
        if target_level_db is not None:
            scale = 10 ** ((target_level_db - level_db) / 20)
            peak = max(float(target.abs().max()), max(float(t.abs().max()) for t in tracks))
            scale = min(scale, peak_ceiling / (peak + 1e-12))

        song_dir = output_root / f"{song_prefix}_{int(start_seconds):06d}s"
        (song_dir / "stems").mkdir(parents=True, exist_ok=True)
        for name, waveform in zip(names, tracks):
            save_wav(song_dir / "stems" / f"{name}.wav", (waveform * scale).unsqueeze(0), sample_rate)
        save_wav(song_dir / "target.wav", target * scale, sample_rate)
        params = {"source": "session_nnls_fit", "r2": [r2, r2], "lag_seconds": lag_seconds,
                  "level_scale": float(scale), "tracks": {}, "anchor_tracks": {}}
        for name, gain, pan in zip(names, gains, pans):
            anchored = any(fnmatchcase(name, p) for p in (anchor_patterns or {}))
            params["anchor_tracks" if anchored else "tracks"][name] = {
                "gain": float(gain), "pan_degrees": float(pan)}
        (song_dir / "mix_params.yaml").write_text(yaml.safe_dump(params))
        written += 1
    return results, lag_seconds
