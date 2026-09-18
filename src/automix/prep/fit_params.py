import math
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
import yaml

from automix.audio_io import load_wav


def _nnls(a: np.ndarray, y: np.ndarray, max_iter: int = 50000, tol: float = 1e-10) -> np.ndarray:
    """Non-negative least squares min ||a x - y||, x >= 0, via accelerated
    projected gradient (FISTA) on the normal equations. The problem is
    tiny (one unknown per stem), so this is fast and needs no scipy."""
    g = a.T @ a
    b = a.T @ y
    step = 1.0 / np.linalg.eigvalsh(g).max()
    x = z = np.zeros(a.shape[1])
    t = 1.0
    for _ in range(max_iter):
        x_new = np.maximum(0.0, z - step * (g @ z - b))
        t_new = (1 + math.sqrt(1 + 4 * t * t)) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        if np.linalg.norm(x_new - x) <= tol * max(1.0, np.linalg.norm(x_new)):
            return x_new
        x, t = x_new, t_new
    return x


def fit_mix_params(song_dir: Path, anchor_patterns: dict = None, stride: int = 2) -> dict:
    """Recovers the static gain + pan a mix was built with, assuming the
    target is a sum of the mono stems under this project's constant-power
    pan law: target_L = sum g_i cos(theta_i) s_i, target_R = sum g_i sin(theta_i) s_i.

    Fits each output channel by non-negative least squares, then converts
    the per-channel coefficients (l_i, r_i) to gain = hypot(l_i, r_i) and
    theta = atan2(r_i, l_i). R^2 per channel is returned so a mix that is
    NOT a static linear sum (reverb, EQ, automation) is easy to spot.

    Returns the mix_params.yaml structure also written by Synthmix:
    `tracks` holds the non-anchored stems (what the model must predict);
    `anchor_tracks` the anchored ones, kept for reference.
    """
    song_dir = Path(song_dir)
    stem_paths = sorted((song_dir / "stems").glob("*.wav"))
    stems = [load_wav(p)[0].mean(dim=0).double().numpy() for p in stem_paths]
    target = load_wav(song_dir / "target.wav")[0].double().numpy()
    n = min(min(len(s) for s in stems), target.shape[1])
    a = np.stack([s[:n:stride] for s in stems], axis=1)

    coeffs, r2 = [], []
    for ch in range(2):
        y = target[ch, :n:stride]
        x = _nnls(a, y)
        resid = y - a @ x
        coeffs.append(x)
        r2.append(float(1 - (resid @ resid) / (y @ y)))

    params = {"source": "nnls_fit", "r2": r2, "tracks": {}, "anchor_tracks": {}}
    for path, left, right in zip(stem_paths, *coeffs):
        entry = {"gain": float(math.hypot(left, right)),
                 "pan_degrees": float(math.degrees(math.atan2(right, left)))}
        anchored = any(fnmatchcase(path.stem, p) for p in (anchor_patterns or {}))
        params["anchor_tracks" if anchored else "tracks"][path.stem] = entry
    return params


def write_fitted_params(processed_root: Path, song_pattern: str, anchor_patterns: dict = None,
                        min_r2: float = 0.99) -> list:
    """Fits and writes mix_params.yaml for every song folder under
    processed_root whose name matches song_pattern. Returns
    (song, r2) pairs; warns when a fit is poor (the mix is then not a
    static gain/pan sum and the recovered values are unreliable)."""
    results = []
    for song_dir in sorted(p for p in Path(processed_root).iterdir()
                           if p.is_dir() and fnmatchcase(p.name, song_pattern)):
        params = fit_mix_params(song_dir, anchor_patterns)
        with open(song_dir / "mix_params.yaml", "w") as f:
            yaml.safe_dump(params, f)
        r2 = min(params["r2"])
        flag = "" if r2 >= min_r2 else f"  WARNING: R^2 below {min_r2}, fit unreliable"
        print(f"{song_dir.name}: {len(params['tracks'])} spots, "
              f"{len(params['anchor_tracks'])} anchored, R^2 {r2:.4f}{flag}")
        results.append((song_dir.name, r2))
    return results
