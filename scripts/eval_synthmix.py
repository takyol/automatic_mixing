#!/usr/bin/env python
"""CLI: evaluates a checkpoint against Synthmix ground truth (mix_params.yaml).

Usage: python scripts/eval_synthmix.py --root data_processed_synthmix_val --checkpoint checkpoints/<run>/best.pt --config configs/kaggle.yaml --out eval.json

`--root` holds Synthmix song folders (stems/, target.wav, mix_params.yaml),
e.g. the val songs regenerated with scripts/generate_synthmix.py.

For each song: predicts gain+pan per stem exactly the way inference does
(full-song VGGish embedding, anchors applied), compares the learned pans
against the known mix_params.yaml ground truth, and measures the L-R
balance of the rendered mix against the target's.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from automix.anchors import anchor_thetas_for
from automix.audio_io import load_wav
from automix.model.automix_model import AutomixModel
from automix.model.context import masked_mean_context
from automix.model.mixer import apply_gain_pan

MAX_SECONDS = 120.0  # cap so long songs still fit in RAM


def balance_db(mix):
    l = mix[0].pow(2).mean().sqrt().item()
    r = mix[1].pow(2).mean().sqrt().item()
    eps = 1e-12
    return 20 * math.log10((l + eps) / (r + eps))


def evaluate_song(song_dir, model, anchor_patterns, device):
    stems_dir = song_dir / "stems"
    stem_paths = sorted(stems_dir.glob("*.wav"))
    params = yaml.safe_load((song_dir / "mix_params.yaml").read_text())
    truth = params["tracks"]

    waveforms, sample_rate = [], None
    for p in stem_paths:
        w, sr = load_wav(p)
        sample_rate = sr
        waveforms.append(w.mean(dim=0))
    max_len = max(w.shape[0] for w in waveforms)
    cap = int(MAX_SECONDS * sample_rate)
    if max_len > cap:  # centered excerpt
        start = (max_len - cap) // 2
        waveforms = [w[start:start + cap] for w in waveforms]
        max_len = max(w.shape[0] for w in waveforms)
        excerpt = True
    else:
        excerpt = False

    padded = [F.pad(w, (0, max_len - w.shape[0])) for w in waveforms]
    stems = torch.stack(padded, 0).unsqueeze(0).to(device)
    mask = torch.ones(1, len(padded), dtype=torch.bool, device=device)
    anchor_theta = anchor_thetas_for(stem_paths, anchor_patterns).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encoder(stems)
        ctx = masked_mean_context(emb, mask).unsqueeze(1).expand(-1, emb.shape[1], -1)
        gain_theta = model.mlp(emb, ctx)
        gain = gain_theta[..., 0]
        theta = gain_theta[..., 1]
        theta_final = torch.where(torch.isnan(anchor_theta), theta, anchor_theta)
        mix = apply_gain_pan(stems, gain, theta_final, mask)[0].cpu()

    target, _ = load_wav(song_dir / "target.wav")
    if excerpt:
        start = (target.shape[1] - cap) // 2
        target = target[:, start:start + cap]

    rows = []
    for i, p in enumerate(stem_paths):
        name = p.stem
        anchored = not math.isnan(anchor_theta[0, i].item())
        pred_deg = math.degrees(theta[0, i].item())
        pred_gain = gain[0, i].item()
        t = truth.get(name)
        rows.append({
            "stem": name, "anchored": anchored,
            "pred_pan_deg": pred_deg, "pred_gain": pred_gain,
            "true_pan_deg": None if t is None else t["pan_degrees"],
            "true_gain": None if t is None else t["gain"],
        })

    learned = [r for r in rows if not r["anchored"] and r["true_pan_deg"] is not None]
    pan_err = [abs(r["pred_pan_deg"] - r["true_pan_deg"]) for r in learned]
    center_err = [abs(45.0 - r["true_pan_deg"]) for r in learned]
    gain_db_err = [20 * math.log10(max(r["pred_gain"], 1e-9) / r["true_gain"]) for r in learned]
    med_off = sorted(gain_db_err)[len(gain_db_err) // 2] if gain_db_err else 0.0
    # a constant gain, scale-corrected the same way, errs by |t - median(t)|
    true_db = sorted(20 * math.log10(r["true_gain"]) for r in learned)
    med_true = true_db[len(true_db) // 2] if true_db else 0.0

    return {
        "song": song_dir.name,
        "n_stems": len(stem_paths),
        "n_learned": len(learned),
        "excerpt": excerpt,
        "pan_mae_deg": sum(pan_err) / len(pan_err) if pan_err else float("nan"),
        "pan_median_deg": sorted(pan_err)[len(pan_err) // 2] if pan_err else float("nan"),
        "pan_max_deg": max(pan_err) if pan_err else float("nan"),
        "center_baseline_mae_deg": sum(center_err) / len(center_err) if center_err else float("nan"),
        "true_pan_spread_deg": (max(r["true_pan_deg"] for r in learned)
                                - min(r["true_pan_deg"] for r in learned)) if learned else float("nan"),
        "pred_pan_spread_deg": (max(r["pred_pan_deg"] for r in learned)
                                - min(r["pred_pan_deg"] for r in learned)) if learned else float("nan"),
        "gain_mae_db_scalecorr": (sum(abs(g - med_off) for g in gain_db_err) / len(gain_db_err)) if gain_db_err else float("nan"),
        "gain_global_offset_db": med_off,
        "gain_const_baseline_mae_db": (sum(abs(t - med_true) for t in true_db) / len(true_db)) if true_db else float("nan"),
        "balance_pred_db": balance_db(mix),
        "balance_target_db": balance_db(target),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    anchor_patterns = yaml.safe_load(args.config.read_text()).get("anchors")
    songs = sorted(p for p in args.root.iterdir() if (p / "mix_params.yaml").exists())

    model = None
    results = []
    for song_dir in songs:
        probe, sr = load_wav(next((song_dir / "stems").glob("*.wav")), num_frames=1)
        if model is None:
            model = AutomixModel(native_sample_rate=sr).to(args.device)
            ck = torch.load(args.checkpoint, map_location=args.device)
            model.mlp.load_state_dict(ck["mlp_state_dict"])
            model.eval()
            print(f"checkpoint epoch {ck['epoch'] + 1}, val_loss {ck['val_loss']:.4f}")
        r = evaluate_song(song_dir, model, anchor_patterns, args.device)
        results.append(r)
        print(f"{r['song']:<38} pan MAE {r['pan_mae_deg']:5.1f} deg "
              f"(center baseline {r['center_baseline_mae_deg']:5.1f}) "
              f"spread {r['pred_pan_spread_deg']:5.1f}/{r['true_pan_spread_deg']:5.1f} "
              f"bal {r['balance_pred_db']:+6.2f} vs {r['balance_target_db']:+6.2f} dB")

    n = len(results)
    print("\n=== mean over %d songs ===" % n)
    for k in ("pan_mae_deg", "center_baseline_mae_deg", "gain_mae_db_scalecorr",
              "gain_const_baseline_mae_db"):
        print(f"  {k:<28} {sum(r[k] for r in results) / n:6.2f}")
    print(f"  {'|balance| pred':<28} {sum(abs(r['balance_pred_db']) for r in results) / n:6.2f} dB")
    print(f"  {'|balance| target':<28} {sum(abs(r['balance_target_db']) for r in results) / n:6.2f} dB")
    args.out.write_text(json.dumps(results, indent=2))
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
