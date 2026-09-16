# Experiment log

One row per training run. Artifacts for run `<name>`: checkpoints and the
exact config in `checkpoints/<name>/`, TensorBoard logs in `runs/<name>/`.
Loss values are only comparable between runs that use the same loss function.

| Run | Data | Loss | Result | Takeaway |
|---|---|---|---|---|
| `2026-07-13_1821_default` | 4 Spheres songs (3 train / 1 val) | sum+diff MR-STFT | best val 2.595 @ epoch 3; val rose afterwards, stopped early at 41/200 | Model overfits tiny dataset almost immediately. Render: gains plausible, panning collapsed hard left. More data needed. |
| `2026-07-14_0042_default` | 24 songs: 4 Spheres + 20 SynthSOD (22 train / 2 val) | sum+diff MR-STFT | best val 2.464 @ epoch 50; LR halved @ 71; stopped at ~epoch 80 for loss change | 5x data fixed early overfitting (bests kept coming until epoch 50). Render @ epoch 50: loudness within 1-2 dB of reference, but stereo image still collapsed left (+23 dB L-R on organ concerto). Diagnosis: sum+diff loss is mirror-symmetric - it cannot penalize wrong-side energy. |
| `2026-07-14_0944_default` | same 24 songs | sum+diff **+ per-channel (L,R)** MR-STFT | stopped at epoch ~7 (losses falling normally); superseded by Kaggle run | Tests whether breaking the loss's left/right symmetry fixes the pan collapse. Loss scale ~2x previous runs (4 terms instead of 2). Too slow on the M1 Pro during the day (~10-18 min/epoch) - moved to Kaggle GPU (run name `kaggle`, see KAGGLE.md). Verify via render L-R balance vs the +23 dB baseline above. |
| `kaggle` (60 epochs, P100) | same 24 songs | sum+diff **+ per-channel (L,R)** MR-STFT | best val 5.1765 @ epoch 59 | **Per-channel loss fixes the pan collapse.** Render L-R balance on the two val songs (never trained on): organ concerto +22.7 dB (old) -> **+7.9 dB** (new), ref -0.3; Mozart1 +9.9 -> **+2.1 dB**, ref +0.6. Mozart near-centered; organ much improved but residual left bias remains on the larger ensemble. Controlled result: only the loss changed. Next: full 200-epoch run, and investigate the organ's residual bias (pan parameterization / more data). |
| `2026-09-15_1202_kaggle` (200 epochs, P100, 2 sessions) | 68 songs: 4 Spheres + 64 **Synthmix** (61 train / 7 val) | sum+diff+per-channel MR-STFT **+ pan anchors** | best val 1.4032 @ epoch 121; never beaten afterwards (epoch 200: val 1.4702, train 0.8841) | **First run measured against ground truth in degrees.** On the 7 held-out Synthmix songs (32 spot mics): pan MAE **5.8°** vs 17.1° for a constant-center predictor and 22.4° for the 60-epoch model — against a 2.5° floor set by the generator's ±5° jitter. Seating order recovered monotonically (Violin 1 25° -> Bass 76°), image slightly center-compressed. Stereo balance matches the target (1.29 vs 1.39 dB mean abs). Training converged ~epoch 120; `last.pt` scores 5.37° despite a worse val loss, so MR-STFT val loss is not a perfect proxy for parameter recovery. Caveat: all 7 val songs are synthetic — the 4 real Spheres songs are all in train, so real-mix generalization is still untested. Loss not comparable to earlier rows (corpus, targets and anchors all changed). See `reports/final_run_evaluation.md`. |

## Conventions

- Run names: `<date>_<time>_<config>` (automatic; override with `run_name` in the config).
- After each run: add a row here, and export final curves/renders worth keeping into `reports/`.
- Smoke tests (`configs/smoke_test.yaml`) go to `runs_smoke/` / `checkpoints_smoke/` and are not logged here.
