# Final run: does the model actually recover the mix parameters?

Run `2026-09-15_1202_kaggle` — 200 epochs on a Kaggle P100 across two
sessions, 68 songs (4 Spheres + 64 Synthmix), per-channel MR-STFT loss
plus pan anchors.

Earlier runs could only be judged by *listening* and by a crude
left/right balance number. This is the first run evaluated against a
**known correct answer**: the Synthmix songs are rendered from a fixed
orchestral seating chart with random gains, and the sampled parameters
are stored per song in `mix_params.yaml`. So we can ask directly: how
many degrees off is the model's placement?

## Headline result

Measured on the **7 validation songs** (held out by song; the model
never trained on them), for the 32 spot mics whose pan the model has to
predict:

| Metric | 60-epoch run | **This run (best.pt)** | Constant-center predictor | Perfect predictor |
|---|---|---|---|---|
| Pan error (MAE) | 22.4° | **5.8°** | 17.1° | 2.5° |
| Gain error (MAE, scale-corrected) | 2.50 dB | **1.37 dB** | — | 0 dB |
| Stereo balance \|L−R\| | 1.79 dB | **1.29 dB** | — | 1.39 dB (target) |

The 2.5° floor is not reachable: the generator jitters each pan by ±5°
uniformly, and that jitter is random per song, so no model can predict
it from the audio. **5.8° is a bit over twice the noise floor** — the
model places instruments essentially where the seating chart puts them.

The 60-epoch model (22.4°) was *worse than simply guessing center*
(17.1°). Its predicted pan spread per song was 2–16° against a true
spread of 25–63°: it had not learned to differentiate tracks at all, it
just put everything in roughly one place. That is the pan collapse,
still visible in the numbers. It is gone now — predicted spread is
47–55° on the large ensembles.

## The model learned the seating chart

Predicted vs. true position, averaged over the val songs
(0° = hard left, 45° = center, 90° = hard right):

| Instrument | Predicted | True |
|---|---|---|
| Violin 1 | 25.0° | 15.5° |
| Harp | 19.8° | 17.1° |
| Horn | 44.4° | 29.9° |
| Violin 2 | 34.1° | 31.3° |
| Flute | 42.4° | 40.0° |
| Clarinet | 40.7° | 45.0° |
| Oboe | 39.6° | 50.9° |
| Bassoon | 48.7° | 51.7° |
| Viola | 52.1° | 54.2° |
| Trombone | 59.9° | 61.1° |
| Trumpet | 57.8° | 62.7° |
| Cello | 65.0° | 71.3° |
| Bass | 76.2° | 79.1° |

The ordering is recovered almost perfectly — strings sweep left to
right from Violin 1 to Bass, exactly as the chart seats them. The
model's image is slightly **compressed toward the center**: it
under-shoots the extremes (Violin 1 by 9.5°, Cello by 6.3°). Horn
(14.4°) and Oboe (11.3°) are the weakest, both instruments whose timbre
is easily confused with their neighbours.

## Training plateaued at ~epoch 120

`best.pt` is from **epoch 121** (val 1.4032). The remaining 79 epochs
never beat it; at epoch 200 val was 1.4702 while train had fallen from
0.97 to 0.88 — mild overfitting.

But the val loss is not the whole story. Evaluating `last.pt`
(epoch 200) on the same metric gives **5.37° pan MAE — slightly better
than best.pt's 5.79°**, despite its worse val loss. The MR-STFT
validation loss is therefore not a perfect proxy for parameter
recovery: the last 80 epochs did not improve the audio reconstruction,
but they did not damage the mix parameters either. Practically the two
checkpoints are equivalent; the honest conclusion is that **training
converged around epoch 120** and a future run can stop there.

## Per-song detail (best.pt)

| Song | Spots | Pan MAE | Center baseline | Pred. spread | True spread | Balance pred / target |
|---|---|---|---|---|---|---|
| organ_concerto_4_3_1 | 5 | 3.2° | 15.5° | 50.4° | 56.8° | −3.56 / −3.05 dB |
| organ_concerto_4_6_1 | 5 | 11.2° | 20.8° | 35.6° | 53.9° | +1.18 / +1.65 dB |
| prelude_and_fugue_hess-30 | 4 | 2.7° | 21.6° | 46.3° | 55.6° | −0.99 / −1.21 dB |
| serenata_375_2 | 3 | 9.7° | 8.4° | 10.2° | 25.3° | −0.39 / −0.71 dB |
| sonata_battalia_4 | 2 | 3.0° | 16.9° | 2.1° | 1.5° | +0.02 / −0.04 dB |
| string_quartet_13_3 | 4 | 4.5° | 21.4° | 47.2° | 62.6° | −0.97 / −1.04 dB |
| symphony_41_551_2 | 9 | 6.2° | 15.3° | 50.0° | 61.8° | −1.95 / −2.02 dB |

`serenata_375_2` is the one song where the model loses to the center
baseline: three wind instruments spanning only 25°, and the model
spreads them just 10°. Small, narrow ensembles are where the
center-compression hurts most.

## Limitations (important)

- **All 7 validation songs are Synthmix** — synthetic targets rendered
  by this project's own gain/pan code. The target is therefore exactly
  representable by the model, which makes this a clean *parameter
  recovery* measurement but **not** evidence about real recordings. The
  four real Spheres songs all landed in the training split, so this run
  has no held-out real-mix evaluation. That is the obvious next
  experiment.
- The loss value (1.4032) is **not comparable** to the 60-epoch run's
  5.1765: the corpus, the target construction (Synthmix instead of the
  SynthSOD tree sum) and the anchors all changed. Only the degree-based
  metrics above are comparable across the two runs, because they are
  measured against ground truth rather than against the loss.
- Anchored tracks (main array / tree channels) are excluded from the
  pan metric by construction — their pan is pinned by config, not
  learned. The 32 numbers above are only the spot mics.

## Artifacts

- `reports/kaggle200_training_curves.png` — loss curves and LR schedule.
- `reports/mixconsole_2026-09-15_1202_kaggle_*.png` — console view of
  the gains and pans for two val songs.
- `checkpoints/2026-09-15_1202_kaggle/` — `best.pt`, `last.pt`, config.
- `scripts/eval_synthmix.py` — produces every number above. The val
  songs are rebuilt with `prepare_synthsod.py` + `generate_synthmix.py`
  (per-song seeded RNG, so `mix_params.yaml` matches the Kaggle run
  exactly); the val split is `split_train_val(seed=0, val_fraction=0.1)`
  over the 68 song names. Predictions use the full song, capped to a
  centered 120 s excerpt.
