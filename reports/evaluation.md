# Results

Consolidated write-up of what the model achieves, what it does not, and how
each number was produced. Per-run bookkeeping lives in `experiment_log.md`;
figures referenced here are in `figures/`.

Two kinds of numbers appear below. The **loss** (multi-resolution STFT) only
says how close the rendered mix is to the target audio and is comparable
across runs only when corpus, targets and anchors are identical. The
**parameter errors** (degrees of pan, dB of gain) compare the model's
decisions against known-correct settings and are comparable everywhere; they
are the numbers to trust.

## Where the ground truth comes from

| Corpus | Target | True gain/pan known because |
|---|---|---|
| Synthmix (64 songs) | rendered by this project's own mixer from a seating chart with random gains | the generator saves them per song in `mix_params.yaml` |
| Spheres (4 songs, 2 distinct mixes) | the engineer's real stereo mix | the mixes are exact static gain/pan sums of the stems — `fit_mix_params.py` recovers the settings at R² = 1.000 |
| Wuppertal concert (own recording) | none | there is no engineer mix; see the feasibility section below |

## Headline: the 200-epoch run

Run `2026-09-15_1202_kaggle` — 68 songs (4 Spheres + 64 Synthmix), per-channel
MR-STFT loss plus pan anchors. Measured on the **7 validation songs** (held out
by song), for the 32 spot mics whose pan the model has to predict:

| Metric | 60-epoch run | **This run (best.pt)** | Constant-center predictor | Perfect predictor |
|---|---|---|---|---|
| Pan error (MAE) | 22.4° | **5.8°** | 17.1° | 2.5° |
| Gain error (MAE, scale-corrected) | 2.50 dB | **1.37 dB** | 1.28 dB | 0 dB |
| Stereo balance \|L−R\| | 1.79 dB | **1.29 dB** | — | 1.39 dB (target) |

The 2.5° floor is not reachable: the generator jitters each pan by ±5°
uniformly and that jitter is random per song, so no model can predict it from
the audio. **5.8° is a bit over twice the noise floor** — the model places
instruments essentially where the seating chart puts them.

The 60-epoch model (22.4°) was *worse than simply guessing center* (17.1°):
its predicted pan spread per song was 2–16° against a true spread of 25–63°.
That is the pan collapse described below, and it is gone — predicted spread is
now 47–55° on the large ensembles.

Gain is the exception: 1.37 dB is no better than predicting one constant level
for every stem (1.28 dB). Synthmix gains are drawn at random, independently of
what the instrument sounds like, so they are unlearnable by construction. Only
Spheres contains real level decisions.

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

The ordering is recovered almost perfectly — strings sweep left to right from
Violin 1 to Bass, exactly as the chart seats them. The image is slightly
**compressed toward the center**: the model under-shoots the extremes (Violin 1
by 9.5°, Cello by 6.3°). Horn (14.4°) and Oboe (11.3°) are the weakest, both
instruments whose timbre is easily confused with their neighbours.

### Per-song detail (best.pt)

| Song | Spots | Pan MAE | Center baseline | Pred. spread | True spread | Balance pred / target |
|---|---|---|---|---|---|---|
| organ_concerto_4_3_1 | 5 | 3.2° | 15.5° | 50.4° | 56.8° | −3.56 / −3.05 dB |
| organ_concerto_4_6_1 | 5 | 11.2° | 20.8° | 35.6° | 53.9° | +1.18 / +1.65 dB |
| prelude_and_fugue_hess-30 | 4 | 2.7° | 21.6° | 46.3° | 55.6° | −0.99 / −1.21 dB |
| serenata_375_2 | 3 | 9.7° | 8.4° | 10.2° | 25.3° | −0.39 / −0.71 dB |
| sonata_battalia_4 | 2 | 3.0° | 16.9° | 2.1° | 1.5° | +0.02 / −0.04 dB |
| string_quartet_13_3 | 4 | 4.5° | 21.4° | 47.2° | 62.6° | −0.97 / −1.04 dB |
| symphony_41_551_2 | 9 | 6.2° | 15.3° | 50.0° | 61.8° | −1.95 / −2.02 dB |

`serenata_375_2` is the one song where the model loses to the center baseline:
three wind instruments spanning only 25°, and the model spreads them just 10°.
Small, narrow ensembles are where the center-compression hurts most.

### Convergence and checkpoint choice

`best.pt` is from **epoch 121** (val 1.4032); the remaining 79 epochs never beat
it, and at epoch 200 val was 1.4702 while train had fallen from 0.97 to 0.88 —
mild overfitting. But `last.pt` (epoch 200) scores **5.37° pan MAE, slightly
better than best.pt's 5.79°** despite the worse val loss. The MR-STFT
validation loss is therefore not a perfect proxy for parameter recovery.
`last.pt` of this run is the best model the project has.

## Against real engineers (Spheres)

`fit_mix_params.py` recovers what the two Spheres mixes were built with, so the
model can be scored against a human decision instead of a generator. The
recovered pans confirm the seating chart used by Synthmix (Violin 1 9.5°/16°,
Viola 59°/57°, Horn 31°/30°, Cello 85°/75°). The gains do **not** follow a
loudness rule: they are piece-dependent (Mozart puts the strings forward and
the winds 17–21 dB under the main array; Tchaikovsky keeps the winds at
−3…−6 dB and pushes viola, cello and bass to −12…−15 dB).

| Checkpoint | Pan MAE | Center baseline | Gain MAE | Constant-level baseline |
|---|---|---|---|---|
| 200-epoch run, best.pt | 8.21° | 16.31° | 2.97 dB | 5.15 dB |
| 200-epoch run, last.pt | **7.89°** | 16.31° | **2.69 dB** | 5.15 dB |
| Spheres-weighted run, last.pt | 8.22° | 16.31° | 2.89 dB | 5.15 dB |

**These are in-sample numbers** — all four Spheres songs are in the training
split, so this measures fit, not generalization. There is no held-out real mix
anywhere in the project.

## Experiment: upweighting the real mixes did not work

Run `2026-09-18_1026_kaggle` gave Spheres 40 % of the training draws instead of
~7 %, and stretched its two mixes with ±6 dB per-stem level jitter and 0.3 stem
dropout (each dropped stem's exact contribution subtracted from the target).
Everything else — split, model, loss, Synthmix targets — stayed identical.

No metric improved: Synthmix val pan MAE 6.6° (was 5.4°), Spheres pan 8.22°
(was 7.89°) and gain 2.89 dB (was 2.69 dB). The informative part is *how* it
failed:

- Training loss plateaued at ~1.63 against 0.97 in the previous run, with the
  learning rate already decayed to 1.9e-5 (`figures/training_curves_comparison.png`).
- The Tchaikovsky gains stay at 4.0–4.8 dB error against a 4.9 dB constant
  baseline **although those songs are trained on**, at six times the previous
  weight.

So the model does not memorize the two real mixes — it cannot fit them at all.
Per-stem level is the bottleneck: a time-averaged VGGish embedding plus a
magnitude-based audio loss give the MLP too weak a handle on level, and the
level augmentation made the task harder without adding the information needed
to solve it.

## Case study: a real concert (Wuppertal, 33–37 tracks)

Own recording, mixed with `render_session.py` at four 90 s excerpts
(`figures/wuppertal/`). Far out of distribution: training songs have 4–22
stems, this has 33–37, some sections doubled with two spot mics, and all spots
carry bleed. Judged by listening, no reference mix exists:

- In the lyrical excerpt the strings are ordered correctly left to right.
- In dense passages the model compresses most spots into 45–60° and picks
  near-uniform gains of −2…−7 dB.
- The Spheres-weighted checkpoint spreads levels more (up to 3.9 dB spread
  instead of 1.4 dB) but pushes the soloists 9 dB further down than the older
  checkpoint — the Mozart "winds far under the main array" rule misfiring on a
  piece where the voice is the subject.

## Can the Wuppertal recording serve as training data?

The model stays audio-loss-only, so the question is purely whether the concert
offers a **target** that a static gain/pan mix can reach. The session carries a
`SUMME` track — the engineer's own stereo mix, full length. It lags the
multitrack by a constant **12.1 ms** (cross-correlation against the main mics);
without compensating that, a fit is meaningless (R² < 0.05).

Aligned, `fit_mix_params.py` explains SUMME well as a static sum, and the
recovered settings independently confirm the main-array layout (A at 0.2°, B at
90.0°, C at 45.7°) and show what the engineer actually did: main array and
outriggers at −7…−11 dB carry the mix, the vocal spots sit 12–14 dB down, and
every instrument spot is below −27 dB.

| Excerpt | Fit R² (L / R) |
|---|---|
| 1 — soloist passage | 0.974 / 0.981 |
| 4 — tutti | 0.804 / 0.851 |

### The loss floor per candidate target

For each candidate, gain and pan were fitted directly to the target with Adam
on the training loss — the best any model could do with this parameterization
(`figures/target_feasibility.png`, 10 s excerpts, 400 steps).

| Target | Best achievable loss | Spot share of mix energy |
|---|---|---|
| Synthmix song (exact solution exists) | 0.01 | 29 % |
| Spheres song (exact solution exists) | 0.02 | 51 % |
| **Wuppertal SUMME**, soloist excerpts | **0.70 / 0.74** | 10–12 % |
| **Wuppertal SUMME**, harp and tutti | **2.17 / 2.55** | 1–2 % |
| Wuppertal main array as target, main mics in the input | 0.06 | **0 %** |
| Wuppertal SUMME, main mics removed from the input | 5.21 | 100 % |

Reading, for reference the trained model's validation loss is 1.40:

- **Using the main array as the target is pointless.** The anchored main mics
  reproduce it exactly, so the optimum mutes every spot (0 % energy). That is
  the same trap raw SynthSOD's tree-sum target sets, and the model would learn
  "turn the spots off".
- **Dropping the main mics from the input makes the target unreachable** (5.21).
  Spot mics alone cannot recreate a spaced main array's stereo image, exactly
  the reason anchors exist.
- **SUMME is a genuine target.** With everything in the input, the floor is
  0.70–0.74 in the two excerpts with spot-miked soloists, well under the
  model's current val loss, and the spots carry 10–12 % of the energy — so
  there is something real to learn.
- **But only in parts of the concert.** In the harp and tutti excerpts the
  floor rises to 2.2–2.6, *above* what the model already achieves elsewhere,
  and the spots contribute 1–2 %. Those passages are dominated by whatever the
  static model cannot express — fader moves within the excerpt, reverb,
  processing — so training on them would mostly fit noise.

### What happened when we trained on it

Run `2026-09-20_1134_wuppertal`: 33 concert windows (16.5 min) added to the 68
songs, everything else unchanged. Two caveats about the setup itself:

- **The validation set changed more than intended.** Forcing 8 windows into val
  takes them out of the random draw, but the 25 remaining windows join the
  shuffle pool, which re-permutes everything. The run ended up with 84 train /
  17 val songs, and **6 of the 7 previously held-out Synthmix songs moved into
  training**. The reported val loss (4.04) is therefore not comparable to the
  earlier 1.40 — neither in content nor in composition.
- **One useful accident**: the draw put `Song3_Tschai1` into val, so the project
  finally has a held-out *real* engineer mix.

Measured per group instead of by the aggregate loss:

| Held out from | Metric | 200-epoch model | Concert-trained model |
|---|---|---|---|
| both runs — `string_quartet_13_3` | pan MAE | **4.20°** | 6.53° |
| both runs — `string_quartet_13_3` | MR-STFT loss | **1.26** | 2.68 |
| new run only — `Song3_Tschai1` (real mix) | pan MAE (center 14.18°) | 6.11° (in-sample) | 8.77° |
| new run only — `Song3_Tschai1` | gain MAE (constant 4.92 dB) | 3.76 dB (in-sample) | 6.29 dB |
| both — 9 concert windows | MR-STFT loss | 18.91 | **3.27** |
| both — 9 concert windows | pan MAE (center 28.6°) | 29.20° | 28.65° |
| both — 9 concert windows | gain MAE (constant 8.70 dB) | 8.62 dB | 9.46 dB |

The loss on concert material improves six-fold, and that is exactly what makes
the result interesting: **the parameter metrics show the model did not learn
the engineer's decisions at all.** Its pans match a constant-center predictor to
within half a degree, and its gains are *worse* than one constant level for
every track (12.69 dB for `last.pt`). What it learned is the gross level regime
for a 33–37 track input — how far down everything has to go — which the loss
rewards heavily and which the old model, never having seen more than 22 tracks,
got badly wrong.

That gain came at a real cost everywhere else: on a song held out by *both*
runs, pan error rose from 4.20° to 6.53° and the loss doubled. On the held-out
real mix, the model's levels are worse than a constant.

**Conclusion:** one concert is one balance philosophy, one hall, one ensemble.
16.5 minutes of it biases the model rather than teaching a transferable rule,
and the MR-STFT loss hides this — it improved six-fold while the actual mixing
decisions stayed at baseline. `2026-09-15_1202_kaggle/last.pt` remains the
project's model.

### Recommendation

1. **Highest value: use it as the project's first held-out real mix.** The
   standing limitation is that no real engineer mix is held out. Wuppertal
   fixes that at zero risk to training: fit `mix_params.yaml` on excerpts that
   pass a `min_r2` gate and score the model against them the same way Spheres
   is scored.
2. **As training data it did not work** (see above). If it is tried again, the
   missing ingredient is diversity, not more minutes of the same concert:
   several sessions, halls and engineers. A single session should at most be
   one corpus among many, and its share of the draws kept small.
3. **What it does not fix:** the level bottleneck. SUMME's spot levels are
   mixing decisions like Spheres', and the experiment above showed the model
   cannot fit those even when trained directly on them.

## History: fixing the "everything pans left" problem

Early runs collapsed the whole stereo image onto one side. The cause was the
loss: comparing only the **sum (L+R)** and **difference (L−R)** signals is
blind to mirroring — a mix with the violins on the left and its mirror image
score identically. The loss constrained how *wide* a mix was but never *which
side* anything went to, and the cheapest way to create width was to dump
everything onto one side.

Adding **per-channel terms** (model's left vs. target's left, right vs. right)
breaks that symmetry. Same data, same model, only the loss changed
(`figures/training_curves_60ep.png`):

| Song (val) | sum/diff only | + per-channel | Reference |
|---|---|---|---|
| Mozart 1 | +9.9 dB | **+2.1 dB** | +0.6 dB |
| Organ concerto | +22.7 dB | **+7.9 dB** | −0.3 dB |

Pan anchors (fixed pan for the main-array stems) attack the same failure
structurally and were added afterwards; the two are complementary.

## Limitations

- **No held-out real mix.** The 7 validation songs are Synthmix, i.e. targets
  rendered by this project's own gain/pan code, so they measure clean
  *parameter recovery* rather than real-world mixing. The four Spheres songs
  are all in the training split.
- **Gains are barely learned.** Synthmix gains are random; Spheres gains are
  real but too few and piece-dependent, and the model underfits them.
- **Center compression.** When unsure, center is the safest guess, so the model
  does not use the full stereo width.
- **Static decisions only.** One gain and one pan per track for a whole piece —
  no automation, EQ, reverb, or spot/main delay compensation.
- **Anchored tracks are excluded from the pan metric** by construction: their
  pan is pinned by config, not learned.

## Reproducing the numbers

```bash
# rebuild the 7 val songs (per-song seeded RNG -> mix_params.yaml matches the Kaggle run)
python scripts/prepare_synthsod.py --config configs/synthsod_prep.yaml
python scripts/generate_synthmix.py --config configs/synthmix_gen.yaml
python scripts/eval_synthmix.py --root data_processed_synthmix --config configs/kaggle.yaml \
    --checkpoint checkpoints/2026-09-15_1202_kaggle/last.pt --out eval_synthmix.json

# Spheres, against the recovered engineer settings
python scripts/fit_mix_params.py --root data_processed --pattern "Song*" --config configs/kaggle.yaml
python scripts/eval_synthmix.py --root data_processed --config configs/kaggle.yaml \
    --checkpoint checkpoints/2026-09-15_1202_kaggle/last.pt --out eval_spheres.json
```

The val split is `split_train_val(seed=0, val_fraction=0.1)` over the 68 song
names. Predictions use the full song, capped to a centered 120 s excerpt.
Checkpoints and the exact config of each run are in `checkpoints/<run>/`.
