# Automatic Mixing

This project trains a small neural network to automatically mix orchestra recordings. You give it separate instrument tracks (stems), and it predicts a gain and a pan position for each track. Those values are applied with simple audio math, and the tracks are summed into a stereo mix.

The network only learns to predict gain and pan. It does not generate audio itself. A frozen, pretrained VGGish model is used to turn each track into an embedding, and a small MLP (the only part that gets trained) turns those embeddings into gain and pan values.

**Anchor tracks**: stems whose filenames match the `anchors` patterns in the training config (e.g. main array `*_L`/`*_C`/`*_R`, room pair, SynthSOD tree channels) get a *fixed* pan position; only their gain is learned. This matters because an orchestral target's stereo width comes from spaced microphones — something panned mono spots can't reproduce — and without anchored width carriers, training collapses to panning everything hard to one side. The same anchor patterns must be passed at inference (`--config`).

## Project layout

- `src/automix/` - the main package
  - `model/` - the VGGish wrapper, context module, MLP, and gain/pan mixer
  - `data/` - dataset loading, manifest building, and batching
  - `losses/` - the multi-resolution STFT loss used for training
  - `prep/` - scripts to convert raw datasets into a common format
  - `anchors.py` - maps stem filenames to fixed pan angles
  - `train_loop.py` - the training loop
  - `inference.py` - renders a mix from stems using a trained checkpoint
- `scripts/` - command line entry points (data prep, training, inference, diagnostics)
- `configs/` - YAML config files used by the scripts
- `reports/` - the experiment journal (`experiment_log.md`), the consolidated results write-up (`evaluation.md`) and the figures both refer to
- `docs/modell_erklaerung/` - a step-by-step explanation of the model in German (LaTeX source + PDF), written as the groundwork for the paper
- `kaggle/` - the notebook used to train on Kaggle's free GPU (see `KAGGLE.md`)

## Setup

Requires Python 3.9+. Create a virtual environment and install the package:

```
python -m venv venv
source venv/bin/activate # for Windows Power Shell run: venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

On Windows, if `python` resolves to an interpreter older than 3.9 or `venv` creation fails, use the `py` launcher to pick a specific installed version instead, e.g. `py -3.11 -m venv venv`.

### GPU on Windows

On Windows, PyPI's default `torch`/`torchaudio` wheels are CPU-only. `pip install -e ".[dev]"` will install and silently succeed without CUDA support even if you have an NVIDIA GPU. To get GPU support, reinstall from PyTorch's own index after the regular install, matching the pinned versions in `pyproject.toml` and a CUDA build supported by your driver (check `nvidia-smi` for your driver's max supported CUDA version):

```
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

Verify with `python -c "import torch; print(torch.cuda.is_available())"`. Note this step isn't captured by `pyproject.toml`, so it needs to be redone any time this venv is recreated from scratch.

On Apple Silicon Macs, `device: auto` picks the MPS backend automatically (requires a native arm64 Python; check with `python -c "import platform; print(platform.machine())"` — it should say `arm64`).

## Preparing data

Each dataset has a prep script that converts the raw download into a common layout: `<output_root>/<song>/stems/*.wav` plus `<output_root>/<song>/target.wav`. The prep configs control where the raw data is read from and where the processed corpus is written.

**Spheres** (real recordings; raw layout `data/spheres/<song>/spot_mics/*.wav` + `mix.wav`):

```
python scripts/prepare_spheres.py --config configs/spheres_prep.yaml    # -> data_processed/
```

**SynthSOD** (synthesized, bleed-free; raw layout `data/synthsod/<song>/Close Mic/*.flac` + `Tree/*.flac`):

```
python scripts/prepare_synthsod.py --config configs/synthsod_prep.yaml  # -> data_processed_synthsod/
```

The SynthSOD target is the summed stereo tree (main array); the tree's channels are also written as `Tree_L.wav`/`Tree_R.wav` stems so training can anchor them.

**Synthetic mixes** (optional, built on top of prepped SynthSOD): renders targets with our own gain+pan math using a fixed orchestral seating chart plus random gains, and saves the sampled values in each song's `mix_params.yaml`. Because the true parameters are known, this corpus is the one place where training accuracy can be measured directly.

```
python scripts/generate_synthmix.py --config configs/synthmix_gen.yaml  # -> data_processed_synthmix/
```

Stems are hard-linked, not copied, so the extra disk cost is only the new target files.

**A raw session with the engineer's own mix** (one WAV per recorder track plus a stereo mix track) is cut into windows by `scripts/prepare_session.py`, each window becoming a song whose target is that mix:

```
python scripts/prepare_session.py --config configs/wuppertal_prep.yaml [--dry-run]
```

Two gates make the result trustworthy, and `--dry-run` reports them without writing audio: the mix track's console latency is compensated (12.1 ms for our concert — an uncompensated offset makes the target meaningless), and every window is fitted by NNLS and kept only above `min_r2`, so training never sees a target the static mixer could not reproduce. `target_level_db` normalizes tracks and target by one factor so a quiet concert cannot dominate a mixed corpus' loss; the fitted gains stay valid because the mix equation is linear in both sides.

## Training

Each training config pairs one processed corpus with its own checkpoint and log directories, so runs never overwrite each other. Pick the config for what you want to train on:

| Scenario | Command | Data root | Checkpoints |
|---|---|---|---|
| Spheres only | `python scripts/train.py --config configs/spheres.yaml` | `data_processed` | `checkpoints_spheres/` |
| SynthSOD only | `python scripts/train.py --config configs/synthsod.yaml` | `data_processed_synthsod` | `checkpoints_synthsod/` |
| Synthetic mixes | `python scripts/train.py --config configs/synthmix.yaml` | `data_processed_synthmix` | `checkpoints_synthmix/` |
| Both datasets | `python scripts/train.py --config configs/combined.yaml` | `data_processed_all` | `checkpoints_combined/` |
| Synthmix + Spheres + a real concert | `python scripts/train.py --config configs/kaggle_wuppertal.yaml` | prepped corpora merged into one root | `checkpoints/` |

`val_song_patterns` forces the listed songs into the validation split and takes them out of the random draw. Windows cut from one session need this: neighbouring windows are musically near-identical, so a random split would leak train material into val. Forcing them out *before* the draw also leaves the rest of the corpus with exactly the split it had without them, so val losses stay comparable to earlier runs.

Every run gets its own name, `<date>_<time>_<config name>` (e.g. `2026-07-13_2352_spheres`), or a custom one via an optional `run_name` key in the config. Checkpoints go to `<checkpoint_dir>/<run name>/` (alongside a copy of the config used), TensorBoard logs to `<log_dir>/<run name>/`. Point TensorBoard at the root to compare runs: `tensorboard --logdir runs/`.

**Training on both datasets**: the manifest simply scans every song folder under one `data_processed_root`, so to combine datasets, prep them into the same folder. Edit `output_root: data_processed_all` in both prep configs (or copy the already-prepped song folders into `data_processed_all/`), then use `configs/combined.yaml`, whose anchor patterns cover both datasets' naming conventions. One caveat: raw SynthSOD's tree-sum target teaches the model to *suppress* close mics (the tree fully explains the target), which conflicts with Spheres mixes where spots genuinely contribute — for combined training, pairing Spheres with the **synthetic-mix** corpus instead is usually the more consistent choice.

Two independently developed defenses against one-sided pan collapse are in place, and they complement each other: the loss carries **per-channel (L, R) terms** in addition to the mirror-invariant sum/diff pair (see `losses/mrstft.py`), giving every track a direct penalty for sitting on the wrong side; and **anchor tracks** pin the main-array stems that carry the target's spaced-mic stereo width. Keep the `anchors` section in any training config you write, and make sure the anchor stems actually exist in the data.

## Rendering a mix

Once you have a trained checkpoint, mix a folder of stems. Pass the training config so the same anchor patterns are applied:

```
python scripts/infer.py --stems-dir path/to/stems --checkpoint checkpoints_spheres/best.pt --output mix.wav --config configs/spheres.yaml
```

## Mixing a raw session

`scripts/render_session.py` skips the prep step and mixes an excerpt straight out of a recorder folder (one WAV per track), which is how the Wuppertal concert in `reports/evaluation.md` was rendered:

```
python scripts/render_session.py --tracks-dir path/to/session --start 1500 --seconds 90 \
    --checkpoint checkpoints/<run>/last.pt --config configs/kaggle.yaml \
    --rename "A=Main_L,B=Main_R,C=Main_C,OUT A=Out_L,OUT B=Out_R" \
    --exclude "TC,SUR A,SUR B,SUMME" --out renders/session_1
```

`--rename` maps recorder track names onto the `anchors` naming so the main array keeps its fixed pan; `--exclude` drops timecode, surround and sum tracks. It writes the mix, a main-array-only reference to listen against, a console figure and the predicted values per track.

## Measuring against ground truth

Where the true gain and pan are known, `scripts/eval_synthmix.py` reports the recovery error in degrees and dB instead of a loss value:

```
python scripts/eval_synthmix.py --root data_processed_synthmix --checkpoint checkpoints/<run>/best.pt \
    --config configs/kaggle.yaml --out eval.json
```

Synthmix songs ship their `mix_params.yaml` from the generator. For a **real** mix that is a static gain/pan sum of its stems, `scripts/fit_mix_params.py` recovers the engineer's settings into the same format by non-negative least squares, and prints the fit R² so a mix that is *not* such a sum is easy to spot (the Spheres mixes fit at R² = 1.000):

```
python scripts/fit_mix_params.py --root data_processed --pattern "Song*" --config configs/kaggle.yaml
```

Those recovered parameters also enable two optional training-config keys, both train-split only (validation sampling stays uniform): `sampling_fractions` ({song glob: share of draws}) upweights a corpus, and `augment` ({patterns, level_db, stem_dropout}) varies stem levels and drops stems while subtracting their exact share from the target. Both keys are off in the shipped configs: the experiment that used them (`reports/evaluation.md`) did not improve the model, and its exact config is preserved in `checkpoints/2026-09-18_1026_kaggle/config.yaml`.

To judge whether some *other* recording could serve as training material, `scripts/target_feasibility.py` fits gain and pan straight to a candidate target and reports the resulting loss floor plus how much mix energy the non-anchored tracks carry. `reports/evaluation.md` walks through the reference values and the verdict for our own concert recording.

## Tests

`./venv/bin/pytest` runs the unit tests in `tests/` (pan law, anchor mapping, sampling weights, parameter recovery, and the dataset's target correction under stem dropout). They need no data and finish in about a second. The end-to-end path — VGGish, training loop, checkpointing — is covered by the smoke config instead:

```
python scripts/train.py --config configs/smoke_test.yaml
```

## Diagnostics

- `scripts/plot_console.py --stems-dir ... --checkpoint ... --output reports/console.png` renders the gain and pan the model chose per stem like a mixing console (requires the dev extras for matplotlib).
- `scripts/dump_gain_pan.py --stems-dir ... --checkpoint ... --config <training config>` prints the gain and pan a checkpoint assigns to each stem, marks anchored tracks, and reports the spread of the learned pans. If every learned pan is nearly the same angle, the model isn't differentiating tracks. For the synthetic-mix corpus, compare against the song's `mix_params.yaml` to measure recovery error.
- `scripts/embedding_similarity.py --stems-dir ...` measures how distinguishable a song's stems are to the VGGish encoder (pairwise cosine similarity). Values near 1.0 mean the mixing MLP can't tell tracks apart.
