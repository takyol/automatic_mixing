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

## Training

Each training config pairs one processed corpus with its own checkpoint and log directories, so runs never overwrite each other. Pick the config for what you want to train on:

| Scenario | Command | Data root | Checkpoints |
|---|---|---|---|
| Spheres only | `python scripts/train.py --config configs/spheres.yaml` | `data_processed` | `checkpoints_spheres/` |
| SynthSOD only | `python scripts/train.py --config configs/synthsod.yaml` | `data_processed_synthsod` | `checkpoints_synthsod/` |
| Synthetic mixes | `python scripts/train.py --config configs/synthmix.yaml` | `data_processed_synthmix` | `checkpoints_synthmix/` |
| Both datasets | `python scripts/train.py --config configs/combined.yaml` | `data_processed_all` | `checkpoints_combined/` |

Logs go to the config's `log_dir`, viewable with TensorBoard (`tensorboard --logdir runs`). Note that TensorBoard layers event files rather than resetting between runs — delete the run's log dir first if you want a clean plot.

**Training on both datasets**: the manifest simply scans every song folder under one `data_processed_root`, so to combine datasets, prep them into the same folder. Edit `output_root: data_processed_all` in both prep configs (or copy the already-prepped song folders into `data_processed_all/`), then use `configs/combined.yaml`, whose anchor patterns cover both datasets' naming conventions. One caveat: raw SynthSOD's tree-sum target teaches the model to *suppress* close mics (the tree fully explains the target), which conflicts with Spheres mixes where spots genuinely contribute — for combined training, pairing Spheres with the **synthetic-mix** corpus instead is usually the more consistent choice.

Heads-up on a known limitation: with only close-mic spots and no anchored width carriers, training converges to panning everything to one side (the loss is blind to the difference between stereo width and a hard-panned mix). Keep the `anchors` section in any training config you write, and make sure the anchor stems actually exist in the data.

## Rendering a mix

Once you have a trained checkpoint, mix a folder of stems. Pass the training config so the same anchor patterns are applied:

```
python scripts/infer.py --stems-dir path/to/stems --checkpoint checkpoints_spheres/best.pt --output mix.wav --config configs/spheres.yaml
```

## Diagnostics

- `scripts/dump_gain_pan.py --stems-dir ... --checkpoint ... --config <training config>` prints the gain and pan a checkpoint assigns to each stem, marks anchored tracks, and reports the spread of the learned pans. If every learned pan is nearly the same angle, the model isn't differentiating tracks. For the synthetic-mix corpus, compare against the song's `mix_params.yaml` to measure recovery error.
- `scripts/embedding_similarity.py --stems-dir ...` measures how distinguishable a song's stems are to the VGGish encoder (pairwise cosine similarity). Values near 1.0 mean the mixing MLP can't tell tracks apart.
