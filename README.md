# Automatic Mixing

This project trains a small neural network to automatically mix orchestra recordings. You give it separate instrument tracks (stems), and it predicts a gain and a pan position for each track. Those values are applied with simple audio math, and the tracks are summed into a stereo mix.

The network only learns to predict gain and pan. It does not generate audio itself. A frozen, pretrained VGGish model is used to turn each track into an embedding, and a small MLP (the only part that gets trained) turns those embeddings into gain and pan values.

## Project layout

- `src/automix/` - the main package
  - `model/` - the VGGish wrapper, context module, MLP, and gain/pan mixer
  - `data/` - dataset loading, manifest building, and batching
  - `losses/` - the multi-resolution STFT loss used for training
  - `prep/` - scripts to convert raw datasets into a common format
  - `train_loop.py` - the training loop
  - `inference.py` - renders a mix from stems using a trained checkpoint
- `scripts/` - command line entry points (data prep, training, inference)
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

## Preparing data

Two datasets are supported: SynthSOD and Spheres. Each has its own prep script that converts the raw dataset into a common folder layout (`data_processed/<song>/stems/*.wav` and `data_processed/<song>/target.wav`).

```
python scripts/prepare_synthsod.py --config configs/synthsod_prep.yaml
python scripts/prepare_spheres.py --config configs/spheres_prep.yaml
```

You'll need to edit the config files to point at wherever you've downloaded the raw datasets, and check the mic folder names match your data.

## Training

Once the data is prepared:

```
python scripts/train.py --config configs/default.yaml
```

Every run gets its own name, `<date>_<time>_<config name>` (e.g. `2026-07-13_2352_default`), or a custom one via an optional `run_name` key in the config. Checkpoints go to `checkpoints/<run name>/` (alongside a copy of the config used), TensorBoard logs to `runs/<run name>/`. Point TensorBoard at the root to compare runs: `tensorboard --logdir runs/`.

## Rendering a mix

Once you have a trained checkpoint, you can mix a folder of stems:

```
python scripts/infer.py --stems-dir path/to/stems --checkpoint checkpoints/best.pt --output mix.wav
```

To see the gain and pan the model chose per stem, rendered like a mixing
console (requires the dev extras for matplotlib):

```
python scripts/plot_console.py --stems-dir path/to/stems \
    --checkpoint checkpoints/<run>/best.pt --output reports/console.png
```
