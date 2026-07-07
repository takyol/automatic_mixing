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

Create a virtual environment and install the package:

```
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

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

This writes checkpoints to `checkpoints/` and logs to `runs/default`, which you can view with TensorBoard.

## Rendering a mix

Once you have a trained checkpoint, you can mix a folder of stems:

```
python scripts/infer.py --stems-dir path/to/stems --checkpoint checkpoints/best.pt --output mix.wav
```
