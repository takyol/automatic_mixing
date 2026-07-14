# Fixing the "everything pans left" problem

Quick recap of where we were: our automixer was collapsing the whole stereo
image to one side — in test renders almost all instruments ended up in the left
channel instead of spread across the stage like the reference mixes.

## What was causing it

Our training loss compared the mix to the target using only the **sum (L+R)** and
**difference (L−R)** signals. This pair is mathematically *blind to left/right
mirroring* — a mix with the violins correctly on the left and its mirror image
with them on the right produce the exact same loss. So the loss constrained how
*wide* the mix was but never *which side* things went, and the model just dumped
everything onto one side to create width the cheapest way.

## What we did

Added two more terms to the loss — comparing the model's **left channel vs the
target's left**, and **right vs right**, directly (see `losses/mrstft.py`). This
breaks the symmetry: putting an instrument on the wrong side now costs loss, so
the model has a reason to learn the real stereo placement (roughly, the
orchestra's seating).

## Result

Measured on the two validation songs the model never trained on
(L−R balance, 0 dB = perfectly centered):

| Song | Old (sum/diff) | New (per-channel) | Reference |
|---|---|---|---|
| Mozart 1 | +9.9 dB | **+2.1 dB** | +0.6 dB |
| Organ concerto | +22.7 dB | **+7.9 dB** | −0.3 dB |

Mozart is basically centered now and matches the reference; the organ concerto
improved a lot but still leans left on the larger ensemble.

We also moved training to Kaggle's free GPU (much faster than the MacBook) and
added checkpoint resume (`--resume`) so long runs survive session limits.

## Status

Clear win — same data and model, only the loss changed, and the collapse is gone.
This was a 60-epoch quick check (`kaggle` run, best val 5.18 @ epoch 59). Next:
run the full 200 epochs, and investigate why the bigger orchestra still has a
residual left bias (more training may help, or the pan parameterization needs a
look). Loss curves (`reports/kaggle_training_curves.png`) and the full run history
(`reports/experiment_log.md`) are in the repo.
