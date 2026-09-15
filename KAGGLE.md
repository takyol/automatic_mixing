# Training on Kaggle (free GPU)

Kaggle gives ~30 GPU-hours per week for free (T4/P100). Epochs run several
times faster than on an M1 Pro, and `num_workers` works there (Linux).

## One-time setup

1. Create an account at kaggle.com and verify it with a phone number
   (Settings -> Phone verification) — required for GPU access.
2. Upload the raw data as a Kaggle Dataset:
   - kaggle.com -> Create -> New Dataset
   - Upload `automix_raw_data.zip` (contains `spheres/` and `synthsod/`;
     Kaggle extracts zips automatically)
   - Title it `automix-raw` (the notebook assumes this name; adjust the
     `RAW` variable in the notebook if you pick another)
   - Keep it **Private** (Spheres is not ours to publish)
3. Create the notebook:
   - kaggle.com -> Create -> New Notebook
   - File -> Import Notebook -> upload `kaggle/train_kaggle.ipynb` from this repo
   - In the panel on the right: **Add Input** -> your `automix-raw` dataset,
     and **Session options -> Accelerator -> GPU**

## Each training session

- Run all cells. The prep cell builds the final-run corpus (Spheres +
  Synthmix — see the README on why raw SynthSOD is only an intermediate),
  and cell 3 starts training, **auto-resuming** from the most recent
  run's `last.pt` when one exists in `/kaggle/working`.
- For long runs use **Save Version -> Save & Run All (Commit)**: the notebook
  runs headless, no browser needed. Checkpoints and TensorBoard logs appear
  on the notebook's **Output** tab afterwards.
- **Batch sessions are killed at 12h and their output is discarded**, so
  cell 3 caps epochs per session (`EPOCHS_THIS_SESSION`, default 120). For
  the full 200-epoch run: session 1 with the default, then session 2 with
  the saved version's Output attached as an input, `PREV_OUTPUT` set to it,
  and `EPOCHS_THIS_SESSION = 200` — it resumes and trains the rest.
- Download `best.pt` from the Output tab to render mixes locally
  (`scripts/infer.py --config configs/kaggle.yaml` so the anchors match),
  and the `runs/` folder to view curves in a local TensorBoard.

## Notes

- Push code changes to GitHub before starting a session — the notebook
  clones this repo fresh each time.
- The weekly GPU quota is visible on your Kaggle profile page.
- Interactive sessions idle out after ~20 min without browser activity;
  the Save & Run All flow avoids this.
