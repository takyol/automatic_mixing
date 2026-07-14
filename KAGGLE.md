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

- Run all cells. Cell 3 starts training and **auto-resumes** from
  `/kaggle/working/checkpoints/kaggle/last.pt` when one exists.
- For long runs use **Save Version -> Save & Run All (Commit)**: the notebook
  runs headless for up to 12 hours, no browser needed. Checkpoints and
  TensorBoard logs appear on the notebook's **Output** tab afterwards.
- Download `best.pt` from the Output tab to render mixes locally
  (`scripts/infer.py`), and the `runs/` folder to view curves in a local
  TensorBoard.

## Notes

- Push code changes to GitHub before starting a session — the notebook
  clones this repo fresh each time.
- The weekly GPU quota is visible on your Kaggle profile page.
- Interactive sessions idle out after ~20 min without browser activity;
  the Save & Run All flow avoids this.
