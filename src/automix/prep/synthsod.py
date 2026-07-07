from pathlib import Path
from automix.prep.common import resample_and_save, sum_and_save


def prepare_synthsod(raw_root: Path, output_root: Path, target_sample_rate: int,
                      close_mic_glob: str, tree_mic_files: list):
    """Converts SynthSOD into the common data_processed layout.

    Input = Close Mic stems. Target = sum of the configured Tree mic files.
    """
    raw_root = Path(raw_root)
    output_root = Path(output_root)

    if not raw_root.is_dir():
        return

    for song_dir in sorted(raw_root.iterdir()):
        if not song_dir.is_dir():
            continue

        close_mic_paths = sorted(song_dir.glob(close_mic_glob))
        tree_paths = [song_dir / rel for rel in tree_mic_files]
        if not close_mic_paths or not all(p.exists() for p in tree_paths):
            continue

        song_output = output_root / song_dir.name
        for src_path in close_mic_paths:
            resample_and_save(src_path, song_output / "stems" / src_path.name, target_sample_rate)

        sum_and_save(tree_paths, song_output / "target.wav", target_sample_rate)
