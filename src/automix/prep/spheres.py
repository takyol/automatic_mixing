from pathlib import Path
from automix.prep.common import resample_and_save


def prepare_spheres(raw_root: Path, output_root: Path, target_sample_rate: int,
                     stems_subdir: str, mix_filename: str):
    """Converts Spheres into the common data_processed layout.

    Input = spot mic stems. Target = the dataset's provided manually
    mixed stereo file, resampled to the canonical rate. 
    """
    raw_root = Path(raw_root)
    output_root = Path(output_root)

    if not raw_root.is_dir():
        return

    for song_dir in sorted(raw_root.iterdir()):
        if not song_dir.is_dir():
            continue

        stems_dir = song_dir / stems_subdir
        mix_path = song_dir / mix_filename
        if not stems_dir.is_dir() or not mix_path.exists():
            continue

        stem_paths = sorted(stems_dir.glob("*.wav"))
        if not stem_paths:
            continue

        song_output = output_root / song_dir.name
        for src_path in stem_paths:
            resample_and_save(src_path, song_output / "stems" / src_path.name, target_sample_rate)

        resample_and_save(mix_path, song_output / "target.wav", target_sample_rate)
