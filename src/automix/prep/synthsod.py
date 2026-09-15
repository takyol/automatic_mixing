from pathlib import Path

from automix.audio_io import load_wav, save_wav
from automix.prep.common import resample_and_save, sum_and_save


def prepare_synthsod(raw_root: Path, output_root: Path, target_sample_rate: int,
                      close_mic_glob: str, tree_mic_glob: str):
    """Converts SynthSOD into the common data_processed layout.

    Input = mono Close Mic stems, one per instrument. Target = sum of the
    per-instrument stereo Tree (main array) renders, which reconstructs
    the full stereo tree mixture. The tree sum's left/right channels are
    additionally written as mono Tree_L/Tree_R stems, so training can
    anchor them at fixed hard-left/hard-right pans. Songs missing close
    or tree files are skipped. Stems are written as .wav regardless of
    source format.
    """
    raw_root = Path(raw_root)
    output_root = Path(output_root)

    if not raw_root.is_dir():
        return

    for song_dir in sorted(raw_root.iterdir()):
        if not song_dir.is_dir():
            continue

        close_mic_paths = sorted(song_dir.glob(close_mic_glob))
        tree_paths = sorted(song_dir.glob(tree_mic_glob))
        if not close_mic_paths or not tree_paths:
            continue

        song_output = output_root / song_dir.name
        for src_path in close_mic_paths:
            resample_and_save(src_path, song_output / "stems" / (src_path.stem + ".wav"),
                              target_sample_rate)

        target_path = song_output / "target.wav"
        sum_and_save(tree_paths, target_path, target_sample_rate)

        target, _ = load_wav(target_path)
        save_wav(song_output / "stems" / "Tree_L.wav", target[0:1], target_sample_rate,
                 subtype="FLOAT")
        save_wav(song_output / "stems" / "Tree_R.wav", target[1:2], target_sample_rate,
                 subtype="FLOAT")
