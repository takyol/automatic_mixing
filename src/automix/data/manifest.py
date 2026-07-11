import random
from dataclasses import dataclass
from pathlib import Path
from automix.audio_io import frame_count


@dataclass
class SongEntry:
    song_id: str
    stem_paths: list
    target_path: Path
    num_frames: int


def build_manifest(processed_root: Path) -> list:
    """Scans processed_root/<song_id>/{stems/*.wav, target.wav} and
    returns a SongEntry per complete song, sorted by song_id."""
    processed_root = Path(processed_root)
    if not processed_root.is_dir():
        return []

    entries = []
    for song_dir in sorted(processed_root.iterdir()):
        if not song_dir.is_dir():
            continue
        stems_dir = song_dir / "stems"
        target_path = song_dir / "target.wav"
        if not stems_dir.is_dir() or not target_path.exists():
            continue
        stem_paths = sorted(stems_dir.glob("*.wav"))
        if not stem_paths:
            continue
        entries.append(SongEntry(
            song_id=song_dir.name,
            stem_paths=stem_paths,
            target_path=target_path,
            num_frames=frame_count(target_path),
        ))
    return entries


def split_train_val(entries: list, val_fraction: float = 0.1, seed: int = 0):
    """Splits songs (not clips) into train/val, deterministic given seed."""
    shuffled = list(entries)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction)) if shuffled else 0
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val
