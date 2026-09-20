import random
from dataclasses import dataclass
from fnmatch import fnmatchcase
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


def split_train_val(entries: list, val_fraction: float = 0.1, seed: int = 0,
                    val_patterns: list = None):
    """Splits songs (not clips) into train/val, deterministic given seed.

    `val_patterns` (song_id globs) forces matching songs into val and takes
    them out of the random draw. Needed when songs are windows cut from one
    long session: neighbouring windows are musically near-identical, so a
    random split would leak train material into val. Because the forced
    songs are removed *before* shuffling, the rest of the corpus keeps the
    exact split it had without them - val losses stay comparable to earlier
    runs on that part.
    """
    forced = [e for e in entries if any(fnmatchcase(e.song_id, p) for p in val_patterns or [])]
    forced_ids = {e.song_id for e in forced}
    shuffled = [e for e in entries if e.song_id not in forced_ids]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction)) if shuffled else 0
    val = forced + shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val
