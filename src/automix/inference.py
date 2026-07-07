from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
from automix.model.automix_model import AutomixModel


def render_mix(stems_dir: Path, checkpoint_path: Path, vggish_backbone=None, device: str = "cpu"):
    """Renders a stereo mix for a folder of stem WAV files using a
    trained checkpoint. Returns (mix: Tensor(2, T), sample_rate: int).

    All stems must share the same sample rate; they're padded to the
    length of the longest stem.
    """
    stems_dir = Path(stems_dir)
    stem_paths = sorted(stems_dir.glob("*.wav"))
    if not stem_paths:
        raise ValueError(f"No .wav files found in {stems_dir}")

    waveforms = []
    sample_rate = None
    for path in stem_paths:
        waveform, sr = torchaudio.load(str(path))
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"Sample rate mismatch: {path} is {sr}Hz, expected {sample_rate}Hz")
        waveforms.append(waveform.mean(dim=0))

    max_len = max(w.shape[0] for w in waveforms)
    padded = [F.pad(w, (0, max_len - w.shape[0])) for w in waveforms]
    stems = torch.stack(padded, dim=0).unsqueeze(0).to(device)  # (1, N, T)
    mask = torch.ones(1, len(padded), dtype=torch.bool, device=device)

    model = AutomixModel(native_sample_rate=sample_rate, vggish_backbone=vggish_backbone).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.mlp.load_state_dict(checkpoint["mlp_state_dict"])
    model.eval()

    with torch.no_grad():
        mix = model(stems, mask)[0]  # (2, T)

    return mix.cpu(), sample_rate
