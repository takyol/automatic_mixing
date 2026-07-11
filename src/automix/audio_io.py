import numpy as np
import soundfile as sf
import torch


def load_wav(path, frame_offset: int = 0, num_frames: int = -1):
    """Loads a WAV file as a (channels, frames) float32 tensor normalized
    to [-1, 1], plus its sample rate.
    """
    data, sample_rate = sf.read(str(path), start=frame_offset, frames=num_frames,
                                 dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T).contiguous()
    return waveform, sample_rate


def save_wav(path, waveform: torch.Tensor, sample_rate: int, subtype: str = "PCM_16"):
    """Writes a (channels, frames) tensor to `path` as a WAV file."""
    data = np.ascontiguousarray(waveform.detach().cpu().numpy().T)
    sf.write(str(path), data, sample_rate, subtype=subtype)


def frame_count(path) -> int:
    return sf.info(str(path)).frames
