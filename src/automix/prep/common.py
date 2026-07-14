import torch
import torchaudio

from automix.audio_io import load_wav, save_wav


def resample_and_save(src_path, dst_path, target_sample_rate: int):
    """Loads an audio file, resamples to target_sample_rate if needed,
    and writes it to dst_path (creating parent directories)."""
    waveform, sample_rate = load_wav(src_path)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    # 32-bit float avoids the clipping/quantization that default 16-bit
    # integer PCM would apply to samples outside [-1, 1].
    save_wav(dst_path, waveform, target_sample_rate, subtype="FLOAT")


def sum_and_save(src_paths, dst_path, target_sample_rate: int):
    """Loads multiple audio files, resamples each to target_sample_rate,
    sums them sample-for-sample (zero-padding the shorter to match the
    longest), and writes the result as dst_path.

    Sums incrementally, one file in memory at a time — inputs can be
    dozens of long orchestral recordings."""
    summed = None
    for src_path in src_paths:
        waveform, sample_rate = load_wav(src_path)
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)

        if summed is None:
            summed = waveform.clone()
        else:
            channels = max(summed.shape[0], waveform.shape[0])
            length = max(summed.shape[1], waveform.shape[1])
            if summed.shape[0] < channels or summed.shape[1] < length:
                grown = torch.zeros(channels, length)
                grown[:summed.shape[0], :summed.shape[1]] = summed
                summed = grown
            summed[:waveform.shape[0], :waveform.shape[1]] += waveform

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    save_wav(dst_path, summed, target_sample_rate, subtype="FLOAT")
