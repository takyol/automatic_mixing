import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    """Steinmetz et al. stereo-aware multi-resolution STFT loss.

    Computed on left+right sum and left-right difference signals, at
    multiple STFT resolutions, combining spectral convergence and
    log-magnitude terms.

    Additionally computed per channel (left vs left, right vs right):
    the sum/diff terms alone are invariant to mirroring the stereo
    image, so they cannot penalize a mix whose energy sits on the wrong
    side. The per-channel terms break that symmetry.
    """

    def __init__(self, fft_sizes=(512, 1024, 2048), eps: float = 1e-8):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.eps = eps
        for n_fft in fft_sizes:
            self.register_buffer(f"window_{n_fft}", torch.hann_window(n_fft), persistent=False)

    def _stft_mag(self, x: torch.Tensor, n_fft: int) -> torch.Tensor:
        window = getattr(self, f"window_{n_fft}")
        spec = torch.stft(
            x, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft,
            window=window, return_complex=True, pad_mode="constant",
        )
        return spec.abs()

    def _single_resolution_loss(self, pred: torch.Tensor, target: torch.Tensor, n_fft: int) -> torch.Tensor:
        pred_mag = self._stft_mag(pred, n_fft)
        target_mag = self._stft_mag(target, n_fft)

        sc_loss = torch.norm(target_mag - pred_mag, p="fro") / (torch.norm(target_mag, p="fro") + self.eps)
        log_pred = torch.log(pred_mag + self.eps)
        log_target = torch.log(target_mag + self.eps)
        sm_loss = F.l1_loss(log_pred, log_target)
        return sc_loss + sm_loss

    def _signal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = [self._single_resolution_loss(pred, target, n_fft) for n_fft in self.fft_sizes]
        return torch.stack(losses).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: (B, 2, T) stereo audio, channel 0 = left, channel 1 = right."""
        pred_sum = pred[:, 0, :] + pred[:, 1, :]
        pred_diff = pred[:, 0, :] - pred[:, 1, :]
        target_sum = target[:, 0, :] + target[:, 1, :]
        target_diff = target[:, 0, :] - target[:, 1, :]

        sum_diff_loss = (self._signal_loss(pred_sum, target_sum)
                         + self._signal_loss(pred_diff, target_diff))
        per_channel_loss = (self._signal_loss(pred[:, 0, :], target[:, 0, :])
                            + self._signal_loss(pred[:, 1, :], target[:, 1, :]))
        return sum_diff_loss + per_channel_loss
