import torch


def apply_gain_pan(stems: torch.Tensor, gain: torch.Tensor, theta: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """
    stems: (B, N, T) raw audio per track
    gain: (B, N) linear gain per track
    theta: (B, N) pan angle in radians, 0 = full left, pi/2 = full right
    mask: (B, N) bool, True = real track, False = padded/silent

    Returns: (B, 2, T) stereo mix, channel 0 = left, channel 1 = right.
    """
    mask_f = mask.to(dtype=gain.dtype)
    left_gain = gain * torch.cos(theta) * mask_f
    right_gain = gain * torch.sin(theta) * mask_f

    left = (stems * left_gain.unsqueeze(-1)).sum(dim=1)
    right = (stems * right_gain.unsqueeze(-1)).sum(dim=1)

    return torch.stack([left, right], dim=1)
