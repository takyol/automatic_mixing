import torch


def masked_mean_context(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    embeddings: (B, N, D) per-track embeddings
    mask: (B, N) bool, True = real track

    Returns: (B, D) mean embedding over real tracks only.
    """
    mask_f = mask.to(dtype=embeddings.dtype).unsqueeze(-1)
    summed = (embeddings * mask_f).sum(dim=1)
    counts = mask_f.sum(dim=1).clamp(min=1.0)
    return summed / counts
