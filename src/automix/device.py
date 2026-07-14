import torch


def resolve_device(requested: str = "auto") -> str:
    """Resolves "auto" to the best available accelerator: CUDA, then Apple
    Silicon's MPS, then CPU. Any other value is passed through unchanged."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
