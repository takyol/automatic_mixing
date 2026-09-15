import math
from fnmatch import fnmatchcase
from pathlib import Path

import torch


def anchor_thetas_for(stem_paths, patterns) -> torch.Tensor:
    """Maps each stem path to a fixed pan angle in radians, or NaN if no
    pattern matches (pan stays model-predicted for those tracks).

    `patterns` is {filename glob -> angle in degrees, 0=left 45=center
    90=right}, matched case-sensitively against the file name without
    its extension. Anchored tracks keep their learned gain; only the pan
    is pinned.
    """
    thetas = torch.full((len(stem_paths),), float("nan"))
    if not patterns:
        return thetas

    for i, path in enumerate(stem_paths):
        name = Path(path).stem
        for pattern, degrees in patterns.items():
            if fnmatchcase(name, pattern):
                thetas[i] = math.radians(degrees)
                break
    return thetas
