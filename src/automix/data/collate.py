import torch


def collate_variable_tracks(batch):
    """
    batch: list of (stems: Tensor(N_i, T), anchor_theta: Tensor(N_i,),
    target: Tensor(2, T)). T is consistent across the whole corpus
    (stems/targets are resampled to a common sample rate during
    preprocessing); N_i (track count) may differ per item. anchor_theta
    holds fixed pan angles in radians, NaN where the pan is learned.

    Returns:
        stems: (B, max_N, T)
        mask: (B, max_N) bool, True where track i < N_i (real track)
        anchor_theta: (B, max_N), NaN-padded
        target: (B, 2, T)
    """
    max_n = max(stems.shape[0] for stems, _, _ in batch)
    t = batch[0][0].shape[1]
    b = len(batch)

    stems_out = torch.zeros(b, max_n, t)
    mask_out = torch.zeros(b, max_n, dtype=torch.bool)
    anchors_out = torch.full((b, max_n), float("nan"))
    targets_out = torch.zeros(b, 2, t)

    for i, (stems, anchor_theta, target) in enumerate(batch):
        n = stems.shape[0]
        stems_out[i, :n] = stems
        mask_out[i, :n] = True
        anchors_out[i, :n] = anchor_theta
        targets_out[i] = target

    return stems_out, mask_out, anchors_out, targets_out
