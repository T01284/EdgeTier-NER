"""Shared CRF Viterbi decoder for Pi 3B ONNX pipeline."""

from __future__ import annotations

import numpy as np


def viterbi_decode(
    emissions: np.ndarray,
    transitions: np.ndarray,
) -> list[int]:
    """
    Args:
        emissions: [seq_len, num_tags]
        transitions: [num_tags, num_tags]
    Returns:
        best path tag ids
    """
    seq_len, num_tags = emissions.shape
    dp = np.zeros((seq_len, num_tags), dtype=np.float64)
    back = np.zeros((seq_len, num_tags), dtype=np.int32)

    dp[0] = emissions[0]
    for t in range(1, seq_len):
        for cur in range(num_tags):
            scores = dp[t - 1] + transitions[:, cur] + emissions[t, cur]
            back[t, cur] = int(np.argmax(scores))
            dp[t, cur] = scores[back[t, cur]]

    path = np.zeros(seq_len, dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(seq_len - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path.tolist()
