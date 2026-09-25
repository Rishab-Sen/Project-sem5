"""
Discrete action-space utilities for QMIX (and any discrete-action MARL method).

A rover's single discrete index k encodes BOTH wheel commands:
    k = i * L + j   ->   (left, right) = (grid[i], grid[j])
i.e. the joint action space is the L x L wheel-level grid (L=5 -> 25 actions).
Continuous methods (MAPPO) bypass this and act directly on the 2-D box.
"""

from __future__ import annotations

import numpy as np


def make_action_grid(n_levels: int = 5) -> np.ndarray:
    """Wheel levels evenly spaced in [-1, 1]. Must include 0 (stop)."""
    assert n_levels % 2 == 1, "odd number of levels so that 0 is included"
    return np.linspace(-1.0, 1.0, n_levels)


def n_joint_actions(n_levels: int = 5) -> int:
    return n_levels * n_levels


def idx_to_wheel_actions(idx: np.ndarray, n_levels: int = 5) -> np.ndarray:
    """
    idx: (N,) integer joint-action indices
    returns: (N, 2) wheel commands in [-1, 1]
    """
    grid = make_action_grid(n_levels)
    idx = np.asarray(idx, dtype=np.int64)
    i = idx // n_levels
    j = idx % n_levels
    return np.stack([grid[i], grid[j]], axis=1)
