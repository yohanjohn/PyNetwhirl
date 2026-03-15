"""
2D grid layout and distance computation for the neural sheet.
"""

import numpy as np
from typing import Tuple


def make_grid(side: int) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Create a square grid of neurons and compute pairwise Euclidean distances.

    Args:
        side: Side length of the square grid (aa in Matlab).

    Returns:
        N: Number of neurons (side * side).
        pos: (N, 2) array of (row, col) positions, 1-indexed style as in Matlab.
        distan: (N, N) symmetric matrix of L2 distances between positions.
    """
    pos = grid_positions(side)
    N = pos.shape[0]
    distan = pairwise_distances(pos)
    return N, pos, distan


def grid_positions(side: int) -> np.ndarray:
    """
    Linear index to 2D grid positions (1-based row, col as in Matlab).

    Args:
        side: Side length of the square grid.

    Returns:
        pos: (N, 2) array, pos[k] = [ii, jj] for ii, jj in 1..side.
    """
    rows = np.arange(1, side + 1, dtype=float)
    cols = np.arange(1, side + 1, dtype=float)
    jj, ii = np.meshgrid(cols, rows)
    pos = np.column_stack([ii.ravel(), jj.ravel()])
    return pos


def pairwise_distances(pos: np.ndarray) -> np.ndarray:
    """
    Symmetric matrix of L2 distances between all pairs of positions.

    Args:
        pos: (N, 2) array of 2D positions.

    Returns:
        distan: (N, N) symmetric matrix; distan[i,j] = norm(pos[i]-pos[j], 2).
    """
    # Vectorized: diff[i,j] = pos[i] - pos[j], then norm per (i,j)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 2)
    distan = np.linalg.norm(diff, axis=2)
    return distan


def pairwise_toroidal_distances(side: int) -> np.ndarray:
    """
    Symmetric matrix of L2 distances on a 2D torus (wrap-around boundary conditions).

    The topology is a square grid of size (side, side) where opposite edges are
    identified. Distances are computed using the shortest displacement along each
    axis, i.e. min(|Δx|, side - |Δx|).

    Args:
        side: Side length of the square grid.

    Returns:
        distan: (N, N) symmetric matrix of toroidal distances, with
            N = side * side and the same linear indexing convention as
            `grid_positions(side)`.
    """
    rows = np.arange(1, side + 1, dtype=float)
    cols = np.arange(1, side + 1, dtype=float)
    jj, ii = np.meshgrid(cols, rows)  # ii=row, jj=col, to match grid_positions
    ii_flat = ii.ravel()
    jj_flat = jj.ravel()

    # Raw displacements
    di = ii_flat[:, np.newaxis] - ii_flat[np.newaxis, :]
    dj = jj_flat[:, np.newaxis] - jj_flat[np.newaxis, :]

    # Wrap-around: choose the shorter displacement on the ring of length `side`
    di = np.abs(di)
    dj = np.abs(dj)
    di = np.minimum(di, side - di)
    dj = np.minimum(dj, side - dj)

    distan = np.sqrt(di ** 2 + dj ** 2)
    return distan
