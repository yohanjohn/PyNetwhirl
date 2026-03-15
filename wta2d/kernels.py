"""
Excitatory and inhibitory weight kernels (Gaussian in distance).
"""

import numpy as np
from typing import Tuple

from .grid import pairwise_toroidal_distances


def make_weights(
    distan: np.ndarray,
    Ae: float = 0.7,
    ke: float = 1.0,
    Ai: float = 0.0085,
    ki: float = 7.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build excitatory (We) and inhibitory (Wi) weight matrices from distances.

    We(i,j) = Ae * exp(-(d(i,j)/ke)^2), We(i,i)=0.
    Wi(i,j) = Ai * exp(-(d(i,j)/ki)^2), Wi(i,i)=0.
    Both matrices are symmetric.

    Args:
        distan: (N, N) symmetric distance matrix.
        Ae: Excitatory kernel max height.
        ke: Excitatory kernel width.
        Ai: Inhibitory kernel max height.
        ki: Inhibitory kernel width.

    Returns:
        We: (N, N) excitatory weights.
        Wi: (N, N) inhibitory weights.
    """
    N = distan.shape[0]
    We = Ae * np.exp(-((distan / ke) ** 2))
    Wi = Ai * np.exp(-((distan / ki) ** 2))
    np.fill_diagonal(Wi, 0)
    # Matlab leaves We diagonal as-is (comment says We(ii,ii)=0 is optional)
    np.fill_diagonal(We, 0)
    return We, Wi


def make_toroidal_weights(
    side: int,
    Ae: float = 0.7,
    ke: float = 1.0,
    Ai: float = 0.0085,
    ki: float = 7.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build excitatory (We) and inhibitory (Wi) weight matrices assuming circular
    (toroidal) boundary conditions on the 2D grid.

    Distances are computed on a square torus of size (side, side) where opposite
    edges are identified. This yields kernels where neighborhoods wrap around
    seamlessly at the borders.

    Args:
        side: Side length of the square grid (aa).
        Ae, ke, Ai, ki: Kernel parameters, as in `make_weights`.

    Returns:
        We: (N, N) excitatory weights with toroidal distances.
        Wi: (N, N) inhibitory weights with toroidal distances.
    """
    distan = pairwise_toroidal_distances(side)
    return make_weights(distan, Ae=Ae, ke=ke, Ai=Ai, ki=ki)
