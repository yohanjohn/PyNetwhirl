"""
Input pattern generators for 2D WTA simulations.
"""

import numpy as np
from typing import Optional, Union


def constant_input(
    tlen: int,
    N: int,
    value: Union[float, np.ndarray],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    (T, N) input: each row is the same vector (e.g. fixed random pattern).

    Args:
        tlen: Number of time steps.
        N: Number of neurons.
        value: scalar or (N,) array; if scalar, broadcast to (N,).
        rng: Optional numpy random generator.

    Returns:
        Inp: (tlen, N).
    """
    if np.isscalar(value):
        row = np.full(N, value)
    else:
        row = np.asarray(value).ravel()
        if row.size != N:
            raise ValueError("value size must be N")
    return np.broadcast_to(row, (tlen, N)).copy()


def random_sparse_input(
    tlen: int,
    N: int,
    sparsity: float = 0.5,
    scale: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Random binary-like input: round(max(rand - (1-sparsity), 0)) * scale.
    Matlab: Inp = round(max(rand(tlen,N) - 0.45, 0)).

    Args:
        tlen, N: Shape.
        sparsity: Subtract this from uniform [0,1]; negative values become 0.
        scale: Multiply by this.
        rng: Optional random generator.

    Returns:
        Inp: (tlen, N).
    """
    rng = rng or np.random.default_rng()
    r = rng.random((tlen, N))
    return (np.maximum(r - (1 - sparsity), 0) * scale).astype(float)


def strip_input(
    tlen: int,
    N: int,
    neuron_inds: Union[range, slice, np.ndarray],
    scale: float = 1.0,
    turn_off_after_frac: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Input to a strip of neurons (e.g. columns 30:50). Optionally zero after a fraction of time.
    Matlab: Inp(:,30:50) = rand(...); Inp(round(msize*0.5):msize,:) = 0.

    Args:
        tlen, N: Shape.
        neuron_inds: Indices (or slice) of neurons that receive input.
        scale: Scale random input.
        turn_off_after_frac: If set (e.g. 0.5), zero all input from step int(tlen*frac) onward.
        rng: Optional random generator.

    Returns:
        Inp: (tlen, N).
    """
    rng = rng or np.random.default_rng()
    Inp = np.zeros((tlen, N))
    inds = np.arange(N)[neuron_inds]
    Inp[:, inds] = scale * rng.random((tlen, len(inds)))
    if turn_off_after_frac is not None:
        t_off = int(tlen * turn_off_after_frac)
        Inp[t_off:, :] = 0
    return Inp


def pattern_input(
    tlen: int,
    pattern_flat: np.ndarray,
    scale: float = 1.0,
    turn_off_after_frac: Optional[float] = None,
) -> np.ndarray:
    """
    Repeat a single (N,) pattern for all time steps. Optionally turn off after a fraction.

    Args:
        tlen: Number of time steps.
        pattern_flat: (N,) pattern.
        scale: Scale applied to pattern.
        turn_off_after_frac: If set, zero from step int(tlen*frac) onward.

    Returns:
        Inp: (tlen, N).
    """
    N = pattern_flat.size
    Inp = np.broadcast_to(scale * np.asarray(pattern_flat).ravel(), (tlen, N)).copy()
    if turn_off_after_frac is not None:
        t_off = int(tlen * turn_off_after_frac)
        Inp[t_off:, :] = 0
    return Inp
