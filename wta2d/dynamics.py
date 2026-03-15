"""
Single-step and full simulation of 2D WTA dynamics.
"""

import numpy as np
from typing import Tuple, Optional


def step(
    x: np.ndarray,
    z: np.ndarray,
    inp: np.ndarray,
    We: np.ndarray,
    Wi: np.ndarray,
    h: float = 0.001,
    B: float = 10.0,
    C: float = 10.0,
    ra: float = 0.01,
    recov: float = 0.7,
    rstrength: float = 0.5,
    zinh: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One Euler step of the WTA dynamics.

    y = max(x, 0)
    inh = y
    x += h * ((B-x)*(Inp + y@We) - (x+C)*(y@Wi + rstrength*z) - x*ra)
    z += recov*h*((B-z)*inh - zinh*z*(1-sign(inh)))
    x, z clipped to [0, B].

    Args:
        x: (N,) current activity.
        z: (N,) adaptation variable.
        inp: (N,) input at this time step.
        We, Wi: (N, N) weight matrices.
        h, B, C, ra, recov, rstrength, zinh: scalar parameters.

    Returns:
        x_new: (N,) updated activity.
        z_new: (N,) updated adaptation.
    """
    y = np.maximum(x, 0)
    inh = y

    # x update: (B-x)*(Inp + y*We) - (x+C)*(y*Wi + rstrength*z) - x*ra
    drive = (B - x) * (inp + y @ We)
    supp = (x + C) * (y @ Wi + rstrength * z)
    x = x + h * (drive - supp - x * ra)
    x = np.clip(x, 0, B)

    # z update: (B-z)*inh - zinh*z*(1-sign(inh))
    sign_inh = np.sign(inh)
    sign_inh[sign_inh == 0] = 1  # so 1-sign(inh) is 0 where inh>0
    z = z + recov * h * ((B - z) * inh - zinh * z * (1 - np.sign(inh)))
    z = np.clip(z, 0, B)

    return x, z


def simulate(
    We: np.ndarray,
    Wi: np.ndarray,
    Inp: np.ndarray,
    *,
    h: float = 0.001,
    B: float = 10.0,
    C: float = 10.0,
    ra: float = 0.01,
    recov: float = 0.7,
    rstrength: float = 0.5,
    zinh: float = 10.0,
    x0: Optional[np.ndarray] = None,
    z0: Optional[np.ndarray] = None,
    save_every: int = 10,
) -> np.ndarray:
    """
    Run simulation for T time steps. Records activity every save_every step.

    Args:
        We, Wi: (N, N) weight matrices.
        Inp: (T, N) input time series.
        h, B, C, ra, recov, rstrength, zinh: dynamics parameters.
        x0, z0: (N,) initial state; if None, zeros.
        save_every: store xs every this many steps (default 10); reduces memory.

    Returns:
        xs: (n_stored, N) activity over time (rectified: max(x,0)),
            with n_stored = ceil(T / save_every).
    """
    T, N = Inp.shape
    if x0 is None:
        x0 = np.zeros(N)
    if z0 is None:
        z0 = np.zeros(N)
    if save_every < 1:
        save_every = 1

    n_stored = (T + save_every - 1) // save_every
    x = x0.copy()
    z = z0.copy()
    xs = np.zeros((n_stored, N))

    out_idx = 0
    for t in range(T):
        x, z = step(
            x, z, Inp[t], We, Wi,
            h=h, B=B, C=C, ra=ra, recov=recov,
            rstrength=rstrength, zinh=zinh,
        )
        if t % save_every == 0:
            xs[out_idx] = np.maximum(x, 0)
            out_idx += 1
    # ensure last step is stored if we didn't land on a multiple of save_every
    if T > 0 and (T - 1) % save_every != 0:
        xs[out_idx - 1] = np.maximum(x, 0)

    return xs
