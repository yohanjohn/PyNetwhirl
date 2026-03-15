"""
High-level runner: build grid, weights, input, run simulation, and optionally plot/save.
"""

import numpy as np
from typing import Optional, Tuple

from .grid import make_grid
from .kernels import make_weights, make_toroidal_weights
from .dynamics import simulate
from .inputs import constant_input, random_sparse_input, strip_input, pattern_input
from . import viz


def run(
    side: int = 20,
    tlen: int = 20000,
    *,
    Ae: float = 0.7,
    ke: float = 1.0,
    Ai: float = 0.0085,
    ki: float = 7.0,
    h: float = 0.001,
    B: float = 10.0,
    C: float = 10.0,
    ra: float = 0.01,
    recov: float = 0.7,
    rstrength: float = 0.5,
    zinh: float = 10.0,
    boundary: str = "open",
    save_every: int = 10,
    Inp: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Run a full 2D WTA simulation (equivalent to one of the Matlab WTA_2D scripts).

    Args:
        side: Grid side length (aa).
        tlen: Number of time steps.
        Ae, ke, Ai, ki: Kernel parameters.
        h, B, C, ra, recov, rstrength, zinh: Dynamics parameters.
        boundary: "open" for standard Euclidean distances, "toroidal" for
            circular wrap-around boundary conditions on the grid.
        save_every: store activity every this many steps (default 10); reduces memory.
        Inp: (tlen, N) input; if None, uses random sparse input.
        rng: Optional random generator.

    Returns:
        xs: (n_stored, N) activity, with n_stored = ceil(tlen / save_every).
        Inp: (n_stored, N) input at saved time points (same length as xs).
        We, Wi: (N, N) weight matrices.
        side: grid side (aa).
    """
    N, pos, distan = make_grid(side)
    if boundary.lower() == "toroidal":
        We, Wi = make_toroidal_weights(side, Ae=Ae, ke=ke, Ai=Ai, ki=ki)
    else:
        We, Wi = make_weights(distan, Ae=Ae, ke=ke, Ai=Ai, ki=ki)
    if Inp is None:
        rng = rng or np.random.default_rng()
        Inp = random_sparse_input(tlen, N, sparsity=0.55, scale=1.0, rng=rng)
    xs = simulate(
        We, Wi, Inp,
        h=h, B=B, C=C, ra=ra, recov=recov,
        rstrength=rstrength, zinh=zinh,
        save_every=save_every,
    )
    # Return input at the same time points as xs for consistent plotting
    tlen = Inp.shape[0]
    indices = np.arange(0, tlen, save_every)
    if len(indices) > xs.shape[0]:
        indices = indices[: xs.shape[0]]
    Inp_saved = Inp[indices].copy()
    if tlen > 0 and (tlen - 1) % save_every != 0 and len(Inp_saved) > 0:
        Inp_saved[-1] = Inp[-1]  # last xs frame is final state; match with final input
    return xs, Inp_saved, We, Wi, side


def run_pattern_strip(
    side: int = 40,
    tlen: int = 20000,
    strip: Tuple[int, int] = (30, 50),
    turn_off_frac: float = 0.5,
    Ae: float = 0.8,
    ke: float = 1.0,
    Ai: float = 0.05,
    ki: float = 3.0,
    boundary: str = "open",
    save_every: int = 10,
    **dyn_kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Run with strip input that turns off halfway (like WTA_2D_pattern6).

    Args mirror `run`, with an additional `strip` range and `turn_off_frac`.
    `boundary` selects "open" vs "toroidal" kernels; `save_every` controls stored steps.
    """
    N, _, distan = make_grid(side)
    if boundary.lower() == "toroidal":
        We, Wi = make_toroidal_weights(side, Ae=Ae, ke=ke, Ai=Ai, ki=ki)
    else:
        We, Wi = make_weights(distan, Ae=Ae, ke=ke, Ai=Ai, ki=ki)
    Inp = strip_input(tlen, N, range(strip[0], strip[1]), scale=1.0, turn_off_after_frac=turn_off_frac)
    xs = simulate(We, Wi, Inp, save_every=save_every, **dyn_kwargs)
    tlen = Inp.shape[0]
    indices = np.arange(0, tlen, save_every)
    if len(indices) > xs.shape[0]:
        indices = indices[: xs.shape[0]]
    Inp_saved = Inp[indices].copy()
    if tlen > 0 and (tlen - 1) % save_every != 0 and len(Inp_saved) > 0:
        Inp_saved[-1] = Inp[-1]
    return xs, Inp_saved, We, Wi, side


if __name__ == "__main__":
    xs, Inp, We, Wi, aa = run(side=20, tlen=2000)
    viz.plot_summary(xs, Inp)
    import matplotlib.pyplot as plt
    plt.savefig("wta2d_summary.png")
    plt.close()
    viz.plot_kernels(We, Wi, aa)
    plt.savefig("wta2d_kernels.png")
    plt.close()
    print("Saved wta2d_summary.png and wta2d_kernels.png. Run show_2d(xs, Inp, aa) for interactive view.")
