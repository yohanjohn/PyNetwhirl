"""
JAX-accelerated 2D WTA dynamics using local Gaussian kernels and 2D convolution.

This backend mirrors the NumPy dynamics in `dynamics.py` but:
  - Uses JAX arrays and JIT compilation.
  - Uses small 2D Gaussian kernels with `jax.scipy.signal.convolve2d` instead of
    full (N, N) weight matrices.

It does not modify the main API; notebooks can import it as:

    import wta2d.jax_backend as wta2d_jax
"""

from typing import Optional, Literal, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.signal import convolve2d
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "wta2d.jax_backend requires JAX. Install with `pip install jax jaxlib`."
    ) from exc


BoundaryKind = Literal["open", "toroidal"]


def _make_gaussian_kernels(
    radius: int,
    Ae: float,
    ke: float,
    Ai: float,
    ki: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Small 2D Gaussian kernels for excitatory and inhibitory weights.

    Shape: (2*radius+1, 2*radius+1), centered at (radius, radius).
    """
    coords = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(coords, coords, indexing="ij")
    dist = jnp.sqrt(xx**2 + yy**2)
    Ke = Ae * jnp.exp(-((dist / ke) ** 2))
    Ki = Ai * jnp.exp(-((dist / ki) ** 2))
    # Zero out the self-connection at the center to mirror make_weights
    center = radius
    Ke = Ke.at[center, center].set(0.0)
    Ki = Ki.at[center, center].set(0.0)
    return Ke, Ki


def _convolve_activity(
    y: jnp.ndarray,
    kernel: jnp.ndarray,
    boundary: BoundaryKind,
) -> jnp.ndarray:
    """
    2D convolution of activity y with given kernel.

    Args:
        y: (side, side) activity (already rectified).
        kernel: (kh, kw) kernel (centered).
        boundary: "open" uses zero padding, "toroidal" uses wrap-around.

    Returns:
        conv: (side, side) array.
    """
    if boundary == "toroidal":
        bmode = "wrap"
    else:
        bmode = "fill"  # zero padding
    return convolve2d(y, kernel, mode="same", boundary=bmode)


def simulate_conv_jax(
    Inp: np.ndarray,
    *,
    side: int,
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
    boundary: BoundaryKind = "open",
    save_every: int = 10,
    radius: Optional[int] = None,
    x0: Optional[np.ndarray] = None,
    z0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    JAX implementation of the 2D WTA dynamics using local Gaussian kernels
    and 2D convolution.

    Args:
        Inp: (T, N) NumPy input (flattened grid: row-major, N = side*side).
        side: Grid side length (aa).
        Ae, ke, Ai, ki: Kernel parameters.
        h, B, C, ra, recov, rstrength, zinh: Dynamics parameters (same as NumPy version).
        boundary: \"open\" (zero padding) or \"toroidal\" (wrap-around).
        save_every: Store activity every this many steps (default 10).
        radius: Kernel radius in grid units. If None, use max(ceil(3*ke), ceil(3*ki)).
        x0, z0: Optional initial state, shape (N,).

    Returns:
        xs: (n_stored, N) NumPy array of rectified activity over time, where
            n_stored = ceil(T / save_every).
    """
    T, N = Inp.shape
    if N != side * side:
        raise ValueError(f"N={N} must equal side*side={side*side}")

    if radius is None:
        radius = int(max(np.ceil(3 * ke), np.ceil(3 * ki)))
        radius = max(radius, 1)

    if save_every < 1:
        save_every = 1

    # Kernels on JAX device
    Ke, Ki = _make_gaussian_kernels(radius, Ae, ke, Ai, ki)

    # Initial state
    if x0 is None:
        x0_np = np.zeros((side, side), dtype=np.float32)
    else:
        x0_np = np.asarray(x0, dtype=np.float32).reshape(side, side)
    if z0 is None:
        z0_np = np.zeros((side, side), dtype=np.float32)
    else:
        z0_np = np.asarray(z0, dtype=np.float32).reshape(side, side)

    x0_j = jnp.asarray(x0_np)
    z0_j = jnp.asarray(z0_np)
    Inp_j = jnp.asarray(Inp.reshape(T, side, side), dtype=jnp.float32)

    params = dict(
        h=h,
        B=B,
        C=C,
        ra=ra,
        recov=recov,
        rstrength=rstrength,
        zinh=zinh,
        boundary=boundary,
    )

    def step_fn(carry, t_and_inp):
        x, z = carry
        t, inp_t = t_and_inp
        y = jnp.maximum(x, 0.0)

        exc = _convolve_activity(y, Ke, boundary=params["boundary"])
        inh = _convolve_activity(y, Ki, boundary=params["boundary"])

        drive = (params["B"] - x) * (inp_t + exc)
        supp = (x + params["C"]) * (inh + params["rstrength"] * z)
        x_new = x + params["h"] * (drive - supp - x * params["ra"])
        x_new = jnp.clip(x_new, 0.0, params["B"])

        # z update (same structure as NumPy version, using sign of inh=y)
        sign_inh = jnp.sign(y)
        sign_inh = jnp.where(sign_inh == 0, 1.0, sign_inh)
        #z_new = z + params["recov"] * params["h"] * (
        #    (params["B"] - z) * y - params["zinh"] * z * (1.0 - sign_inh)
        #)
        z_new = z + params["recov"] * params["h"] * (
            (params["B"] - z) * y - params["zinh"] * z 
        )
        
        z_new = jnp.clip(z_new, 0.0, params["B"])

        # Decide whether to store this step
        store_mask = (t % save_every) == 0
        return (x_new, z_new), (store_mask, jnp.maximum(x_new, 0.0))

    ts = jnp.arange(T, dtype=jnp.int32)
    carry0 = (x0_j, z0_j)
    (_, _), (store_masks, xs_all) = jax.lax.scan(
        step_fn, carry0, (ts, Inp_j)
    )

    # Compress stored steps
    xs_stored = xs_all[store_masks]
    if xs_stored.shape[0] == 0 and T > 0:
        xs_stored = xs_all[-1:,:]

    xs_flat = xs_stored.reshape(xs_stored.shape[0], N)
    return np.asarray(xs_flat)

