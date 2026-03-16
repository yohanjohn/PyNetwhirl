"""
Visualization: time slices (show_2D), summary plots, kernels, and GIF export.
"""

import os
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import matplotlib.cm as cm


def _reshape_to_grid(vec: np.ndarray, aa: int) -> np.ndarray:
    """Reshape (N,) to (aa, aa) row-major as in Matlab (ii=row, jj=col)."""
    return vec.reshape(aa, aa)


def plot_activity_heatmap(
    activity: np.ndarray,
    aa: Optional[int] = None,
    *,
    title: Optional[str] = "Activity",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Simple 2D heatmap of activity on the grid. Useful for testing resizing and layout.

    Args:
        activity: Either (N,) for one time step or (aa, aa) already gridded.
        aa: Grid side length. If None, inferred from activity (sqrt of size for 1D, or shape for 2D).
        title: Axis title.
        vmin, vmax: Color scale; if None, use data min/max.
        figsize: Figure size; default (5, 5).
        ax: If given, draw on this axes; otherwise create a new figure.

    Returns:
        The figure (existing or newly created).
    """
    if activity.ndim == 2:
        grid = np.asarray(activity)
        aa = grid.shape[0]
    else:
        activity = np.asarray(activity).ravel()
        if aa is None:
            n = activity.size
            aa = int(np.sqrt(n))
            if aa * aa != n:
                raise ValueError(f"activity size {n} is not a perfect square; pass aa explicitly")
        grid = _reshape_to_grid(activity, aa)
    if vmin is None:
        vmin = np.nanmin(grid)
    if vmax is None:
        vmax = np.nanmax(grid)
    if vmax <= vmin:
        vmax = vmin + 1
    if ax is None:
        figsize = figsize or (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.imshow(grid, aspect="equal", vmin=vmin, vmax=vmax)
    if title:
        ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    return fig


def show_2d(
    xs: np.ndarray,
    Inp: np.ndarray,
    aa: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Interactive figure: slider over time to show input and activity as 2D grids and surfs.
    Matches Matlab show_2D(xs, Inp, aa).
    """
    if figsize is None:
        figsize = (10, 8)
    fig = plt.figure(figsize=figsize)
    ax_surf_inp = fig.add_subplot(2, 2, 1, projection="3d")
    ax_surf_xs = fig.add_subplot(2, 2, 2, projection="3d")
    ax_im_inp = fig.add_subplot(2, 2, 3)
    ax_im_xs = fig.add_subplot(2, 2, 4)
    Imax = np.nanmax(Inp) if Inp.size else 0
    xmax = np.nanmax(xs) if xs.size else 0
    if Imax == 0:
        Imax = 1
    if xmax == 0:
        xmax = 1
    L = xs.shape[0]

    def update(frame: int) -> None:
        frame = int(np.clip(frame, 0, L - 1))
        inp_frame = _reshape_to_grid(Inp[frame], aa)
        xs_frame = _reshape_to_grid(xs[frame], aa)

        ax_surf_inp.clear()
        ax_surf_inp.plot_surface(
            np.arange(aa), np.arange(aa), inp_frame,
            cmap=cm.viridis, rstride=1, cstride=1,
        )
        ax_surf_inp.set_xlim(0, aa - 1)
        ax_surf_inp.set_ylim(0, aa - 1)
        ax_surf_inp.set_zlim(0, Imax)
        ax_surf_inp.axis("off")

        ax_surf_xs.clear()
        ax_surf_xs.plot_surface(
            np.arange(aa), np.arange(aa), xs_frame,
            cmap=cm.viridis, rstride=1, cstride=1,
        )
        ax_surf_xs.set_xlim(0, aa - 1)
        ax_surf_xs.set_ylim(0, aa - 1)
        ax_surf_xs.set_zlim(0, xmax)
        ax_surf_xs.axis("off")

        ax_im_inp.clear()
        ax_im_inp.imshow(inp_frame, aspect="equal", vmin=0, vmax=Imax)
        ax_im_inp.set_title("Input")
        ax_im_inp.axis("off")

        ax_im_xs.clear()
        ax_im_xs.imshow(xs_frame, aspect="equal", vmin=0, vmax=xmax)
        ax_im_xs.set_title("Activity")
        ax_im_xs.axis("off")
        plt.draw()

    update(0)
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, "time", 0, L - 1, valinit=0, valstep=1)
    slider.on_changed(lambda v: update(v))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()


def plot_summary(
    xs: np.ndarray,
    Inp: np.ndarray,
    title_activities: str = "Activities",
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Three-panel summary: imagesc(Inp'), imagesc(xs'), and plot(xs).
    Matches the summary subplots in the Matlab scripts.
    """
    if figsize is None:
        figsize = (10, 8)
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    axes[0].imshow(Inp.T, aspect="auto")
    axes[0].set_title("Input")
    axes[1].imshow(xs.T, aspect="auto")
    axes[1].set_title("Activities")
    axes[2].plot(xs)
    axes[2].set_title(title_activities)
    axes[2].set_xlabel("time")
    plt.tight_layout()
    return fig


def plot_kernels(We: np.ndarray, Wi: np.ndarray, aa: int) -> plt.Figure:
    """
    Show excitatory and inhibitory kernel centered at middle of grid.
    Matlab: inde = round(N/2 - aa/2); imagesc(reshape(We(inde,:),aa,aa)), etc.
    """
    N = We.shape[0]
    inde = int(round(N / 2 - aa / 2))
    inde = max(0, min(inde, N - 1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(We[inde].reshape(aa, aa), aspect="equal")
    axes[0].set_title("Excitatory kernel")
    axes[0].axis("off")
    axes[1].imshow(Wi[inde].reshape(aa, aa), aspect="equal")
    axes[1].set_title("Inhibitory kernel")
    axes[1].axis("off")
    plt.tight_layout()
    return fig


def save_gif(
    xs: np.ndarray,
    aa: int,
    filepath: str,
    nframe: Optional[int] = None,
    delay: float = 0.1,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    border: int = 1,
    every_n_steps: int = 10,
) -> None:
    """
    Save activity time series as an animated GIF.

    Frames are subsampled from the time series: one frame every every_n_steps
    (default 10). Optionally cap the number of frames with nframe. Output is
    written to the Examples subfolder. Colormap and dark background are
    configurable.
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("save_gif requires imageio: pip install imageio") from None

    # Save under Examples subfolder
    examples_dir = "Examples"
    os.makedirs(examples_dir, exist_ok=True)
    filepath = os.path.join(examples_dir, os.path.basename(filepath))

    T = xs.shape[0]
    if every_n_steps < 1:
        every_n_steps = 1
    indices = np.arange(0, T, every_n_steps)
    if nframe is not None and len(indices) > nframe:
        pick = np.linspace(0, len(indices) - 1, nframe).astype(int)
        indices = indices[pick]
    if len(indices) == 0:
        indices = np.array([0])
    vmin = float(vmin) if vmin is not None else 0
    vmax = float(vmax) if vmax is not None else np.nanmax(xs)
    if np.isnan(vmax) or vmax <= vmin:
        vmax = vmin + 1

    frames = []
    for i in indices:
        frame = _reshape_to_grid(xs[i], aa)
        fig, ax = plt.subplots(figsize=(5, 5), facecolor="black")
        ax.set_facecolor("black")
        ax.imshow(frame, aspect="equal", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.axis("off")
        fig.canvas.draw()
        #buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        #buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, 1:] 

        # Optionally crop away uniform background, leaving a minimal border
        if border is not None and border >= 0:
            bg = buf[0, 0].copy()
            mask = (buf != bg).any(axis=2)
            if np.any(mask):
                ys, xs_idx = np.where(mask)
                y0 = max(int(ys.min()) - border, 0)
                y1 = min(int(ys.max()) + border, buf.shape[0] - 1)
                x0 = max(int(xs_idx.min()) - border, 0)
                x1 = min(int(xs_idx.max()) + border, buf.shape[1] - 1)
                buf = buf[y0 : y1 + 1, x0 : x1 + 1]

        frames.append(buf)
        plt.close(fig)

    imageio.mimsave(filepath, frames, duration=delay, loop=0)
