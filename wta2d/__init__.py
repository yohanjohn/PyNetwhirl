"""
wta2d: 2D Winner-Take-All neural network (Python port of Matlab WTA_2D* scripts).
"""

from .grid import make_grid, grid_positions, pairwise_distances, pairwise_toroidal_distances
from .kernels import make_weights, make_toroidal_weights
from .dynamics import step, simulate
from .inputs import constant_input, random_sparse_input, strip_input, pattern_input
from .viz import show_2d, plot_summary, plot_kernels, plot_activity_heatmap, save_gif
from .run import run, run_pattern_strip

__all__ = [
    "make_grid",
    "grid_positions",
    "pairwise_distances",
    "pairwise_toroidal_distances",
    "make_weights",
    "make_toroidal_weights",
    "step",
    "simulate",
    "constant_input",
    "random_sparse_input",
    "strip_input",
    "pattern_input",
    "show_2d",
    "plot_summary",
    "plot_kernels",
    "plot_activity_heatmap",
    "save_gif",
    "run",
    "run_pattern_strip",
]

__version__ = "0.1.0"
