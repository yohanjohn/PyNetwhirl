# wta2d

Python package that reproduces the 2D Winner-Take-All (WTA) neural network behavior from the Matlab scripts (`WTA_2D_*.m`, `show_2D.m`, `gifmaker_2D.m`).

## Structure

- **`grid.py`** — 2D square grid of neurons and pairwise Euclidean distance matrix.
- **`kernels.py`** — Excitatory (We) and inhibitory (Wi) Gaussian kernels in distance.
- **`dynamics.py`** — Single-step update and full simulation (activity `x`, adaptation `z`).
- **`inputs.py`** — Input pattern generators (constant, random sparse, strip, pattern).
- **`viz.py`** — Plotting: interactive time slider (`show_2d`), summary panels, kernel view, GIF export.
- **`run.py`** — High-level `run()` and `run_pattern_strip()` to run full simulations.

## Install

From the repo root (Netwhirl directory that contains `wta2d/` and `pyproject.toml`):

```bash
pip install -e .
```

Or install dependencies only:

```bash
pip install -r wta2d/requirements.txt
```

## Usage

```python
from wta2d import make_grid, make_weights, simulate, random_sparse_input, show_2d, plot_summary, plot_kernels

# Build grid and kernels
N, pos, distan = make_grid(side=20)
We, Wi = make_weights(distan, Ae=0.7, ke=1, Ai=0.0085, ki=7)

# Input and simulate
tlen = 5000
Inp = random_sparse_input(tlen, N, sparsity=0.55)
xs = simulate(We, Wi, Inp, h=0.001, B=10, C=10, ra=0.01, recov=0.7, rstrength=0.5, zinh=10)

# Visualize
plot_summary(xs, Inp)
plot_kernels(We, Wi, 20)
show_2d(xs, Inp, 20)  # interactive slider over time
```

High-level runner (matches Matlab script flow):

```python
from wta2d.run import run, run_pattern_strip
import wta2d.viz as viz

xs, Inp, We, Wi, aa = run(side=20, tlen=20000)
viz.plot_summary(xs, Inp)
viz.plot_kernels(We, Wi, aa)
viz.show_2d(xs, Inp, aa)

# Strip input that turns off halfway (like WTA_2D_pattern6)
xs, Inp, We, Wi, aa = run_pattern_strip(side=40, tlen=20000, strip=(30, 50), turn_off_frac=0.5)
```

Save activity as GIF (written to the `Examples` folder; frames subsampled every 10 steps by default):

```python
from wta2d import save_gif
save_gif(xs, aa=20, filepath="activity.gif", every_n_steps=10, delay=0.1)
# Optional: cap total frames with nframe=50
```

## Parameters (Matlab correspondence)

| Parameter   | Typical | Description |
|------------|---------|-------------|
| `side` (aa) | 20–40  | Grid side length; N = side² |
| `Ae`, `ke`  | 0.6–0.8, 1 | Excitatory kernel height and width |
| `Ai`, `ki`  | 0.005–0.05, 3–7 | Inhibitory kernel height and width |
| `h`         | 0.001 | Time step |
| `B`, `C`    | 10    | Saturation and decay |
| `ra`        | 0.01  | Spontaneous decay rate |
| `recov`     | 0.7–1.2 | Adaptation recovery rate |
| `rstrength` | 0.25–0.5 | Strength of adaptation feedback |
| `zinh`      | 10    | Adaptation time constant in z dynamics |

## License

Same as the parent Netwhirl project.
