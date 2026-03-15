# Netwhirl

2D Winner-Take-All (WTA) neural network: Python package and demos (ported from Matlab WTA_2D scripts).

## Quick start

```bash
pip install -e .
```

```python
import wta2d
xs, Inp, We, Wi, side = wta2d.run(side=20, tlen=2000)
wta2d.viz.plot_summary(xs, Inp)
```

See **[wta2d/README.md](wta2d/README.md)** for the full API and **[wta2d_demo.ipynb](wta2d_demo.ipynb)** / **wta2d_toroidal_demo*.ipynb** for examples.
