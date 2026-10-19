# PyperPlot

PyperPlot provides a Matplotlib `GridSpec` subclass for laying out figures with
absolute measurements. Lengths are specified in inches; use
`AbsoluteGridSpec.cm` to convert centimetres.

```python
import matplotlib.pyplot as plt

from pyper_plot.abs_gridspec import AbsoluteGridSpec

fig = plt.figure(figsize=(6, 4))
grid = AbsoluteGridSpec(
    nrows=2,
    ncols=2,
    figure=fig,
    abs_margin_left=0.5,
    abs_margin_right=0.5,
    abs_margin_bottom=0.5,
    abs_margin_top=0.5,
    abs_wspace=0.25,
    abs_hspace=0.25,
)

ax = fig.add_subplot(grid[0, 0])
```

Image helpers and common axes-layout helpers live in `pyper_plot.image` and
`pyper_plot.utilities`, respectively.
