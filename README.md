# PyperPlot

PyperPlot extends Matplotlib's `GridSpec` with physical dimensions. It is useful
for paper figures where margins, panel sizes, and spacing must be specified in
inches or centimetres rather than tuned as relative fractions.

## Quick start

```python
import matplotlib.pyplot as plt

from pyper_plot import AbsoluteGridSpec

fig = plt.figure()
grid = AbsoluteGridSpec(
    nrows=2,
    ncols=3,
    figure=fig,
    unit="cm",
    abs_content_width=12,
    abs_content_height=8,
    abs_margin_left=1.5,
    abs_margin_right=0.5,
    abs_margin_bottom=1,
    abs_margin_top=0.5,
    abs_wspace=0.4,
    abs_hspace=0.6,
)

axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
fig.savefig("figure.pdf")
```

Here, the six panels occupy a 12 cm by 8 cm content area. PyperPlot adds the
requested margins and spacing and resizes the Matplotlib figure accordingly.
Matplotlib still stores the resulting figure size in inches internally.

## Units and figure sizing

Set `unit="in"` (the default) or `unit="cm"`. The selected unit applies to every
argument whose name starts with `abs_`.

- When `abs_content_width` or `abs_content_height` is given, PyperPlot computes
  and sets that total figure dimension.
- When a content dimension is omitted, PyperPlot keeps the corresponding
  dimension from `figure.get_size_inches()` and computes the available content
  space after subtracting the supplied margins and spacing.
- `AbsoluteGridSpec.cm` remains available as an inches-per-centimetre conversion
  factor for older code that manually converts dimensions.

## Fixed and relative panel sizes

`abs_widths` and `abs_heights` can mix physical sizes with relative weights:

- A non-negative value is an absolute size in the selected unit.
- A negative value is a dimensionless weight for sharing the remaining space.

```python
grid = AbsoluteGridSpec(
    nrows=2,
    ncols=3,
    figure=fig,
    unit="cm",
    abs_content_width=14,
    abs_content_height=8,
    abs_margin_left=1,
    abs_margin_right=1,
    abs_margin_bottom=1,
    abs_margin_top=1,
    abs_widths=[4, -2, -1],
    abs_heights=[3, -1],
)
```

The first column is 4 cm wide. The other columns divide the remaining 10 cm in
a 2:1 ratio. The first row is 3 cm high and the second receives the remaining
5 cm.

If every entry is absolute, the values must add up to the corresponding content
extent. Each sequence must also match the number of rows or columns.

## Axes utilities

Common helpers are available directly from `pyper_plot` and from
`pyper_plot.utilities`:

```python
from pyper_plot import create_inset_axis, create_n_axes, shade_grid_cell

axes_with_indices = create_n_axes(grid, 4)
first_ax, row, col = axes_with_indices[0]

inset = create_inset_axis(
    fig,
    first_ax,
    rel_width=0.4,
    rel_height=0.3,
    margin_x=0.05,
    margin_y=0.05,
    x_align="right",
    y_align="top",
)

shade_grid_cell(fig, grid[1, :], facecolor="0.95")
```

Other helpers include `gs_row`, `gs_col`, `label_subplot`, `spine_axis`,
`clear_spines`, and `add_box_around_image`.

## Image utilities

Image helpers are available directly from `pyper_plot` and from
`pyper_plot.image`:

```python
from pyper_plot import crop_to_content, image_to_ax, open_image

image = open_image("diagram.png")
image = crop_to_content(image)
image_to_ax(first_ax, image)
```

`crop_to_content` infers the background from the most common corner color unless
one is supplied explicitly. `replace_background_color` can replace that color
and add or preserve an alpha channel. Exact channel values are used for
background matching.

## Public API

- Layout: `AbsoluteGridSpec`, `LengthUnit`
- Axes: `create_n_axes`, `gs_row`, `gs_col`, `create_inset_axis`,
  `label_subplot`, `spine_axis`, `clear_spines`, `shade_grid_cell`,
  `add_box_around_image`
- Images: `open_image`, `save_image`, `image_to_ax`, `crop`,
  `crop_to_content`, `replace_background_color`
