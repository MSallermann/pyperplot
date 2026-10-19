from __future__ import annotations

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.text import Text
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.typing import ColorType, LineStyleType
from collections.abc import Sequence
from typing import Literal
from matplotlib.patches import FancyBboxPatch
from typing import Any

_DEFAULT_SLICE = slice(None, None, None)


def create_n_axes(
    gs: GridSpec, n: int, fig: Figure | None = None
) -> list[tuple[Axes, int, int]]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return []

    max_axes = gs.nrows * gs.ncols
    if n > max_axes:
        raise ValueError(
            f"Cannot create {n} axes on this gridspec with {gs.nrows} rows and {gs.ncols} cols. Maximum is {max_axes}."
        )

    if fig is None:
        fig = gs.figure
    assert fig is not None

    result = []
    for irow in range(gs.nrows):
        for icol in range(gs.ncols):
            ax = fig.add_subplot(gs[irow, icol])
            result.append((ax, irow, icol))
            if len(result) == n:
                return result

    return result


def gs_row(
    idx_row: int,
    gs: GridSpec,
    fig: Figure | None = None,
    idx_slice: slice = _DEFAULT_SLICE,
) -> list[Axes]:
    if fig is None:
        fig = gs.figure
    assert fig is not None
    return [
        fig.add_subplot(gs[idx_row, idx_col])
        for idx_col in range(gs.ncols)[idx_slice]
    ]


def gs_col(
    idx_col: int,
    gs: GridSpec,
    fig: Figure | None = None,
    idx_slice: slice = _DEFAULT_SLICE,
) -> list[Axes]:
    if fig is None:
        fig = gs.figure
    assert fig is not None
    return [
        fig.add_subplot(gs[idx_row, idx_col])
        for idx_row in range(gs.nrows)[idx_slice]
    ]


def label_subplot(
    ax: Axes,
    text: str,
    pad_x: float = 0.0,
    pad_y: float = 0.025,
    **kwargs: Any,
) -> Text:
    """Labels a subplot at the left upper corner. Wrapper around ax.text"""
    if pad_x >= 0.0:
        ha = "left"
    else:
        ha = "right"

    if pad_y >= 0.0:
        va = "bottom"
    else:
        va = "top"

    return ax.text(  # pyright: ignore[reportUnknownMemberType]
        x=pad_x,
        y=1.0 + pad_y,
        s=text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        **kwargs,
    )


def add_box_around_image(
    ax: Axes, axes_image: AxesImage, **kwargs: Any
) -> FancyBboxPatch:
    """Adds a box patch around an axis.

    Args:
        ax (plt.ax): The ax
        axes_image (_type_): The image

    Returns:
        FancyBboxPatch: Returns the box patch

    """
    extent = axes_image.get_extent()
    left, right, bottom, top = extent
    width = right - left
    height = top - bottom
    fancy = FancyBboxPatch((left, bottom), width, height, **kwargs)
    ax.add_patch(fancy)
    return fancy


def spine_axis(
    fig: Figure,
    spec: Axes | SubplotSpec,
    color: ColorType = "black",
    which: Sequence[Literal["left", "right", "top", "bottom"]] = (
        "left",
        "right",
        "top",
        "bottom",
    ),
    zorder: int = 2,
    label: str = "spine",
    linewidth: float | None = None,
    linestyle: LineStyleType | None = None,
    alpha: float | None = None,
) -> Axes:
    if isinstance(spec, Axes):
        bounds = spec.get_position().bounds
    else:
        bounds = spec.get_position(fig).bounds
    a = fig.add_axes(bounds, zorder=zorder, label=label)

    a.set_facecolor((0, 0, 0, 0))
    a.tick_params(
        axis="both", which="both", bottom=0, left=0, labelbottom=0, labelleft=0
    )
    for k in ["left", "right", "top", "bottom"]:
        s = a.spines[k]
        if k in which:
            s.set_linewidth(linewidth)
            s.set_linestyle(linestyle)
            s.set_visible(True)
            s.set_alpha(alpha)
            s.set_color(color)
        else:
            s.set_visible(False)
    return a


def create_inset_axis(
    fig: Figure,
    containing_ax: Axes,
    rel_width: float = 0.5,
    rel_height: float = 0.5,
    margin_x: float = 0.0,
    margin_y: float = 0.0,
    x_align: Literal["left", "right", "center"] = "left",
    y_align: Literal["bottom", "top", "center"] = "bottom",
) -> Axes:
    """Creates an axis for an inset.

    Args:
        containing_ax (plt.Axes): The Axes object relative to which to place the inset
        rel_width (float, optional): Width of the inset as a fraction of the containing ax. Defaults to 0.5.
        rel_height (float, optional): Height of the inset as a fraction of the containing ax. Defaults to 0.5.
        margin_x (float, optional): X margin from the selected left or right edge, as a fraction of the containing ax. Ignored when centered. Defaults to 0.0.
        margin_y (float, optional): Y margin from the selected bottom or top edge, as a fraction of the containing ax. Ignored when centered. Defaults to 0.0.
        x_align (str, optional): Where to align the inset horizontally. One of ["left", "right", "center"]. Defaults to "left".
        y_align (str, optional): Where to align the inset horizontally. One of ["bottom", "top", "center"]. Defaults to "bottom".

    Returns:
        plt.Axes: The inset Axes object

    """
    if containing_ax.figure is not fig:
        raise ValueError("containing_ax must belong to fig")
    if rel_width <= 0.0 or rel_height <= 0.0:
        raise ValueError("relative width and height must be positive")

    bbox = containing_ax.get_position()
    width = rel_width * bbox.width
    height = rel_height * bbox.height

    if x_align == "left":
        left = bbox.x0 + margin_x * bbox.width
    elif x_align == "right":
        left = bbox.x1 - margin_x * bbox.width - width
    elif x_align == "center":
        left = bbox.x0 + 0.5 * (bbox.width - width)
    else:
        raise ValueError(f"unknown horizontal alignment: {x_align}")

    if y_align == "bottom":
        bottom = bbox.y0 + margin_y * bbox.height
    elif y_align == "top":
        bottom = bbox.y1 - margin_y * bbox.height - height
    elif y_align == "center":
        bottom = bbox.y0 + 0.5 * (bbox.height - height)
    else:
        raise ValueError(f"unknown vertical alignment: {y_align}")

    return fig.add_axes(rect=(left, bottom, width, height))


def clear_spines(ax: Axes) -> Axes:
    ax.set_facecolor((0, 0, 0, 0))
    ax.tick_params(  # pyright: ignore[reportUnknownMemberType]
        axis="both", which="both", bottom=0, left=0, labelbottom=0, labelleft=0
    )
    for k in ["left", "right", "top", "bottom"]:
        s = ax.spines[k]
        s.set_visible(False)
    return ax
