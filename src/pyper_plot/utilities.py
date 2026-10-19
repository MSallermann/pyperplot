"""Convenience functions for creating and decorating Matplotlib axes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.image import AxesImage
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.text import Text
from matplotlib.typing import ColorType, LineStyleType

_DEFAULT_SLICE = slice(None, None, None)


def create_n_axes(
    gs: GridSpec, n: int, fig: Figure | None = None
) -> list[tuple[Axes, int, int]]:
    """Create the first *n* axes in a grid, in row-major order.

    Args:
        gs: Grid whose cells receive the new axes.
        n: Number of axes to create.
        fig: Owning figure. Defaults to ``gs.figure``.

    Returns:
        Tuples containing each new axes and its zero-based row and column.

    Raises:
        ValueError: If *n* is negative or exceeds the number of grid cells.
    """
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
    """Create axes in selected columns of one grid row.

    Args:
        idx_row: Row index accepted by the GridSpec, including negative indices.
        gs: Grid whose cells receive the new axes.
        fig: Owning figure. Defaults to ``gs.figure``.
        idx_slice: Slice applied to the grid's column indices.

    Returns:
        New axes ordered from the first selected column to the last.
    """
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
    """Create axes in selected rows of one grid column.

    Args:
        idx_col: Column index accepted by the GridSpec, including negative indices.
        gs: Grid whose cells receive the new axes.
        fig: Owning figure. Defaults to ``gs.figure``.
        idx_slice: Slice applied to the grid's row indices.

    Returns:
        New axes ordered from the first selected row to the last.
    """
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
    """Add a label near the upper-left corner of an axes.

    ``pad_x`` and ``pad_y`` use axes coordinates. Their signs also determine
    text alignment so the label extends away from the axes boundary.

    Additional keyword arguments are forwarded to ``Axes.text``.
    """
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
    """Draw a ``FancyBboxPatch`` around the displayed extent of an image.

    Args:
        ax: Axes containing the image.
        axes_image: Image whose data extent defines the box.
        **kwargs: Styling forwarded to ``FancyBboxPatch``.

    Returns:
        The patch added to *ax*.
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
    """Overlay a transparent axes that draws selected spines.

    This is useful when images or artists obscure the original axes frame.
    The overlay has no ticks or labels and occupies the position of *spec*.

    Args:
        fig: Figure that owns the original and overlay axes.
        spec: Existing axes or subplot specification to outline.
        color: Spine color.
        which: Spines to show.
        zorder: Drawing order of the overlay axes.
        label: Matplotlib label assigned to the overlay axes.
        linewidth: Optional spine width.
        linestyle: Optional spine style.
        alpha: Optional spine opacity.

    Returns:
        The transparent overlay axes.
    """
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
    """Create an inset axes positioned relative to another axes.

    Width, height, and margins are fractions of the containing axes. Margins
    are measured inward from the selected edge and ignored for centered
    alignment. The inset position is fixed in figure coordinates when this
    function is called.

    Args:
        fig: Figure that will own the inset.
        containing_ax: Axes used as the positioning reference.
        rel_width: Inset width divided by the containing axes width.
        rel_height: Inset height divided by the containing axes height.
        margin_x: Horizontal margin from a selected edge.
        margin_y: Vertical margin from a selected edge.
        x_align: Horizontal alignment within the containing axes.
        y_align: Vertical alignment within the containing axes.

    Returns:
        The newly created inset axes.

    Raises:
        ValueError: If the containing axes belongs to another figure, a size
            is non-positive, or an alignment is unknown.
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
    """Hide all spines, ticks, labels, and the opaque axes background."""
    ax.set_facecolor((0, 0, 0, 0))
    ax.tick_params(  # pyright: ignore[reportUnknownMemberType]
        axis="both", which="both", bottom=0, left=0, labelbottom=0, labelleft=0
    )
    for k in ["left", "right", "top", "bottom"]:
        s = ax.spines[k]
        s.set_visible(False)
    return ax


def shade_grid_cell(
    fig: Figure,
    spec: SubplotSpec,
    facecolor: ColorType,
    edgecolor: ColorType = "None",
    alpha: float = 1.0,
    zorder: int = -1,
) -> None:
    """Add a figure-level rectangle behind a GridSpec cell or cell span.

    Args:
        fig: Figure containing the grid.
        spec: Cell or cell span to shade, such as ``grid[0, :]``.
        facecolor: Rectangle fill color.
        edgecolor: Rectangle edge color.
        alpha: Rectangle opacity.
        zorder: Drawing order. The default places the rectangle behind axes.
    """
    bbox = spec.get_position(fig)

    fig.add_artist(
        Rectangle(
            (bbox.x0, bbox.y0),
            bbox.width,
            bbox.height,
            facecolor=facecolor,
            transform=fig.transFigure,
            edgecolor=edgecolor,
            alpha=alpha,
            zorder=zorder,
        )
    )
