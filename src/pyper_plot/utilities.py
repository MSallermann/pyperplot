from __future__ import annotations

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.text import Text
from matplotlib.gridspec import GridSpecBase
import numpy as np
import numpy.typing as npt
from typing import Literal
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from typing import Any
import os
from pathlib import Path


def gs_row(
    idx_row: int, gs: GridSpecBase, fig: Figure, idx_slice: slice
) -> list[Axes]:
    return [
        fig.add_subplot(gs[idx_row, idx_col])
        for idx_col in range(gs.ncols)[idx_slice]
    ]


def gs_col(
    idx_col: int, gs: GridSpecBase, fig: Figure, idx_slice: slice
) -> list[Axes]:
    return [
        fig.add_subplot(gs[idx_col, idx_row])
        for idx_row in range(gs.nrows)[idx_slice]
    ]


def label_subplot(
    ax: Axes,
    text: str,
    pad_x: float = 0.0,
    pad_y: float = 0.025,
    **kwargs: dict[str, Any],
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
    ax: Axes, axes_image: AxesImage, **kwargs: dict[str, Any]
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
        margin_x (float, optional): X margin of the inset as a fraction of the containing ax. Defaults to 0.0.
        margin_y (float, optional): Y margin of the inset as a fraction of the containing ax. Defaults to 0.0.
        x_align (str, optional): Where to align the inset horizontally. One of ["left", "right", "center"]. Defaults to "left".
        y_align (str, optional): Where to align the inset horizontally. One of ["bottom", "top", "center"]. Defaults to "bottom".

    Returns:
        plt.Axes: The inset Axes object

    """
    bbox = containing_ax.get_position()

    w = bbox.x1 - bbox.x0
    h = bbox.y1 - bbox.y0

    def helper(
        old_var0: float,
        old_var1: float,
        align: str,
        wh: float,
        rel_width_height: float,
        margin_xy: float,
    ) -> tuple[float, float]:
        if align == "left" or align == "bottom":
            new_var0 = old_var0 + margin_xy * rel_width_height * wh
            new_var1 = new_var0 + rel_width_height * wh
        elif align == "right" or align == "top":
            new_var1 = old_var1 - margin_xy * rel_width_height * wh
            new_var0 = new_var1 - rel_width_height * wh
        elif align == "center":
            new_var0 = old_var0 + 0.5 * (1.0 - rel_width_height) * wh
            new_var1 = old_var1 - 0.5 * (1.0 - rel_width_height) * wh
        else:
            raise Exception(f"unknown align: {align}")
        return new_var0, new_var1

    x0, x1 = helper(bbox.x0, bbox.x1, x_align, w, rel_width, margin_x)
    y0, y1 = helper(bbox.y0, bbox.y1, y_align, h, rel_height, margin_y)

    return fig.add_axes(  # pyright: ignore[reportUnknownMemberType]
        rect=(x0, y0, x1, y1)
    )


def open_image(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path))


def image_to_ax(ax: Axes, image: str | Path | npt.NDArray[np.floating]) -> AxesImage:
    if isinstance(image, str | Path):
        if os.path.exists(image):
            image = open_image(image)
        else:
            raise Exception(f"`{image}` does not exist")

    ax.tick_params(  # type: ignore
        axis="both", which="both", bottom=0, left=0, labelbottom=0, labelleft=0
    )
    ax.set_facecolor((0, 0, 0, 0))

    for s in ax.spines.values():
        s.set_visible(False)

    return ax.imshow(image)  # pyright: ignore[reportUnknownMemberType]


def clear_spines(ax: Axes) -> Axes:
    ax.set_facecolor((0, 0, 0, 0))
    ax.tick_params(  # pyright: ignore[reportUnknownMemberType]
        axis="both", which="both", bottom=0, left=0, labelbottom=0, labelleft=0
    )
    for k in ["left", "right", "top", "bottom"]:
        s = ax.spines[k]
        s.set_visible(False)
    return ax
