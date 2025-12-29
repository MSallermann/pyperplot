from __future__ import annotations

from collections.abc import Sequence

from matplotlib.axes import Axes
from matplotlib.image import AxesImage
import numpy as np
import numpy.typing as npt
from typing import Literal
from matplotlib.patches import FancyBboxPatch
from typing import Any
from pathlib import Path


class PyperPlot:
    """A class for laying out plots in a grid. (Plus some utilities)."""

    # one cm in inches
    cm = 1.0 / 2.54

    # one inch in inches
    inch = 1.0

    # Annotations
    offset_u = 1.5 * np.array([0, 10])
    offset_r = 1.5 * np.array([10, 0])
    offset_l = -1.5 * offset_r
    offset_d = -1.5 * offset_u
    offset_ur = (offset_r + offset_u) / np.sqrt(2)
    offset_ul = (offset_l + offset_u) / np.sqrt(2)
    offset_dr = (offset_r + offset_d) / np.sqrt(2)
    offset_dl = (offset_l + offset_d) / np.sqrt(2)

    offset_dict: dict[str, Any] = {
        "u": offset_u,
        "r": offset_r,
        "l": offset_l,
        "d": offset_d,
        "ur": offset_ur,
        "ul": offset_ul,
        "dr": offset_dr,
        "dl": offset_dl,
    }

    def __init__(
        self,
        width: float,
        height: float | None = None,
        nrows: int = 1,
        ncols: int = 1,
    ) -> None:
        self._ncols: int = ncols
        self._nrows: int = nrows
        self.width: float = width
        self.height: float | None = height

        self.annotate_letter: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        self._horizontal_margins: list[float] = [0.1, 0.1]
        self._vertical_margins: list[float] = [0.1, 0.1]
        self._wspace: float = 0.1
        self._hspace: float = 0.1

        self._width_ratios: list[float] | None = None
        self._height_ratios: list[float] | None = None

        self._fig = None
        self._gs = None

        self.annotation_dict: dict[str, dict[str, Any]] = {}

    @staticmethod
    def label_subplot(
        ax: Axes,
        text: str,
        pad_x: float = 0.0,
        pad_y: float = 0.025,
        **kwargs: dict[str, Any],
    ):
        """Labels a subplot at the left upper corner. Wrapper around ax.text"""
        if pad_x >= 0.0:
            ha = "left"
        else:
            ha = "right"

        if pad_y >= 0.0:
            va = "bottom"
        else:
            va = "top"

        ax.text(  # pyright: ignore[reportUnknownMemberType]
            x=pad_x,
            y=1.0 + pad_y,
            s=text,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            **kwargs,
        )

    @staticmethod
    def annotate(
        ax: Axes,
        text: str,
        pos: Sequence[float] = (0.0, 0.98),
        **kwargs: dict[str, Any],
    ):
        """Annotate an ax with some text. Wrapper around ax.text.

        Args:
            ax (plt.ax): the ax
            text (str): the text

        """
        ax.text(  # pyright: ignore[reportUnknownMemberType]
            pos[0],
            pos[1],
            text,
            transform=ax.transAxes,
            **kwargs,
        )

    def add_box_around_image(
        self, ax: Axes, axes_image: AxesImage, **kwargs: dict[str, Any]
    ):
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
        self,
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

        assert self._fig is not None
        return self._fig.add_axes(  # pyright: ignore[reportUnknownMemberType]
            rect=(x0, y0, x1, y1)
        )

    @staticmethod
    def image_to_ax(
        ax: Axes, image: str | Path | npt.NDArray[np.floating]
    ) -> AxesImage:
        import os

        if isinstance(image, str | Path):
            if os.path.exists(image):
                image = PyperPlot.open_image(image)
            else:
                raise Exception(f"`{image}` does not exist")

        ax.tick_params(  # type: ignore
            axis="both", which="both", bottom=0, left=0, labelbottom=0, labelleft=0
        )
        ax.set_facecolor((0, 0, 0, 0))

        for s in ax.spines.values():
            s.set_visible(False)

        return ax.imshow(image)  # pyright: ignore[reportUnknownMemberType]

    @staticmethod
    def clear_spines(ax: Axes) -> Axes:
        ax.set_facecolor((0, 0, 0, 0))
        ax.tick_params(  # pyright: ignore[reportUnknownMemberType]
            axis="both", which="both", bottom=0, left=0, labelbottom=0, labelleft=0
        )
        for k in ["left", "right", "top", "bottom"]:
            s = ax.spines[k]
            s.set_visible(False)
        return ax

    def xy_text_auto(
        self, ax: Axes, xy: npt.ArrayLike, deriv: float, scale: float = 15
    ):
        trans_deriv = ax.transData.transform(
            [[0.0, 0.0], [1.0, deriv]]
        )  # transform from data coordinates to display coordinates
        display_deriv = (trans_deriv[1, 1] - trans_deriv[0, 1]) / (
            trans_deriv[1, 0] - trans_deriv[0, 0]
        )

        direction = np.array([display_deriv, -1])
        direction = direction / np.linalg.norm(direction)

        # Where do we hit the upper ylim in display coords
        lim_lower = ax.transData.transform([ax.get_xlim()[0], ax.get_ylim()[0]])
        lim_upper = ax.transData.transform([ax.get_xlim()[1], ax.get_ylim()[1]])
        xy = ax.transData.transform(xy)

        intersect_upper = (lim_upper - xy) / direction
        dist_upper = np.min(np.abs(intersect_upper))
        sign_upper = np.sign(intersect_upper[0])

        intersect_lower = (lim_lower - xy) / direction
        dist_lower = np.min(np.abs(intersect_lower))
        sign_lower = np.sign(intersect_lower[0])

        if dist_upper > dist_lower:
            return scale * sign_upper * direction
        return scale * sign_lower * direction

    def annotate_graph(
        self,
        ax: Axes,
        xy: tuple[float, float],
        xy_text: tuple[float, float],
        text: str | None = None,
        key: str | None = "key1",
        offset_scale: float = 1.0,
    ):
        if key is not None:
            if key not in self.annotation_dict:
                self.annotation_dict[key] = {
                    "annotate_increment": 0,
                    "annotation_list": [],
                }
        elif text is None:
            raise Exception("Need to specify text if `key` is None")

        arrowprops = dict(arrowstyle="-")

        if text is None:
            text = self.annotate_letter[
                self.annotation_dict[key]["annotate_increment"]
            ]
            self.annotation_dict[key]["annotate_increment"] += 1

        if isinstance(xy_text, str):
            xy_text = PyperPlot.offset_dict[xy_text.lower()]

        if key is not None:
            self.annotation_dict[key]["annotation_list"].append([xy, text])

        ax.annotate(  # pyright: ignore[reportUnknownMemberType]
            text,
            xy,
            (xy_text[0] * offset_scale, xy_text[1] * offset_scale),
            arrowprops=arrowprops,
            verticalalignment="center",
            horizontalalignment="center",
            textcoords="offset points",
        )
