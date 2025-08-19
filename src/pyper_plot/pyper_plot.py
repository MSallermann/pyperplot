from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from typing import Any, TypeVar, Callable

T = TypeVar("T")


def greater_than_zero(w: float) -> bool:
    return w > 0


def smaller_eq_than_zero(w: float) -> bool:
    return w <= 0


def count_predicate(seq: Sequence[T], pred: Callable[[T], bool]) -> int:
    count: int = 0
    for t in seq:
        if pred(t):
            count += 1
    return count


def filter_predicate(seq: Sequence[T], pred: Callable[[T], bool]) -> list[T]:
    res: list[T] = []
    for t in seq:
        if pred(t):
            res.append(t)
    return res


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

    offset_dict = {
        "u": offset_u,
        "r": offset_r,
        "l": offset_l,
        "d": offset_d,
        "ur": offset_ur,
        "ul": offset_ul,
        "dr": offset_dr,
        "dl": offset_dl,
    }

    default_rcParams = {
        "font.size": 8,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.labelsize": 8,
    }

    def __init__(
        self,
        width: float,
        height: float | None = None,
        nrows: int = 1,
        ncols: int = 1,
        rcParams: dict[str, Any] | None = None,
    ) -> None:
        self._ncow: int = ncols
        self._nrow: int = nrows
        self.width: float = width
        self.height: float | None = height

        if rcParams is None:
            rcParams = self.default_rcParams

        mpl.rcParams.update(rcParams)

        self.annotate_letter: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        self._horizontal_margins: list[float] = [0.1, 0.1]
        self._vertical_margins: list[float] = [0.1, 0.1]
        self._wspace: float = 0.1
        self._hspace: float = 0.1

        self._width_ratios: list[float] | None = None
        self._height_ratios: list[float] | None = None

        self._fig = None
        self._gs = None

        self.annotation_dict = {}

    def info_string(self) -> str:
        """Return a string with information about the plot."""

        res = "Paper_Plot\n"
        res += f"\t width  = {self.width:.3f} inch ({self.width / self.cm:.3f} cm)\n"
        if self.height is not None:
            res += (
                f"\t height = {self.height:.3f} inch ({self.height / self.cm:.3f} cm)\n"
            )
        res += f"\t ncols  = {self.ncols}\n"
        res += f"\t nrows  = {self.nrows}\n"
        res += f"\t wspace = {self.wspace:.3f}\n"
        res += f"\t hspace = {self.hspace:.3f}\n"
        res += f"\t horizontal_margins =  {self.horizontal_margins[0]:.3f}, {self.horizontal_margins[1]:.3f}\n"
        res += f"\t vertical_margins = {self.vertical_margins[0]:.3f}, {self.vertical_margins[1]:.3f}\n"
        res += f"\t width_ratios = {self.width_ratios}\n"
        res += f"\t height_ratios = {self.height_ratios}\n"
        return res

    @property
    def ncols(self) -> int:
        """Number of columns."""
        return self._ncols

    @ncols.setter
    def ncols(self, value: int):
        if value != self._ncols and self._width_ratios:
            print("WARNING: changing ncols resets width_ratios")
            self._width_ratios = None
        self._ncols = value

    @property
    def nrows(self) -> int:
        """Number of rows."""
        return self._nrows

    @nrows.setter
    def nrows(self, value: int):
        if value != self._nrows and self._height_ratios:
            print("WARNING: changing ncols resets height_ratios")
            self._height_ratios = None
        self._nrows = value

    @property
    def font_size(self) -> float:
        return mpl.rcParams["font.size"]

    @font_size.setter
    def font_size(self, value: float):
        mpl.rcParams["font.size"] = value

    @property
    def axes_labelsize(self) -> float:
        return mpl.rcParams["axes.labelsize"]

    @axes_labelsize.setter
    def axes_labelsize(self, value: float):
        mpl.rcParams["axes.labelsize"] = value

    @property
    def xtick_labelsize(self) -> float:
        return mpl.rcParams["xtick.labelsize"]

    @xtick_labelsize.setter
    def xtick_labelsize(self, value: float):
        mpl.rcParams["xtick.labelsize"] = value

    @property
    def ytick_labelsize(self) -> float:
        return mpl.rcParams["ytick.labelsize"]

    @ytick_labelsize.setter
    def ytick_labelsize(self, value: float):
        mpl.rcParams["ytick.labelsize"] = value

    @property
    def width_ratios(self):
        return self._width_ratios

    @width_ratios.setter
    def width_ratios(self, value: Sequence[float] | None):
        """Relative column width ratios."""
        if value is None:
            self._width_ratios = None
            return

        if len(value) == self.ncols:
            self._width_ratios = list(value)
        else:
            raise Exception(f"Length of width_ratios has to match ncols {self.ncols}")

    @property
    def height_ratios(self) -> list[float] | None:
        """Relative row height ratios."""
        return self._height_ratios

    @height_ratios.setter
    def height_ratios(self, value: Sequence[float] | None):
        if value is None:
            self._height_ratios = None
            return

        if len(value) == self.nrows:
            self._height_ratios = list(value)
        else:
            msg = f"Length of height_ratios has to match nrows {self.nrows}"
            raise Exception(msg)

    @property
    def wspace(self) -> float:
        """Space between columns as a fraction of the average column width."""
        return self._wspace

    @wspace.setter
    def wspace(self, value: float):
        self._wspace = value

    @property
    def hspace(self) -> float:
        """Space between rows as a fraction of the average row height."""
        return self._hspace

    @hspace.setter
    def hspace(self, value: float):
        self._hspace = value

    @property
    def horizontal_margins(self):
        """Horizontal margins (left and right) as fraction of figure width"""
        return self._horizontal_margins.copy()

    @horizontal_margins.setter
    def horizontal_margins(self, value: Sequence[float]):
        if len(value) == 2:
            self._horizontal_margins = list(value)
        else:
            msg = f"horizontal_margins has to have shape (2,) but you specified {value}"
            raise Exception(msg)

    @property
    def vertical_margins(self) -> list[float]:
        """Horizontal margins (bottom and top) as fraction of figure height"""
        return self._vertical_margins.copy()

    @vertical_margins.setter
    def vertical_margins(self, value: Sequence[float]):
        if len(value) == 2:
            self._vertical_margins = list(value)
        else:
            msg = f"vertical_margins has to have shape (2,) but you specified {value}"
            raise Exception(msg)

    def height_from_aspect_ratio(self, aspect_ratio: float):
        """The width of the figure (including all margins) is fixed. The aspect ratio is the aspect ratio of all content, excluding hspace, wspace and margins"""
        # Deal with width
        rel_margin_w = sum(self.horizontal_margins)
        rel_width_minus_margins = 1.0 - rel_margin_w

        rel_average_subplot_width = rel_width_minus_margins / (
            self.ncols + self.wspace * (self.ncols - 1)
        )

        rel_total_wspace_between_subplots = (
            rel_average_subplot_width * self.wspace * (self.ncols - 1)
        )

        width_prefactor = rel_width_minus_margins - rel_total_wspace_between_subplots

        rel_margin_h = sum(self.vertical_margins)
        rel_height_minus_margins = 1 - rel_margin_h

        rel_average_subplot_height = rel_height_minus_margins / (
            self.nrows + self.hspace * (self.nrows - 1)
        )

        rel_total_hspace_between_subplots = (
            rel_average_subplot_height * self.hspace * (self.nrows - 1)
        )
        height_prefactor = rel_height_minus_margins - rel_total_hspace_between_subplots

        self.height = self.width / aspect_ratio * width_prefactor / height_prefactor

    @staticmethod
    def compute_content_extent(
        total_extent: float,
        ncols_or_rows: int,
        abs_hwspace: float,
        abs_margin_hw: float,
    ) -> float:
        content_extent = (
            total_extent - abs_hwspace * (ncols_or_rows - 1) - abs_margin_hw
        )

        if content_extent <= 0.0:
            msg = (
                "Margins and space are too big. There is no space left for the content."
            )
            raise Exception(msg)

        return content_extent

    @staticmethod
    def compute_total_extent(
        content_extent: float,
        ncols_or_rows: int,
        abs_hwspace: float,
        abs_margin_hw: float,
    ) -> float:
        return content_extent + abs_hwspace * (ncols_or_rows - 1) + abs_margin_hw

    @staticmethod
    def compute_content_ratios(
        abs_content_width_or_height: float, abs_widths_or_heights: Sequence[float]
    ) -> list[float]:

        # Count how many of the widths are greater than zero
        num_widths_greater_than_zero = count_predicate(
            abs_widths_or_heights, greater_than_zero
        )

        if num_widths_greater_than_zero == 0:
            # If there are no relative widths left to distribute, we have no slack.
            # This means we check that the sum of absolute widths matches the content width
            if not np.isclose(
                np.sum(abs_widths_or_heights), abs_content_width_or_height
            ):
                raise Exception(
                    "The absolute widths do not match the expected width of the plot content. You should make at least one of them relative by specifying a negative number."
                )

        content_ratios = [
            w / abs_content_width_or_height for w in abs_widths_or_heights
        ]

        # Compute the remaining width, to be distributed according to the relative weights
        remaining_width_or_height = abs_content_width_or_height - sum(
            filter_predicate(abs_widths_or_heights, greater_than_zero)
        )

        if remaining_width_or_height < 0.0:
            raise Exception("Absolute widths/heights are larger than total width")

        # Then we compute the total weight of the negative widths
        total_weight_of_relative_widths = sum(
            filter_predicate(abs_widths_or_heights, smaller_eq_than_zero)
        )

        # Iterate over all the widths
        for idx_w, w in enumerate(abs_widths_or_heights):
            # If it is a relative width, we distribute the remaining width according to its relative weight
            if w < 0.0:
                content_ratios[idx_w] = (
                    w
                    / total_weight_of_relative_widths
                    * remaining_width_or_height
                    / abs_content_width_or_height
                )

        return content_ratios

    @staticmethod
    def check_height_specifiers(
        abs_content_height: float | None,
        height: float | None,
    ):
        # Check if overspecified
        height_specifiers: list[str] = []
        if abs_content_height is not None:
            height_specifiers.append("abs_content_height")
        if height is not None:
            height_specifiers.append("height")

        if len(height_specifiers) > 1:
            msg = f"The height of the plot is overspecified, because you have set {height_specifiers})"
            raise Exception(msg)

    def apply_absolute_margins(
        self,
        abs_hspace: float = 0.5 * cm,
        abs_wspace: float = 0.5 * cm,
        abs_vertical_margins: Sequence[float] = [0.15 * cm, 0.15 * cm],
        abs_horizontal_margins: Sequence[float] = [0.15 * cm, 0.15 * cm],
        abs_heights: Sequence[float] | None = None,
        abs_widths: Sequence[float] | None = None,
        abs_content_width: float | None = None,
        abs_content_height: float | None = None,
    ):
        """Set the layout of the figure in *absolute* units. All lengths are given in inches. Call this *before* fig() and gs().

        Args:
            abs_hspace (float): The space between each row of the grid. Defaults to 0.5*cm.
            abs_wspace (float): The space between each column of the grid. Defaults to 0.5*cm.
            abs_vertical_margins (list): Bottom and top margins. Defaults to [0.15 * cm, 0.15 * cm].
            abs_horizontal_margins (list): Left and right margins. Defaults to [0.15 * cm, 0.15 * cm].
            abs_heights (list, optional): Heights of the rows. Defaults to None. Negative values will be used as relative weights.
            abs_widths (list, optional): Widths of the columns. Defaults to None. Negative values will be used as relative weights.
            abs_content_width (float, optional): Absolute width of the content. !WARNING will recompute the total width of the figure, based on wspace and margins. It is usually better to keep the width fixed. Defaults to None.
            abs_content_height (float, optional): Absolute height of the content. This will only be used if aspect_ratio is None. !WARNING will recompute the total height of the figure.

        """
        assert len(abs_horizontal_margins) == 2  # noqa: PLR2004
        assert len(abs_vertical_margins) == 2

        # Compute the absolute space, taken up by the margins
        abs_margin_w = float(sum(abs_horizontal_margins))
        abs_margin_h = float(sum(abs_vertical_margins))

        # If the content width has not been specified, we compute it based on the current figure width
        # If it has been specified we compute the total figure width based on it
        if abs_content_width is None:
            abs_content_width = PyperPlot.compute_content_extent(
                total_extent=self.width,
                ncols_or_rows=self.ncols,
                abs_hwspace=abs_wspace,
                abs_margin_hw=abs_margin_w,
            )
        else:
            self.width = PyperPlot.compute_total_extent(
                content_extent=abs_content_width,
                ncols_or_rows=self.ncols,
                abs_hwspace=abs_wspace,
                abs_margin_hw=abs_margin_w,
            )

        # Compute the width ratios from absolute heights
        if abs_widths is not None:
            self.width_ratios = PyperPlot.compute_content_ratios(
                abs_content_width_or_height=abs_content_width,
                abs_widths_or_heights=abs_widths,
            )

        if abs_content_height is None and self.height is None:
            raise Exception(
                "Either `abs_content_height` or `self.height` need to be not `None`"
            )

        if abs_content_height is None and self.height is not None:
            abs_content_height = PyperPlot.compute_content_extent(
                total_extent=self.height,
                ncols_or_rows=self.nrows,
                abs_hwspace=abs_hspace,
                abs_margin_hw=abs_margin_h,
            )
        elif abs_content_height is not None:
            self.height = PyperPlot.compute_total_extent(
                content_extent=abs_content_height,
                ncols_or_rows=self.nrows,
                abs_hwspace=abs_hspace,
                abs_margin_hw=abs_margin_h,
            )

        if abs_heights is not None and abs_content_height is not None:
            self.height_ratios = PyperPlot.compute_content_ratios(
                abs_content_width_or_height=abs_content_height,
                abs_widths_or_heights=abs_heights,
            )
        else:  # We should never get here
            raise Exception(
                "Something went wrong! `abs_heights` and `abs_content_height` are both None. We should never get here..."
            )

        # Compute the relative quantities that gridpsec needs
        self.hspace = abs_hspace / abs_content_height * self.nrows
        self.wspace = abs_wspace / abs_content_width * self.ncols
        self.height = abs_content_height + abs_margin_h + abs_hspace * (self.nrows - 1)
        self.horizontal_margins = [m / self.width for m in abs_horizontal_margins]
        self.vertical_margins = [m / self.height for m in abs_vertical_margins]

    def reset(self):
        """Resets the internal fig and gs to None"""
        self._fig = None
        self._gs = None

    def fig(self):
        """Get the underlying figure object"""
        if self._fig is None:
            self._fig = plt.figure(figsize=(self.width, self.height))
        return self._fig

    def gs(self):
        """Get the underlying GridSpec object"""
        if self._gs is None:
            self._gs = GridSpec(
                figure=self._fig,
                nrows=self.nrows,
                ncols=self.ncols,
                left=self.horizontal_margins[0],
                bottom=self.vertical_margins[0],
                right=1.0 - self.horizontal_margins[1],
                top=1.0 - self.vertical_margins[1],
                hspace=self.hspace,
                wspace=self.wspace,
                width_ratios=self.width_ratios,
                height_ratios=self.height_ratios,
            )
        return self._gs

    @staticmethod
    def label_subplot(ax, text, pad_x=0.0, pad_y=0.025, **kwargs):
        """Labels a subplot at the left upper corner. Wrapper around ax.text"""
        if pad_x >= 0.0:
            ha = "left"
        else:
            ha = "right"

        if pad_y >= 0.0:
            va = "bottom"
        else:
            va = "top"

        ax.text(
            x=pad_x,
            y=1.0 + pad_y,
            s=text,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            **kwargs,
        )

    @staticmethod
    def annotate(ax, text, pos=[0, 0.98], **kwargs):
        """Annotate an ax with some text. Wrapper around ax.text.

        Args:
            ax (plt.ax): the ax
            text (str): the text

        """
        ax.text(
            *pos,
            text,
            transform=ax.transAxes,
            **kwargs,
        )

    def add_box_around_image(self, ax, axes_image, **kwargs):
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

    @staticmethod
    def open_image(path):
        return np.array(Image.open(path))

    @staticmethod
    def replace_background_color(image, replacement_color, background_color=None):
        """Replaced the backgroudn color of an image, specified as a numpy array.

        Args:
            image (np.Array): The image array
            replacement_color (the color): The color wich replaces the background color.
            background_color (the background color, optional): The backgroudn color. If None it is inferred from the corners. Defaults to None.

        Returns:
            np.Array: the new image array

        """
        N_CHANNELS = image.shape[-1]  # Number of channels in the picture
        image_shape = image.shape

        # If no background color is specified, we take the most frequent color among the corners
        if background_color is None:
            corner_colors = [image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]]
            # As soon as we find the first repeated color we can stop, since a count of two will always be at least 50%
            # TODO: these loops look stupid but they work
            for cc in corner_colors:
                for cc2 in corner_colors:
                    if np.all(cc == cc2):
                        background_color = cc
                        break
                if np.all(cc == cc2):
                    break
            background_color = corner_colors[0]

        N_CHANNELS_BG = len(replacement_color)

        # If the background color has more channels than the picture, we need to introduce the alpha channel
        if N_CHANNELS_BG > N_CHANNELS and replacement_color[-1] != 1.0:
            image_copy = np.ones(shape=(image_shape[0], image_shape[1], 4))
            image_copy[:, :, :3] = image
            image = image_copy
            background_color = [
                *background_color,
                1.0,
            ]  # Extend the background color with the alpha channel
        elif N_CHANNELS_BG < N_CHANNELS:
            replacement_color = [*replacement_color, 1.0]

        indices = np.argwhere(np.all(image[:, :] == background_color, axis=2))
        image[indices[:, 0], indices[:, 1], :] = replacement_color
        return image

    @staticmethod
    def crop_to_content(image, background_color=None, replace_background_color=None):
        """Crops an image array to its content."""
        N_CHANNELS = image.shape[-1]  # Number of channels in the picture
        image_shape = image.shape

        # If no background color is specified, we take the most frequent color among the corners
        if background_color is None:
            corner_colors = [image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]]

            # As soon as we find the first repeated color we can stop, since a count of two will always be at least 50%
            # TODO: these loops look stupid but they work
            for cc in corner_colors:
                for cc2 in corner_colors:
                    if np.all(cc == cc2):
                        background_color = cc
                        break
                if np.all(cc == cc2):
                    break

            background_color = corner_colors[0]

        # I think I know why this works
        indices = np.argwhere(np.any(image[:, :] != background_color, axis=2))

        lower_height = np.min(indices[:, 0])
        upper_height = np.max(indices[:, 0])
        lower_width = np.min(indices[:, 1])
        upper_width = np.max(indices[:, 1])

        if replace_background_color is not None:
            N_CHANNELS_BG = len(replace_background_color)

            # If the background color has more channels than the picture, we need to introduce the alpha channel
            if N_CHANNELS_BG > N_CHANNELS and replace_background_color[-1] != 1.0:
                image_copy = np.ones(shape=(image_shape[0], image_shape[1], 4))
                image_copy[:, :, :3] = image
                image = image_copy
                background_color = [
                    *background_color,
                    1.0,
                ]  # Extend the background color with the alpha channel
            elif N_CHANNELS_BG < N_CHANNELS:
                replace_background_color = [*replace_background_color, 1.0]

            indices = np.argwhere(np.all(image[:, :] == background_color, axis=2))
            image[indices[:, 0], indices[:, 1], :] = replace_background_color

        return image[lower_height : upper_height + 1, lower_width : upper_width + 1, :]

    @staticmethod
    def crop(image, left=0, right=0, top=0, bottom=0):
        """Crops an image by removing pixels from the left, right, top and bottom"""
        assert left >= 0 and right >= 0 and top >= 0 and bottom >= 0

        height_old = image.shape[0]
        height_new = height_old - top - bottom

        width_old = image.shape[1]
        width_new = width_old - left - right

        if height_new <= 0:
            raise Exception(
                "Cropping too much at top and/or bottom. No pixels remaining."
            )

        if width_new <= 0:
            raise Exception(
                "Cropping too much at left and/or right. No pixels remaining."
            )

        return image[top : height_old - bottom, left : width_old - right, :]

    def create_inset_axis(
        self,
        containing_ax,
        rel_width=0.5,
        rel_height=0.5,
        margin_x=0.0,
        margin_y=0.0,
        x_align="left",
        y_align="bottom",
    ):
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
        pos = containing_ax.get_position(self._fig)

        w = pos.x1 - pos.x0
        h = pos.y1 - pos.y0

        def helper(old_var0, old_var1, align, wh, rel_width_height, margin_xy):
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

        pos.x0, pos.x1 = helper(pos.x0, pos.x1, x_align, w, rel_width, margin_x)
        pos.y0, pos.y1 = helper(pos.y0, pos.y1, y_align, h, rel_height, margin_y)

        return self._fig.add_axes(pos)

    @staticmethod
    def image_to_ax(ax, image):
        import os

        if isinstance(image, str):
            if os.path.exists(image):
                image = Paper_Plot.open_image(image)
            else:
                raise Exception(f"`{image}` does not exist")

        ax.tick_params(
            axis="both", which="both", bottom=0, left=0, labelbottom=0, labelleft=0
        )
        ax.set_facecolor([0, 0, 0, 0])
        for k, s in ax.spines.items():
            s.set_visible(False)

        return ax.imshow(image)

    @staticmethod
    def clear_spines(ax):
        ax.set_facecolor([0, 0, 0, 0])
        ax.tick_params(
            axis="both", which="both", bottom=0, left=0, labelbottom=0, labelleft=0
        )
        for k in ["left", "right", "top", "bottom"]:
            s = ax.spines[k]
            s.set_visible(False)
        return ax

    def spine_axis(
        self,
        spec,
        color="black",
        which=["left", "right", "top", "bottom"],
        zorder=2,
        label="spine",
    ):
        try:
            a = self._fig.add_axes(
                spec.get_position(self._fig), zorder=zorder, label=label
            )
        except:
            a = self._fig.add_axes(spec, zorder=zorder, label=label)

        a.set_facecolor([0, 0, 0, 0])
        a.tick_params(
            axis="both", which="both", bottom=0, left=0, labelbottom=0, labelleft=0
        )
        for k in ["left", "right", "top", "bottom"]:
            s = a.spines[k]
            if k in which:
                s.set_visible(True)
                s.set_color(color)
            else:
                s.set_visible(False)
        return a

    def row(self, row_idx, sl=slice(None, None, None), gs=None):
        if gs is None:
            gs = self._gs

        col_indices = range(gs.ncols)[sl]
        return [self._fig.add_subplot(gs[row_idx, col_idx]) for col_idx in col_indices]

    def col(self, col_idx, sl=slice(None, None, None), gs=None):
        if gs is None:
            gs = self._gs

        row_indices = list(range(gs.nrows)[sl])
        return [self._fig.add_subplot(gs[row_idx, col_idx]) for row_idx in row_indices]

    def xy_text_auto(self, ax, xy, deriv, scale=15):
        trans_deriv = ax.transData.transform(
            [[0, 0], [1, deriv]]
        )  # transform from data coordinates to display coordinates
        display_deriv = (trans_deriv[1, 1] - trans_deriv[0, 1]) / (
            trans_deriv[1, 0] - trans_deriv[0, 0]
        )

        # display_deriv = trans_deriv[1]/trans_deriv[0] # transform to display derivative
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

    def annotate_graph(self, ax, xy, xy_text, text=None, key="key1", offset_scale=1):
        if key is not None:
            if key not in self.annotation_dict:
                self.annotation_dict[key] = {
                    "annotate_increment": 0,
                    "annotation_list": [],
                }
        elif text is None:
            raise Exception("Need to specify text if key is None")

        arrowprops = dict(arrowstyle="-")

        if text is None:
            text = self.annotate_letter[self.annotation_dict[key]["annotate_increment"]]
            self.annotation_dict[key]["annotate_increment"] += 1

        if type(xy_text) is str:
            xy_text = Paper_Plot.offset_dict[xy_text.lower()]

        if key is not None:
            self.annotation_dict[key]["annotation_list"].append([xy, text])

        ax.annotate(
            text,
            xy,
            xy_text * offset_scale,
            arrowprops=arrowprops,
            verticalalignment="center",
            horizontalalignment="center",
            textcoords="offset points",
        )
