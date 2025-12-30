from __future__ import annotations
from matplotlib.gridspec import GridSpec
from collections.abc import Sequence, Iterable
from dataclasses import dataclass
from matplotlib.figure import Figure
import numpy as np

from typing import TypeVar, Callable

T = TypeVar("T")


def greater_than_zero(w: float) -> bool:
    return w > 0


def smaller_eq_than_zero(w: float) -> bool:
    return w <= 0


def count_predicate(seq: Iterable[T], pred: Callable[[T], bool]) -> int:
    count: int = 0
    for t in seq:
        if pred(t):
            count += 1
    return count


def filter_predicate(seq: Iterable[T], pred: Callable[[T], bool]) -> list[T]:
    res: list[T] = []
    for t in seq:
        if pred(t):
            res.append(t)
    return res


@dataclass
class GridSpecParams:
    fig_height: float
    fig_width: float
    height_ratios: list[float] | None
    width_ratios: list[float] | None
    left: float | None
    bottom: float | None
    right: float | None
    top: float | None
    wspace: float | None
    hspace: float | None


class AbsoluteGridSpec(GridSpec):
    cm = 1.0 / 2.54

    def __init__(
        self,
        nrows: int,
        ncols: int,
        figure: Figure,
        abs_margin_left: float,
        abs_margin_bottom: float,
        abs_margin_right: float,
        abs_margin_top: float,
        abs_wspace: float = 0.1,
        abs_hspace: float = 0.1,
        abs_widths: Sequence[float] | None = None,
        abs_heights: Sequence[float] | None = None,
        abs_content_width: float | None = None,
        abs_content_height: float | None = None,
    ) -> None:
        gridpsec_params = AbsoluteGridSpec.compute_gridspec_params(
            nrows=nrows,
            ncols=ncols,
            figure=figure,
            abs_hspace=abs_hspace,
            abs_wspace=abs_wspace,
            abs_vertical_margins=[abs_margin_bottom, abs_margin_top],
            abs_horizontal_margins=[abs_margin_left, abs_margin_right],
            abs_heights=abs_heights,
            abs_widths=abs_widths,
            abs_content_width=abs_content_width,
            abs_content_height=abs_content_height,
        )

        figure.set_size_inches(
            gridpsec_params.fig_width,
            gridpsec_params.fig_height,
        )

        super().__init__(
            nrows=nrows,
            ncols=ncols,
            figure=figure,
            left=gridpsec_params.left,
            right=gridpsec_params.right,
            bottom=gridpsec_params.bottom,
            top=gridpsec_params.top,
            wspace=gridpsec_params.wspace,
            hspace=gridpsec_params.hspace,
            width_ratios=gridpsec_params.width_ratios,
            height_ratios=gridpsec_params.height_ratios,
        )

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
            msg = "Margins and space are too big. There is no space left for the content."
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

        if num_widths_greater_than_zero != 0:
            # If there are no relative widths to distribute, we have no slack.
            # This means we check that the sum of absolute widths matches the content width
            if not np.isclose(
                np.sum(abs_widths_or_heights), abs_content_width_or_height
            ):
                msg = "The absolute widths (or heights) do not match the expected width (or height) of the plot content. You should make at least one of them relative by specifying a negative number."
                raise Exception(msg)

        content_ratios = [
            w / abs_content_width_or_height for w in abs_widths_or_heights
        ]

        # Compute the remaining width, to be distributed according to the relative weights
        remaining_width_or_height = abs_content_width_or_height - sum(
            filter_predicate(abs_widths_or_heights, greater_than_zero)
        )

        if remaining_width_or_height < 0.0:
            msg = "Absolute widths/heights are larger than total width"
            raise Exception(msg)

        # Then, we compute the total weight of the negative widths
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

    @staticmethod
    def compute_gridspec_params(
        figure: Figure,
        nrows: int,
        ncols: int,
        abs_hspace: float,
        abs_wspace: float,
        abs_vertical_margins: Sequence[float],
        abs_horizontal_margins: Sequence[float],
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
        fig_width, fig_height = figure.get_size_inches()

        # Compute the absolute space taken up by the margins
        assert len(abs_horizontal_margins) == 2
        abs_margin_w = float(sum(abs_horizontal_margins))

        assert len(abs_vertical_margins) == 2
        abs_margin_h = float(sum(abs_vertical_margins))

        # If the content width has not been specified, we compute it based on the current figure width
        # If it has been specified we compute the total figure width based on it
        if abs_content_width is None:
            abs_content_width = AbsoluteGridSpec.compute_content_extent(
                total_extent=fig_width,
                ncols_or_rows=ncols,
                abs_hwspace=abs_wspace,
                abs_margin_hw=abs_margin_w,
            )
        else:
            fig_width = AbsoluteGridSpec.compute_total_extent(
                content_extent=abs_content_width,
                ncols_or_rows=ncols,
                abs_hwspace=abs_wspace,
                abs_margin_hw=abs_margin_w,
            )

        # Compute the width ratios from absolute widths
        if abs_widths is not None:
            width_ratios = AbsoluteGridSpec.compute_content_ratios(
                abs_content_width_or_height=abs_content_width,
                abs_widths_or_heights=abs_widths,
            )
        else:
            width_ratios = None

        # If the content height has not been specified, we compute it based on the current figure height
        # If it has been specified we compute the total figure height based on it
        if abs_content_height is None:
            abs_content_height = AbsoluteGridSpec.compute_content_extent(
                total_extent=fig_height,
                ncols_or_rows=nrows,
                abs_hwspace=abs_hspace,
                abs_margin_hw=abs_margin_h,
            )
        else:
            fig_height = AbsoluteGridSpec.compute_total_extent(
                content_extent=abs_content_height,
                ncols_or_rows=nrows,
                abs_hwspace=abs_hspace,
                abs_margin_hw=abs_margin_h,
            )

        if abs_heights is not None:
            height_ratios = AbsoluteGridSpec.compute_content_ratios(
                abs_content_width_or_height=abs_content_height,
                abs_widths_or_heights=abs_heights,
            )
        else:
            height_ratios = None

        # Compute the relative quantities that gridpsec needs
        hspace = abs_hspace / abs_content_height * nrows
        wspace = abs_wspace / abs_content_width * ncols
        fig_height = abs_content_height + abs_margin_h + abs_hspace * (nrows - 1)

        left = abs_horizontal_margins[0] / fig_width
        right = 1.0 - abs_horizontal_margins[1] / fig_width

        bottom = abs_vertical_margins[0] / fig_height
        top = 1.0 - abs_vertical_margins[0] / fig_height

        return GridSpecParams(
            fig_height=fig_height,
            fig_width=fig_width,
            height_ratios=height_ratios,
            width_ratios=width_ratios,
            left=left,
            bottom=bottom,
            right=right,
            top=top,
            wspace=wspace,
            hspace=hspace,
        )
