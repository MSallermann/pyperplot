"""A Matplotlib ``GridSpec`` with physically sized layout parameters."""

from __future__ import annotations
from matplotlib.gridspec import GridSpec
from collections.abc import Sequence, Iterable
from dataclasses import dataclass
from matplotlib.figure import Figure
import numpy as np

from typing import Callable, Literal, TypeVar

T = TypeVar("T")
LengthUnit = Literal["in", "cm"]


def greater_eq_than_zero(w: float) -> bool:
    """Return whether *w* represents an absolute, non-negative size."""
    return w >= 0


def smaller_than_zero(w: float) -> bool:
    """Return whether *w* represents a negative relative weight."""
    return w < 0


def count_predicate(seq: Iterable[T], pred: Callable[[T], bool]) -> int:
    """Count the values in *seq* for which *pred* is true."""
    count: int = 0
    for t in seq:
        if pred(t):
            count += 1
    return count


def filter_predicate(seq: Iterable[T], pred: Callable[[T], bool]) -> list[T]:
    """Return the values in *seq* for which *pred* is true."""
    res: list[T] = []
    for t in seq:
        if pred(t):
            res.append(t)
    return res


@dataclass
class GridSpecParams:
    """Normalized values needed to initialize a Matplotlib ``GridSpec``."""

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
    """Lay out a Matplotlib grid using inches or centimetres.

    All ``abs_*`` arguments use ``unit``. Matplotlib figures are measured in
    inches internally, so centimetre inputs are converted before the grid is
    created.

    If ``abs_content_width`` or ``abs_content_height`` is provided, the
    corresponding figure dimension is resized to contain the requested
    content, margins, and inter-cell spacing. If omitted, the content extent
    is derived from the figure's current size.

    Entries in ``abs_widths`` and ``abs_heights`` have mixed semantics:
    non-negative values are physical sizes, while negative values are relative
    weights used to divide the remaining space. For example, ``[4, -2, -1]``
    creates one 4-unit column and splits the remaining width 2:1.

    Args:
        nrows: Number of grid rows.
        ncols: Number of grid columns.
        figure: Figure that owns the grid. Its size may be updated.
        abs_margin_left: Left figure margin.
        abs_margin_bottom: Bottom figure margin.
        abs_margin_right: Right figure margin.
        abs_margin_top: Top figure margin.
        abs_wspace: Horizontal space between adjacent columns.
        abs_hspace: Vertical space between adjacent rows.
        abs_widths: Optional size or relative weight for each column. Its
            length must equal ``ncols``.
        abs_heights: Optional size or relative weight for each row. Its length
            must equal ``nrows``.
        abs_content_width: Optional total width occupied by the columns,
            excluding margins and spacing.
        abs_content_height: Optional total height occupied by the rows,
            excluding margins and spacing.
        unit: Unit for every ``abs_*`` argument: ``"in"`` or ``"cm"``.

    Raises:
        ValueError: If dimensions are inconsistent or leave no content area.
    """

    ATOL = 1e-5

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
        unit: LengthUnit = "in",
    ) -> None:
        self.unit: LengthUnit = unit
        gridspec_params = AbsoluteGridSpec.compute_gridspec_params(
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
            unit=unit,
        )

        figure.set_size_inches(
            gridspec_params.fig_width,
            gridspec_params.fig_height,
        )

        super().__init__(
            nrows=nrows,
            ncols=ncols,
            figure=figure,
            left=gridspec_params.left,
            right=gridspec_params.right,
            bottom=gridspec_params.bottom,
            top=gridspec_params.top,
            wspace=gridspec_params.wspace,
            hspace=gridspec_params.hspace,
            width_ratios=gridspec_params.width_ratios,
            height_ratios=gridspec_params.height_ratios,
        )

    @staticmethod
    def compute_content_extent(
        total_extent: float,
        ncols_or_rows: int,
        abs_hwspace: float,
        abs_margin_hw: float,
    ) -> float:
        """Return content size after subtracting margins and cell spacing.

        All arguments must use the same unit.

        Raises:
            ValueError: If margins and spacing consume the entire extent.
        """
        content_extent = (
            total_extent - abs_hwspace * (ncols_or_rows - 1) - abs_margin_hw
        )

        if content_extent <= 0.0:
            msg = "Margins and space are too big. There is no space left for the content."
            raise ValueError(msg)

        return content_extent

    @staticmethod
    def compute_total_extent(
        content_extent: float,
        ncols_or_rows: int,
        abs_hwspace: float,
        abs_margin_hw: float,
    ) -> float:
        """Return total size after adding margins and cell spacing."""
        return content_extent + abs_hwspace * (ncols_or_rows - 1) + abs_margin_hw

    @staticmethod
    def compute_content_ratios(
        abs_content_width_or_height: float, abs_widths_or_heights: Sequence[float]
    ) -> list[float]:
        """Convert mixed absolute sizes and relative weights to ratios.

        Non-negative entries are fixed physical sizes. Negative entries are
        weights that share the space left after fixed sizes are subtracted.
        The returned ratios are suitable for Matplotlib's ``width_ratios`` or
        ``height_ratios``.

        Raises:
            ValueError: If fixed sizes exceed the content extent or, without
                relative entries, do not add up to it.
        """
        # Count how many of the widths are smaller than zero
        num_widths_smaller_than_zero = count_predicate(
            abs_widths_or_heights, smaller_than_zero
        )

        if num_widths_smaller_than_zero == 0:
            # If there are no relative widths to distribute, we have no slack.
            # This means we check that the sum of absolute widths matches the content width

            if not np.isclose(
                np.sum(abs_widths_or_heights),
                abs_content_width_or_height,
                atol=AbsoluteGridSpec.ATOL,
            ):
                msg = f"You have specified absolute heights(widths) for the rows(columns) and an absolute height(width) of the content. The problem is that the sum of them is {np.sum(abs_widths_or_heights):.2f}, while the specified height(width) of the content is {abs_content_width_or_height:.2f}. These should be the same! A solution would be to adjust your specification or to make at least one of the row(column) heights(widths) relative by specifying a negative number."

                raise ValueError(msg)

        content_ratios = [
            w / abs_content_width_or_height for w in abs_widths_or_heights
        ]

        # Compute the remaining width, to be distributed according to the relative weights
        remaining_width_or_height = abs_content_width_or_height - sum(
            filter_predicate(abs_widths_or_heights, greater_eq_than_zero)
        )

        if remaining_width_or_height < -AbsoluteGridSpec.ATOL:
            msg = f"Absolute widths/heights are larger than total height/width. Remaining height/width = {remaining_width_or_height}."
            raise ValueError(msg)

        remaining_width_or_height = max(remaining_width_or_height, 0.0)

        # Then, we compute the total weight of the negative widths
        total_weight_of_relative_widths = sum(
            filter_predicate(abs_widths_or_heights, smaller_than_zero)
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
    ) -> None:
        """Reject simultaneous total-height and content-height specifications."""
        # Check if overspecified
        height_specifiers: list[str] = []
        if abs_content_height is not None:
            height_specifiers.append("abs_content_height")
        if height is not None:
            height_specifiers.append("height")

        if len(height_specifiers) > 1:
            msg = f"The height of the plot is overspecified, because you have set {height_specifiers})"
            raise ValueError(msg)

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
        unit: LengthUnit = "in",
    ) -> GridSpecParams:
        """Compute GridSpec parameters from dimensions in the selected unit.

        Args:
            figure: Figure providing dimensions when content extents are
                omitted.
            nrows: Number of rows.
            ncols: Number of columns.
            abs_hspace: Space between adjacent rows.
            abs_wspace: Space between adjacent columns.
            abs_vertical_margins: Bottom and top margins.
            abs_horizontal_margins: Left and right margins.
            abs_heights: Optional sizes or relative weights for the rows.
            abs_widths: Optional sizes or relative weights for the columns.
            abs_content_width: Optional total width occupied by columns.
            abs_content_height: Optional total height occupied by rows.
            unit: Unit used by every ``abs_*`` argument.

        Returns:
            Figure dimensions in inches and normalized GridSpec parameters.

        Raises:
            ValueError: If the unit, dimensions, or sequence lengths are
                invalid.
        """
        if unit == "in":
            inches_per_unit = 1.0
        elif unit == "cm":
            inches_per_unit = AbsoluteGridSpec.cm
        else:
            raise ValueError(f"unknown length unit: {unit!r}")

        def convert_length(value: float) -> float:
            return value * inches_per_unit

        def convert_sizes(values: Sequence[float] | None) -> list[float] | None:
            if values is None:
                return None
            return [
                convert_length(value) if value >= 0.0 else value for value in values
            ]

        abs_hspace = convert_length(abs_hspace)
        abs_wspace = convert_length(abs_wspace)
        abs_vertical_margins = [
            convert_length(value) for value in abs_vertical_margins
        ]
        abs_horizontal_margins = [
            convert_length(value) for value in abs_horizontal_margins
        ]
        abs_heights = convert_sizes(abs_heights)
        abs_widths = convert_sizes(abs_widths)
        if abs_content_width is not None:
            abs_content_width = convert_length(abs_content_width)
        if abs_content_height is not None:
            abs_content_height = convert_length(abs_content_height)

        fig_width, fig_height = figure.get_size_inches()

        # Compute the absolute space taken up by the margins
        if len(abs_horizontal_margins) != 2:
            raise ValueError("abs_horizontal_margins must contain left and right")
        abs_margin_w = float(sum(abs_horizontal_margins))

        if len(abs_vertical_margins) != 2:
            raise ValueError("abs_vertical_margins must contain bottom and top")
        abs_margin_h = float(sum(abs_vertical_margins))

        if abs_widths is not None and len(abs_widths) != ncols:
            raise ValueError("abs_widths length must match ncols")
        if abs_heights is not None and len(abs_heights) != nrows:
            raise ValueError("abs_heights length must match nrows")

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

        # Compute the relative quantities that GridSpec needs
        hspace = abs_hspace / abs_content_height * nrows
        wspace = abs_wspace / abs_content_width * ncols
        fig_height = abs_content_height + abs_margin_h + abs_hspace * (nrows - 1)

        left = abs_horizontal_margins[0] / fig_width
        right = 1.0 - abs_horizontal_margins[1] / fig_width

        bottom = abs_vertical_margins[0] / fig_height
        top = 1.0 - abs_vertical_margins[1] / fig_height

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
