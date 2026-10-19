import matplotlib.pyplot as plt
import pytest
from matplotlib.gridspec import GridSpec

from pyper_plot.utilities import create_inset_axis, create_n_axes, gs_col


def test_create_n_axes_handles_zero_and_rejects_negative_counts():
    fig = plt.figure()
    gs = GridSpec(2, 2, figure=fig)

    assert create_n_axes(gs, 0) == []
    with pytest.raises(ValueError, match="non-negative"):
        create_n_axes(gs, -1)

    plt.close(fig)


def test_gs_col_creates_axes_down_the_requested_column():
    fig = plt.figure()
    gs = GridSpec(3, 2, figure=fig)

    axes = gs_col(1, gs)

    for expected_row, ax in enumerate(axes):
        spec = ax.get_subplotspec()
        assert spec is not None
        assert spec.rowspan.start == expected_row
        assert spec.colspan.start == 1
    plt.close(fig)


@pytest.mark.parametrize(
    ("x_align", "expected_left"),
    [("left", 0.1), ("right", 0.4), ("center", 0.25)],
)
@pytest.mark.parametrize(
    ("y_align", "expected_bottom"),
    [("bottom", 0.2), ("top", 0.55), ("center", 0.375)],
)
def test_create_inset_axis_geometry(
    x_align, expected_left, y_align, expected_bottom
):
    fig, ax = plt.subplots()
    parent = ax.get_position()

    inset = create_inset_axis(
        fig,
        ax,
        rel_width=0.5,
        rel_height=0.25,
        margin_x=0.1,
        margin_y=0.2,
        x_align=x_align,
        y_align=y_align,
    )
    bounds = inset.get_position()

    assert bounds.x0 == pytest.approx(parent.x0 + expected_left * parent.width)
    assert bounds.y0 == pytest.approx(parent.y0 + expected_bottom * parent.height)
    assert bounds.width == pytest.approx(0.5 * parent.width)
    assert bounds.height == pytest.approx(0.25 * parent.height)
    plt.close(fig)


def test_create_inset_axis_validates_figure_and_size():
    fig, ax = plt.subplots()
    other_fig = plt.figure()

    with pytest.raises(ValueError, match="must belong"):
        create_inset_axis(other_fig, ax)
    with pytest.raises(ValueError, match="must be positive"):
        create_inset_axis(fig, ax, rel_width=0.0)

    plt.close(fig)
    plt.close(other_fig)
