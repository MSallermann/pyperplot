from pyper_plot.abs_gridspec import AbsoluteGridSpec
import pyper_plot.utilities
import pyper_plot.image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

IMAGE_FOLDER = Path(__file__).parent / "images"
OUTPUT_FOLDER = Path(__file__).parent / "output2"
OUTPUT_FOLDER.mkdir(exist_ok=True)

CM = AbsoluteGridSpec.cm


def test_random():
    fig = plt.figure(figsize=(15 * CM, 15 * CM))

    NROWS = 3
    NCOLS = 5
    gs = AbsoluteGridSpec(
        nrows=NROWS,
        ncols=NCOLS,
        figure=fig,
        abs_margin_left=0.1,
        abs_margin_right=0.1,
        abs_margin_bottom=0.1,
        abs_margin_top=0.1,
        abs_hspace=0.1,
        abs_wspace=0.1,
        abs_widths=[-1] * NCOLS,
        abs_heights=[-1] * NROWS,
    )

    x = np.linspace(0, 2 * np.pi)

    for a in pyper_plot.utilities.gs_row(-1, gs, fig, slice(1, None, 2)):
        a.plot(x, np.sin(x))
        a.set_xlabel("x")
        a.set_ylabel("y")

    for a in pyper_plot.utilities.gs_col(0, gs, fig, slice(0, 1)):
        a.plot(x, np.sin(x))
        a.set_xlabel("x")
        a.set_ylabel("y")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(x, x**2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    pyper_plot.utilities.spine_axis(
        fig,
        gs[0, 2],
        color="red",
        linewidth=1.0,
        linestyle="--",
        which=["left", "right"],
    )

    # Picture with three channels that we extend by transparency
    image = pyper_plot.image.open_image(IMAGE_FOLDER / "hopfion_cut_vertical.png")
    image_crop = pyper_plot.image.crop_to_content(image)
    ax_img = fig.add_subplot(gs[-1, 0])
    pyper_plot.image.image_to_ax(ax_img, image_crop)

    image = pyper_plot.image.open_image(IMAGE_FOLDER / "spins.png")
    image_crop = pyper_plot.image.crop_to_content(image)
    ax_img = fig.add_subplot(gs[-2, 0])
    pyper_plot.image.image_to_ax(ax_img, image_crop)
    fig.savefig(OUTPUT_FOLDER / "test_fig.png", dpi=300)


def test_aspect_ratio():
    fig = plt.figure(figsize=(20 * CM, 15 * CM))

    NROWS = 3
    NCOLS = 5
    gs = AbsoluteGridSpec(
        nrows=NROWS,
        ncols=NCOLS,
        abs_content_width=NCOLS * 2 * CM,
        abs_content_height=NROWS * 2 * CM,
        figure=fig,
        abs_margin_left=0.05,
        abs_margin_right=0.1,
        abs_margin_bottom=0.05,
        abs_margin_top=0.1,
        abs_hspace=0.1,
        abs_wspace=0.3,
    )

    for irow in range(gs.nrows):
        for ax in pyper_plot.utilities.gs_row(irow, gs):
            pyper_plot.image.image_to_ax(ax, IMAGE_FOLDER / "square.png")
            pyper_plot.utilities.spine_axis(fig, ax)

    fig.suptitle("All diamonds should be perfectly inscribed in the squares!")
    fig.savefig(OUTPUT_FOLDER / "test_fig_aspect.png", dpi=300)


def test_abs_margins():

    fig = plt.figure(figsize=(30 * CM, 30 * CM))

    gs = AbsoluteGridSpec(
        figure=fig,
        nrows=2,
        ncols=2,
        abs_content_width=28 * CM,
        abs_content_height=28 * CM,
        abs_hspace=2.5 * CM,
        abs_wspace=2.5 * CM,
        abs_margin_left=2 * CM,
        abs_margin_right=2 * CM,
        abs_margin_bottom=2 * CM,
        abs_margin_top=2 * CM,
    )

    for irow in range(gs.nrows):
        for ax in pyper_plot.utilities.gs_row(irow, gs):
            pyper_plot.image.image_to_ax(
                ax, (IMAGE_FOLDER / "square.png").as_posix()
            )
            pyper_plot.utilities.spine_axis(fig, ax)

    fig.savefig(OUTPUT_FOLDER / "test_fig_abs_margins.png", dpi=300)


def test_abs_margins_advanced():
    fig = plt.figure(figsize=(30 * CM, 30 * CM))

    gs = AbsoluteGridSpec(
        figure=fig,
        nrows=3,
        ncols=4,
        abs_content_width=28 * CM,
        abs_content_height=10 * CM,
        abs_hspace=1.0 * CM,
        abs_wspace=1.0 * CM,
        abs_margin_left=2 * CM,
        abs_margin_right=2 * CM,
        abs_margin_bottom=2 * CM,
        abs_margin_top=2 * CM,
        abs_widths=[11 * CM, -2, -1, -1],
        abs_heights=[-1 * CM, -2 *CM, 5 * CM],
    )

    for irow in range(gs.nrows):
        for ax in pyper_plot.utilities.gs_row(irow, gs):
            pyper_plot.image.image_to_ax(
                ax, IMAGE_FOLDER / "square.png"
            )
            pyper_plot.utilities.spine_axis(fig, ax)

    ax = fig.add_subplot(gs[:, 0])
    pyper_plot.utilities.spine_axis(fig, ax)
    pyper_plot.image.image_to_ax(ax, IMAGE_FOLDER / "square.png")

    fig.savefig(
        OUTPUT_FOLDER / "test_fig_abs_margins_advanced.png",
        dpi=300,
    )
