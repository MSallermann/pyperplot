import os
import numpy as np
from pyper_plot.pyper_plot import PyperPlot
import matplotlib.pyplot as plt
from pathlib import Path

IMAGE_FOLDER = Path(__file__).parent / "images"
OUTPUT_FOLDER = Path(__file__).parent / "output"
OUTPUT_FOLDER.mkdir(exist_ok=True)
CM = PyperPlot.cm


def test_random():
    ### There is not much to test except that no exceptions get thrown, I guess
    pplot = PyperPlot(15 * CM)
    pplot.ncols = 5

    pplot.hspace = 0.2
    pplot.wspace = 0.3
    pplot.nrows = 3
    pplot.width_ratios = np.ones(pplot.ncols)
    pplot.width_ratios[0] = 2

    pplot.height_from_aspect_ratio(1)

    print(pplot.info_string())

    fig = pplot.fig()
    gs = pplot.gs()

    x = np.linspace(0, 2 * np.pi)

    # row
    for a in pplot.row(-1, slice(1, None, 2)):
        a.plot(x, np.sin(x))
        a.set_xlabel("x")
        a.set_ylabel("y")

    # col
    for a in pplot.col(0, slice(0, 1)):
        a.plot(x, np.cos(x))
        a.set_xlabel("x")
        a.set_ylabel("y")

    # annotate
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(x, x**2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    for i in range(6):
        xy = (i, i**2)
        xy_text = pplot.xy_text_auto(ax, xy, deriv=2 * i)
        pplot.annotate_graph(ax, xy, xy_text)

    # spine axes
    pplot.spine_axis(gs[0, 2], color="red", which=["left", "right"])

    # Picture with three channels that we extend by transparency
    image = plt.imread(os.path.join(IMAGE_FOLDER, "hopfion_cut_vertical.png"))
    image_crop = pplot.crop_to_content(
        image, replace_background_color=[0.0, 0.0, 0.0, 0.0]
    )
    plt.imsave(os.path.join(OUTPUT_FOLDER, "cropped_hopfion.png"), image_crop)
    ax_img = fig.add_subplot(gs[-1, 0])
    pplot.image_to_ax(ax_img, image_crop)

    image = plt.imread(os.path.join(IMAGE_FOLDER, "spins.png"))
    image_crop = pplot.crop_to_content(
        image, replace_background_color=[0.0, 0.0, 0.0, 0.0]
    )
    plt.imsave(os.path.join(OUTPUT_FOLDER, "cropped_spins.png"), image_crop)
    ax_img = fig.add_subplot(gs[-2, 0])
    pplot.image_to_ax(ax_img, image_crop)

    fig.savefig(os.path.join(OUTPUT_FOLDER, "test_fig.png"), dpi=300)


def test_aspect_ratio():
    pplot = PyperPlot(20 * PyperPlot.cm, height=15 * CM)

    pplot.ncols = 5
    pplot.nrows = 3
    pplot.hspace = 0.1
    pplot.wspace = 0.3

    pplot.vertical_margins = [0.05, 0.1]
    pplot.horizontal_margins = [0.05, 0.1]

    pplot.height_from_aspect_ratio(pplot.ncols / pplot.nrows)
    print(pplot.info_string())

    fig = pplot.fig()
    gs = pplot.gs()

    for irow in range(pplot.nrows):
        for ax in pplot.row(irow):
            pplot.image_to_ax(ax, os.path.join(IMAGE_FOLDER, "square.png"))
            pplot.spine_axis(ax)

    fig.suptitle("All diamonds should be perfectly inscribed in the squares!")
    fig.savefig(os.path.join(OUTPUT_FOLDER, "test_fig_aspect.png"), dpi=300)


def test_abs_margins():
    pplot = PyperPlot(30 * CM)

    pplot.ncols = 2
    pplot.nrows = 2

    print(pplot.info_string())

    pplot.height_from_aspect_ratio(
        pplot.ncols / pplot.nrows,
    )

    print(pplot.height)

    pplot.apply_absolute_margins(
        abs_hspace=2.5 * CM,
        abs_wspace=2.5 * CM,
        abs_horizontal_margins=np.ones(2) * CM,
        abs_vertical_margins=np.ones(2) * CM,
    )
    print(pplot.info_string())

    fig = pplot.fig()
    gs = pplot.gs()

    for irow in range(pplot.nrows):
        for ax in pplot.row(irow):
            pplot.image_to_ax(ax, (IMAGE_FOLDER / "square.png").as_posix())
            pplot.spine_axis(ax)

    # fig.suptitle("All diamonds should be perfectly inscribed in the squares!")
    fig.savefig(os.path.join(OUTPUT_FOLDER, "test_fig_abs_margins.png"), dpi=300)


def test_abs_margins_advanced():
    pplot = PyperPlot(30 * CM)

    pplot.ncols = 4
    pplot.nrows = 2

    print(pplot.info_string())

    pplot.height_from_aspect_ratio(pplot.ncols / pplot.nrows)
    pplot.apply_absolute_margins(
        abs_hspace=1.0 * CM * 1,
        abs_wspace=1.0 * CM * 1,
        abs_horizontal_margins=np.ones(2) * CM * 1,
        abs_vertical_margins=np.ones(2) * CM * 1,
        abs_widths=[11 * CM, -1, -1, -1],
        abs_heights=[5 * CM, 5 * CM],
    )
    print(pplot.info_string())

    fig = pplot.fig()
    gs = pplot.gs()

    for irow in range(pplot.nrows):
        for ax in pplot.row(irow, slice(1, None)):
            pplot.image_to_ax(ax, os.path.join(IMAGE_FOLDER, "square.png"))
            pplot.spine_axis(ax)

    ax = fig.add_subplot(gs[:, 0])
    pplot.image_to_ax(ax, os.path.join(IMAGE_FOLDER, "square.png"))
    pplot.spine_axis(ax)

    # fig.suptitle("All diamonds should be perfectly inscribed in the squares!")
    fig.savefig(
        os.path.join(OUTPUT_FOLDER, "test_fig_abs_margins_advanced.png"),
        dpi=300,
    )


def test_misc():
    image = plt.imread(os.path.join(IMAGE_FOLDER, "hopfion_cut_vertical.png"))
    image_crop = PyperPlot.crop(image, left=10, right=500, top=500, bottom=500)
    plt.imsave(os.path.join(OUTPUT_FOLDER, "cropped_hopfion_test_2.png"), image_crop)

    image = plt.imread(os.path.join(IMAGE_FOLDER, "circle.png"))
    image_crop = PyperPlot.crop_to_content(image)
    plt.imsave(os.path.join(OUTPUT_FOLDER, "cropped_circle.png"), image_crop)
