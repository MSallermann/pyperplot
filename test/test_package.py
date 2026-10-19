import pyper_plot
from pyper_plot.abs_gridspec import AbsoluteGridSpec


def test_public_api():
    assert pyper_plot.AbsoluteGridSpec is AbsoluteGridSpec
    assert pyper_plot.__version__
    assert set(pyper_plot.__all__) == {
        "AbsoluteGridSpec",
        "LengthUnit",
        "add_box_around_image",
        "clear_spines",
        "create_inset_axis",
        "create_n_axes",
        "crop",
        "crop_to_content",
        "gs_col",
        "gs_row",
        "image",
        "image_to_ax",
        "label_subplot",
        "open_image",
        "replace_background_color",
        "save_image",
        "shade_grid_cell",
        "spine_axis",
        "utilities",
        "__version__",
    }
