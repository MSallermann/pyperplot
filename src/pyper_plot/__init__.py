"""Tools for building precisely sized Matplotlib figures.

The main entry point is :class:`AbsoluteGridSpec`. Common axes and image
helpers are re-exported here for convenience; the ``image`` and ``utilities``
modules remain available for namespaced use.
"""

from importlib.metadata import PackageNotFoundError, version

from . import image, utilities
from .abs_gridspec import AbsoluteGridSpec
from .image import (
    crop,
    crop_to_content,
    image_to_ax,
    open_image,
    replace_background_color,
    save_image,
)
from .utilities import (
    add_box_around_image,
    clear_spines,
    create_inset_axis,
    create_n_axes,
    gs_col,
    gs_row,
    label_subplot,
    spine_axis,
)

try:
    __version__ = version("PyperPlot")
except (
    PackageNotFoundError
):  # pragma: no cover - only for an uninstalled source tree
    __version__ = "0+unknown"

__all__ = [
    "AbsoluteGridSpec",
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
    "spine_axis",
    "utilities",
    "__version__",
]
