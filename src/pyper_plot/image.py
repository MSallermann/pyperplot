"""Small image helpers for figure assembly and transparent backgrounds."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from collections.abc import Sequence
import numpy as np
import numpy.typing as npt
from PIL import Image
from pathlib import Path


def open_image(path: str | Path) -> np.ndarray:
    """Load an image with Pillow and return it as a NumPy array.

    The array dtype and channel count follow the image's Pillow mode.
    """
    return np.array(Image.open(path))


def save_image(
    path: str | Path, image: npt.NDArray[np.generic] | Image.Image
) -> None:
    """Save a NumPy image array or Pillow image to *path*.

    The output format is inferred by Pillow from the filename extension.
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image.save(path)


def image_to_ax(ax: Axes, image: str | Path | npt.NDArray[np.generic]) -> AxesImage:
    """Display an image on an axes prepared for frameless image content.

    Ticks, tick labels, spines, and the opaque axes background are hidden.

    Args:
        ax: Axes on which to display the image.
        image: Image array or path accepted by :func:`open_image`.

    Returns:
        The ``AxesImage`` created by ``Axes.imshow``.

    Raises:
        FileNotFoundError: If an image path does not exist.
    """
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(path)
        image = open_image(path)

    ax.tick_params(  # type: ignore
        axis="both", which="both", bottom=0, left=0, labelbottom=0, labelleft=0
    )
    ax.set_facecolor((0, 0, 0, 0))

    for s in ax.spines.values():
        s.set_visible(False)

    return ax.imshow(image)  # pyright: ignore[reportUnknownMemberType]


def _infer_background_color(image: np.ndarray) -> np.ndarray:
    """Infer a background color from the most common corner color."""
    if image.ndim != 3:
        raise ValueError("image must have shape (height, width, channels)")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must not be empty")

    corners = (image[0, 0], image[0, -1], image[-1, 0], image[-1, -1])
    return max(
        corners,
        key=lambda color: sum(np.array_equal(color, other) for other in corners),
    )


def crop_to_content(
    image: np.ndarray,
    background_color: Sequence[float] | None = None,
) -> np.ndarray:
    """Crop uniform background pixels from the outer edges of an image.

    Content is the smallest rectangle containing every pixel that differs
    exactly from ``background_color``. If no color is supplied, the most
    common color among the four corners is used.

    Args:
        image: Array with shape ``(height, width, channels)``.
        background_color: Optional channel values identifying background.

    Returns:
        A view of *image* cropped to its non-background content.

    Raises:
        ValueError: If the image shape is invalid, the image is empty, or it
            contains only background pixels.
    """
    if background_color is None:
        resolved_background = _infer_background_color(image)
    else:
        resolved_background = background_color

    indices = np.argwhere(np.any(image != resolved_background, axis=2))
    if len(indices) == 0:
        raise ValueError("image contains only background pixels")

    lower_height = np.min(indices[:, 0])
    upper_height = np.max(indices[:, 0])
    lower_width = np.min(indices[:, 1])
    upper_width = np.max(indices[:, 1])

    return image[lower_height : upper_height + 1, lower_width : upper_width + 1, :]


def replace_background_color(
    image: np.ndarray,
    replacement_color: Sequence[float],
    background_color: Sequence[float] | None = None,
) -> np.ndarray:
    """Return a copy of an image with its background color replaced.

    Background matching is exact. If ``background_color`` is omitted, the
    most common corner color is used. A replacement with one extra channel
    adds an alpha channel; a replacement with one fewer channel inherits an
    opaque alpha value. Integer images use their dtype maximum as opaque,
    while floating-point images use ``1.0``.

    Args:
        image: Array with shape ``(height, width, channels)``.
        replacement_color: Channel values assigned to background pixels.
        background_color: Optional channel values identifying background.

    Returns:
        A new array; *image* is not modified.

    Raises:
        ValueError: If the image shape or replacement channel count is invalid.
    """
    if image.ndim != 3:
        raise ValueError("image must have shape (height, width, channels)")

    n_channels = image.shape[-1]
    replacement_color = list(replacement_color)

    if background_color is None:
        resolved_background = _infer_background_color(image).tolist()
    else:
        resolved_background = list(background_color)

    if np.issubdtype(image.dtype, np.integer):
        opaque_alpha: float | int = np.iinfo(image.dtype).max
    else:
        opaque_alpha = 1.0

    if len(replacement_color) == n_channels + 1:
        result = np.empty((*image.shape[:2], n_channels + 1), dtype=image.dtype)
        result[..., :n_channels] = image
        result[..., -1] = opaque_alpha
        resolved_background.append(opaque_alpha)
    elif len(replacement_color) == n_channels - 1:
        result = image.copy()
        replacement_color.append(opaque_alpha)
    elif len(replacement_color) == n_channels:
        result = image.copy()
    else:
        raise ValueError(
            "replacement color must have the same number of channels as the "
            "image, or differ by one alpha channel"
        )

    mask = np.all(result == resolved_background, axis=2)
    result[mask] = replacement_color
    return result


def crop(
    image: np.ndarray,
    left: int = 0,
    right: int = 0,
    top: int = 0,
    bottom: int = 0,
) -> np.ndarray:
    """Remove a fixed number of pixels from each image edge.

    Args:
        image: Image array whose first two dimensions are height and width.
        left: Columns removed from the left edge.
        right: Columns removed from the right edge.
        top: Rows removed from the top edge.
        bottom: Rows removed from the bottom edge.

    Returns:
        A cropped view of *image*.

    Raises:
        ValueError: If a crop value is negative or removes all pixels along an
            axis.
    """
    if min(left, right, top, bottom) < 0:
        raise ValueError("crop values must be non-negative")

    height_old = image.shape[0]
    height_new = height_old - top - bottom

    width_old = image.shape[1]
    width_new = width_old - left - right

    if height_new <= 0:
        raise ValueError(
            "Cropping too much at top and/or bottom. No pixels remaining."
        )

    if width_new <= 0:
        raise ValueError(
            "Cropping too much at left and/or right. No pixels remaining."
        )

    return image[top : height_old - bottom, left : width_old - right, :]
