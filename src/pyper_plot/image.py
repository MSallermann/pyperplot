from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from collections.abc import Sequence
import numpy as np
import numpy.typing as npt
from PIL import Image
from pathlib import Path


def open_image(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path))


def save_image(
    path: str | Path, image: npt.NDArray[np.generic] | Image.Image
) -> None:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image.save(path)


def image_to_ax(ax: Axes, image: str | Path | npt.NDArray[np.generic]) -> AxesImage:
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
    """Crops an image array to its content."""
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
    """Replaced the background color of an image, specified as a numpy array.

    Args:
        image (np.Array): The image array
        replacement_color (the color): The color which replaces the background color.
        background_color (the background color, optional): The background color. If None it is inferred from the corners. Defaults to None.

    Returns:
        np.Array: the new image array

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
    """Crops an image by removing pixels from the left, right, top and bottom"""
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
