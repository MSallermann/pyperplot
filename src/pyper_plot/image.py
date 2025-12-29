from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from collections.abc import Sequence
import numpy as np
import numpy.typing as npt
from PIL import Image
import os
from pathlib import Path


def open_image(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path))


def save_image(path: str | Path, image: npt.NDArray[np.floating] | Image.Image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image.save(path)


def image_to_ax(ax: Axes, image: str | Path | npt.NDArray[np.floating]) -> AxesImage:
    if isinstance(image, str | Path):
        if os.path.exists(image):
            image = open_image(image)
        else:
            msg = f"`{image}` does not exist"
            raise Exception(msg)

    ax.tick_params(  # type: ignore
        axis="both", which="both", bottom=0, left=0, labelbottom=0, labelleft=0
    )
    ax.set_facecolor((0, 0, 0, 0))

    for s in ax.spines.values():
        s.set_visible(False)

    return ax.imshow(image)  # pyright: ignore[reportUnknownMemberType]


def crop_to_content(
    image: np.ndarray,
    background_color: Sequence[float] | None = None,
):
    """Crops an image array to its content."""
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
            if np.all(cc == corner_colors[-1]):
                break

        background_color = corner_colors[0]

    # I think I know why this works
    indices = np.argwhere(np.any(image[:, :] != background_color, axis=2))

    lower_height = np.min(indices[:, 0])
    upper_height = np.max(indices[:, 0])
    lower_width = np.min(indices[:, 1])
    upper_width = np.max(indices[:, 1])

    return image[lower_height : upper_height + 1, lower_width : upper_width + 1, :]


def replace_background_color(
    image: np.ndarray,
    replacement_color: Sequence[float],
    background_color: Sequence[float] | None = None,
):
    """Replaced the background color of an image, specified as a numpy array.

    Args:
        image (np.Array): The image array
        replacement_color (the color): The color which replaces the background color.
        background_color (the background color, optional): The background color. If None it is inferred from the corners. Defaults to None.

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
            if np.all(cc == corner_colors[-1]):
                break
        background_color = corner_colors[0]

    N_CHANNELS_BG = len(replacement_color)

    # If the background color has more channels than the picture, we need to introduce the alpha channel
    if N_CHANNELS_BG > N_CHANNELS and replacement_color[-1] != 1.0:
        image_copy = np.ones(shape=(image_shape[0], image_shape[1], 4))
        image_copy[:, :, :3] = image
        image = image_copy
        assert background_color is not None
        background_color = [
            *background_color,
            1.0,
        ]  # Extend the background color with the alpha channel
    elif N_CHANNELS_BG < N_CHANNELS:
        replacement_color = [*replacement_color, 1.0]

    indices = np.argwhere(np.all(image[:, :] == background_color, axis=2))
    image[indices[:, 0], indices[:, 1], :] = replacement_color
    return image


def crop(
    image: np.ndarray,
    left: int = 0,
    right: int = 0,
    top: int = 0,
    bottom: int = 0,
):
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
