import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyper_plot.image import (
    crop,
    crop_to_content,
    image_to_ax,
    replace_background_color,
)


def test_background_color_is_inferred_from_all_corners():
    image = np.full((4, 4, 3), 255, dtype=np.uint8)
    image[0, 0] = [128, 0, 0]
    image[2, 2] = [0, 0, 0]

    result = replace_background_color(image, [0, 0, 0, 0])

    assert result.shape == (4, 4, 4)
    assert np.array_equal(result[0, 0], [128, 0, 0, 255])
    assert np.array_equal(result[0, -1], [0, 0, 0, 0])
    assert np.array_equal(image[0, -1], [255, 255, 255])


def test_crop_to_content_rejects_an_image_without_content():
    image = np.ones((3, 3, 3), dtype=float)

    with pytest.raises(ValueError, match="only background"):
        crop_to_content(image)


def test_crop_validates_crop_amounts():
    image = np.zeros((3, 3, 3))

    with pytest.raises(ValueError, match="non-negative"):
        crop(image, left=-1)
    with pytest.raises(ValueError, match="No pixels remaining"):
        crop(image, top=3)


def test_image_to_ax_raises_file_not_found_error(tmp_path):
    fig, ax = plt.subplots()

    with pytest.raises(FileNotFoundError):
        image_to_ax(ax, tmp_path / "missing.png")

    plt.close(fig)
