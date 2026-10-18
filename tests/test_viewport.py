"""Zoom anchoring, bounds and source-coordinate invariance."""

import numpy as np

from src.viewport import ImageViewport


def test_zoom_keeps_cursor_over_same_image_point_and_fit_is_minimum():
    view = ImageViewport()
    image, canvas = (1600, 1200), (800, 600)
    scale, x, y = view.transform(image, canvas)
    point_before = ((300 - x) / scale, (240 - y) / scale)
    view.zoom_at(2, 300, 240, image, canvas)
    scale, x, y = view.transform(image, canvas)
    assert np.allclose(point_before, ((300 - x) / scale, (240 - y) / scale))
    view.zoom_at(.001, 300, 240, image, canvas)
    assert view.zoom == 1
    assert view.pan_x == view.pan_y == 0
    assert np.allclose(view.transform(image, canvas), (.5, 0, 0))


def test_pan_and_maximum_zoom_cannot_lose_the_image():
    view = ImageViewport()
    image, canvas = (1600, 1200), (800, 600)
    view.zoom_at(1000, 400, 300, image, canvas)
    assert view.zoom == view.max_zoom
    view.pan_x, view.pan_y = 1e9, -1e9
    scale, x, y = view.transform(image, canvas)
    assert x <= 0 and y <= 0
    assert x + image[0] * scale >= canvas[0]
    assert y + image[1] * scale >= canvas[1]
