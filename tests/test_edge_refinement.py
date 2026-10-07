"""Regression tests for reflection-safe physical edge refinement."""

import cv2
import numpy as np

from src.classifier import refine_contour_to_physical_edge


def test_outer_refinement_prefers_silhouette_over_nearby_shadow():
    gray = np.full((180, 220), 225, dtype=np.uint8)
    cv2.rectangle(gray, (45, 35), (165, 145), 65, -1)
    # A secondary high-contrast line outside the real right edge simulates a
    # cast shadow / dark halo that an unconstrained strongest-gradient search
    # used to follow.
    cv2.rectangle(gray, (170, 38), (174, 142), 145, -1)
    coarse = np.array(
        [[[42, 32]], [[168, 32]], [[168, 148]], [[42, 148]]], dtype=np.int32
    )

    refined = refine_contour_to_physical_edge(gray, coarse, max_shift=6.0)
    x, y, w, h = cv2.boundingRect(refined)

    assert abs(x - 45) <= 2
    assert abs(y - 35) <= 2
    assert abs((x + w - 1) - 165) <= 2
    assert abs((y + h - 1) - 145) <= 2


def test_outer_refinement_rejects_surface_texture():
    gray = np.full((160, 210), 230, dtype=np.uint8)
    cv2.rectangle(gray, (35, 30), (175, 130), 80, -1)
    for x in range(48, 170, 12):
        cv2.line(gray, (x, 36), (x, 124), 145, 2)
    coarse = np.array(
        [[[33, 28]], [[177, 28]], [[177, 132]], [[33, 132]]], dtype=np.int32
    )

    refined = refine_contour_to_physical_edge(gray, coarse, max_shift=6.0)
    x, y, w, h = cv2.boundingRect(refined)

    assert abs(x - 35) <= 2
    assert abs((x + w - 1) - 175) <= 2
    assert abs(y - 30) <= 2
    assert abs((y + h - 1) - 130) <= 2
