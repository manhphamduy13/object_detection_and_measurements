"""Regression tests for taught internal hole/slot metrology."""

import cv2
import numpy as np

from src.measurer import WorkpieceMeasurer


def _synthetic_part():
    gray = np.full((220, 240), 25, dtype=np.uint8)
    cv2.rectangle(gray, (20, 30), (220, 190), 220, -1)
    cv2.circle(gray, (90, 105), 25, 25, -1)
    cv2.rectangle(gray, (145, 92), (205, 118), 25, -1)
    # A closed, high-contrast specular spot is deliberately hole-like in
    # shape, but its interior does not expose the dark scene background.
    cv2.circle(gray, (55, 155), 14, 255, -1)
    contour = np.array([[[20, 30]], [[220, 30]], [[220, 190]], [[20, 190]]], dtype=np.int32)
    return gray, contour


def _to_runtime_wcs(point, contour):
    center, (rw, rh), angle = cv2.minAreaRect(contour)
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0
    dx = float(point[0]) - float(center[0])
    dy = float(point[1]) - float(center[1])
    rad = np.radians(-angle)
    rx = dx * np.cos(rad) - dy * np.sin(rad)
    ry = dx * np.sin(rad) + dy * np.cos(rad)
    return [float(rx / rw), float(ry / rh)]


def test_snap_circle_and_slot_inner_edges():
    gray, contour = _synthetic_part()
    measurer = WorkpieceMeasurer()

    inner_contours = measurer.detect_inner_contours(gray, contour)
    assert len(inner_contours) >= 2
    centers = []
    for inner in inner_contours:
        moments = cv2.moments(inner)
        if abs(moments["m00"]) > 1e-6:
            centers.append((moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]))
    assert any(np.hypot(cx - 90.0, cy - 105.0) < 5.0 for cx, cy in centers)
    assert any(np.hypot(cx - 175.0, cy - 105.0) < 8.0 for cx, cy in centers)
    assert not any(np.hypot(cx - 55.0, cy - 155.0) < 8.0 for cx, cy in centers)

    c1, c2, circle_contour = measurer.snap_scanline_to_inner_contour(
        gray, contour, (65.0, 105.0), (115.0, 105.0)
    )
    assert circle_contour is not None
    circle_diameter = np.hypot(c2[0] - c1[0], c2[1] - c1[1])
    # Endpoints are projected back onto the visualized contour, so a radius
    # 25 circle has an exact polygon-to-polygon diameter of 50 px.
    assert abs(circle_diameter - 50.0) < 1.0
    assert abs(cv2.pointPolygonTest(circle_contour, c1, True)) < 1e-5
    assert abs(cv2.pointPolygonTest(circle_contour, c2, True)) < 1e-5

    s1, s2, slot_contour = measurer.snap_scanline_to_inner_contour(
        gray, contour, (145.0, 105.0), (205.0, 105.0)
    )
    assert slot_contour is not None
    slot_length = np.hypot(s2[0] - s1[0], s2[1] - s1[1])
    assert abs(slot_length - 60.0) < 1.0
    assert abs(cv2.pointPolygonTest(slot_contour, s1, True)) < 1e-5
    assert abs(cv2.pointPolygonTest(slot_contour, s2, True)) < 1e-5

    w1, w2, _ = measurer.snap_scanline_to_inner_contour(
        gray, contour, (175.0, 92.0), (175.0, 118.0)
    )
    slot_width = np.hypot(w2[0] - w1[0], w2[1] - w1[1])
    assert abs(slot_width - 26.0) < 1.0


def test_inner_recipe_participates_in_spec_verdict():
    gray, contour = _synthetic_part()
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    measurer = WorkpieceMeasurer()
    p1, p2 = (65.0, 105.0), (115.0, 105.0)
    sku = "SYNTHETIC_INNER"
    measurer.caliper_recipes[sku] = {
        "sku": sku,
        "wcs": {
            "centroid_offset": [0.0, 0.0],
            "angle": 0.0,
        },
        "calipers": [{
            "label": "ID",
            "nominal_mm": 5.1,
            "p1_wcs": _to_runtime_wcs(p1, contour),
            "p2_wcs": _to_runtime_wcs(p2, contour),
            "measure_type": "inner_contour",
            "color": [210, 80, 230],
        }],
    }
    spec = {
        "sku": sku,
        "optical_scale_mm_per_px": 0.1,
        "dimensions": [{
            "label": "ID",
            "nominal": 5.1,
            "min": 5.0,
            "max": 5.2,
            "diameter": True,
            "axis": "2D_in_plane",
        }],
    }

    result = measurer.measure(image, contour, spec)
    inner = result["recipe_calipers"]["ID"]
    assert inner["measure_type"] == "inner_contour"
    assert abs(inner["length_mm"] - 5.1) <= 0.1
    assert result["spec_report"][0]["status"] == "PASS"
    assert result["overall_pass"] is True
    assert result["active_draw_calipers"][0]["measure_type"] == "inner_contour"
    assert result["active_draw_calipers"][0]["inner_contour"] is not None


def test_outer_caliper_endpoints_are_on_visualized_contour():
    gray = np.full((140, 180), 230, dtype=np.uint8)
    cv2.rectangle(gray, (30, 25), (150, 115), 35, -1)
    contour = np.array(
        [[[30, 25]], [[150, 25]], [[150, 115]], [[30, 115]]], dtype=np.int32
    )

    q1, q2 = WorkpieceMeasurer.snap_scanline_to_edges(
        gray, contour, (55.0, 70.0), (125.0, 70.0)
    )
    assert abs(cv2.pointPolygonTest(contour, q1, True)) < 1e-5
    assert abs(cv2.pointPolygonTest(contour, q2, True)) < 1e-5

    # Even a stale/misaligned recipe that misses the polygon is projected to
    # the nearest visible edge instead of being returned as a raw background point.
    r1, r2 = WorkpieceMeasurer.snap_scanline_to_edges(
        gray, contour, (10.0, 5.0), (170.0, 5.0)
    )
    assert abs(cv2.pointPolygonTest(contour, r1, True)) < 1e-5
    assert abs(cv2.pointPolygonTest(contour, r2, True)) < 1e-5

    # The automatically generated overall L/W annotations must use the same
    # green physical contour, not the floating midpoints of minAreaRect.
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    measured = WorkpieceMeasurer().measure(image, contour, spec=None)
    outer = [
        cal for cal in measured["active_draw_calipers"]
        if cal.get("measure_type") == "outer_dimension"
    ]
    assert {cal["label"] for cal in outer} == {"L", "W"}
    for cal in outer:
        assert abs(cv2.pointPolygonTest(contour, cal["p1"], True)) < 1e-5
        assert abs(cv2.pointPolygonTest(contour, cal["p2"], True)) < 1e-5
