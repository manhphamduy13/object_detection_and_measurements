"""Regression tests for golden-template contour metrology."""

import cv2
import numpy as np

from src.measurer import WorkpieceMeasurer
from src.template_geometry import check_profile, offset_contours, template_intersections


def _rectangle(x1, y1, x2, y2):
    return np.asarray(
        [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]],
        dtype=np.int32,
    )


def _recipe(measurer, golden, tolerance_mm=0.25):
    frame = measurer._contour_frame(golden)
    local = [
        measurer._point_to_frame_local(point, frame).tolist()
        for point in golden.reshape(-1, 2)
    ]
    return {
        "sku": "SYNTHETIC_TEMPLATE",
        "wcs": frame,
        "golden_contour_local_px": local,
        "contour_tolerance_mm": tolerance_mm,
        "calipers": [{
            "label": "W",
            "p1_ref": [30.0, 100.0],
            "p2_ref": [170.0, 100.0],
            "p1_wcs": [-0.5, 0.0],
            "p2_wcs": [0.5, 0.0],
            "golden_length_mm": 14.0,
            "measure_type": "template_contour",
        }],
    }


def test_template_caliper_adds_signed_edge_deviations_to_golden_length():
    golden = _rectangle(30, 40, 170, 160)
    measurer = WorkpieceMeasurer()
    measurer.caliper_recipes["SYNTHETIC_TEMPLATE"] = _recipe(measurer, golden)
    gray = np.full((220, 240), 230, dtype=np.uint8)

    outward = _rectangle(28, 40, 172, 160)
    result = measurer.apply_caliper_recipes(
        gray, outward, 0.1, "SYNTHETIC_TEMPLATE"
    )["W"]
    assert abs(result["delta_p1_mm"] - 0.2) < 1e-6
    assert abs(result["delta_p2_mm"] - 0.2) < 1e-6
    assert abs(result["length_mm"] - 14.4) < 1e-6

    inward = _rectangle(32, 40, 168, 160)
    result = measurer.apply_caliper_recipes(
        gray, inward, 0.1, "SYNTHETIC_TEMPLATE"
    )["W"]
    assert abs(result["delta_p1_mm"] + 0.2) < 1e-6
    assert abs(result["delta_p2_mm"] + 0.2) < 1e-6
    assert abs(result["length_mm"] - 13.6) < 1e-6


def test_contour_profile_gate_is_bidirectional_and_strict():
    golden = _rectangle(30, 40, 170, 160)
    measurer = WorkpieceMeasurer()
    measurer.caliper_recipes["SYNTHETIC_TEMPLATE"] = _recipe(measurer, golden)

    within_band = _rectangle(28, 40, 172, 160)
    profile = measurer.evaluate_contour_profile(
        within_band, 0.1, "SYNTHETIC_TEMPLATE"
    )
    assert profile["pass"] is True
    assert abs(profile["max_deviation_mm"] - 0.2) < 1e-6

    outside_band = _rectangle(26, 40, 174, 160)
    profile = measurer.evaluate_contour_profile(
        outside_band, 0.1, "SYNTHETIC_TEMPLATE"
    )
    assert profile["pass"] is False
    assert profile["outside_point_count"] > 0
    assert abs(profile["max_deviation_mm"] - 0.4) < 1e-6


def test_rigid_alignment_does_not_treat_translation_as_size_error():
    golden = _rectangle(30, 40, 170, 160)
    measurer = WorkpieceMeasurer()
    measurer.caliper_recipes["SYNTHETIC_TEMPLATE"] = _recipe(measurer, golden)
    shifted = golden + np.asarray([[[37, -19]]], dtype=np.int32)
    profile = measurer.evaluate_contour_profile(
        shifted, 0.1, "SYNTHETIC_TEMPLATE"
    )
    assert profile["pass"] is True
    assert profile["max_deviation_mm"] < 1e-6

    rotation = cv2.getRotationMatrix2D((100.0, 100.0), 37.0, 1.0)
    rotated = cv2.transform(golden.astype(np.float32), rotation)
    rotated += np.asarray([[[21.0, 13.0]]], dtype=np.float32)
    profile = measurer.evaluate_contour_profile(
        rotated, 0.1, "SYNTHETIC_TEMPLATE"
    )
    assert profile["pass"] is True
    assert profile["max_deviation_mm"] < 1e-4


def test_profile_failure_participates_in_final_metrology_verdict():
    golden = _rectangle(30, 40, 170, 160)
    measured = _rectangle(26, 40, 174, 160)
    gray = np.full((220, 240), 230, dtype=np.uint8)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    measurer = WorkpieceMeasurer()
    measurer.caliper_recipes["SYNTHETIC_TEMPLATE"] = _recipe(measurer, golden)
    spec = {
        "sku": "SYNTHETIC_TEMPLATE",
        "optical_scale_mm_per_px": 0.1,
        "dimensions": [{
            "label": "W",
            "nominal": 14.8,
            "min": 14.0,
            "max": 15.5,
            "axis": "2D_in_plane",
        }],
    }

    result = measurer.measure(image, measured, spec)
    assert result["spec_report"][0]["status"] == "PASS"
    assert result["dimension_pass"] is True
    assert result["profile_pass"] is False
    assert result["caliper_pass"] is False
    assert result["overall_pass"] is False


def test_asymmetric_offsets_apply_per_edge_not_to_total_dimension():
    golden = _rectangle(30, 40, 170, 160)
    m = WorkpieceMeasurer()
    recipe = _recipe(m, golden)
    recipe.update(contour_lower_mm=.1, contour_upper_mm=.3)
    m.caliper_recipes["SYNTHETIC_TEMPLATE"] = recipe
    assert m.evaluate_contour_profile(_rectangle(28, 40, 172, 160), .1,
                                      "SYNTHETIC_TEMPLATE")["pass"] is True
    assert m.evaluate_contour_profile(_rectangle(32, 40, 168, 160), .1,
                                      "SYNTHETIC_TEMPLATE")["pass"] is False


def test_erosion_at_square_corners_is_not_reverse_hausdorff_distance():
    # A 2px inset on all sides is valid for erosion by 2px, even though the
    # old corner-to-corner Hausdorff distance would incorrectly be sqrt(8).
    golden = _rectangle(30, 40, 170, 160)
    inset = _rectangle(32, 42, 168, 158)
    profile = check_profile(inset, golden, .1, .2, 0)
    assert profile["pass"] is True
    assert abs(profile["max_inward_mm"] - .2) < 1e-6


def test_band_checks_required_material_not_only_a_small_patch_near_golden_edge():
    golden = _rectangle(30, 40, 170, 160)
    patch = _rectangle(30, 70, 32, 90)
    profile = check_profile(patch, golden, .1, .3, .3)
    assert profile["pass"] is False
    assert len(profile["missing_core_points"]) > 0


def test_subpixel_limits_and_narrow_burr_vertices_are_not_skipped():
    golden = _rectangle(30, 40, 170, 160).astype(np.float32)
    measured = _rectangle(28, 40, 172, 160).astype(np.float32)
    assert check_profile(measured, golden, .1, .1, .20001)["pass"] is True
    assert check_profile(measured, golden, .1, .1, .199)["pass"] is False
    burr = np.array([[[30, 40]], [[170, 40]], [[170, 99]], [[175, 100]],
                     [[170, 101]], [[170, 160]], [[30, 160]]], dtype=np.float32)
    assert check_profile(burr, golden, .1, .3, .3)["pass"] is False


def test_offsets_have_different_extents_and_preserve_concave_split_regions():
    golden = _rectangle(30, 40, 170, 160)
    inner, outer = offset_contours(golden, 2, 4)
    ip, op = inner[0].reshape(-1, 2), outer[0].reshape(-1, 2)
    assert np.allclose(ip.min(0), (32, 42), atol=.5)
    assert np.allclose(ip.max(0), (168, 158), atol=.5)
    assert np.allclose(op.min(0), (26, 36), atol=.5)
    concave = np.array([[[0, 0]], [[20, 0]], [[20, 8]], [[40, 8]], [[40, 0]],
                        [[60, 0]], [[60, 20]], [[40, 20]], [[40, 12]],
                        [[20, 12]], [[20, 20]], [[0, 20]]], dtype=np.float32)
    split, _ = offset_contours(concave, 3, 2)
    assert len(split) == 2


def test_infinite_scanline_finds_larger_parts_but_does_not_shift_a_missing_line():
    golden = _rectangle(30, 40, 170, 160)
    larger = _rectangle(20, 40, 180, 160)
    pair = template_intersections(larger, (30, 100), (170, 100))
    assert np.allclose(pair, ((20, 100), (180, 100)))
    assert template_intersections(golden, (30, 300), (170, 300)) is None


def test_actual_golden_override_and_blank_fallback_do_not_use_nominal():
    m = WorkpieceMeasurer()
    golden = _rectangle(30, 40, 170, 160)
    recipe = _recipe(m, golden)
    recipe["calipers"][0].update(golden_actual_mm=15.2, nominal_mm=100)
    m.caliper_recipes["SYNTHETIC_TEMPLATE"] = recipe
    gray = np.zeros((220, 240), dtype=np.uint8)
    larger = _rectangle(28, 40, 172, 160)
    result = m.apply_caliper_recipes(gray, larger, .1, "SYNTHETIC_TEMPLATE")["W"]
    assert result["golden_source"] == "actual"
    assert abs(result["length_mm"] - 15.6) < 1e-6
    assert result["length_px"] == 144
    recipe["calipers"][0]["golden_actual_mm"] = None
    result = m.apply_caliper_recipes(gray, larger, .1, "SYNTHETIC_TEMPLATE")["W"]
    assert result["golden_source"] == "image"
    assert abs(result["length_mm"] - 14.4) < 1e-6


def test_missing_scanline_and_missing_profile_are_fail_closed_and_renderable():
    from src.measurer import annotate_measurements
    m = WorkpieceMeasurer()
    contour = _rectangle(30, 40, 170, 160)
    recipe = _recipe(m, contour)
    recipe["calipers"][0].update(p1_ref=[30, 300], p2_ref=[170, 300])
    m.caliper_recipes["SYNTHETIC_TEMPLATE"] = recipe
    image = np.zeros((440, 540, 3), dtype=np.uint8)
    spec = {"sku": "SYNTHETIC_TEMPLATE", "optical_scale_mm_per_px": .1,
            "dimensions": [{"label": "W", "nominal": 14, "min": 13, "max": 15}]}
    result = m.measure(image, contour, spec)
    assert result["caliper_pass"] is False
    assert result["spec_report"][0]["measured_mm"] is None
    assert result["spec_report"][0]["status"] == "FAIL"
    assert annotate_measurements(image, contour, result).shape == image.shape
    del recipe["golden_contour_local_px"]
    assert m.evaluate_contour_profile(contour, .1, "SYNTHETIC_TEMPLATE")["pass"] is False


def test_zero_offsets_allow_exact_golden():
    golden = _rectangle(30, 40, 170, 160)
    assert check_profile(golden, golden, .1, 0, 0)["pass"] is True


def test_profile_enabled_independently_of_scanline_type():
    m = WorkpieceMeasurer()
    golden = _rectangle(30, 40, 170, 160)
    recipe = _recipe(m, golden)
    recipe["contour_profile_enabled"] = True
    recipe["calipers"][0]["measure_type"] = "taught_caliper"
    m.caliper_recipes["SYNTHETIC_TEMPLATE"] = recipe
    profile = m.evaluate_contour_profile(_rectangle(26, 40, 174, 160), .1, "SYNTHETIC_TEMPLATE")
    assert profile["pass"] is False
    recipe["contour_profile_enabled"] = False
    assert m.evaluate_contour_profile(golden, .1, "SYNTHETIC_TEMPLATE") is None
