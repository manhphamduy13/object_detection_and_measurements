"""The normal report path must include template QC for legacy SCAN recipes."""

import json

import cv2
import numpy as np
from PIL import Image

from predict import inspect_known_sample


def test_known_sku_report_uses_golden_profile_with_legacy_recipe(tmp_path):
    golden = np.full((300, 420, 3), 230, dtype=np.uint8)
    cv2.rectangle(golden, (60, 60), (320, 200), (35, 35, 35), -1)
    sample = golden.copy()
    cv2.rectangle(sample, (298, 110), (330, 150), (230, 230, 230), -1)
    recipe = {
        "contour_lower_mm": .3, "contour_upper_mm": .3,
        "calipers": [{"label": "W", "measure_type": "taught_caliper",
                      "p1_wcs": [-.5, 0], "p2_wcs": [.5, 0]}],
    }
    spec = {"sku": "TEST_REPORT", "optical_scale_mm_per_px": .1,
            "dimensions": [{"label": "W", "nominal": 26, "min": 25.9, "max": 26.1}]}
    control, _ = inspect_known_sample(golden, recipe, spec, tmp_path, "golden", golden_image=golden)
    measured, paths = inspect_known_sample(sample, recipe, spec, tmp_path, "notched", golden_image=golden)
    assert control["profile_pass"] is True
    assert measured["profile_pass"] is False
    assert measured["spec_report"][0]["golden_length_mm"] is not None
    assert measured["spec_report"][0]["total_delta_mm"] < 0
    assert recipe.get("wcs") is None  # No mutation of the production recipe.
    assert recipe.get("contour_profile_enabled") is None
    assert paths["panel_summary"].parent.name == "visualizations"
    assert Image.open(paths["panel_summary"]).width > 1500
    assert Image.open(paths["measured_image"]).size == (420, 300)
    record = json.loads(paths["prediction"].read_text(encoding="utf-8"))
    assert record["expected_sku"] == "TEST_REPORT"
    assert record["classification_performed"] is False
    assert record["predicted_class"] is None
    assert record["inspection_recipe"]["golden_contour_local_px"]
