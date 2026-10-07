"""
Metrology Subpixel Caliper Test.
Validates caliper edge extraction, pin diameter phi metrology, and CAD tolerance verifications.
"""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

import cv2
import config
import classifier as clf
import measurer

def test_caliper_measurements():
    candidates = sorted((Path(config.TEST_DIR) / "98003P" / "images").glob("*.png"))
    assert candidates, "No rotation-only 98003P test sample found"
    sample_img_path = candidates[0]

    img = cv2.imread(str(sample_img_path))
    described = clf.describe_image(img)
    assert described is not None, "Segmentation failed"

    # This is a metrology test, so use the known fixture's specification
    # directly.  Classification/fusion has separate tests and should not make
    # an edge/caliper regression fail for the wrong reason.
    library, _, _ = clf.build_database(config.REFERENCE_DIR)
    spec = library["98003P"].get("spec")
    assert spec is not None, "Missing 98003P specification"

    m = measurer.WorkpieceMeasurer()
    meas = m.measure(img, described["contour"], spec)
    assert meas["Length_L_px"] > 600, "Length measurement abnormal"
    assert meas["Width_W_px"] > 500, "Width measurement abnormal"

    # Only dimensions with a taught, image-supported recipe may be reported as
    # measured.  Diameter values must never be copied from their nominal CAD
    # value merely to make a report pass.
    report = meas.get("spec_report", [])
    labels = {r["label"]: r for r in report}
    unmeasured = {u["label"]: u for u in meas.get("unmeasured_dimensions", [])}

    assert "L" in labels, "Missing length L"
    assert "W" in labels, "Missing width W"
    assert "D" in unmeasured, "Untaught diameter D must be explicitly unmeasured"
    assert "D1" in unmeasured, "Untaught diameter D1 must be explicitly unmeasured"
    assert "D2" in unmeasured, "Untaught diameter D2 must be explicitly unmeasured"
    assert "H" in unmeasured, "H must be in unmeasured_dimensions (Z-axis skipped from 2D comparison)"

    print(f"Caliper measurement test passed: L={meas['Length_L_px']:.2f}px, W={meas['Width_W_px']:.2f}px")
    print(">>> TAUGHT METROLOGY PASSED; UNSUPPORTED DIAMETERS REMAIN UNMEASURED <<<")


def test_user_custom_recipe_adherence():
    """
    Verifies that when a user deletes proposed dimensions and keeps only their custom selection:
    1. The metrology engine measures ONLY what the user kept.
    2. Deleted dimensions are NOT drawn on the image and NOT compared in spec_report.
    3. Deleted dimensions are accurately tracked in unmeasured_dimensions.
    """
    candidates = sorted((Path(config.TEST_DIR) / "98661BBS-1" / "images").glob("*.png"))
    assert candidates, "No rotation-only 98661BBS-1 test sample found"
    sample_img_path = candidates[0]

    img = cv2.imread(str(sample_img_path))
    described = clf.describe_image(img)
    assert described is not None

    m = measurer.WorkpieceMeasurer()
    spec_path = PROJECT_DIR / "assets" / "references" / "98661BBS-1" / "spec.json"
    with open(spec_path, "r", encoding="utf-8") as f:
        import json
        spec = json.load(f)

    # Simulate user having customized recipe to ONLY measure ['L', 'W'] (deleted F, J, K, d, etc.)
    original_recipes = m.caliper_recipes.copy()
    try:
        m.caliper_recipes["98661BBS-1"] = {
            "sku": "98661BBS-1",
            "calipers": [
                {"label": "L", "nominal_mm": 28.5},
                {"label": "W", "nominal_mm": 4.3}
            ]
        }

        meas = m.measure(img, described["contour"], spec)
        report = meas.get("spec_report", [])
        measured_labels = [r["label"] for r in report]
        unmeasured_labels = [u["label"] for u in meas.get("unmeasured_dimensions", [])]
        active_draw_labels = [c["label"] for c in meas.get("active_draw_calipers", [])]

        # Must measure ONLY L and W
        assert set(measured_labels) == {"L", "W"}, f"Expected {{'L', 'W'}}, got {measured_labels}"
        assert "L" in active_draw_labels, "L must be drawn"
        assert "W" in active_draw_labels, "W must be drawn"
        assert "F" not in active_draw_labels, "F was deleted by user and must NOT be drawn"
        assert "J" not in active_draw_labels, "J was deleted by user and must NOT be drawn"
        assert "d" not in active_draw_labels, "Hole d was deleted by user and must NOT be drawn"

        # Deleted CAD specs must be in unmeasured_dimensions with clear reason
        assert "J" in unmeasured_labels, "J must be in unmeasured_dimensions"
        assert any(k in unmeasured_labels for k in ("K", "D1")), "K or D1 must be in unmeasured_dimensions"
        for u in meas.get("unmeasured_dimensions", []):
            if u["label"] in ("J", "F", "d", "K", "T", "D1", "D5", "D6"):
                assert u["reason"] == "Người dùng đã xóa/không chọn đo kích thước này"

        print(">>> USER RECIPE ADHERENCE & DELETION TEST PASSED: Only chosen dimensions measured & drawn! <<<")
    finally:
        m.caliper_recipes = original_recipes


if __name__ == "__main__":
    test_caliper_measurements()
    test_user_custom_recipe_adherence()
