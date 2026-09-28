"""
Metrology Subpixel Caliper Test.
Validates caliper edge extraction and tolerance verifications.
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
    sample_img_path = Path(config.TEST_DIR) / "98003P" / "images" / "98003P_1_rot23.png"
    assert sample_img_path.exists(), f"Sample not found: {sample_img_path}"

    img = cv2.imread(str(sample_img_path))
    described = clf.describe_image(img)
    assert described is not None, "Segmentation failed"

    library, contours, _ = clf.build_database(config.REFERENCE_DIR)
    ref_items = clf.load_dataset(config.REFERENCE_DIR)
    thresholds, _ = clf.calibrate_thresholds(ref_items, library, contours)
    result = clf.predict(img, library, contours, thresholds)
    assert result["class"] == "98003P", f"Predicted {result['class']}, expected 98003P"

    meas = measurer.WorkpieceMeasurer().measure(img, described["contour"], result.get("spec"))
    assert meas["Length_L_px"] > 600, "Length measurement abnormal"
    assert meas["Width_W_px"] > 500, "Width measurement abnormal"
    print(f"Caliper measurement test passed: L={meas['Length_L_px']:.2f}px, W={meas['Width_W_px']:.2f}px")
    print(">>> ALL METROLOGY TESTS PASSED <<<")

if __name__ == "__main__":
    test_caliper_measurements()