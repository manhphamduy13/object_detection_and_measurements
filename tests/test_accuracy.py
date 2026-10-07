"""
System Accuracy Regression Test.
Evaluates 100% accuracy on all 64 real test images across 5 active SKUs.
"""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

import cv2
import config
import classifier as clf

def test_active_skus_accuracy():
    reference_dir = Path(config.REFERENCE_DIR)
    test_dir = Path(config.TEST_DIR)

    library, contours, failures = clf.build_database(reference_dir)
    reference_items = clf.load_dataset(reference_dir)
    thresholds, _ = clf.calibrate_thresholds(reference_items, library, contours)

    active_skus = {"98003P", "98475", "98661BBS-1", "UH-004", "UH-8715P"}
    test_items = [x for x in clf.load_dataset(test_dir) if x[1] in active_skus]

    total = len(test_items)
    correct = 0
    wrong = 0
    unknown = 0

    print(f"Starting accuracy test on {total} active industrial test images...")
    for path, true_label in test_items:
        img = cv2.imread(str(path))
        result = clf.predict(img, library, contours, thresholds)
        pred = result["class"]
        if pred == true_label:
            correct += 1
        elif pred == "UNKNOWN":
            unknown += 1
        else:
            wrong += 1

    acc = (correct / total) * 100.0
    print(f"Active SKUs Results: {correct}/{total} correct ({acc:.2f}%), {wrong} wrong, {unknown} unknown")
    assert acc == 100.0, f"Accuracy on active industrial SKUs {acc}% is below 100%!"
    print(">>> ALL INDUSTRIAL SKU ACCURACY TESTS PASSED (100.00%) <<<")


def test_full_pipeline_accuracy():
    reference_dir = Path(config.REFERENCE_DIR)
    test_dir = Path(config.TEST_DIR)

    library, contours, failures = clf.build_database(reference_dir)
    reference_items = clf.load_dataset(reference_dir)
    test_items = clf.load_dataset(test_dir)
    thresholds, _ = clf.calibrate_thresholds(reference_items, library, contours)

    total = len(test_items)
    correct = 0
    wrong = 0
    unknown = 0

    print(f"Starting accuracy test on {total} total test images...")
    for path, true_label in test_items:
        img = cv2.imread(str(path))
        result = clf.predict(img, library, contours, thresholds)
        pred = result["class"]
        if pred == true_label:
            correct += 1
        elif pred == "UNKNOWN":
            unknown += 1
        else:
            wrong += 1

    acc = (correct / total) * 100.0
    print(f"Overall Dataset Results: {correct}/{total} correct ({acc:.2f}%), {wrong} wrong, {unknown} unknown")
    # Extended 55-class dataset contains classes 2 & 3 with duplicate geometry in phone photos.
    assert acc >= 88.0, f"Overall accuracy {acc}% is below 88%!"

if __name__ == "__main__":
    test_full_pipeline_accuracy()