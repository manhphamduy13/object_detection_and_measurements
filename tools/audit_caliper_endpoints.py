"""Audit that every rendered outer-dimension endpoint lies on the workpiece edge."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import cv2


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src import classifier as clf  # noqa: E402
from src.measurer import WorkpieceMeasurer  # noqa: E402


OUTER_TYPES = {
    "taught_caliper", "length", "overall_length", "width", "gap",
    "step_length", "thickness", "outer_dimension",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check outer caliper endpoints against their detected contour."
    )
    parser.add_argument("--reference-dir", default="assets/references")
    parser.add_argument("--test-dir", default="assets/tests")
    parser.add_argument(
        "--all", action="store_true",
        help="audit every test image (default: one representative per SKU)",
    )
    parser.add_argument("--tolerance-px", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    library, _, _ = clf.build_database(args.reference_dir)
    items = clf.load_dataset(args.test_dir)
    if not args.all:
        representatives = {}
        for path, label in items:
            representatives.setdefault(label, (path, label))
        items = list(representatives.values())

    audited_images = 0
    audited_endpoints = 0
    violations = []
    type_counts = Counter()
    measurer = WorkpieceMeasurer()

    for path, label in items:
        image = cv2.imread(str(path))
        described = clf.describe_image(image) if image is not None else None
        if described is None:
            violations.append((path.name, "SEGMENTATION", "-", float("inf")))
            continue
        spec = library.get(label, {}).get("spec")
        measured = measurer.measure(image, described["contour"], spec)
        audited_images += 1
        for caliper in measured.get("active_draw_calipers", []):
            measure_type = caliper.get("measure_type", "taught_caliper")
            type_counts[measure_type] += 1
            if measure_type in ("inner_contour", "hole", "slot"):
                target_cnt = caliper.get("inner_contour")
                if target_cnt is None:
                    violations.append(
                        (path.name, caliper.get("label", "?"), "inner_contour_missing", 999.0)
                    )
                    continue
            else:
                target_cnt = described["contour"]

            for endpoint_name in ("p1", "p2"):
                audited_endpoints += 1
                distance = abs(cv2.pointPolygonTest(
                    target_cnt, tuple(caliper[endpoint_name]), True
                ))
                if distance > args.tolerance_px:
                    violations.append(
                        (path.name, caliper.get("label", "?"), endpoint_name, distance)
                    )

    print(f"images={audited_images}/{len(items)}")
    print(f"outer_endpoints={audited_endpoints}")
    print(f"types={dict(type_counts)}")
    print(f"violations={len(violations)} tolerance_px={args.tolerance_px:.3f}")
    for filename, label, endpoint, distance in violations[:30]:
        print(f"  {filename}: {label}.{endpoint} distance={distance:.4f}px")
    raise SystemExit(1 if violations else 0)


if __name__ == "__main__":
    main()
