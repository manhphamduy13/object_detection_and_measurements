"""Preview an inner-hole/slot caliper without changing the saved recipe."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import classifier as clf
from src.measurer import WorkpieceMeasurer


def parse_point(value: str):
    try:
        x, y = value.split(",", 1)
        return float(x), float(y)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("point must use x,y format") from exc


def main():
    parser = argparse.ArgumentParser(description="Preview an internal contour measurement")
    parser.add_argument("--image", required=True, help="Input workpiece image")
    parser.add_argument("--p1", required=True, type=parse_point, help="First inner edge, x,y")
    parser.add_argument("--p2", required=True, type=parse_point, help="Opposite inner edge, x,y")
    parser.add_argument("--scale", type=float, default=0.0, help="Optional mm/px")
    parser.add_argument("--show-all", action="store_true", help="Draw every detected candidate")
    parser.add_argument(
        "--output",
        default=str(PROJECT_DIR / "outputs" / "inner_contour_validation.png"),
        help="Output visualization",
    )
    args = parser.parse_args()

    image = cv2.imread(str(Path(args.image)))
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")
    described = clf.describe_image(image)
    if described is None:
        raise SystemExit("Could not segment the outer workpiece contour")

    inner_contours = WorkpieceMeasurer.detect_inner_contours(
        described["gray"], described["contour"], image_bgr=image
    )
    q1, q2, selected_inner = WorkpieceMeasurer.snap_scanline_to_inner_contour(
        described["gray"], described["contour"], args.p1, args.p2, image_bgr=image
    )
    length_px = float(np.hypot(q2[0] - q1[0], q2[1] - q1[1]))
    length_mm = length_px * args.scale if args.scale > 0 else None

    vis = image.copy()
    cv2.drawContours(vis, [described["contour"]], -1, (0, 255, 0), 2, cv2.LINE_AA)
    if args.show_all and inner_contours:
        cv2.drawContours(vis, inner_contours, -1, (255, 210, 0), 2, cv2.LINE_AA)
    if selected_inner is not None:
        cv2.drawContours(vis, [selected_inner], -1, (210, 80, 230), 3, cv2.LINE_AA)
    a = tuple(int(round(v)) for v in q1)
    b = tuple(int(round(v)) for v in q2)
    cv2.line(vis, a, b, (210, 80, 230), 3, cv2.LINE_AA)
    cv2.circle(vis, a, 6, (210, 80, 230), -1, cv2.LINE_AA)
    cv2.circle(vis, b, 6, (210, 80, 230), -1, cv2.LINE_AA)
    label = f"INNER: {length_px:.2f}px"
    if length_mm is not None:
        label += f" / {length_mm:.3f}mm"
    midpoint = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
    cv2.putText(vis, label, (midpoint[0] + 8, midpoint[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, label, (midpoint[0] + 8, midpoint[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 80, 230), 1, cv2.LINE_AA)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), vis)
    value = f"{length_px:.3f}px"
    if length_mm is not None:
        value += f" ({length_mm:.4f}mm)"
    print(
        f"inner contours={len(inner_contours)}, selected={'yes' if selected_inner is not None else 'fallback'}, "
        f"p1={q1}, p2={q2}, length={value}, output={output}"
    )


if __name__ == "__main__":
    main()
