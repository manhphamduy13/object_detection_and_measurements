"""Batch visual audit for closed contours detected inside workpieces."""

import argparse
from pathlib import Path
import sys

import cv2

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import classifier as clf
from src.measurer import WorkpieceMeasurer


def main():
    parser = argparse.ArgumentParser(description="Audit inner-contour detection on multiple SKUs")
    parser.add_argument(
        "--input-dir", default=str(PROJECT_DIR / "assets" / "references"),
        help="Reference/test root containing <SKU>/images",
    )
    parser.add_argument(
        "--output-dir", default=str(PROJECT_DIR / "outputs" / "inner_contour_batch"),
        help="Directory for audit PNGs",
    )
    parser.add_argument("--max-per-sku", type=int, default=1)
    parser.add_argument("--max-side", type=int, default=1600, help="Resize audit image only")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    total = segmented = with_inner = 0

    for sku_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        image_dir = sku_dir / "images"
        images = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )[:max(args.max_per_sku, 0)] if image_dir.exists() else []
        for path in images:
            total += 1
            image = cv2.imread(str(path))
            described = clf.describe_image(image) if image is not None else None
            if described is None:
                print(f"{sku_dir.name}: {path.name}: segmentation failed")
                continue
            segmented += 1
            inner_contours = WorkpieceMeasurer.detect_inner_contours(
                described["gray"], described["contour"], image_bgr=image
            )
            with_inner += int(bool(inner_contours))

            vis = image.copy()
            cv2.drawContours(vis, [described["contour"]], -1, (0, 255, 0), 2, cv2.LINE_AA)
            for index, contour in enumerate(inner_contours, start=1):
                cv2.drawContours(vis, [contour], -1, (255, 210, 0), 2, cv2.LINE_AA)
                moments = cv2.moments(contour)
                if abs(moments["m00"]) > 1e-6:
                    cx = int(round(moments["m10"] / moments["m00"]))
                    cy = int(round(moments["m01"] / moments["m00"]))
                    cv2.putText(vis, f"I{index}", (cx + 4, cy - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(vis, f"I{index}", (cx + 4, cy - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 210, 0), 1, cv2.LINE_AA)

            title = f"{sku_dir.name}: closed inner contours={len(inner_contours)}"
            cv2.putText(vis, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2, cv2.LINE_AA)
            output_path = output_dir / f"{sku_dir.name}_{path.stem}_inner_edges.png"
            if args.max_side > 0 and max(vis.shape[:2]) > args.max_side:
                scale = args.max_side / float(max(vis.shape[:2]))
                vis = cv2.resize(
                    vis,
                    (int(round(vis.shape[1] * scale)), int(round(vis.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imwrite(str(output_path), vis)
            print(f"{sku_dir.name}: {path.name}: {len(inner_contours)} -> {output_path.name}")

    print(
        f"inner contour batch: total={total}, segmented={segmented}, "
        f"with_candidates={with_inner}, output={output_dir}"
    )


if __name__ == "__main__":
    main()
