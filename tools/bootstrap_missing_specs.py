"""Create clearly-marked provisional L/W specs for references without a PDF.

These specs are intended for rotation-repeatability testing only.  They do not
claim traceable physical dimensions: a fixed provisional scale is applied to
the median reference silhouette and the provenance is recorded in each file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src import classifier as clf  # noqa: E402


REFERENCE_DIR = PROJECT_DIR / "assets" / "references"
PRODUCTS_PATH = PROJECT_DIR / "configs" / "products.json"
PROVISIONAL_SCALE_MM_PER_PX = 0.05


def dimension(label: str, nominal: float) -> dict:
    # These are synthetic smoke-test limits, not supplier tolerances.  Five
    # percent covers rasterisation/OBB variation on small or concave objects
    # while still exposing gross segmentation failures.
    tolerance = max(0.15, nominal * 0.05)
    return {
        "raw_spec": f"PROVISIONAL {label} {nominal:.3f} +/- {tolerance:.3f} mm",
        "label": label,
        "nominal": round(nominal, 3),
        "diameter": False,
        "upper_tol": round(tolerance, 3),
        "lower_tol": round(-tolerance, 3),
        "min": round(nominal - tolerance, 3),
        "max": round(nominal + tolerance, 3),
        "tolerance_type": "provisional_symmetric",
        "axis": "2D_in_plane",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-provisional", action="store_true",
        help="Recompute existing provisional specs after segmentation changes.",
    )
    args = parser.parse_args()
    products = json.loads(PRODUCTS_PATH.read_text(encoding="utf-8")) if PRODUCTS_PATH.exists() else {}
    created = []
    skipped = []
    for sku_dir in sorted(path for path in REFERENCE_DIR.iterdir() if path.is_dir()):
        spec_path = sku_dir / "spec.json"
        if spec_path.exists():
            existing = json.loads(spec_path.read_text(encoding="utf-8"))
            is_provisional = (
                existing.get("review_status") == "PROVISIONAL_NOT_FOR_PRODUCTION_RELEASE"
            )
            if not (args.refresh_provisional and is_provisional):
                products[sku_dir.name] = existing
                continue
        image_dir = sku_dir / "images"
        image_paths = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in clf.config.IMAGE_EXTENSIONS
        ) if image_dir.exists() else []
        dimensions_px = []
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            described = clf.describe_image(image) if image is not None else None
            if described is None:
                continue
            (_, _), (width, height), _ = cv2.minAreaRect(described["contour"])
            dimensions_px.append((max(float(width), float(height)), min(float(width), float(height))))
        if not dimensions_px:
            skipped.append({"sku": sku_dir.name, "reason": "no segmentable reference"})
            continue
        long_px = float(np.median([item[0] for item in dimensions_px]))
        short_px = float(np.median([item[1] for item in dimensions_px]))
        length_mm = long_px * PROVISIONAL_SCALE_MM_PER_PX
        width_mm = short_px * PROVISIONAL_SCALE_MM_PER_PX
        spec = {
            "sku": sku_dir.name,
            "source_file": "generated_from_reference_silhouette",
            "item_name": sku_dir.name,
            "part_no": sku_dir.name,
            "description": "Provisional rotation-repeatability specification",
            "material": "unknown",
            "unit": "mm (provisional)",
            "review_status": "PROVISIONAL_NOT_FOR_PRODUCTION_RELEASE",
            "dimensions": [dimension("L", length_mm), dimension("W", width_mm)],
            "optical_scale_mm_per_px": PROVISIONAL_SCALE_MM_PER_PX,
            "golden_sample": {
                "filenames": [path.name for path in image_paths],
                "median_obb_long_px": round(long_px, 3),
                "median_obb_short_px": round(short_px, 3),
                "sample_count": len(dimensions_px),
            },
        }
        spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
        products[sku_dir.name] = spec
        created.append(sku_dir.name)

    PRODUCTS_PATH.write_text(json.dumps(products, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"provisional specs: created={len(created)}, skipped={len(skipped)}")
    print("created:", ", ".join(created))
    if skipped:
        print("skipped:", json.dumps(skipped, ensure_ascii=False))


if __name__ == "__main__":
    main()
