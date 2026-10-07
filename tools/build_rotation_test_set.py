"""Build a lossless rotation-only test set from reference images.

The output canvas is expanded to the source diagonal before rotation, so no
workpiece pixel is clipped.  The affine scale is always exactly 1.0 and PNG is
used to avoid adding JPEG edge artefacts to the benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src import classifier as clf  # noqa: E402
DEFAULT_ANGLES = (17, 43, 79, 137, 223, 311)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def border_colour(image: np.ndarray) -> tuple[int, int, int]:
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    return tuple(int(value) for value in np.median(border, axis=0))


def rotate_without_crop(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    side = int(math.ceil(math.hypot(width, height))) + 20
    if side % 2:
        side += 1
    matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0), float(angle), 1.0
    )
    matrix[0, 2] += (side - width) / 2.0
    matrix[1, 2] += (side - height) / 2.0
    background = border_colour(image)
    return cv2.warpAffine(
        image,
        matrix,
        (side, side),
        flags=cv2.INTER_LINEAR,
        # Replicating a clipped foreground pixel creates long artificial
        # streaks after rotation and may connect two otherwise separate
        # objects.  A robust border-colour estimate pads only the new canvas;
        # object pixels and affine scale remain unchanged.
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background,
    )


def focus_on_workpiece(image: np.ndarray):
    """Crop background only, retaining the full segmented object and margin."""
    described = clf.describe_image(image)
    if described is None:
        return image, [0, 0, image.shape[1], image.shape[0]], "segmentation_failed"
    x, y, width, height = cv2.boundingRect(described["contour"])
    object_side = max(width, height)
    margin = max(32, int(round(object_side * 0.35)))

    # Textured paper/large shadows need more context for a reliable perimeter
    # background model. Expand until the ROI border is reasonably uniform or
    # the source limits are reached; the workpiece pixel scale stays unchanged.
    while True:
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1 = min(image.shape[1], x + width + margin)
        y1 = min(image.shape[0], y + height + margin)
        candidate = image[y0:y1, x0:x1]
        gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
        band = max(4, int(round(min(gray.shape[:2]) * 0.02)))
        border = np.concatenate((
            gray[:band].ravel(), gray[-band:].ravel(),
            gray[:, :band].ravel(), gray[:, -band:].ravel(),
        )).astype(np.float32)
        border_mad = float(np.median(np.abs(border - np.median(border))))
        if border_mad <= 5.0 or margin >= int(round(object_side * 1.25)):
            break
        next_margin = min(int(round(object_side * 1.25)), int(round(margin * 1.6)))
        if next_margin <= margin:
            break
        margin = next_margin
    focused = image[y0:y1, x0:x1].copy()
    focused_mask = described["object_mask"][y0:y1, x0:x1]
    background_pixels = focused[focused_mask == 0]
    background = tuple(
        int(value) for value in np.median(background_pixels, axis=0)
    ) if background_pixels.size else border_colour(focused)
    # Isolate the already-segmented golden workpiece before rotating it.  This
    # changes no foreground pixel and prevents paper boundaries, other partial
    # instances and shadows from becoming artificial rotation-dependent edges.
    focused[focused_mask == 0] = background
    touches_source = (
        x <= 2 or y <= 2
        or x + width >= image.shape[1] - 2
        or y + height >= image.shape[0] - 2
    )
    status = "segmented_roi_background_matted"
    if touches_source:
        # The source itself is tightly clipped.  Padding cannot reconstruct
        # missing material, but it prevents BORDER_REPLICATE from stretching a
        # metal strip/object edge across the augmented canvas.
        fh, fw = focused.shape[:2]
        padding = max(24, int(round(max(fh, fw) * 0.30)))
        focused = cv2.copyMakeBorder(
            focused, padding, padding, padding, padding,
            cv2.BORDER_CONSTANT, value=background,
        )
        status = "source_clipped_background_matted_and_padded"
    return focused, [x0, y0, x1, y1], status


def build(reference_dir: Path, test_dir: Path, angles: tuple[float, ...]) -> dict:
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    manifest = {
        "policy": "rotation_only_foreground_geometry_no_scale_lossless",
        "affine_scale": 1.0,
        "angles_deg": list(angles),
        "images": [],
        "skipped": [],
    }
    for sku_dir in sorted(path for path in reference_dir.iterdir() if path.is_dir()):
        image_dir = sku_dir / "images"
        sources = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ) if image_dir.exists() else []
        if not sources:
            manifest["skipped"].append({"sku": sku_dir.name, "reason": "no reference image"})
            continue
        output_dir = test_dir / sku_dir.name / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        for source in sources:
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                manifest["skipped"].append({"sku": sku_dir.name, "source": source.name, "reason": "decode failed"})
                continue
            focused, source_roi, focus_status = focus_on_workpiece(image)
            for angle in angles:
                rotated = rotate_without_crop(focused, angle)
                angle_token = f"{angle:g}".replace("-", "m").replace(".", "p")
                destination = output_dir / f"{source.stem}_rot{angle_token}.png"
                if not cv2.imwrite(str(destination), rotated, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    raise RuntimeError(f"Could not write {destination}")
                manifest["images"].append({
                    "sku": sku_dir.name,
                    "source": source.name,
                    "output": str(destination.relative_to(PROJECT_DIR)),
                    "angle_deg": angle,
                    "affine_scale": 1.0,
                    "source_shape": list(image.shape[:2]),
                    "source_roi_xyxy": source_roi,
                    "focused_shape": list(focused.shape[:2]),
                    "focus_status": focus_status,
                    "output_shape": list(rotated.shape[:2]),
                })

    manifest_path = test_dir / "rotation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=PROJECT_DIR / "assets" / "references")
    parser.add_argument("--test-dir", type=Path, default=PROJECT_DIR / "assets" / "tests")
    parser.add_argument("--angles", type=float, nargs="+", default=DEFAULT_ANGLES)
    args = parser.parse_args()
    manifest = build(args.reference_dir, args.test_dir, tuple(args.angles))
    print(
        f"rotation test set: {len(manifest['images'])} images, "
        f"{len(manifest['skipped'])} skipped, output={args.test_dir}"
    )


if __name__ == "__main__":
    main()
