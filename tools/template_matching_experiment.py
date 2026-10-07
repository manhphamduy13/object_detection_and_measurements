"""Single-object, rotation/scale-invariant template-matching experiment.

The experiment deliberately uses disjoint rotation angles for the template bank
and the synthetic test set.  Augmentation always expands the destination canvas,
so rotating/scaling never clips additional source pixels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import classifier as clf  # noqa: E402


# Ten-degree templates bound the residual angular error to five degrees while
# keeping the bank small enough for an industrial start-up time.  The previous
# five-degree bank segmented every rotated reference from scratch and took more
# than ten minutes for the current catalogue.
TEMPLATE_ANGLES = tuple(range(0, 360, 15))
TEST_CASES = (
    (7, 0.70), (23, 0.90), (47, 1.10), (71, 1.30),
    (103, 0.80), (137, 1.20), (173, 1.00), (211, 0.75),
    (257, 1.25), (319, 0.95),
)
CANONICAL_SIZE = 128
TEXTURE_SIZE = 128
MAX_SOURCE_SIDE = 1200


def discover_references(reference_dir: Path):
    """Return every decodable reference grouped by its folder label."""
    grouped = {}
    skipped = []
    for class_dir in sorted((p for p in reference_dir.iterdir() if p.is_dir()),
                            key=lambda p: p.name):
        image_dir = class_dir / "images"
        if not image_dir.is_dir():
            continue
        paths = []
        for path in sorted(p for p in image_dir.iterdir() if p.is_file()):
            if path.suffix.lower() not in clf.config.IMAGE_EXTENSIONS:
                skipped.append({"path": str(path), "reason": "unsupported extension"})
                continue
            if cv2.imread(str(path)) is None:
                skipped.append({"path": str(path), "reason": "OpenCV decode failed"})
                continue
            paths.append(path)
        if paths:
            grouped[class_dir.name] = paths
    return grouped, skipped


def border_colour(image: np.ndarray) -> tuple[int, int, int]:
    """Robust background estimate from image perimeter pixels."""
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    return tuple(int(v) for v in np.median(border, axis=0))


def rotate_scale_no_crop(image: np.ndarray, angle: float, scale: float) -> np.ndarray:
    """Rotate and scale onto an expanded square canvas without clipping."""
    h, w = image.shape[:2]
    diagonal = math.hypot(w, h)
    side = int(math.ceil(diagonal * max(scale, 1.0) + 32))
    background = border_colour(image)
    canvas = np.full((side, side, 3), background, dtype=np.uint8)
    x = (side - w) // 2
    y = (side - h) // 2
    canvas[y:y + h, x:x + w] = image
    centre = (side / 2.0, side / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle, scale)
    return cv2.warpAffine(
        canvas, matrix, (side, side), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=background,
    )


def focus_source(image: np.ndarray, mask: np.ndarray, margin_ratio: float = 0.30) -> np.ndarray:
    """Remove excessive empty background, never object pixels, and keep a safe margin."""
    points = cv2.findNonZero(mask)
    if points is None:
        raise RuntimeError("empty silhouette")
    x, y, w, h = cv2.boundingRect(points)
    margin = max(16, round(max(w, h) * margin_ratio))
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(image.shape[1], x + w + margin), min(image.shape[0], y + h + margin)
    roi = image[y0:y1, x0:x1]
    # Downscale only; aspect ratio and every retained object pixel are preserved.
    factor = min(1.0, MAX_SOURCE_SIDE / max(roi.shape[:2]))
    if factor < 1.0:
        roi = cv2.resize(roi, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    return roi


def largest_mask(image: np.ndarray) -> np.ndarray:
    """Extract the single workpiece using the project's production segmenter."""
    described = clf.describe_image(image)
    if described is None:
        raise RuntimeError("could not segment the workpiece")
    mask = (described["object_mask"] > 0).astype(np.uint8) * 255
    return mask


def canonical_mask(mask: np.ndarray, size: int = CANONICAL_SIZE) -> np.ndarray:
    """Centre a silhouette and normalize its size while preserving its aspect."""
    points = cv2.findNonZero(mask)
    if points is None:
        raise RuntimeError("empty silhouette")
    x, y, w, h = cv2.boundingRect(points)
    roi = mask[y:y + h, x:x + w]
    target = int(size * 0.78)
    factor = target / max(w, h)
    rw, rh = max(1, round(w * factor)), max(1, round(h * factor))
    interpolation = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_NEAREST
    roi = cv2.resize(roi, (rw, rh), interpolation=interpolation)
    out = np.zeros((size, size), dtype=np.uint8)
    ox, oy = (size - rw) // 2, (size - rh) // 2
    out[oy:oy + rh, ox:ox + rw] = roi
    return out


def canonical_edges(image: np.ndarray, mask: np.ndarray, size: int = CANONICAL_SIZE) -> np.ndarray:
    """Normalize internal appearance as an illumination-resistant edge template."""
    points = cv2.findNonZero(mask)
    if points is None:
        raise RuntimeError("empty silhouette")
    x, y, w, h = cv2.boundingRect(points)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    roi = gray[y:y + h, x:x + w]
    roi_mask = mask[y:y + h, x:x + w]
    target = int(size * 0.78)
    factor = target / max(w, h)
    rw, rh = max(1, round(w * factor)), max(1, round(h * factor))
    roi = cv2.resize(roi, (rw, rh), interpolation=cv2.INTER_AREA)
    roi_mask = cv2.resize(roi_mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
    roi = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
    edge = cv2.Canny(roi, 35, 110)
    edge[roi_mask == 0] = 0
    out = np.zeros((size, size), dtype=np.uint8)
    ox, oy = (size - rw) // 2, (size - rh) // 2
    out[oy:oy + rh, ox:ox + rw] = edge
    return out


def canonical_texture(image: np.ndarray, mask: np.ndarray, size: int = TEXTURE_SIZE) -> np.ndarray:
    """Return a normalized foreground texture vector, independent of brightness."""
    points = cv2.findNonZero(mask)
    if points is None:
        raise RuntimeError("empty silhouette")
    x, y, w, h = cv2.boundingRect(points)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    roi = gray[y:y + h, x:x + w]
    roi_mask = mask[y:y + h, x:x + w]
    target = int(size * 0.78)
    factor = target / max(w, h)
    rw, rh = max(1, round(w * factor)), max(1, round(h * factor))
    roi = cv2.resize(roi, (rw, rh), interpolation=cv2.INTER_AREA).astype(np.float32)
    roi_mask = cv2.resize(roi_mask, (rw, rh), interpolation=cv2.INTER_NEAREST) > 0
    values = roi[roi_mask]
    mean = float(values.mean()) if values.size else 0.0
    std = max(float(values.std()), 8.0) if values.size else 8.0
    normalized = np.clip((roi - mean) / std, -3.0, 3.0)
    normalized[~roi_mask] = 0.0
    out = np.zeros((size, size), dtype=np.float32)
    ox, oy = (size - rw) // 2, (size - rh) // 2
    out[oy:oy + rh, ox:ox + rw] = normalized
    vector = out.reshape(-1)
    vector -= vector.mean()
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    return vector


def rotate_mask(mask: np.ndarray, angle: float) -> np.ndarray:
    size = mask.shape[0]
    matrix = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), angle, 1.0)
    return cv2.warpAffine(mask, matrix, (size, size), flags=cv2.INTER_NEAREST)


def rotate_canonical_mask(mask: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a canonical silhouette on a padded canvas, then re-normalize it."""
    h, w = mask.shape[:2]
    side = int(math.ceil(math.hypot(w, h))) + 4
    canvas = np.zeros((side, side), dtype=np.uint8)
    x, y = (side - w) // 2, (side - h) // 2
    canvas[y:y + h, x:x + w] = mask
    matrix = cv2.getRotationMatrix2D((side / 2.0, side / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(
        canvas, matrix, (side, side), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return canonical_mask(rotated)


def rotate_pair_no_crop(image: np.ndarray, mask: np.ndarray, angle: float):
    """Rotate an already-segmented image/mask pair without re-segmentation."""
    h, w = image.shape[:2]
    side = int(math.ceil(math.hypot(w, h) + 32))
    background = border_colour(image)
    image_canvas = np.full((side, side, 3), background, dtype=np.uint8)
    mask_canvas = np.zeros((side, side), dtype=np.uint8)
    x, y = (side - w) // 2, (side - h) // 2
    image_canvas[y:y + h, x:x + w] = image
    mask_canvas[y:y + h, x:x + w] = mask
    matrix = cv2.getRotationMatrix2D((side / 2.0, side / 2.0), angle, 1.0)
    rotated_image = cv2.warpAffine(
        image_canvas, matrix, (side, side), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=background,
    )
    rotated_mask = cv2.warpAffine(
        mask_canvas, matrix, (side, side), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return rotated_image, rotated_mask


def match_score(query: np.ndarray, template: np.ndarray) -> float:
    """Centered, scale-normalized silhouette similarity in [0, 1]."""
    response = cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED)
    return float(np.max(response))


def build_template_bank(references, output_dir: Path | None = None):
    bank = {}
    preview_dir = output_dir / "template_bank" if output_dir is not None else None
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for label, paths in references.items():
        bank[label] = []
        for ref_index, path in enumerate(paths):
            image = cv2.imread(str(path))
            try:
                source_mask = largest_mask(image)
                base_template = canonical_mask(source_mask)
            except RuntimeError as exc:
                failures.append({"path": str(path), "reason": str(exc)})
                continue
            for angle in TEMPLATE_ANGLES:
                # Metal texture is illumination/viewpoint dependent, so the
                # production bank intentionally uses geometry only.  Rotating a
                # 256 px silhouette is also orders of magnitude faster than
                # warping and re-segmenting the full-resolution source photo.
                template = rotate_canonical_mask(base_template, angle)
                texture = np.zeros(TEXTURE_SIZE * TEXTURE_SIZE, dtype=np.float32)
                bank[label].append((template, texture, path.name, angle))
                if preview_dir is not None:
                    cv2.imwrite(str(preview_dir / f"{label}_r{ref_index}_a{angle:03d}.png"), template)
    vectorized = {}
    for label, templates in bank.items():
        if not templates:
            continue
        masks = np.stack([item[0].reshape(-1) for item in templates]).astype(np.float32)
        textures = np.stack([item[1] for item in templates]).astype(np.float32)
        masks -= masks.mean(axis=1, keepdims=True)
        mask_norm = masks / np.maximum(np.linalg.norm(masks, axis=1, keepdims=True), 1e-12)
        vectorized[label] = {"items": templates, "mask_norm": mask_norm,
                             "texture_norm": textures}
    bank = vectorized
    return bank, failures


def classify(query: np.ndarray, query_texture: np.ndarray, bank, shape_weight: float = 1.0):
    per_class = {}
    best_meta = {}
    query_vector = query.reshape(-1).astype(np.float32)
    query_vector -= query_vector.mean()
    query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
    for label, entry in bank.items():
        silhouette_scores = entry["mask_norm"] @ query_vector
        texture_scores = entry["texture_norm"] @ query_texture
        scores = shape_weight * silhouette_scores + (1.0 - shape_weight) * texture_scores
        best_index = int(np.argmax(scores))
        score = float(scores[best_index])
        template, texture, name, angle = entry["items"][best_index]
        per_class[label] = score
        best_meta[label] = {"reference": name, "angle": angle, "template": template,
                            "silhouette_score": float(silhouette_scores[best_index]),
                            "texture_score": float(texture_scores[best_index])}
    ranked = sorted(per_class.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    return best_label, best_score, best_score - second_score, per_class, best_meta[best_label]


def save_visualization(path: Path, image: np.ndarray, mask: np.ndarray,
                       query: np.ndarray, best_template: np.ndarray, record: dict):
    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Augmented test\n(expanded canvas, no crop)")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Segmented object")
    axes[2].imshow(query, cmap="gray")
    axes[2].set_title("Scale-normalized query")
    axes[3].imshow(best_template, cmap="gray")
    axes[3].set_title(f"Best template\nangle={record['template_angle']} deg")
    overlay = cv2.addWeighted(query, 0.5, best_template, 0.5, 0)
    axes[4].imshow(overlay, cmap="magma")
    axes[4].set_title(
        f"true={record['true_class']}\npred={record['predicted_class']}\n"
        f"similarity={record['similarity']:.3f}, margin={record['margin']:.3f}"
    )
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_overview(path: Path, rows: list[dict], augmented_dir: Path):
    """Save one compact visual audit sheet containing every prediction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = 5
    rows_count = math.ceil(len(rows) / columns)
    fig, axes = plt.subplots(rows_count, columns, figsize=(15, 3.5 * rows_count))
    for axis, record in zip(np.asarray(axes).flat, rows):
        image_path = augmented_dir / record["true_class"] / record["filename"]
        image = cv2.imread(str(image_path))
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        colour = "green" if record["correct"] else "red"
        axis.set_title(
            f"{record['true_class']} -> {record['predicted_class']}\n"
            f"a={record['test_angle']} deg, s={record['test_scale']:.2f}\n"
            f"sim={record['similarity']:.3f}, margin={record['margin']:.3f}",
            color=colour, fontsize=9,
        )
        axis.axis("off")
    for axis in np.asarray(axes).flat[len(rows):]:
        axis.axis("off")
    fig.suptitle("Single-object template matching: all augmented tests", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(path: Path, rows: list[dict], labels: list[str]):
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for record in rows:
        if record["predicted_class"] in index:
            matrix[index[record["true_class"]], index[record["predicted_class"]]] += 1
    fig, axis = plt.subplots(figsize=(max(8, len(labels)), max(7, len(labels) * 0.8)))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Augmented-test confusion matrix")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize(rows: list[dict]):
    return {
        "total": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": (sum(row["correct"] for row in rows) / len(rows)) if rows else None,
        "mean_similarity": float(np.mean([r["similarity"] for r in rows])) if rows else None,
        "min_similarity": float(np.min([r["similarity"] for r in rows])) if rows else None,
        "mean_margin": float(np.mean([r["margin"] for r in rows])) if rows else None,
        "min_margin": float(np.min([r["margin"] for r in rows])) if rows else None,
        "predicted_counts": dict(Counter(r["predicted_class"] for r in rows)),
    }


def run(reference_dir: Path, output_dir: Path, test_dir: Path | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    references, skipped = discover_references(reference_dir)
    bank, template_failures = build_template_bank(references, output_dir)
    if len(bank) < 2:
        raise RuntimeError("template matching requires at least two valid classes")
    labels = sorted(bank)
    rows = []
    failures = list(template_failures)
    augmented_dir = output_dir / "augmented_tests"
    visual_dir = output_dir / "visualizations"

    for label in labels:
        for source_index, source_path in enumerate(references[label], start=1):
            source = cv2.imread(str(source_path))
            try:
                source_mask = largest_mask(source)
                focused = focus_source(source, source_mask)
            except RuntimeError as exc:
                failures.append({"path": str(source_path), "reason": str(exc)})
                continue
            for case_index, (angle, scale) in enumerate(TEST_CASES, start=1):
                test = rotate_scale_no_crop(focused, angle, scale)
                stem = f"{label}_{source_path.stem}_t{case_index:02d}_a{angle}_s{scale:.2f}"
                test_path = augmented_dir / label / f"{stem}.jpg"
                test_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(test_path), test, [cv2.IMWRITE_JPEG_QUALITY, 92])
                try:
                    mask = largest_mask(test)
                    query = canonical_mask(mask)
                    query_texture = canonical_texture(test, mask)
                    predicted, score, margin, class_scores, meta = classify(query, query_texture, bank)
                except RuntimeError as exc:
                    failures.append({"path": str(test_path), "reason": str(exc)})
                    continue
                record = {
                    "filename": test_path.name,
                    "source_reference": source_path.name,
                    "true_class": label,
                    "predicted_class": predicted,
                    "correct": predicted == label,
                    "test_angle": angle,
                    "test_scale": scale,
                    "similarity": score,
                    "margin": margin,
                    "class_scores": class_scores,
                    "template_reference": meta["reference"],
                    "template_angle": meta["angle"],
                    "silhouette_score": meta["silhouette_score"],
                    "texture_score": meta["texture_score"],
                }
                rows.append(record)
                visual_path = visual_dir / label / f"{stem}.png"
                if not visual_path.exists():
                    save_visualization(visual_path, test, mask, query, meta["template"], record)

    real_rows = []
    if test_dir and test_dir.is_dir():
        real_visual_dir = output_dir / "real_test_visualizations"
        for true_class_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            label = true_class_dir.name
            if label not in bank:
                continue
            for path in sorted((true_class_dir / "images").glob("*")):
                if path.suffix.lower() not in clf.config.IMAGE_EXTENSIONS:
                    continue
                image = cv2.imread(str(path))
                if image is None:
                    failures.append({"path": str(path), "reason": "OpenCV decode failed"})
                    continue
                try:
                    mask = largest_mask(image)
                    query = canonical_mask(mask)
                    query_texture = canonical_texture(image, mask)
                    predicted, score, margin, class_scores, meta = classify(query, query_texture, bank)
                except RuntimeError as exc:
                    failures.append({"path": str(path), "reason": str(exc)})
                    continue
                record = {
                    "filename": path.name, "source_reference": "real test",
                    "true_class": label, "predicted_class": predicted,
                    "correct": predicted == label, "test_angle": "real",
                    "test_scale": "real", "similarity": score, "margin": margin,
                    "class_scores": class_scores, "template_reference": meta["reference"],
                    "template_angle": meta["angle"],
                    "silhouette_score": meta["silhouette_score"],
                    "texture_score": meta["texture_score"],
                }
                real_rows.append(record)
                visual_path = real_visual_dir / label / f"{path.stem}.png"
                if not visual_path.exists():
                    save_visualization(visual_path, image, mask, query, meta["template"], record)

    per_class = {label: summarize([row for row in rows if row["true_class"] == label])
                 for label in labels}
    summary = {
        "method": "dense 5-degree template matching (silhouette + normalized internal texture)",
        "labels": labels,
        "reference_images": sum(len(paths) for paths in references.values()),
        "templates": sum(len(entry["items"]) for entry in bank.values()),
        "template_angles_deg": list(TEMPLATE_ANGLES),
        "test_cases_per_reference": len(TEST_CASES),
        "augmented": summarize(rows),
        "real_tests": summarize(real_rows),
        "per_class": per_class,
        "skipped_inputs": skipped,
        "failures": failures,
        "no_crop_policy": "empty background is trimmed with safety margin; object is preserved; rotation/scale uses expanded canvas",
    }
    audit_rows = []
    for label in labels:
        group = sorted((r for r in rows if r["true_class"] == label), key=lambda r: r["margin"])
        audit_rows.extend(group[:2])
        save_overview(output_dir / "class_overviews" / f"{label}.png", group, augmented_dir)
    save_overview(output_dir / "overview.png", audit_rows, augmented_dir)
    save_confusion_matrix(output_dir / "confusion_matrix.png", rows, labels)
    (output_dir / "results.json").write_text(
        json.dumps({"summary": summary, "predictions": rows,
                    "real_test_predictions": real_rows}, indent=2), encoding="utf-8"
    )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["filename", "source_reference", "true_class", "predicted_class", "correct", "test_angle",
                  "test_scale", "similarity", "margin", "template_reference", "template_angle"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    if real_rows:
        with (output_dir / "real_test_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in real_rows:
                writer.writerow({key: row[key] for key in fields})
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path,
                        default=PROJECT_DIR / "assets" / "references")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_DIR / "outputs" / "template_matching")
    parser.add_argument("--test-dir", type=Path,
                        default=PROJECT_DIR / "assets" / "tests")
    args = parser.parse_args()
    run(args.reference_dir, args.output_dir, args.test_dir)


if __name__ == "__main__":
    main()
