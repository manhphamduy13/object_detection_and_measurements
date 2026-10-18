"""Geometry shared by template teaching, inspection and visualization."""

import cv2
import numpy as np


INNER_COLOR = (255, 150, 40)  # BGR: blue
OUTER_COLOR = (0, 145, 255)  # BGR: orange
GOLDEN_COLOR = (160, 160, 160)


def profile_limits(recipe):
    """Read asymmetric per-edge offsets, including old symmetric recipes."""
    legacy = recipe.get("contour_tolerance_mm", 0.1)
    lower = float(recipe.get("contour_lower_mm", legacy))
    upper = float(recipe.get("contour_upper_mm", legacy))
    if not np.isfinite([lower, upper]).all() or min(lower, upper) < 0:
        raise ValueError("Contour offsets must be finite, non-negative numbers")
    return lower, upper


def golden_baseline(caliper, length_px, mm_per_px):
    """An actual measurement overrides image calibration only when supplied."""
    actual = caliper.get("golden_actual_mm")
    if actual not in (None, ""):
        value = float(actual)
        source = "actual"
    else:
        value = float(length_px) * float(mm_per_px)
        source = "image"
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Golden length must be finite and greater than zero")
    return value, source


def dense_boundary(contour, spacing=1.0):
    """Include every vertex and sample long polygon segments at <= spacing px."""
    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    pieces = []
    for start, end in zip(points, np.roll(points, -1, axis=0)):
        count = max(1, int(np.ceil(np.linalg.norm(end - start) / spacing)))
        pieces.append(start + np.arange(count)[:, None] / count * (end - start))
    return np.concatenate(pieces)


def check_profile(target, golden, scale, lower, upper):
    """Check target boundary in band AND containment of the required inner core.

    Positive means extra material, negative means missing material. All
    polygon vertices are checked, including narrow burrs; long segments are
    sampled every pixel. This checks the external silhouette, not inner holes.
    """
    target_cv = np.asarray(target, dtype=np.float32).reshape(-1, 1, 2)
    golden_cv = np.asarray(golden, dtype=np.float32).reshape(-1, 1, 2)
    target_points = dense_boundary(target)
    forward = np.array([
        -cv2.pointPolygonTest(golden_cv, (float(p[0]), float(p[1])), True)
        for p in target_points
    ]) * scale
    inner, outer = offset_contours(golden, lower / scale, upper / scale)
    core_points = np.concatenate([dense_boundary(c) for c in inner]) if inner else np.empty((0, 2))
    # Preview offsets are rasterized. Only samples confirmed to belong to the
    # exact polygon's eroded core may cause a missing-material failure.
    core_points = np.array([
        p for p in core_points
        if cv2.pointPolygonTest(golden_cv, (float(p[0]), float(p[1])), True) * scale >= lower - 1e-5
    ]).reshape(-1, 2)
    core_depths = np.array([
        cv2.pointPolygonTest(target_cv, (float(p[0]), float(p[1])), True) * scale
        for p in core_points
    ])
    missing_core = core_depths < -1e-5
    # Reverse Hausdorff distance is not a valid erosion test at corners: a
    # square inset by r on both axes has corner distance sqrt(2)*r but is a
    # valid r-offset. Test the required eroded material directly instead.
    missing_distances = -lower + core_depths[missing_core]
    distances = np.concatenate((forward, missing_distances))
    # Tiny epsilon covers float32 rigid-transform round-off, not pixel noise.
    outside = (distances < -lower - 1e-5) | (distances > upper + 1e-5)
    invalid_target = (forward < -lower - 1e-5) | (forward > upper + 1e-5)
    checked_count = len(target_points) + len(core_points)
    outside_count = int(np.count_nonzero(invalid_target) + np.count_nonzero(missing_core))
    return {
        "enabled": True,
        "pass": bool(not np.any(outside)),
        "status": "FAIL" if np.any(outside) else "PASS",
        "lower_mm": lower,
        "upper_mm": upper,
        "lower_px": lower / scale,
        "upper_px": upper / scale,
        "max_deviation_mm": float(np.max(np.abs(distances))),
        "max_inward_mm": max(0.0, -float(np.min(distances))),
        "max_outward_mm": max(0.0, float(np.max(distances))),
        "p95_deviation_mm": float(np.percentile(np.abs(distances), 95)),
        "outside_point_count": outside_count,
        "outside_point_ratio": outside_count / max(1, checked_count),
        "sampled_point_count": checked_count,
        "sample_spacing_px": 1.0,
        "aligned_golden_contour": golden_cv,
        "outside_target_points": target_points[invalid_target],
        "missing_core_points": core_points[missing_core],
        "inner_contours": inner,
        "outer_contours": outer,
    }


def line_intersections(contour, p1, p2):
    """Intersections with the infinite taught line, sorted along P1 -> P2."""
    origin = np.asarray(p1, dtype=float)
    axis = np.asarray(p2, dtype=float) - origin
    length = np.linalg.norm(axis)
    if length < 1e-9:
        return np.empty((0, 2))
    axis /= length
    pts = np.asarray(contour, dtype=float).reshape(-1, 2)
    hits = []
    cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        segment = b - a
        denom = cross(axis, segment)
        if abs(denom) < 1e-9:
            if abs(cross(a - origin, axis)) < 1e-6:
                hits.extend([float((a - origin) @ axis), float((b - origin) @ axis)])
            continue
        u = cross(a - origin, axis) / denom
        if -1e-8 <= u <= 1 + 1e-8:
            hits.append(float(cross(a - origin, segment) / denom))
    unique = []
    for value in sorted(hits):
        if not unique or abs(value - unique[-1]) > 1e-5:
            unique.append(value)
    return origin + np.asarray(unique)[:, None] * axis


def template_intersections(contour, p1, p2):
    """Find distinct ordered edges nearest golden landmarks without moving the line."""
    hits = line_intersections(contour, p1, p2)
    if len(hits) < 2:
        return None
    # Pair each side of the golden midpoint with its closest physical edge.
    # A vanished feature cannot be silently replaced by a distant edge on the
    # same side of the workpiece. Profile checking rejects excessive shifts.
    axis = np.asarray(p2) - np.asarray(p1)
    midpoint = (np.asarray(p1) + np.asarray(p2)) / 2
    positions = (hits - midpoint) @ axis
    left, right = hits[positions < -1e-7], hits[positions > 1e-7]
    if not len(left) or not len(right):
        return None
    q1 = left[np.argmin(np.linalg.norm(left - p1, axis=1))]
    q2 = right[np.argmin(np.linalg.norm(right - p2, axis=1))]
    return tuple(q1), tuple(q2)


def offset_contours(golden, lower_px, upper_px):
    """Raster preview of Euclidean offsets, on a padded local subpixel grid.

    Boundary inspection uses polygon distances above. Inner-core samples
    from this grid are confirmed against the exact polygon before inspection.
    Concave offsets can split into multiple contours.
    """
    points = np.asarray(golden, dtype=np.float32).reshape(-1, 2)
    padding = int(np.ceil(max(lower_px, upper_px))) + 4
    origin = np.floor(points.min(axis=0)) - padding
    extent = np.ceil(points.max(axis=0) - origin + padding + 1).astype(int)
    factor = min(2.0, np.sqrt(8_000_000 / max(1, int(extent[0]) * int(extent[1]))))
    shape = np.maximum(3, np.ceil(extent[::-1] * factor).astype(int))
    polygon = np.rint((points - origin) * factor).astype(np.int32).reshape(-1, 1, 2)
    mask = np.zeros(tuple(shape), dtype=np.uint8)
    boundary = np.ones(tuple(shape), dtype=np.uint8)
    cv2.drawContours(mask, [polygon], -1, 1, -1)
    cv2.drawContours(boundary, [polygon], -1, 0, 1)
    distance = cv2.distanceTransform(boundary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) / factor
    signed = np.where(mask > 0, distance, -distance)

    def extract(threshold):
        region = (signed >= threshold).astype(np.uint8)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c.astype(np.float32) / factor + origin for c in contours]

    inner = [np.asarray(golden, dtype=np.float32)] if lower_px == 0 else extract(lower_px)
    outer = [np.asarray(golden, dtype=np.float32)] if upper_px == 0 else extract(-upper_px)
    return inner, outer


def draw_profile(image, profile, thickness=1):
    """Draw golden, separate inner/outer limits and out-of-band points in BGR."""
    golden = profile.get("aligned_golden_contour")
    if golden is None:
        return
    inner = profile.get("inner_contours")
    outer = profile.get("outer_contours")
    if inner is None or outer is None:
        inner, outer = offset_contours(golden, profile["lower_px"], profile["upper_px"])
    for contours, color in ((inner, INNER_COLOR), (outer, OUTER_COLOR), ([golden], GOLDEN_COLOR)):
        cv2.drawContours(image, [np.rint(c).astype(np.int32) for c in contours],
                         -1, color, thickness, cv2.LINE_AA)
    for point in profile.get("outside_target_points", []):
        cv2.circle(image, tuple(np.rint(point).astype(int)), 1, (0, 0, 255), -1)
