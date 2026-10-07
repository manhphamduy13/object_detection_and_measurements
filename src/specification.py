"""Helpers for editing metrology dimensions without coupling them to the GUI."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def build_dimension(
    label: str,
    nominal: float,
    tolerance_minus: float = 0.0,
    tolerance_plus: Optional[float] = None,
    *,
    description: str = "",
    diameter: bool = False,
    axis: str = "2D_in_plane",
) -> Dict[str, Any]:
    """Create the canonical dimension structure stored in ``spec.json``.

    Tolerances are accepted as positive magnitudes.  ``tolerance_plus``
    defaults to the minus tolerance, which makes the common +/- workflow
    concise while still supporting asymmetric drawings.
    """
    clean_label = str(label).strip()
    if not clean_label:
        raise ValueError("dimension label is required")

    nominal = float(nominal)
    tol_minus = abs(float(tolerance_minus))
    tol_plus = tol_minus if tolerance_plus is None else abs(float(tolerance_plus))
    if nominal < 0:
        raise ValueError("nominal value must be non-negative")

    if tol_minus == tol_plus:
        tolerance_type = "symmetric"
        tolerance_text = f"±{tol_plus:g}"
    else:
        tolerance_type = "asymmetric"
        tolerance_text = f"+{tol_plus:g}/-{tol_minus:g}"

    raw_spec = description.strip() or f"{clean_label} {nominal:g}{tolerance_text}mm"
    return {
        "raw_spec": raw_spec,
        "label": clean_label,
        "nominal": nominal,
        "diameter": bool(diameter),
        "upper_tol": tol_plus,
        "lower_tol": -tol_minus,
        "min": nominal - tol_minus,
        "max": nominal + tol_plus,
        "tolerance_type": tolerance_type,
        "axis": axis,
    }


def upsert_dimension(dimensions: Iterable[Dict[str, Any]], dimension: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Replace a dimension with the same label, or append a new one."""
    result = [dict(item) for item in dimensions]
    label = str(dimension.get("label", "")).strip().casefold()
    if not label:
        raise ValueError("dimension label is required")

    for index, item in enumerate(result):
        if str(item.get("label", "")).strip().casefold() == label:
            result[index] = dict(dimension)
            return result
    result.append(dict(dimension))
    return result


def tolerance_magnitudes(dimension: Dict[str, Any]) -> tuple[float, float]:
    """Return (minus, plus) tolerance magnitudes for form population."""
    nominal = float(dimension.get("nominal", dimension.get("nominal_mm", 0.0)) or 0.0)
    minus = dimension.get("lower_tol")
    plus = dimension.get("upper_tol")
    if minus is None:
        minus = nominal - float(dimension.get("min", nominal) or nominal)
    if plus is None:
        plus = float(dimension.get("max", nominal) or nominal) - nominal
    return abs(float(minus or 0.0)), abs(float(plus or 0.0))
