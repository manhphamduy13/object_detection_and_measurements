"""
Metrology Measurer for Object Classification Pipeline
Inherits optical caliper subpixel precision from QC_CAM stage 3.
Measures primary workpiece dimensions:
- Length (L) [px]
- Width (W) [px]
- Top Edge, Bottom Edge, Left Edge, Right Edge [px]
- Feature specific specs from spec.json
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np


class CaliperTool:
    def __init__(self, min_gradient: float = 20.0):
        self.min_gradient = min_gradient

    def sample_profile_1d(
        self,
        gray: np.ndarray,
        p_start: Tuple[float, float],
        p_end: Tuple[float, float],
        num_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x1, y1 = p_start
        x2, y2 = p_end
        length = float(np.hypot(x2 - x1, y2 - y1))
        if num_samples is None:
            num_samples = max(int(np.ceil(length)), 5)

        t = np.linspace(0.0, 1.0, num_samples)
        xs = x1 + t * (x2 - x1)
        ys = y1 + t * (y2 - y1)

        xs_f = xs.astype(np.float32).reshape(1, -1)
        ys_f = ys.astype(np.float32).reshape(1, -1)
        profile = cv2.remap(
            gray, xs_f, ys_f,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        ).flatten().astype(np.float64)

        return profile, xs, ys

    def compute_gradient(self, profile: np.ndarray) -> np.ndarray:
        n = len(profile)
        grad = np.zeros(n, dtype=np.float64)
        if n > 2:
            grad[1:-1] = profile[2:] - profile[:-2]
            grad[0] = profile[1] - profile[0]
            grad[-1] = profile[-1] - profile[-2]
        return grad

    def find_subpixel_edge(
        self,
        profile: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray
    ) -> Tuple[Tuple[float, float], bool]:
        grad = self.compute_gradient(profile)
        mag = np.abs(grad)
        n = len(mag)
        if n < 3:
            return (float(xs[0]), float(ys[0])), False

        i_peak = int(np.argmax(mag))
        peak_val = mag[i_peak]
        if peak_val < self.min_gradient:
            return (float(xs[i_peak]), float(ys[i_peak])), False

        # Parabolic interpolation
        if 0 < i_peak < n - 1:
            a = mag[i_peak - 1]
            b = mag[i_peak]
            c = mag[i_peak + 1]
            denom = a - 2.0 * b + c
            offset = 0.5 * (a - c) / denom if abs(denom) > 1e-6 else 0.0
            offset = max(-0.5, min(0.5, offset))
        else:
            offset = 0.0

        sub_idx = float(i_peak) + offset
        sub_idx = max(0.0, min(float(n - 1), sub_idx))

        t = sub_idx / float(n - 1) if n > 1 else 0.0
        x_sub = float(xs[0] + t * (xs[-1] - xs[0]))
        y_sub = float(ys[0] + t * (ys[-1] - ys[0]))
        return (x_sub, y_sub), True


class WorkpieceMeasurer:
    def __init__(self, min_gradient: float = 20.0):
        self.caliper = CaliperTool(min_gradient=min_gradient)

    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]      # Top-left
        rect[2] = pts[np.argmax(s)]      # Bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]   # Top-right
        rect[3] = pts[np.argmax(diff)]   # Bottom-left
        return rect

    def refine_edge(
        self,
        gray: np.ndarray,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        margin: float = 25.0
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = float(np.hypot(dx, dy))
        if length < 1e-4:
            return p1, p2, 0.0

        nx = -dy / length
        ny = dx / length

        # Scanline across p1
        s1_start = (p1[0] - nx * margin, p1[1] - ny * margin)
        s1_end = (p1[0] + nx * margin, p1[1] + ny * margin)
        prof1, xs1, ys1 = self.caliper.sample_profile_1d(gray, s1_start, s1_end)
        sub_p1, valid1 = self.caliper.find_subpixel_edge(prof1, xs1, ys1)
        final_p1 = sub_p1 if valid1 else p1

        # Scanline across p2
        s2_start = (p2[0] - nx * margin, p2[1] - ny * margin)
        s2_end = (p2[0] + nx * margin, p2[1] + ny * margin)
        prof2, xs2, ys2 = self.caliper.sample_profile_1d(gray, s2_start, s2_end)
        sub_p2, valid2 = self.caliper.find_subpixel_edge(prof2, xs2, ys2)
        final_p2 = sub_p2 if valid2 else p2

        dist = float(np.hypot(final_p2[0] - final_p1[0], final_p2[1] - final_p1[1]))
        return final_p1, final_p2, dist

    def measure(
        self,
        img_bgr: np.ndarray,
        contour: np.ndarray,
        spec: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Measures workpiece dimensions in pixels.
        Returns:
            {
                "Length_L_px": float,
                "Width_W_px": float,
                "Top_Edge_px": float,
                "Bottom_Edge_px": float,
                "Left_Edge_px": float,
                "Right_Edge_px": float,
                "dimensions": [
                    {"label": ..., "measured_px": ..., "p1": ..., "p2": ...},
                    ...
                ]
            }
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr

        # Oriented bounding box
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        corners = self._order_corners(box)
        tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]

        p1_top, p2_top, len_top = self.refine_edge(gray, tuple(tl), tuple(tr))
        p1_bot, p2_bot, len_bot = self.refine_edge(gray, tuple(bl), tuple(br))
        p1_left, p2_left, len_left = self.refine_edge(gray, tuple(tl), tuple(bl))
        p1_right, p2_right, len_right = self.refine_edge(gray, tuple(tr), tuple(br))

        mid_top = ((p1_top[0] + p2_top[0]) / 2.0, (p1_top[1] + p2_top[1]) / 2.0)
        mid_bot = ((p1_bot[0] + p2_bot[0]) / 2.0, (p1_bot[1] + p2_bot[1]) / 2.0)
        mid_left = ((p1_left[0] + p2_left[0]) / 2.0, (p1_left[1] + p2_left[1]) / 2.0)
        mid_right = ((p1_right[0] + p2_right[0]) / 2.0, (p1_right[1] + p2_right[1]) / 2.0)

        side_h = (len_top + len_bot) / 2.0
        side_v = (len_left + len_right) / 2.0

        if side_h >= side_v:
            overall_len = side_h
            overall_wid = side_v
            len_p1, len_p2 = mid_left, mid_right
            wid_p1, wid_p2 = mid_top, mid_bot
        else:
            overall_len = side_v
            overall_wid = side_h
            len_p1, len_p2 = mid_top, mid_bot
            wid_p1, wid_p2 = mid_left, mid_right

        dims = [
            {"label": "Length (L)", "measured_px": round(overall_len, 2), "p1": len_p1, "p2": len_p2},
            {"label": "Width (W)", "measured_px": round(overall_wid, 2), "p1": wid_p1, "p2": wid_p2},
            {"label": "Top Edge", "measured_px": round(len_top, 2), "p1": p1_top, "p2": p2_top},
            {"label": "Bottom Edge", "measured_px": round(len_bot, 2), "p1": p1_bot, "p2": p2_bot},
            {"label": "Left Edge", "measured_px": round(len_left, 2), "p1": p1_left, "p2": p2_left},
            {"label": "Right Edge", "measured_px": round(len_right, 2), "p1": p1_right, "p2": p2_right},
        ]

        # Map to spec dimensions if available
        if spec and "dimensions" in spec:
            for item in spec["dimensions"]:
                lbl = item.get("label", "")
                # If spec mentions Length/L or Width/W
                if lbl in ("L", "L_plug", "Pin_L"):
                    dims.append({"label": f"Spec_{lbl}", "measured_px": round(overall_len, 2), "p1": len_p1, "p2": len_p2, "nominal_mm": item.get("nominal")})
                elif lbl in ("W", "W_plug", "Width", "D"):
                    dims.append({"label": f"Spec_{lbl}", "measured_px": round(overall_wid, 2), "p1": wid_p1, "p2": wid_p2, "nominal_mm": item.get("nominal")})

        optical_scale = None
        if spec and "optical_scale_mm_per_px" in spec:
            optical_scale = spec["optical_scale_mm_per_px"]

        meas_dict = {
            "Length_L_px": round(overall_len, 2),
            "Width_W_px": round(overall_wid, 2),
            "Top_Edge_px": round(len_top, 2),
            "Bottom_Edge_px": round(len_bot, 2),
            "Left_Edge_px": round(len_left, 2),
            "Right_Edge_px": round(len_right, 2),
            "dimensions": dims,
            "oriented_box": box.tolist(),
        }

        if optical_scale:
            meas_dict["optical_scale_mm_per_px"] = optical_scale
            meas_dict["Length_L_mm"] = round(overall_len * optical_scale, 3)
            meas_dict["Width_W_mm"] = round(overall_wid * optical_scale, 3)
            for d in dims:
                d["measured_mm"] = round(d["measured_px"] * optical_scale, 3)

        # Detect internal circular holes / pin features
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        bx, by, bw, bh = cv2.boundingRect(contour)
        crop_mask = mask[by:by+bh, bx:bx+bw]
        crop_gray = gray[by:by+bh, bx:bx+bw]
        contours_internal, _ = cv2.findContours(crop_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        diameters_mm = []
        if optical_scale:
            for c in contours_internal:
                area = cv2.contourArea(c)
                if 40 < area < (bw * bh * 0.4):
                    (x, y), r = cv2.minEnclosingCircle(c)
                    d_mm = round((2 * r) * optical_scale, 2)
                    if 0.5 < d_mm < 15.0:
                        diameters_mm.append(d_mm)
            diameters_mm = sorted(diameters_mm, reverse=True)

            # Check brass pins if color image
            pin_widths_mm = []
            if len(img_bgr.shape) == 3:
                b, g, r = cv2.split(img_bgr.astype(float))
                brass = (((r - b) > 20.0) & (r > 60.0)).astype(np.uint8) * 255
                brass_crop = brass[by:by+bh, bx:bx+bw]
                n_lbl, _, b_stats, _ = cv2.connectedComponentsWithStats(brass_crop)
                for i in range(1, n_lbl):
                    a = b_stats[i, cv2.CC_STAT_AREA]
                    if 200 < a < 20000:
                        pw = min(b_stats[i, cv2.CC_STAT_WIDTH], b_stats[i, cv2.CC_STAT_HEIGHT]) * optical_scale
                        pin_widths_mm.append(round(pw, 2))
                pin_widths_mm = sorted(pin_widths_mm, reverse=True)

            # Full mapping for every dimension in spec
            spec_report = []
            used_dia, used_pin = 0, 0
            for item in (spec.get("dimensions", []) if spec else []):
                lbl = item.get("label", "DIM")
                nom = item.get("nominal")
                mi = item.get("min", nom)
                ma = item.get("max", nom)
                is_dia = item.get("diameter", False)

                val = None
                note = ""
                if lbl in ("L", "L_plug") or (not is_dia and nom == max(d["nominal"] for d in spec["dimensions"] if not d.get("diameter"))):
                    val = meas_dict["Length_L_mm"]
                    note = "Truc doc chinh (Longitudinal L)"
                elif lbl == "H":
                    val = None
                    note = "Chieu cao truc Z (Can camera goc nghieng/ngang)"
                elif lbl in ("W", "W_plug"):
                    val = meas_dict["Width_W_mm"]
                    note = "Truc ngang chinh (Transverse W)"
                elif is_dia or "D" in lbl:
                    if diameters_mm and used_dia < len(diameters_mm):
                        val = diameters_mm[used_dia]
                        used_dia += 1
                        note = "Duong kinh lo tron (Optical Circle)"
                    elif pin_widths_mm and used_pin < len(pin_widths_mm):
                        val = pin_widths_mm[used_pin]
                        used_pin += 1
                        note = "Duong kinh chan pin (Pin Caliper)"
                    else:
                        val = None
                        note = "Duong kinh (Hinh chieu 2D)"
                else:
                    if abs(nom - meas_dict["Width_W_mm"]) < 2.0:
                        val = meas_dict["Width_W_mm"]
                        note = "Mep bien ngoai W"
                    else:
                        val = None
                        note = "Chi tiet bien trong"

                if val is not None:
                    ok = (mi <= val <= ma) if (mi is not None and ma is not None) else True
                    status = "PASS (OK)" if ok else "FAIL (NG)"
                else:
                    status = "CAN CAMERA PHU (N/A)"

                spec_report.append({
                    "label": lbl,
                    "measured_mm": val,
                    "nominal_mm": nom,
                    "min_mm": mi,
                    "max_mm": ma,
                    "status": status,
                    "note": note
                })
            meas_dict["spec_report"] = spec_report

        return meas_dict


def annotate_measurements(
    img_bgr: np.ndarray,
    contour: np.ndarray,
    measurements: Dict[str, Any],
    sku_label: str = "UNKNOWN"
) -> np.ndarray:
    """
    Renders high-contrast engineering annotation overlay on the workpiece.
    """
    vis = img_bgr.copy()

    # Draw contour
    cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)

    # Draw oriented box
    if "oriented_box" in measurements:
        box = np.array(measurements["oriented_box"], dtype=np.int32)
        cv2.drawContours(vis, [box], 0, (255, 140, 0), 2)

    # Draw primary dimension lines (L and W)
    dims = measurements.get("dimensions", [])
    colors = {
        "Length (L)": (0, 255, 255),      # Yellow
        "Width (W)": (255, 0, 255),       # Magenta
        "Top Edge": (0, 165, 255),        # Orange
        "Bottom Edge": (0, 165, 255),
        "Left Edge": (255, 200, 0),       # Cyan-ish
        "Right Edge": (255, 200, 0),
    }

    for d in dims[:2]:  # Length & Width
        label = d["label"]
        p1 = (int(d["p1"][0]), int(d["p1"][1]))
        p2 = (int(d["p2"][0]), int(d["p2"][1]))
        val = d["measured_px"]
        color = colors.get(label, (0, 255, 0))

        # Line + ticks
        cv2.line(vis, p1, p2, color, 2, cv2.LINE_AA)
        cv2.circle(vis, p1, 4, color, -1)
        cv2.circle(vis, p2, 4, color, -1)

        # Label
        mid_x = int((p1[0] + p2[0]) / 2)
        mid_y = int((p1[1] + p2[1]) / 2)
        val_mm = d.get("measured_mm")
        if val_mm:
            text = f"{label}: {val_mm:.2f}mm ({val:.1f}px)"
        else:
            text = f"{label}: {val}px"
        cv2.putText(vis, text, (mid_x + 10, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, (mid_x + 10, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Header banner
    len_mm = measurements.get('Length_L_mm')
    wid_mm = measurements.get('Width_W_mm')
    if len_mm and wid_mm:
        header = f"QC METROLOGY: {sku_label} | L: {len_mm:.2f}mm ({measurements.get('Length_L_px')}px) | W: {wid_mm:.2f}mm ({measurements.get('Width_W_px')}px)"
    else:
        header = f"QC METROLOGY: {sku_label} | L: {measurements.get('Length_L_px')}px | W: {measurements.get('Width_W_px')}px"
    cv2.putText(vis, header, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, header, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    return vis
