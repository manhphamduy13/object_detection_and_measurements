"""
Industrial Metrology Scanline Blueprint Generator.
100% DYNAMIC & RECIPE-DRIVEN - NO HARDCODED SKUS OR POSITIONS.

Reads directly from:
1. assets/references/{sku}/images/{sku}_1.png (Golden Sample)
2. assets/references/{sku}/drawing.png (CAD Technical Drawing from PDF)
3. assets/references/{sku}/spec.json (CAD Specifications)
4. configs/caliper_recipes.json (Taught Caliper Measurements from Teaching Studio)

Outputs:
- outputs/scanline_diagrams/{sku}_scanline_diagram.png (High-Res 2-Panel Side-by-Side Blueprint)
- outputs/scanline_diagrams/ALL_PARTS_scanline_overview.png (Comprehensive Overview Portfolio)
"""
import os
import sys
import json
from pathlib import Path
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

import classifier as clf
import measurer

OUT_DIR = PROJECT_DIR / "outputs" / "scanline_diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS_DIR = PROJECT_DIR / "configs"
RECIPE_FILE = CONFIGS_DIR / "caliper_recipes.json"
PRODUCTS_FILE = CONFIGS_DIR / "products.json"

def clean_text(s: str) -> str:
    """Replaces Unicode characters that OpenCV cv2.putText cannot render with ASCII equivalents."""
    return s.replace("φ", "phi ").replace("±", "+/-").replace("—", "-")

def draw_caliper_arrow(vis, p1, p2, color, label="", tick_size=10, text_offset=(10, -8)):
    """Draws an industrial metrology caliper line with tick marks at endpoints and text label."""
    p1_i = (int(round(p1[0])), int(round(p1[1])))
    p2_i = (int(round(p2[0])), int(round(p2[1])))
    
    cv2.line(vis, p1_i, p2_i, color, 2, cv2.LINE_AA)
    cv2.circle(vis, p1_i, 3, color, -1, cv2.LINE_AA)
    cv2.circle(vis, p2_i, 3, color, -1, cv2.LINE_AA)
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = np.hypot(dx, dy)
    if dist > 1.0:
        nx = -dy / dist * tick_size
        ny = dx / dist * tick_size
        for pt in (p1, p2):
            t1 = (int(round(pt[0] + nx)), int(round(pt[1] + ny)))
            t2 = (int(round(pt[0] - nx)), int(round(pt[1] - ny)))
            cv2.line(vis, t1, t2, color, 2, cv2.LINE_AA)
            
    if label:
        clean_lbl = clean_text(label)
        mx = int(round((p1[0] + p2[0]) / 2.0)) + text_offset[0]
        my = int(round((p1[1] + p2[1]) / 2.0)) + text_offset[1]
        cv2.putText(vis, clean_lbl, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, clean_lbl, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

def generate_dynamic_schematic(sku: str, spec_data: dict, recipes: dict):
    """
    Renders the exact scanlines taught in configs/caliper_recipes.json onto the golden sample.
    100% dynamic for ANY SKU.
    """
    ref_imgs = list((PROJECT_DIR / "assets" / "references" / sku / "images").glob("*.png"))
    if not ref_imgs:
        ref_imgs = list((PROJECT_DIR / "assets" / "tests" / sku).glob("**/*.png"))
    if not ref_imgs:
        print(f"Warning: No reference image found for SKU {sku}")
        return None, {}

    img_path = ref_imgs[0]
    img = cv2.imread(str(img_path))
    if img is None:
        return None, {}

    described = clf.describe_image(img)
    cnt = described["contour"]
    gray = described["gray"]
    optical_scale = spec_data.get("optical_scale_mm_per_px", 0.044)

    m = measurer.WorkpieceMeasurer()
    meas = m.measure(img, cnt, spec_data)

    vis = img.copy()
    cv2.drawContours(vis, [cnt], -1, (60, 60, 60), 1, cv2.LINE_AA)

    # 1. Draw all calipers directly from taught caliper_recipes.json
    sku_recipe = recipes.get(sku, {})
    calipers = sku_recipe.get("calipers", [])

    palette = [
        (0, 230, 255), (255, 160, 0), (0, 255, 180), (255, 0, 255),
        (0, 200, 255), (200, 255, 0), (255, 120, 120), (180, 180, 255)
    ]

    drawn_labels = set()
    for idx, cal in enumerate(calipers):
        lbl = cal["label"]
        nom = cal.get("nominal_mm", 0.0)
        p1 = cal.get("p1_ref")
        p2 = cal.get("p2_ref")

        # If p1_ref not present, compute from WCS
        if not p1 or not p2:
            wcs = sku_recipe.get("wcs", {})
            u1, v1 = cal.get("p1_wcs", (0.0, 0.0))
            u2, v2 = cal.get("p2_wcs", (0.0, 0.0))
            w = wcs.get("width", 1.0)
            h = wcs.get("height", 1.0)
            c = wcs.get("center", (img.shape[1]/2, img.shape[0]/2))
            ang = np.radians(wcs.get("angle", 0.0))
            cos_a, sin_a = np.cos(ang), np.sin(ang)
            dx1, dy1 = u1 * w * cos_a - v1 * h * sin_a, u1 * w * sin_a + v1 * h * cos_a
            dx2, dy2 = u2 * w * cos_a - v2 * h * sin_a, u2 * w * sin_a + v2 * h * cos_a
            p1 = (c[0] + dx1, c[1] + dy1)
            p2 = (c[0] + dx2, c[1] + dy2)

        col = tuple(cal.get("color", palette[idx % len(palette)]))
        tag = f"{lbl} (Nom: {nom:.2f}mm)"
        draw_caliper_arrow(vis, p1, p2, col, tag, tick_size=10, text_offset=(12, -6))
        drawn_labels.add(lbl)

    # 2. Draw circular holes ONLY IF 'd' or a hole caliper is in the recipe calipers and not already drawn
    if any(c.get("label", "").lower() == "d" or c.get("measure_type") == "hole" for c in calipers):
        if "d" not in drawn_labels and "D" not in drawn_labels:
            holes = meas.get("circles_detected", [])
            for h in holes:
                cx, cy = int(round(h["center"][0])), int(round(h["center"][1]))
                r = int(round(h["radius"]))
                d_mm = h["d_mm"]
                cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.line(vis, (cx - 8, cy), (cx + 8, cy), (0, 255, 255), 1, cv2.LINE_AA)
                cv2.line(vis, (cx, cy - 8), (cx, cy + 8), (0, 255, 255), 1, cv2.LINE_AA)

                tag = f"HOLE phi d: {d_mm:.2f}mm"
                cv2.putText(vis, tag, (cx - r - 30, cy - r - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, tag, (cx - r - 30, cy - r - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                drawn_labels.add("d")

    # 3. Draw pins ONLY IF in recipe calipers AND NOT already drawn
    user_labels = set(c["label"] for c in calipers)
    pin_res = meas.get("pin_analysis", {})
    if pin_res.get("has_pins", False):
        for c in pin_res.get("calipers", []):
            c_lbl = c["label"]
            # Only draw if the user kept this pin diameter label in their recipe
            if any(k == c_lbl or k in c_lbl for k in user_labels):
                lbl = clean_text(c_lbl)
                if lbl in drawn_labels or c_lbl in drawn_labels:
                    continue
                draw_caliper_arrow(vis, c["p1"], c["p2"], (0, 255, 255), f"{lbl}: {c['val_mm']:.2f}mm", tick_size=8, text_offset=(10, 4))
                drawn_labels.add(lbl)

    return vis, meas

def assemble_full_blueprint(sku: str, spec_data: dict, camera_vis: np.ndarray, meas: dict):
    """
    Composes a side-by-side industrial blueprint:
    Left (Panel 1): 2D CAD Technical Drawing extracted from PDF
    Right (Panel 2): Live Camera Scanline Metrology Schematic
    Bottom-Right: CAD Specification & Measured Values Table
    """
    drawing_path = PROJECT_DIR / "assets" / "references" / sku / "drawing.png"
    if drawing_path.exists():
        cad_img = cv2.imread(str(drawing_path))
    else:
        cad_img = np.zeros((600, 600, 3), dtype=np.uint8)
        cv2.putText(cad_img, f"CAD Drawing: {sku}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    target_h = 850
    h_cad, w_cad = cad_img.shape[:2]
    new_w_cad = int(round(w_cad * (target_h / float(h_cad))))
    cad_resized = cv2.resize(cad_img, (new_w_cad, target_h), interpolation=cv2.INTER_AREA)

    h_cam, w_cam = camera_vis.shape[:2]
    new_w_cam = int(round(w_cam * (target_h / float(h_cam))))
    cam_resized = cv2.resize(camera_vis, (new_w_cam, target_h), interpolation=cv2.INTER_AREA)

    banner_h = 100
    total_w = new_w_cad + new_w_cam
    canvas = np.zeros((target_h + banner_h, total_w, 3), dtype=np.uint8)

    # Header
    cv2.rectangle(canvas, (0, 0), (total_w, banner_h), (20, 24, 30), -1)
    cv2.line(canvas, (0, banner_h), (total_w, banner_h), (0, 200, 255), 2)

    title = f"OPTICAL SCANLINE METROLOGY BLUEPRINT - SKU: {sku}"
    sub = f"Description: {spec_data.get('description', '')} | Optical Scale: {spec_data.get('optical_scale_mm_per_px', 0.044)*1000:.2f} um/px | Source: {spec_data.get('source_file', '')}"
    cv2.putText(canvas, title, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, clean_text(sub), (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 215, 255), 1, cv2.LINE_AA)

    # Place Panels
    canvas[banner_h:banner_h+target_h, :new_w_cad] = cad_resized
    canvas[banner_h:banner_h+target_h, new_w_cad:] = cam_resized
    cv2.line(canvas, (new_w_cad, banner_h), (new_w_cad, banner_h + target_h), (0, 200, 255), 2)

    # Panel titles
    cv2.rectangle(canvas, (10, banner_h + 10), (320, banner_h + 45), (15, 15, 20), -1)
    cv2.putText(canvas, "[PANEL 1: 2D CAD DRAWING (PDF)]", (15, banner_h + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (new_w_cad + 10, banner_h + 10), (new_w_cad + 440, banner_h + 45), (15, 15, 20), -1)
    cv2.putText(canvas, "[PANEL 2: DYNAMIC CALIPER SCANLINE SCHEMATIC]", (new_w_cad + 15, banner_h + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    # Overlay CAD Specification Table
    report = meas.get("spec_report", [])
    if report:
        tbl_w = 430
        tbl_h = 35 + len(report) * 22 + 10
        tbl_x = total_w - tbl_w - 15
        tbl_y = banner_h + target_h - tbl_h - 15

        overlay = canvas.copy()
        cv2.rectangle(overlay, (tbl_x, tbl_y), (tbl_x + tbl_w, tbl_y + tbl_h), (10, 15, 20), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (tbl_x, tbl_y), (tbl_x + tbl_w, tbl_y + tbl_h), (0, 200, 255), 1)

        cv2.putText(canvas, "CAD SPECIFICATION & MEASURED VALUES", (tbl_x + 12, tbl_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (tbl_x + 12, tbl_y + 28), (tbl_x + tbl_w - 12, tbl_y + 28), (60, 60, 70), 1)

        curr_y = tbl_y + 46
        for row in report:
            lbl = row.get("label", "")
            raw = clean_text(row.get("raw_spec", f"{lbl} {row.get('nominal')}mm"))
            meas_val = row.get("measured_mm")
            status = row.get("status", "")

            val_str = f"{meas_val:.2f}mm" if meas_val is not None else "  N/A  "
            line_str = f"{lbl:4s}: {raw:18s} -> {val_str:8s} [{status}]"
            st_color = (0, 255, 0) if status == "PASS" else ((0, 200, 255) if status == "Z-AXIS" else (0, 0, 255))

            cv2.putText(canvas, line_str, (tbl_x + 12, curr_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, st_color, 1, cv2.LINE_AA)
            curr_y += 22

    return canvas

def generate_overview_collage(blueprints: dict):
    """Combines all generated blueprints into a high-level overview portfolio image."""
    thumbs = []
    target_th_h = 360
    for sku, bp_path in blueprints.items():
        if bp_path and Path(bp_path).exists():
            b_img = cv2.imread(bp_path)
            h, w = b_img.shape[:2]
            new_w = int(round(w * (target_th_h / float(h))))
            thumb = cv2.resize(b_img, (new_w, target_th_h), interpolation=cv2.INTER_AREA)
            thumbs.append((sku, thumb))

    if not thumbs:
        return

    # Stack in rows of up to 3 items
    row_count = (len(thumbs) + 2) // 3
    rows = [thumbs[i*3:(i+1)*3] for i in range(row_count)]

    gap = 20
    margin = 30
    max_w = max(sum(t[1].shape[1] for t in r) + gap * (len(r) - 1) + margin * 2 for r in rows)
    canvas_w = max(max_w, 1600)
    canvas_h = target_th_h * len(rows) + 120 + gap * (len(rows) - 1)

    collage = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    collage[:] = (20, 22, 26)

    cv2.putText(collage, "INDUSTRIAL METROLOGY SCANLINE PORTFOLIO - DYNAMIC BLUEPRINTS",
                (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(collage, "100% Dynamic Recipe Metrology: Taught Scanlines, Holes, Pins, & CAD Tolerances",
                (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 1, cv2.LINE_AA)

    y_curr = 100
    for r in rows:
        row_w = sum(t[1].shape[1] for t in r) + gap * (len(r) - 1)
        x_curr = (canvas_w - row_w) // 2
        for sku, th in r:
            th_w = th.shape[1]
            collage[y_curr:y_curr+target_th_h, x_curr:x_curr+th_w] = th
            cv2.rectangle(collage, (x_curr, y_curr), (x_curr+th_w, y_curr+target_th_h), (80, 80, 80), 1)
            x_curr += th_w + gap
        y_curr += target_th_h + gap

    out_file = OUT_DIR / "ALL_PARTS_scanline_overview.png"
    cv2.imwrite(str(out_file), collage)
    print(f"\n  -> Saved Combined Overview Portfolio: {out_file}")

def main():
    print("=" * 70)
    print(" 100% DYNAMIC SCANLINE BLUEPRINT GENERATOR")
    print("=" * 70)

    # Load recipes
    recipes = {}
    if RECIPE_FILE.exists():
        try:
            with open(RECIPE_FILE, "r", encoding="utf-8") as f:
                recipes = json.load(f)
        except Exception:
            pass

    # Load products / specs
    products = {}
    if PRODUCTS_FILE.exists():
        try:
            with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
                products = json.load(f)
        except Exception:
            pass

    # Also check references dir for any SKU not yet in products.json
    ref_dir = PROJECT_DIR / "assets" / "references"
    for sku_dir in sorted(ref_dir.iterdir()):
        if sku_dir.is_dir() and sku_dir.name not in products:
            spec_f = sku_dir / "spec.json"
            if spec_f.exists():
                try:
                    with open(spec_f, "r", encoding="utf-8") as f:
                        products[sku_dir.name] = json.load(f)
                except Exception:
                    pass

    blueprints = {}
    for sku, data in products.items():
        print(f"\nProcessing SKU: {sku}...")
        cam_vis, meas = generate_dynamic_schematic(sku, data, recipes)
        if cam_vis is None:
            continue

        full_bp = assemble_full_blueprint(sku, data, cam_vis, meas)
        out_file = OUT_DIR / f"{sku}_scanline_diagram.png"
        cv2.imwrite(str(out_file), full_bp)
        blueprints[sku] = str(out_file)
        print(f"  -> Saved Blueprint: {out_file}")

    print("\nGenerating combined portfolio overview...")
    generate_overview_collage(blueprints)
    print("\n" + "=" * 70)
    print(" ALL BLUEPRINTS GENERATED SUCCESSFULLY FROM RECIPES")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
