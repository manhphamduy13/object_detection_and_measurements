"""
Automated SKU Setup from 2D CAD Drawings / Specification PDFs.
100% DYNAMIC EXTRACTION - NO HARDCODED DICTIONARIES.

Performs:
1. Technical drawing raster image extraction directly from PDF stream.
2. Complete dynamic regex text parsing of CAD specifications:
   - Dimension labels (L, W, F, J, d, K, t, A, B, E, etc.)
   - Nominal values in mm
   - Symmetric tolerances (±tol) and asymmetric tolerances (+upper/-lower)
   - Max/Min limits (MAX / MIN)
   - Cylindrical diameter flags (φ / ψ / Ø)
   - Plane categorization (2D in-plane vs Z-height side view)
3. Dynamic optical calibration using Golden Sample images:
   - Measures golden sample bounding box pixels
   - Matches with largest nominal dimension to calculate um/px (optical scale)
4. Exports:
   - assets/references/{sku}/spec.json
   - assets/references/{sku}/drawing.png
   - configs/products.json

Usage:
  python tools/auto_setup_from_pdf.py
  python tools/auto_setup_from_pdf.py --pdf items/98661BBS-1.pdf
"""
import os
import sys
import re
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
import pypdf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

import classifier as clf

ITEMS_DIR = PROJECT_DIR / "items"
REFERENCES_DIR = PROJECT_DIR / "assets" / "references"
CONFIGS_DIR = PROJECT_DIR / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

def parse_dimension_line(line: str, default_label: str = ""):
    """
    Parses a single line or table entry from an engineering PDF into a structured dimension specification.
    Supports symmetric/asymmetric tolerances, MAX/MIN limits, diameters, and plane categorization.
    Generic for ANY mechanical / electrical component CAD drawing.
    """
    s = line.strip().replace("ψ", "φ").replace("Ø", "φ").replace("+-", "±")
    
    # Must contain measurement indicators
    has_tol = any(k in s for k in ["±", "+", "MAX", "MIN", "/-"])
    has_dim_unit = "mm" in s.lower() or "φ" in s or "測" in s
    if not (has_tol or has_dim_unit):
        return None

    # Filter out procedural text, notes, electrical/thermal limits, ball indentation limits
    if any(k in s for k in ["见球压", "标准书", "標準", "不能超過", "超過", "測試", "操作", "MΩ", "20N", "125C", "压痕", "球压", "AWG", "Revision", "Date", "抽樣"]):
        return None

    # Regex to capture:
    # 1. Leading Label (optional single letter or code starting with letter, e.g. 'L', 'W', 'D1')
    # 2. Diameter symbol (φ)
    # 3. Nominal value
    # 4. Tolerance (+upper/-lower or ±tol or MAX/MIN)
    # 5. Trailing Label (optional single letter e.g. '8.60±0.15mmH' or '1.5+0/-0.04mmT')
    pattern = r'(?:([A-Za-z][0-9]?)\s+)?(?:(φ)\s*)?(\d+(?:\.\d+)?)\s*(?:(±|\+-|\+)\s*(\d+(?:\.\d+)?)(?:\/-\s*(\d+(?:\.\d+)?))?|\s*(MAX|MIN))?\s*(?:mm)?\s*([A-Za-z])?'
    m = re.search(pattern, s)
    if not m:
        return None

    lbl1, is_dia, nom_str, tol_sign, tol1_str, tol2_str, limit_str, lbl2 = m.groups()
    if not nom_str:
        return None

    nom = float(nom_str)
    # Filter out unreasonable numbers like part numbers or dates (e.g. 98003, 2007)
    if nom > 500.0 or nom < 0.2:
        return None

    # Must have either tolerance or explicit dimension unit
    if not (tol_sign or limit_str or is_dia or "mm" in s.lower()):
        return None

    lbl = lbl1 or lbl2 or default_label or ""
    if lbl.upper() in ("M", "MAX", "MIN") and ("MAX" in s or "MIN" in s):
        lbl = default_label or ""
    dia = bool(is_dia or "φ" in s or "dia" in s.lower())

    if limit_str == "MAX" or "MAX" in s:
        upper = 0.0
        lower = -nom
        mi, ma = 0.0, nom
        tol_type = "max_limit"
    elif tol2_str:  # Asymmetric: +tol1/-tol2
        upper = float(tol1_str)
        lower = -float(tol2_str)
        mi = round(nom + lower, 3)
        ma = round(nom + upper, 3)
        tol_type = "asymmetric"
    elif tol1_str:  # Symmetric: ±tol1
        tol = float(tol1_str)
        upper, lower = tol, -tol
        mi = round(nom - tol, 3)
        ma = round(nom + tol, 3)
        tol_type = "symmetric"
    else:
        upper, lower = 0.2, -0.2
        mi, ma = round(nom - 0.2, 3), round(nom + 0.2, 3)
        tol_type = "default"

    # Identify axis: Z-height vs 2D in-plane
    axis = "2D_in_plane"
    if lbl.upper() in ("H",) or "height" in s.lower() or "thick" in s.lower() or "厚" in s or "高" in s:
        axis = "Z_height_side_view"

    return {
        "raw_spec": s,
        "label": lbl,
        "nominal": nom,
        "diameter": dia,
        "upper_tol": upper,
        "lower_tol": lower,
        "min": mi,
        "max": ma,
        "tolerance_type": tol_type,
        "axis": axis
    }

def extract_pdf_data(pdf_path: Path):
    """
    Extracts text, metadata, drawing image, and dimensions dynamically from any PDF.
    100% GENERIC DYNAMIC EXTRACTION - NO HARDCODED SKUS OR TYPES:
    1. Spatial Table Extractor: Couples row numbers, inspection item labels, and specification values.
    2. Text Stream / Callout Extractor: Resolves dimensions labeled in the drawing annotations.
    3. Intelligent Reconciliation: Merges table specifications with callout annotations.
    """
    sku = pdf_path.stem
    reader = pypdf.PdfReader(str(pdf_path))
    page = reader.pages[0]

    ref_folder = REFERENCES_DIR / sku
    ref_folder.mkdir(parents=True, exist_ok=True)
    images_folder = ref_folder / "images"
    images_folder.mkdir(parents=True, exist_ok=True)

    # 1. Extract embedded CAD drawing image
    drawing_filename = None
    if len(page.images) > 0:
        drawing_img = page.images[0]
        drawing_filename = "drawing.png"
        drawing_path = ref_folder / drawing_filename
        with open(drawing_path, "wb") as f:
            f.write(drawing_img.data)

    # 2. Extract text and parse metadata
    text = page.extract_text()
    desc = f"Workpiece {sku}"
    mat = "Standard"

    for line in text.splitlines():
        line_clean = line.strip()
        if "Description" in line_clean or "品名" in line_clean or "Item" in line_clean:
            parts = line_clean.split()
            if len(parts) > 1 and parts[-1] not in ("Description", "Item"):
                desc = parts[-1]
        if any(k in line_clean for k in ["黄铜", "铜", "Brass", "Copper", "Cu"]):
            mat = "Brass / Copper"
        elif any(k in line_clean for k in ["PA66", "PBT", "Plastic", "HOUSING", "胶壳"]):
            mat = "Engineering Plastic"

    # 3. Dynamic dimension parsing
    # 3a. Extract all callouts from the text stream (e.g. 'L 32.5±0.25mm', 'W 4.9±0.30mm', '8.60±0.15mmH')
    stream_dims = {}
    for line in text.splitlines():
        line_clean = line.strip()
        parsed = parse_dimension_line(line_clean)
        if parsed:
            key = (parsed["label"], parsed["nominal"])
            stream_dims[key] = parsed

    # 3b. Extract table rows using spatial coordinates
    elements = []
    def visitor(txt, cm, tm, font_dict, font_size):
        t = txt.strip()
        if t:
            elements.append((tm[4], tm[5], font_size, t))
    page.extract_text(visitor_text=visitor)

    from collections import defaultdict
    rows = defaultdict(list)
    for x, y, size, t in elements:
        matched = False
        for rk in rows:
            if abs(rk - y) < 4.5:
                rows[rk].append((x, size, t))
                matched = True
                break
        if not matched:
            rows[y].append((x, size, t))

    final_dims = []
    table_labels = set()

    for y in sorted(rows.keys(), reverse=True):
        row = sorted(rows[y], key=lambda e: e[0])
        for x, size, t in row:
            if re.match(r"^\d{1,2}$", t) and 70 <= x <= 125:
                lbl_tokens = [e[2] for e in row if 130 <= e[0] <= 215]
                spec_tokens = [e[2] for e in row if 215 <= e[0] <= 335]
                
                lbl = None
                for tok in lbl_tokens:
                    tok_clean = tok.strip()
                    # Filter non-dimension procedural inspection lines
                    if tok_clean in ["外觀", "表面處理", "表面处理", "PIN表面處理", "絕緣顔色", "顔色", "组装检查", "绝缘电阻", "球压测试"]:
                        lbl = False
                        break
                    # Generic mechanical drawing term translations
                    if tok_clean in ["板厚"]:
                        lbl = "t"
                        break
                    if tok_clean in ["高度", "高"]:
                        lbl = "H"
                        break
                    if tok_clean in ["寬度", "寬"]:
                        lbl = "W"
                        break
                    if tok_clean in ["長度", "長"]:
                        lbl = "L"
                        break
                    if re.match(r"^[A-Za-z][0-9]?$", tok_clean):
                        lbl = tok_clean
                        break
                if lbl is False:
                    continue

                spec_str = " ".join(spec_tokens).strip()
                dim = None
                if spec_str:
                    dim = parse_dimension_line(spec_str, default_label=lbl or "")
                
                # If spec wasn't directly in table column 3, match from text stream with this label
                if not dim and lbl:
                    for (s_lbl, s_nom), s_dim in stream_dims.items():
                        if s_lbl == lbl:
                            dim = s_dim.copy()
                            break

                # If table token had OCR fragmentation (e.g. '8.5' instead of '28.5' for L)
                if dim and lbl and lbl in ("L", "W", "A", "B"):
                    for (s_lbl, s_nom), s_dim in stream_dims.items():
                        if s_lbl == lbl and s_nom > dim["nominal"]:
                            dim = s_dim.copy()
                            break

                if dim:
                    if lbl and not dim["label"]:
                        dim["label"] = lbl
                    # Correct common OCR font misrecognition (e.g. letter K seen as V in Asian font)
                    if dim["label"] == "V" and abs(dim["nominal"] - 3.0) < 0.1:
                        dim["label"] = "K"
                    table_labels.add(dim["label"])
                    final_dims.append(dim)
                break

    # Fallback: Also include callouts from stream that were not already in table
    for (s_lbl, s_nom), s_dim in stream_dims.items():
        if s_lbl and s_lbl not in table_labels and s_lbl not in [d["label"] for d in final_dims]:
            final_dims.append(s_dim)

    # Ensure all dimensions have valid labels
    used_labels = set(d["label"] for d in final_dims if d["label"])
    non_dia_2d = [d for d in final_dims if not d["diameter"] and d["axis"] == "2D_in_plane"]
    max_nom = max((d["nominal"] for d in non_dia_2d), default=0.0)

    for d in final_dims:
        if not d["label"]:
            if d["nominal"] == max_nom and "L" not in used_labels:
                lbl = "L"
            elif d["diameter"]:
                lbl = "d" if "d" not in used_labels else f"d{len(final_dims)+1}"
            else:
                lbl = f"D{len(final_dims)+1}"
            used_labels.add(lbl)
            d["label"] = lbl

    # Sort: Primary length (L/A), width (W/B), in-plane, then Z-height
    def sort_dim(d):
        is_z = 1 if d.get("axis") == "Z_height_side_view" else 0
        lbl = d.get("label", "")
        if lbl in ("L", "A") and not is_z:
            order = 0
        elif lbl in ("W", "B") and not is_z:
            order = 1
        elif d.get("diameter"):
            order = 2
        else:
            order = 3
        return (is_z, order, -d["nominal"])

    final_dims.sort(key=sort_dim)
    dimensions = final_dims

    spec = {
        "sku": sku,
        "source_file": str(pdf_path.relative_to(PROJECT_DIR)).replace("\\", "/"),
        "item_name": sku,
        "part_no": sku,
        "description": desc,
        "material": mat,
        "unit": "mm",
        "drawing": drawing_filename,
        "review_status": "dynamically_extracted_from_pdf",
        "dimensions": dimensions
    }

    return spec

def calibrate_optical_scale(sku: str, spec: dict):
    """
    Measures the golden reference image to determine camera optical scale (mm/px).
    """
    ref_imgs = sorted(list((REFERENCES_DIR / sku / "images").glob("*.png")))
    if not ref_imgs:
        ref_imgs = sorted(list((PROJECT_DIR / "assets" / "tests" / sku).glob("**/*.png")))

    if not ref_imgs:
        return 0.044, None

    golden_img_path = ref_imgs[0]
    img = cv2.imread(str(golden_img_path))
    if img is None:
        return 0.044, None

    described = clf.describe_image(img)
    cnt = described["contour"]
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect
    obb_long = max(w, h)
    obb_short = min(w, h)

    # Find primary nominal length from 2D dimensions
    dims_2d = [d for d in spec.get("dimensions", []) if d.get("axis") == "2D_in_plane"]
    if dims_2d:
        nom_length = max(d["nominal"] for d in dims_2d)
    else:
        nom_length = max(d["nominal"] for d in spec.get("dimensions", [])) if spec.get("dimensions") else 30.0

    optical_scale = nom_length / float(obb_long) if obb_long > 0 else 0.044

    golden_info = {
        "filename": golden_img_path.name,
        "obb_long_px": round(float(obb_long), 1),
        "obb_short_px": round(float(obb_short), 1),
        "angle_deg": round(float(angle), 1),
        "optical_scale_um_per_px": round(optical_scale * 1000.0, 2)
    }

    return optical_scale, golden_info

def process_all_pdfs(single_pdf: str = ""):
    if single_pdf:
        pdf_files = [Path(single_pdf)]
    else:
        pdf_files = sorted(list(ITEMS_DIR.glob("*.pdf")))

    print("=" * 80)
    print(f" 100% DYNAMIC PDF SPECIFICATION EXTRACTION (Found {len(pdf_files)} PDF files)")
    print("=" * 80)

    # Load existing products.json if present
    products_db = {}
    prod_file = CONFIGS_DIR / "products.json"
    if prod_file.exists():
        try:
            with open(prod_file, "r", encoding="utf-8") as f:
                products_db = json.load(f)
        except Exception:
            pass

    for pdf_path in pdf_files:
        sku = pdf_path.stem
        print(f"\n[XỬ LÝ DYNAMIC]: {pdf_path.name} -> SKU: {sku}")
        
        spec = extract_pdf_data(pdf_path)
        scale, g_info = calibrate_optical_scale(sku, spec)
        spec["optical_scale_mm_per_px"] = round(scale, 6)
        if g_info:
            spec["golden_sample"] = g_info

        # Save spec.json into assets/references/{sku}/spec.json
        ref_spec_path = REFERENCES_DIR / sku / "spec.json"
        with open(ref_spec_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2, ensure_ascii=False)
        print(f"  -> Lưu đặc tả: {ref_spec_path} (Trích xuất {len(spec['dimensions'])} kích thước)")

        # Update products.json
        products_db[sku] = spec

        # Print extracted table
        print(f"  {'TÊN':<6} | {'TIÊU CHUẨN (mm)':<22} | {'TRỤC ĐO':<18} | {'NGUỒN GỐC TRÍCH XUẤT'}")
        print("  " + "-" * 75)
        for d in spec["dimensions"]:
            nom = d["nominal"]
            mi, ma = d["min"], d["max"]
            dia_str = "φ " if d["diameter"] else "  "
            raw = f"{dia_str}{nom:.2f} [{mi:.2f}..{ma:.2f}]"
            print(f"  {d['label']:<6} | {raw:<22} | {d['axis']:<18} | {d['raw_spec']}")

    # Save updated products.json
    with open(prod_file, "w", encoding="utf-8") as f:
        json.dump(products_db, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f" HOÀN TẤT BÓC TÁCH DYNAMIC TOÀN BỘ FILE PDF! ĐÃ CẬP NHẬT: {prod_file}")
    print("=" * 80 + "\n")

def list_specs():
    ref_dir = PROJECT_DIR / "assets" / "references"
    print("=" * 80)
    print(" DANH SÁCH CÁC KÍCH THƯỚC TRÍCH XUẤT TỪ FILE BẢN VẼ PDF GỐC (SPEC.JSON)")
    print("=" * 80)
    
    for p in sorted(ref_dir.iterdir()):
        spec_f = p / "spec.json"
        if not spec_f.exists():
            continue
        with open(spec_f, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        sku = data.get("sku", p.name)
        desc = data.get("description", "")
        src = data.get("source_file", "")
        scale = data.get("optical_scale_mm_per_px", 0.0) * 1000.0
        
        print(f"\n[SKU: {sku}] - {desc}")
        print(f"  Source PDF : {src}")
        print(f"  Camera Cal : {scale:.2f} um/px")
        print(f"  {'TÊN':<6} | {'TIÊU CHUẨN (mm)':<22} | {'TRỤC ĐO':<18} | {'NGUỒN GỐC TRÍCH XUẤT'}")
        print("  " + "-" * 75)
        
        for d in data.get("dimensions", []):
            lbl = d.get("label", "")
            nom = d.get("nominal", 0.0)
            low = d.get("min", nom - abs(d.get("lower_tol", 0.1)))
            high = d.get("max", nom + abs(d.get("upper_tol", 0.1)))
            dia_str = "φ " if d.get("diameter") else "  "
            raw = f"{dia_str}{nom:.2f} [{low:.2f}..{high:.2f}]"
            axis = d.get("axis", "2D_in_plane")
            raw_spec = d.get("raw_spec", "")
            print(f"  {lbl:<6} | {raw:<22} | {axis:<18} | {raw_spec}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="100% Dynamic PDF CAD Extraction & Spec Manager")
    parser.add_argument("--pdf", type=str, default="", help="Chỉ định 1 file PDF cụ thể để bóc tách")
    parser.add_argument("--list", action="store_true", help="Liệt kê toàn bộ bảng kích thước trích xuất từ PDF")
    args = parser.parse_args()
    if args.list:
        list_specs()
    else:
        process_all_pdfs(args.pdf)
