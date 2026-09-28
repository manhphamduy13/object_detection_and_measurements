import os
import sys
import re
import json
from pathlib import Path
import cv2
import numpy as np
import pypdf

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from classifier import segment_qc_cam_full

ITEMS_DIR = PROJECT_DIR / "items"
REFERENCES_DIR = PROJECT_DIR / "assets" / "references"
CONFIGS_DIR = PROJECT_DIR / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)


def parse_dimension_string(text):
    norm = text.replace("\xb1", "+-").replace("\u03c8", "D_").replace("\xd8", "D_").replace("I^", "D_")
    pattern = r'([A-Za-z]\w*)?\s*(?:D_)?\s*(\d+(?:\.\d+)?)\s*(?:\+-(\d+(?:\.\d+)?)|(\+\d+(?:\.\d+)?)\s*/\s*(-\d+(?:\.\d+)?)|(MAX|MIN))?\s*(?:mm)?'
    m = re.search(pattern, norm, re.IGNORECASE)
    if not m:
        return None

    lbl, nom_str, sym_tol, pos_tol, neg_tol, limit_type = m.groups()
    if not nom_str:
        return None

    nom = float(nom_str)
    dim = {
        "raw_spec": text.strip(),
        "label": lbl or "DIM",
        "nominal": nom,
        "diameter": ("D_" in norm or "O" in norm[:3])
    }

    if sym_tol:
        tol = float(sym_tol)
        dim["upper_tol"] = tol
        dim["lower_tol"] = -tol
        dim["min"] = round(nom - tol, 4)
        dim["max"] = round(nom + tol, 4)
        dim["tolerance_type"] = "symmetric"
    elif pos_tol and neg_tol:
        up = float(pos_tol)
        lo = float(neg_tol)
        dim["upper_tol"] = up
        dim["lower_tol"] = lo
        dim["min"] = round(nom + lo, 4)
        dim["max"] = round(nom + up, 4)
        dim["tolerance_type"] = "asymmetric"
    elif limit_type:
        dim["upper_tol"] = None
        dim["lower_tol"] = None
        dim["min"] = None if limit_type.upper() == "MAX" else nom
        dim["max"] = nom if limit_type.upper() == "MAX" else None
        dim["tolerance_type"] = limit_type.lower() + "_only"
    else:
        dim["upper_tol"] = 0.0
        dim["lower_tol"] = 0.0
        dim["min"] = nom
        dim["max"] = nom
        dim["tolerance_type"] = "exact"

    return dim


def process_pdf(pdf_path: Path):
    sku = pdf_path.stem
    reader = pypdf.PdfReader(str(pdf_path))
    page = reader.pages[0]
    text = page.extract_text()

    dimensions = []
    seen = set()

    for line in text.splitlines():
        line = line.strip()
        if "mm" in line.lower() or "max" in line.lower() or "\xb1" in line or "+/" in line:
            clean_line = re.sub(r'[\u4e00-\u9fff\u3000-\u303f].*', '', line).strip()
            if not clean_line:
                continue
            dim = parse_dimension_string(clean_line)
            if dim and dim["nominal"] > 0.1:
                key = (dim["label"], dim["nominal"], dim["min"], dim["max"])
                if key not in seen:
                    seen.add(key)
                    dimensions.append(dim)

    ref_folder = REFERENCES_DIR / sku
    ref_folder.mkdir(parents=True, exist_ok=True)
    images_folder = ref_folder / "images"
    images_folder.mkdir(parents=True, exist_ok=True)

    drawing_filename = None
    if len(page.images) > 0:
        drawing_img = page.images[0]
        drawing_filename = "drawing.png"
        drawing_path = ref_folder / drawing_filename
        with open(drawing_path, "wb") as f:
            f.write(drawing_img.data)
        print(f"  [Drawing] Extracted {drawing_filename} ({len(drawing_img.data)} bytes)")

    spec_data = {
        "sku": sku,
        "source_file": f"items/{pdf_path.name}",
        "item_name": sku,
        "part_no": sku,
        "unit": "mm",
        "drawing": drawing_filename,
        "review_status": "auto_extracted_from_pdf",
        "dimensions": dimensions
    }

    optical_scale = None
    ref_imgs = list(images_folder.glob("*.*"))
    if ref_imgs:
        golden_path = sorted(ref_imgs)[0]
        img_bgr = cv2.imread(str(golden_path))
        if img_bgr is not None:
            cnt, mask, _ = segment_qc_cam_full(img_bgr)
            if cnt is not None:
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (rw, rh), angle = rect
                long_px = max(rw, rh)
                short_px = min(rw, rh)

                linear_dims = [d for d in dimensions if not d["diameter"] and d["nominal"] > 3.0]
                linear_dims.sort(key=lambda x: x["nominal"], reverse=True)
                if linear_dims:
                    nom_l = linear_dims[0]["nominal"]
                    optical_scale = float(nom_l / long_px)
                    spec_data["optical_scale_mm_per_px"] = round(optical_scale, 6)
                    spec_data["golden_sample"] = {
                        "filename": golden_path.name,
                        "obb_long_px": round(long_px, 2),
                        "obb_short_px": round(short_px, 2),
                        "angle_deg": round(angle, 2),
                        "optical_scale_um_per_px": round(optical_scale * 1000.0, 2)
                    }
                    print(f"  [Golden Sample] {golden_path.name} -> Scale: {optical_scale*1000:.2f} um/px (L_px={long_px:.1f}, L_nom={nom_l:.2f}mm)")

    spec_path = ref_folder / "spec.json"
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec_data, f, indent=2, ensure_ascii=False)
    print(f"  [Spec] Saved {spec_path.name} with {len(dimensions)} dimensions.")

    return sku, spec_data


def main():
    print("===================================================================")
    print("  AUTOMATED CAD-TO-INSPECTION SETUP (100% Classical CV & Geometry)")
    print("===================================================================")

    if not ITEMS_DIR.exists():
        print(f"Error: items directory not found at {ITEMS_DIR}")
        return

    pdf_files = sorted(list(ITEMS_DIR.glob("*.pdf")))
    if not pdf_files:
        print(f"No PDF files found in {ITEMS_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF drawing files in items/:\n")

    all_specs = {}
    for pdf in pdf_files:
        print(f"--> Processing {pdf.name} ...")
        sku, spec = process_pdf(pdf)
        all_specs[sku] = spec
        print()

    prod_config_path = CONFIGS_DIR / "products.json"
    with open(prod_config_path, "w", encoding="utf-8") as f:
        json.dump(all_specs, f, indent=2, ensure_ascii=False)

    print("===================================================================")
    print(f"SUCCESS: Processed {len(all_specs)} SKUs.")
    print(f"Updated specifications in assets/references/<sku>/spec.json")
    print(f"Saved global product configurations to: {prod_config_path}")
    print("===================================================================\n")


if __name__ == "__main__":
    main()