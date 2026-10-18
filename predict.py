"""
Single Image Quality Control & Metrology Inspection Pipeline.

Usage:
    python predict.py path/to/sample_photo.png

Features:
- Subpixel edge detection & contour segmentation (QC_CAM Stage 2 zero-bleed).
- Rotation-invariant Hu moment + geometric fingerprint shape matching.
- Automatic spec loading (spec.json / 2D drawing dimensions).
- Caliper metrology (pixels -> micrometers/millimeters using optical scale).
- Per-sample overwrite mechanism:
  Re-predicting the SAME individual sample (keyed by unique sample stem/ID)
  overwrites its previous results, while preserving all other samples of the same SKU!
"""

import argparse
import copy
import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# Enable UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config
import classifier as clf
import measurer
from src.template_matcher import TemplateMatcher, public_result as public_template_result
from src.template_geometry import draw_profile


def parse_args():
    parser = argparse.ArgumentParser(description='Classify one photo and print its metrology report.')
    parser.add_argument('image', nargs='?', default=None, help='Path to the part photo to inspect')
    parser.add_argument('--image', dest='image_opt', default=None, help='Optional explicit image flag')
    parser.add_argument('--reference-dir', default=str(config.REFERENCE_DIR),
                        help='Class database folder (default: assets/references)')
    parser.add_argument('--output-dir', default=str(config.OUTPUT_DIR),
                        help='Base output folder (default: outputs)')
    parser.add_argument('--true-class', default=None,
                        help='Optional ground truth label for validation')
    parser.add_argument('--sku', default=None,
                        help='Inspect against this known SKU/golden recipe; skip automatic classification')
    parser.add_argument('--profile-lower', type=float, default=None,
                        help='Override inward contour tolerance in mm (with --sku)')
    parser.add_argument('--profile-upper', type=float, default=None,
                        help='Override outward contour tolerance in mm (with --sku)')
    args = parser.parse_args()
    if args.image_opt:
        args.image = args.image_opt
    if not args.image:
        parser.error("Vui lòng cung cấp đường dẫn ảnh (ví dụ: python predict.py <đường_dẫn_ảnh> hoặc python predict.py --image <đường_dẫn_ảnh>)")
    if not args.sku and (args.profile_lower is not None or args.profile_upper is not None):
        parser.error('--profile-lower / --profile-upper require --sku')
    return args


def to_python(obj):
    """Make numpy values JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (obj == float('inf') or np.isinf(obj) or np.isnan(obj)):
        return None
    return obj


def inspect_known_sample(img, recipe, spec, output_dir, stem, *, described=None, golden_image=None):
    """Studio and --sku share metrology, contour inspection and standard reports.

    The SKU is selected by the operator; recognition is explicitly not performed.
    Missing legacy golden geometry is taught from its reference photo in memory.
    Production recipes/specs are never changed by this function.
    """
    recipe, spec = copy.deepcopy(recipe), copy.deepcopy(spec)
    sku = spec['sku']
    recipe.setdefault('contour_profile_enabled', True)
    if described is None:
        described = clf.describe_image(img)
    if described is None:
        raise ValueError('Failed to segment sample image')
    engine = measurer.WorkpieceMeasurer()
    if golden_image is not None:
        golden_desc = clf.describe_image(golden_image)
        if golden_desc is None:
            raise ValueError('Failed to segment golden image')
        golden_contour = golden_desc['contour']
        frame = engine._contour_frame(golden_contour)
        if not recipe.get('golden_contour_local_px'):
            recipe['wcs'] = frame
            recipe['golden_contour_local_px'] = [
                engine._point_to_frame_local(p, frame).tolist()
                for p in golden_contour.reshape(-1, 2)
            ]
    elif not recipe.get('golden_contour_local_px'):
        raise ValueError('Golden contour is missing; select a reference photo or teach the SKU again')
    engine.caliper_recipes[sku] = recipe
    measurements = engine.measure(img, described['contour'], spec)
    # Legacy SCAN calipers can also report their golden comparison. P1/P2
    # signed deltas remain exclusive to rigid template calipers.
    if golden_image is not None:
        control = engine.measure(golden_image, golden_contour, spec)
        control_cals = control.get('recipe_calipers', {})
        for row in measurements.get('spec_report', []):
            if row.get('golden_length_mm') is not None:
                continue
            original = control_cals.get(row['label'], {})
            current = measurements.get('recipe_calipers', {}).get(row['label'], {})
            baseline = original.get('length_mm')
            if baseline is None:
                continue
            delta = row['measured_mm'] - baseline if row.get('measured_mm') is not None else None
            comparison = {'golden_length_mm': baseline, 'golden_source': 'image',
                          'total_delta_mm': delta}
            row.update(comparison)
            current.update(comparison)
    result = {
        'class': sku, 'spec': spec, 'accepted': True, 'shape_accepted': True,
        'classification_performed': False, 'inspection_mode': 'known_sku',
        'template_matching': {}, 'final_pass': bool(measurements['overall_pass']),
        'reference_image': recipe.get('reference_image', ''),
    }
    out = Path(output_dir)
    paths = {
        'panel_summary': out / 'visualizations' / f'{stem}.png',
        'measured_image': out / 'visualizations' / f'{stem}_measured.png',
        'prediction': out / 'predictions' / f'{stem}.json',
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    annotated = measurer.annotate_measurements(img, described['contour'], measurements, sku)
    if not cv2.imwrite(str(paths['measured_image']), annotated):
        raise OSError(f"Cannot write {paths['measured_image']}")
    save_visualization_panel(img, described, result, measurements, stem, paths['panel_summary'])
    record = {
        'stem': stem, 'predicted_class': None, 'expected_sku': sku,
        'inspection_mode': 'known_sku', 'classification_performed': False,
        'measurement_pass': measurements['caliper_pass'],
        'profile_pass': measurements.get('profile_pass'), 'final_pass': result['final_pass'],
        'measurements_px': measurements, 'spec_report': measurements.get('spec_report', []),
        'spec': spec, 'inspection_recipe': recipe,
        'outputs': {k: str(v) for k, v in paths.items()},
    }
    paths['prediction'].write_text(json.dumps(to_python(record), indent=2, ensure_ascii=False),
                                   encoding='utf-8')
    return measurements, paths


def save_visualization_panel(img, described, result, meas, stem, output_png_path, true_class=None):
    """
    Generates and saves the full 6-panel inspection summary figure:
    1. Original photo
    2. Grayscale + Edge Gradient / Otsu
    3. Classification and template/shape matching on the source image
    4. PDF specification with nominal values and tolerance ranges
    5. Edge measurement overlay and tolerance HUD
    6. Step-by-step QC Pipeline & Tolerance Verdict
    """
    features = described.get('features', {})
    contour = described['contour']
    profile = meas.get('contour_profile')
    known_sku = result.get('inspection_mode') == 'known_sku'

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # 1. Original
    axes[0][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0][0].set_title(f'1. Original Image\n{stem}', fontsize=10, fontweight='bold')

    # 2. Grayscale / Edge
    stage2_steps = described.get('stage2_steps')
    if stage2_steps is not None:
        axes[0][1].imshow(stage2_steps.step2_edge_map, cmap='gray')
        axes[0][1].set_title('2. Sobel Edge Gradient Map (Stage 2)', fontsize=10, fontweight='bold')
    else:
        axes[0][1].imshow(described['gray'], cmap='gray')
        axes[0][1].set_title(f'2. Grayscale (Otsu = {described.get("otsu", 0):.0f})', fontsize=10, fontweight='bold')

    # 3. Classification / template-shape matching on the same source image.
    matching_vis = img.copy()
    cv2.drawContours(matching_vis, [contour], -1, (0, 255, 0), 2, cv2.LINE_AA)
    tm = result.get('template_matching', {})
    shortlist = result.get('shortlist', [])[:3]
    match_lines = [f"MATCH: {result.get('class', 'UNKNOWN')}"]
    match_lines.extend(
        f"{i}. {cand['label']}  score={cand['shape_score']:.4f}"
        for i, cand in enumerate(shortlist, start=1)
    )
    match_lines.extend([
        f"TM: {tm.get('predicted_class', 'UNKNOWN')} sim={tm.get('similarity', 0.0):.3f}",
        f"TM margin={tm.get('margin', 0.0):.3f} pass={tm.get('accepted', False)}",
    ])
    for line_no, line in enumerate(match_lines):
        y = 28 + line_no * 22
        cv2.putText(matching_vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(matching_vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 1, cv2.LINE_AA)
    defect_map = tm.get('_defect_overlay_rgb')
    if profile and profile.get('aligned_golden_contour') is not None:
        profile_vis = img.copy()
        cv2.drawContours(profile_vis, [contour], -1, (0, 255, 0), 2, cv2.LINE_AA)
        draw_profile(profile_vis, profile)
        x, y, w, h = cv2.boundingRect(contour)
        margin = int(max(w, h) * .08) + 10
        roi = profile_vis[max(0, y-margin):min(img.shape[0], y+h+margin),
                          max(0, x-margin):min(img.shape[1], x+w+margin)]
        axes[0][2].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        axes[0][2].set_title(
            f"3. Golden Contour / Tolerance Band: {profile['status']}\n"
            f"in={profile.get('lower_mm', 0):.3f} / out={profile.get('upper_mm', 0):.3f} mm\n"
            "blue=inner, orange=outer, gray=golden, green=sample, red=out of band",
            fontsize=8.5, fontweight='bold',
        )
    elif defect_map is not None:
        axes[0][2].imshow(defect_map)
    else:
        axes[0][2].imshow(cv2.cvtColor(matching_vis, cv2.COLOR_BGR2RGB))
    if not (profile and profile.get('aligned_golden_contour') is not None):
        axes[0][2].set_title(
        '3. Template Similarity / Defect Map\n'
        f"sim={tm.get('similarity', 0.0):.3f} | defect={tm.get('defect_ratio', 0.0)*100:.1f}% "
        f"(max {tm.get('max_defect_ratio', 0.0)*100:.0f}%) | "
        f"missing={tm.get('missing_ratio', 0.0)*100:.1f}% | extra={tm.get('extra_ratio', 0.0)*100:.1f}%\n"
        'green=match, red=missing, blue=extra', fontsize=8.5, fontweight='bold'
        )

    # 4. Specification loaded from PDF, including all nominal/tolerance values.
    spec = result.get('spec') or {}
    spec_lines = [
        '4. Specification / Golden Recipe', '',
        f"SKU   : {spec.get('sku', 'N/A')}",
        f"Unit  : {spec.get('unit', 'mm')}",
        f"Scale : {float(spec.get('optical_scale_mm_per_px', 0.0) or 0.0) * 1000:.2f} um/px",
        '', 'Dimensions:'
    ]
    for dim in spec.get('dimensions', []):
        if dim.get('axis') == 'Z_height_side_view':
            suffix = ' [Z/not measured]'
        else:
            suffix = ''
        spec_lines.append(
            f" {dim.get('label', 'DIM'):<4} {float(dim.get('nominal', 0.0)):6.2f} mm  "
            f"[{float(dim.get('min', 0.0)):6.2f}..{float(dim.get('max', 0.0)):6.2f}]{suffix}"
        )
    if profile:
        spec_lines.extend([
            '', 'Contour offsets (per edge):',
            f" Inward : {profile.get('lower_mm', 0):.3f} mm",
            f" Outward: {profile.get('upper_mm', 0):.3f} mm",
        ])
    axes[1][0].text(0.02, 0.98, '\n'.join(spec_lines), va='top', ha='left',
                    family='monospace', fontsize=7.7)
    axes[1][0].axis('off')

    # 5. Caliper overlay and detailed tolerance table share one image.
    overlay = measurer.annotate_measurements(
        img, contour, meas, result['class'], template_matching=tm
    )
    axes[1][1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1][1].set_title('5. Edge Measurement + Dual QC Verdict', fontsize=10, fontweight='bold')

    # 6. Step-by-Step Inspection Pipeline
    caliper_ok = meas.get('caliper_pass', meas.get('overall_pass', True))
    template_ok = tm.get('template_quality_pass', tm.get('accepted', True)) if tm else True

    lines = [
        '6. QC Inspection Metrology Summary\n',
        '[STEP 1: EDGE & MORPHOLOGY]',
        f'  Contour Points: {len(contour)}',
        f'  Area          : {features.get("area", 0.0):.0f} px',
        f'  Perimeter     : {features.get("perimeter", 0.0):.1f} px',
        f'  Extent        : {features.get("extent", 0.0):.3f}',
        f'  Solidity      : {features.get("solidity", 0.0):.3f}',
        f'  Centroid      : ({features.get("centroid", (0, 0))[0]:.1f}, {features.get("centroid", (0, 0))[1]:.1f})',
        '',
        '[STEP 2: MULTI-CUE CLASSIFICATION]',
        '  Top Candidates (matchShapes + Hu):'
    ]
    for i, cand in enumerate(result.get('shortlist', [])[:3], start=1):
        lines.append(f'   {i}. {cand["label"]} score={cand["shape_score"]:.4f}')

    lines.append(f'  Shape score : {result.get("best_score", 0.0):.4f}')
    lines.append(f'  Margin      : {result.get("margin", 0.0):.4f}')
    lines.append(f'  TM Matched  : {tm.get("predicted_class", "UNKNOWN")}')
    lines.append(f'  Similarity  : {tm.get("similarity", 0.0):.4f} (min {tm.get("min_similarity", 0.60):.2f})')
    lines.append(f'  Defect area : {tm.get("defect_ratio", 0.0)*100:.2f}% (max {tm.get("max_defect_ratio", 0.45)*100:.0f}%)')
    lines.append(f'  Missing     : {tm.get("missing_ratio", 0.0)*100:.2f}%')
    lines.append(f'  Extra/Burr  : {tm.get("extra_ratio", 0.0)*100:.2f}%')
    lines.append('')
    lines.append('[STEP 3: SPECIFICATION & TOLERANCES]')
    spec = result.get('spec')
    if spec:
        lines.append(f'  SKU         : {spec.get("sku")}')
        scale_val = spec.get("optical_scale_mm_per_px")
        if scale_val:
            lines.append(f'  Scale       : {scale_val*1000:.2f} um/px')
    else:
        lines.append('  Spec        : No spec.json loaded (UNKNOWN)')

    lines.append('')
    lines.append('[STEP 4: CALIPER METROLOGY GATE]')
    lines.append(f'  Length (L)  : {meas.get("Length_L_px", 0.0):.2f} px ({meas.get("Length_L_mm", 0.0):.2f} mm)')
    lines.append(f'  Width (W)   : {meas.get("Width_W_px", 0.0):.2f} px ({meas.get("Width_W_mm", 0.0):.2f} mm)')
    lines.append(f'  Caliper Gate: {"PASS" if caliper_ok else "FAIL"}')

    lines.append('')
    lines.append('[STEP 5: TEMPLATE DEFECT GATE]')
    lines.append(f'  Template QC : {"PASS" if template_ok else "FAIL"}')

    if known_sku:
        lines = [
            '6. Inspection Summary', '',
            f"Selected SKU: {result['class']}",
            'Recognition: not run (operator selected SKU)', '',
            f"Dimensions: {'PASS' if meas.get('dimension_pass', False) else 'FAIL'}",
            f"Golden contour: {profile['status'] if profile else 'DISABLED'}",
        ]
        if profile:
            lines.extend([
                f"Max inward : {profile.get('max_inward_mm', 0):.3f} mm",
                f"Max outward: {profile.get('max_outward_mm', 0):.3f} mm",
                f"Outside band: {profile.get('outside_point_count', 0)} points",
            ])
            if profile.get('error'):
                lines.append(profile['error'])
        lines.extend(['', 'Measurement vs Golden (mm):'])
        for row in meas.get('spec_report', [])[:7]:
            val, baseline, delta = row.get('measured_mm'), row.get('golden_length_mm'), row.get('total_delta_mm')
            lines.append(f" {row['label']}: {val:.3f}  {row['status']}" if val is not None
                         else f" {row['label']}: N/A FAIL")
            if baseline is not None and delta is not None:
                lines.append(f"   golden={baseline:.3f}, delta={delta:+.3f}")
            if row.get('delta_p1_mm') is not None:
                lines.append(f"   P1={row['delta_p1_mm']:+.3f}, P2={row['delta_p2_mm']:+.3f}")

    lines.append('')
    verdict = result['class']
    overall_pass = bool(result.get('final_pass', caliper_ok and template_ok and result.get('accepted', False)))
    if known_sku:
        headline = f"VERDICT: {verdict} ({'PASS' if overall_pass else 'FAIL'})"
    elif overall_pass and (not true_class or verdict == true_class):
        headline = f'VERDICT: {verdict} (DUAL PASS - OK)'
    elif not result.get('shape_accepted', False):
        headline = 'VERDICT: SHAPE REJECTED (UNKNOWN)'
    elif not caliper_ok and not template_ok:
        headline = 'VERDICT: REJECTED (CALIPER + DEFECT FAIL)'
    elif not caliper_ok:
        headline = 'VERDICT: REJECTED (CALIPER TOLERANCE FAIL)'
    elif not template_ok:
        headline = 'VERDICT: REJECTED (TEMPLATE DEFECT EXCEEDED)'
    else:
        headline = f'VERDICT: {verdict} (MISMATCH)'

    axes[1][2].text(0.02, 0.98, '\n'.join(lines), va='top', ha='left',
                    family='monospace', fontsize=8.2)
    axes[1][2].set_title(
        headline, color='green' if overall_pass else 'red',
        fontsize=10.5, fontweight='bold', pad=8
    )
    axes[1][2].axis('off')

    plt.tight_layout()
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_png_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def update_predictions_json(json_path: Path, new_entry: dict, stem: str) -> tuple:
    """
    Updates predictions.json atomically.
    If an entry with the same sample stem exists, it OVERWRITES that exact entry.
    Otherwise, it appends the new entry.
    Preserves all other samples of the same SKU and different SKUs.
    Returns (is_overwritten: bool, total_count: int).
    """
    entries = []
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    entries = data
        except Exception:
            entries = []

    found_idx = -1
    for idx, e in enumerate(entries):
        e_fn = e.get('filename') or e.get('image') or ''
        e_stem = e.get('stem') or Path(e_fn).stem
        if e_stem == stem:
            found_idx = idx
            break

    is_overwritten = False
    if found_idx >= 0:
        entries[found_idx] = new_entry
        is_overwritten = True
    else:
        entries.append(new_entry)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(to_python(entries), f, indent=2, ensure_ascii=False)

    return is_overwritten, len(entries)


def update_evaluation_csv(csv_path: Path, row_data: dict, stem: str) -> bool:
    """
    Updates evaluation.csv atomically.
    Overwrites the record if sample stem matches, otherwise appends.
    """
    rows = []
    fieldnames = ['filename', 'true_class', 'predicted_class', 'accepted',
                  'best_score', 'second_score', 'margin',
                  'template_class', 'template_similarity', 'template_margin',
                  'template_defect_ratio', 'template_missing_ratio', 'template_extra_ratio', 'template_pass',
                  'measurement_pass', 'final_pass', 'latency_ms',
                  'length_px', 'width_px']

    if csv_path.exists():
        try:
            with open(csv_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
        except Exception:
            rows = []

    found_idx = -1
    for idx, r in enumerate(rows):
        r_stem = Path(r.get('filename', '')).stem
        if r_stem == stem:
            found_idx = idx
            break

    is_overwritten = False
    if found_idx >= 0:
        rows[found_idx] = row_data
        is_overwritten = True
    else:
        rows.append(row_data)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                clean_row = {k: r.get(k, '') for k in fieldnames}
                writer.writerow(clean_row)
    except PermissionError:
        print(f"  [CẢNH BÁO] Không thể ghi đè file {csv_path.name} do đang được mở bởi ứng dụng khác (Excel).")

    return is_overwritten


def main():
    args = parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.exists():
        sys.exit(f'ERROR: File not found: {args.image}')

    t0 = time.perf_counter()
    img = cv2.imread(str(image_path))
    if img is None:
        sys.exit(f'ERROR: Could not decode image: {args.image}')

    stem = image_path.stem
    filename = image_path.name
    out_dir = Path(args.output_dir).resolve()
    vis_dir = out_dir / 'visualizations'
    steps_base_dir = out_dir / 'steps'
    preds_dir = out_dir / 'predictions'
    pred_summary_file = out_dir / 'predictions.json'
    eval_csv_file = out_dir / 'evaluation.csv'

    # Build reference library & calibrate
    reference_dir = Path(args.reference_dir).resolve()
    if args.sku:
        spec_path = reference_dir / args.sku / 'spec.json'
        spec = json.loads(spec_path.read_text(encoding='utf-8'))
        recipe = copy.deepcopy(measurer.WorkpieceMeasurer().caliper_recipes.get(args.sku, {}))
        reference_path = recipe.get('reference_image')
        reference_path = (Path(config.PROJECT_DIR) / reference_path) if reference_path else None
        if reference_path is None or not reference_path.is_file():
            photos = sorted(p for p in (reference_dir / args.sku / 'images').iterdir()
                            if p.suffix.lower() in config.IMAGE_EXTENSIONS)
            if not photos:
                sys.exit(f'No golden image for {args.sku}')
            reference_path = photos[0]
        golden_image = cv2.imread(str(reference_path))
        if golden_image is None:
            sys.exit(f'Cannot read golden image: {reference_path}')
        recipe['reference_image'] = str(reference_path)
        if args.profile_lower is not None:
            recipe['contour_lower_mm'] = args.profile_lower
        if args.profile_upper is not None:
            recipe['contour_upper_mm'] = args.profile_upper
        measured, paths = inspect_known_sample(img, recipe, spec, out_dir, stem,
                                               golden_image=golden_image)
        print(f"SKU {args.sku}: {'PASS' if measured['overall_pass'] else 'FAIL'} (known SKU)")
        profile = measured.get('contour_profile')
        print(f"Golden contour: {profile['status'] if profile else 'DISABLED'}")
        for name, path in paths.items():
            print(f'{name}: {path}')
        return
    library, contours, failures = clf.build_database(reference_dir)
    for path, reason in failures:
        print(f'  skipped {path.name}: {reason}')
    if not any(contours.values()):
        sys.exit(f'No reference photos found under {reference_dir} -- '
                 'add images to assets/references/<class>/images/ first')

    reference_items = clf.load_dataset(reference_dir)
    thresholds, _info = clf.calibrate_thresholds(reference_items, library, contours)
    template_matcher = TemplateMatcher(
        reference_dir,
        min_similarity=config.TEMPLATE_MIN_SIMILARITY,
        min_margin=config.TEMPLATE_MIN_MARGIN,
        max_defect_ratio=config.TEMPLATE_MAX_DEFECT_RATIO,
    )

    # Step 1: Preprocessing & segmentation
    described = clf.describe_image(img)
    if described is None:
        sys.exit(f'Failed to segment workpiece in {args.image}')

    print('=' * 75)
    print(' QUALITY CONTROL MACHINE VISION INSPECTION REPORT')
    print('=' * 75)
    print(f'SẢN PHẨM (SAMPLE ID) : {stem}')
    print(f'FILE ẢNH GỐC         : {filename}')
    print(f'THỜI ĐIỂM KIỂM TRA    : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

    # Step 1: Preprocessing output
    det_method = 'QC_CAM Stage 2 Edge Detection (Sobel gradient + zero-bleed)' if described.get('stage2_steps') else 'Laboratory Backlight Otsu Segmentation'
    feat = described['features']
    print('[BƯỚC 1: TIỀN XỬ LÝ & BÓC TÁCH BIÊN DẠNG VẬT THỂ]')
    print(f'  Phương pháp  : {det_method}')
    print(f'  Diện tích    : {feat["area"]:.1f} px')
    print(f'  Chu vi       : {feat["perimeter"]:.1f} px')
    print(f'  Tỷ lệ cạnh   : {feat["rect_aspect"]:.3f} (xoay bất biến)')
    print(f'  Số lỗ hổng   : {feat["hole_count"]}')

    # Step 2: Classification
    result = clf.predict(img, library, contours, thresholds)
    shape_accepted = bool(result.get('accepted'))
    expected_label = result.get('best_label')
    template_result = template_matcher.match(img, described['object_mask'], expected_label)
    result['shape_accepted'] = shape_accepted
    result['template_matching'] = template_result
    result['accepted'] = bool(shape_accepted and template_result['accepted'])
    result['class'] = expected_label if result['accepted'] else 'UNKNOWN'
    # Keep the best candidate's spec available for the measurement gate.
    entry = library.get(expected_label, {})
    result['spec'] = entry.get('spec')
    result['drawing'] = entry.get('drawing')
    print('\n[BƯỚC 2: PHÂN LOẠI LINH KIỆN & SO KHỚP MẪU CHUẨN]')
    print(f'  Kết quả nhận diện : {result["class"]}')
    print(f'  Trạng thái        : {"CHẤP NHẬN (PASS)" if result["accepted"] else "KHÔNG XÁC ĐỊNH (UNKNOWN)"}')
    if result.get('shortlist'):
        print(f'  Điểm khớp (Hu)    : {result["best_score"]:.4f} (ngưỡng chấp nhận < {thresholds["match_threshold"]:.2f})')
        print(f'  Biên độ an toàn   : {result["margin"]:.4f} (khoảng cách so với ứng viên thứ 2)')

    print(f'  Template matching : {template_result["predicted_class"]}')
    print(f'  Similarity        : {template_result["similarity"]:.4f} (min {template_result["min_similarity"]:.2f})')
    print(f'  Template margin   : {template_result["margin"]:.4f} (min {template_result["min_margin"]:.2f})')
    print(f'  Template gate     : {"PASS" if template_result["accepted"] else "FAIL"}')

    # Step 3: Spec Loading
    print('\n[BƯỚC 3: TẢI ĐẶC TẢ KỸ THUẬT TIÊU CHUẨN (spec.json)]')
    spec = result.get('spec')
    if spec:
        print(f'  Mã SKU            : {spec.get("sku")}')
        print(f'  Tên sản phẩm      : {spec.get("item_name")}')
        print(f'  Vật liệu          : {spec.get("material")}')
        print(f'  Đơn vị đo tiêu chuẩn : {spec.get("unit", "mm")}')
    else:
        print('  Không có file spec.json tương ứng hoặc kết quả là UNKNOWN')

    # Step 4: Subpixel Caliper Metrology & Quality Evaluation
    meas = measurer.WorkpieceMeasurer().measure(
        img, described['contour'], spec, template_matching=template_result
    )
    caliper_pass = bool(meas.get('caliper_pass', meas.get('overall_pass', False)))
    template_pass = bool(template_result.get('template_quality_pass', template_result.get('accepted', True)))
    final_pass = bool(result['accepted'] and caliper_pass and template_pass)
    result['caliper_pass'] = caliper_pass
    result['template_pass'] = template_pass
    result['measurement_pass'] = caliper_pass
    result['final_pass'] = final_pass
    active_cals = meas.get('active_draw_calipers', [])
    print('\n[BƯỚC 4: ĐO ĐẠC KÍCH THƯỚC QUANG HỌC CALIPER (THEO CẤU HÌNH NGƯỜI DÙNG)]')
    print('  ' + '-' * 65)
    print(f'  {"TÊN KÍCH THƯỚC / TIA ĐO":<32} | {"KÍCH THƯỚC ĐO ĐƯỢC":<28}')
    print('  ' + '-' * 65)
    if active_cals:
        for c in active_cals:
            lbl = c['label']
            v_mm = c.get('val_mm', 0.0)
            p1, p2 = c['p1'], c['p2']
            px_len = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            value_text = f'{v_mm:6.3f} mm' if v_mm is not None else 'N/A - FAIL'
            print(f'  Kích thước [{lbl}]: {lbl:<18} | {px_len:8.2f} px ({value_text})')
            if c.get('measure_type') == 'template_contour' and c.get('valid', True):
                detail = meas.get('recipe_calipers', {}).get(lbl, {})
                print(
                    f'    TEMPLATE: golden={detail.get("golden_length_mm", 0.0):.3f} mm '
                    f'+ delta_P1={detail.get("delta_p1_mm", 0.0):+.3f} mm '
                    f'+ delta_P2={detail.get("delta_p2_mm", 0.0):+.3f} mm'
                )
    else:
        print(f'  {"Length (L) - Chiều dài tổng":<32} | {meas["Length_L_px"]:>8.2f} px')
        print(f'  {"Width (W) - Chiều rộng tổng":<32} | {meas["Width_W_px"]:>8.2f} px')
    print('  ' + '-' * 65)

    scale = spec.get('optical_scale_mm_per_px') if spec else None
    if scale:
        print('\n[BƯỚC 4B: QUY ĐỔI SANG MILIMET & KIỂM TRA DUNG SAI (THEO BẢN VẼ)]')
        print(f'  Hệ số quang học camera (Optical Scale): {scale * 1000.0:.2f} um/px ({scale:.6f} mm/px)')
        print('  ' + '-' * 85)
        print(f'  {"KÍCH THƯỚC":<10} | {"ĐO ĐƯỢC (mm)":<14} | {"TIÊU CHUẨN (mm)":<24} | {"KẾT LUẬN":<12} | GHI CHÚ')
        print('  ' + '-' * 85)

        spec_report = meas.get('spec_report', [])
        if spec_report:
            for item in spec_report:
                lbl = item['label']
                val = item['measured_mm']
                nom = item['nominal_mm']
                mi = item['min_mm']
                ma = item['max_mm']
                status = item['status']
                note = item['note']

                val_str = f'{val:>7.2f} mm' if val is not None else '     N/A  '
                range_str = f'{nom:>5.2f} [{mi:.2f}, {ma:.2f}] mm' if (mi is not None and ma is not None) else f'{nom:>5.2f} MAX mm'
                print(f'  {lbl:<10} | {val_str} | {range_str:<24} | {status:<12} | {note}')
        else:
            print('  (Không có kích thước nào được cấu hình để kiểm tra dung sai)')
        print('  ' + '-' * 85)

        unmeas = meas.get('unmeasured_dimensions', [])
        if unmeas:
            unmeas_labels = [f"{u['label']} ({u['reason']})" for u in unmeas]
            print(f'  [THÔNG SỐ BỎ QUA SO SÁNH DO NGƯỜI DÙNG ĐÃ XÓA/KHÔNG CHỌN]: {", ".join(unmeas_labels)}')
            print('  ' + '-' * 85)

        profile = meas.get('contour_profile')
        if profile:
            print('\n[BƯỚC 4B.1: GOLDEN CONTOUR TOLERANCE BAND]')
            print(f'  Lệch vào cho phép: {profile.get("lower_mm", 0.0):.3f} mm')
            print(f'  Lệch ra cho phép : {profile.get("upper_mm", 0.0):.3f} mm')
            print(f'  Sai lệch lớn nhất: {profile.get("max_deviation_mm", 0.0):.3f} mm')
            print(f'  P95 deviation    : {profile.get("p95_deviation_mm", 0.0):.3f} mm')
            print(
                f'  Điểm ngoài band  : {profile.get("outside_point_count", 0)}/'
                f'{profile.get("sampled_point_count", 0)}'
            )
            print(f'  Profile gate     : {profile["status"]}')
            if profile.get('error'):
                print(f'  Error            : {profile["error"]}')

    # Step 4C: Template Matching Quality Inspection (Defect analysis)
    print('\n[BƯỚC 4C: ĐÁNH GIÁ CHẤT LƯỢNG BỀ MẶT & BIÊN DẠNG (TEMPLATE MATCHING QC)]')
    print('  ' + '-' * 85)
    sim_val = template_result.get('similarity', 0.0)
    min_sim_val = template_result.get('min_similarity', 0.60)
    def_val = template_result.get('defect_ratio', 0.0) * 100.0
    max_def_val = template_result.get('max_defect_ratio', 0.45) * 100.0
    mis_val = template_result.get('missing_ratio', 0.0) * 100.0
    max_mis_val = template_result.get('max_missing_ratio', 0.35) * 100.0
    ext_val = template_result.get('extra_ratio', 0.0) * 100.0
    max_ext_val = template_result.get('max_extra_ratio', 0.35) * 100.0

    print(f'  Độ tương đồng biên dạng (Similarity) : {sim_val:.4f} (Tiêu chuẩn >= {min_sim_val:.2f}) -> {"PASS" if sim_val >= min_sim_val else "FAIL"}')
    print(f'  Tỷ lệ khuyết tật tổng (Defect Ratio) : {def_val:.2f}% (Tối đa <= {max_def_val:.1f}%) -> {"PASS" if def_val <= max_def_val else "FAIL"}')
    print(f'  Vật liệu bị khuyết/mẻ (Missing)      : {mis_val:.2f}% (Tối đa <= {max_mis_val:.1f}%) -> {"PASS" if mis_val <= max_mis_val else "FAIL"}')
    print(f'  Bavia/vật liệu thừa (Extra/Burr)     : {ext_val:.2f}% (Tối đa <= {max_ext_val:.1f}%) -> {"PASS" if ext_val <= max_ext_val else "FAIL"}')
    print(f'  Cổng đánh giá Template QC Gate       : {"PASS (ĐẠT CHUẨN)" if template_pass else "FAIL (KHÔNG ĐẠT)"}')
    print('  ' + '-' * 85)

    print('\n[KẾT LUẬN KIỂM SOÁT CHẤT LƯỢNG TOÀN DIỆN (DUAL QC VERDICT)]')
    if final_pass:
        print('  >>> KẾT QUẢ: CHẤP NHẬN (PASS - CALIPER METROLOGY OK + TEMPLATE QC OK) <<<')
    elif not caliper_pass and not template_pass:
        print('  >>> KẾT QUẢ: TỪ CHỐI (FAIL - CẢ DUNG SAI KÍCH THƯỚC VÀ KHUYẾT TẬT BỀ MẶT ĐỀU LỖI) <<<')
    elif not caliper_pass:
        print('  >>> KẾT QUẢ: TỪ CHỐI (FAIL - KÍCH THƯỚC CALIPER VƯỢT DUNG SAI CAD CHO PHÉP) <<<')
    else:
        print('  >>> KẾT QUẢ: TỪ CHỐI (FAIL - BIÊN DẠNG KHUYẾT TẬT VƯỢT TIÊU CHUẨN TEMPLATE) <<<')

    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Step 5: Export Files with Per-Sample Overwrite Mechanism
    vis_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    step_dir = steps_base_dir / stem

    # Overwrite check: If step_dir exists, clear old step files of this sample
    if step_dir.exists():
        for old_f in step_dir.glob('*.*'):
            try:
                old_f.unlink()
            except Exception:
                pass
    step_dir.mkdir(parents=True, exist_ok=True)

    # Save 5 intermediate step images
    gray = described['gray']
    stage2_steps = described.get('stage2_steps')
    if stage2_steps is not None:
        cv2.imwrite(str(step_dir / 'step1_roi_crop.png'), stage2_steps.step1_roi)
        cv2.imwrite(str(step_dir / 'step2_edge_gradient_map.png'), stage2_steps.step2_edge_map)
        cv2.imwrite(str(step_dir / 'step3_workpiece_mask.png'), stage2_steps.step3_mask)
        cv2.imwrite(str(step_dir / 'step3_contours_extracted.png'), stage2_steps.step3_contours_vis)
    else:
        cv2.imwrite(str(step_dir / 'step1_roi_crop.png'), gray)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        sobelx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.clip(cv2.magnitude(sobelx, sobely), 0, 255).astype(np.uint8)
        cv2.imwrite(str(step_dir / 'step2_edge_gradient_map.png'), mag)
        cv2.imwrite(str(step_dir / 'step3_workpiece_mask.png'), described['mask'])
        vis_cnt = img.copy()
        cv2.drawContours(vis_cnt, [described['contour']], -1, (0, 255, 0), 2)
        cv2.imwrite(str(step_dir / 'step3_contours_extracted.png'), vis_cnt)

    # High-resolution annotated measurement overlay (Caliper Metrology + Template Defect HUD)
    annotated = measurer.annotate_measurements(
        img, described['contour'], meas, result['class'],
        template_matching=template_result
    )
    cv2.imwrite(str(step_dir / 'step4_caliper_scanlines.png'), annotated)
    cv2.imwrite(str(step_dir / 'step5_final_inspection.png'), annotated)

    # Save visualization files for this specific sample (Ghi đè cho chính mẫu này)
    out_measured_stem = vis_dir / f'{stem}_measured.png'
    out_measured_latest = out_dir / 'prediction_measured.png'
    out_panel_stem = vis_dir / f'{stem}.png'

    cv2.imwrite(str(out_measured_stem), annotated)
    cv2.imwrite(str(out_measured_latest), annotated)
    save_visualization_panel(img, described, result, meas, stem, out_panel_stem, args.true_class)

    # Save per-sample detailed JSON
    sample_json_path = preds_dir / f'{stem}.json'
    sample_record = {
        'stem': stem,
        'filename': filename,
        'image_path': str(image_path),
        'timestamp': datetime.now().isoformat(),
        'predicted_class': result['class'],
        'true_class': args.true_class or result['class'],
        'accepted': result['accepted'],
        'best_score': result['best_score'],
        'margin': result['margin'],
        'template_matching': public_template_result(template_result),
        'caliper_pass': caliper_pass,
        'template_pass': template_pass,
        'measurement_pass': caliper_pass,
        'final_pass': final_pass,
        'latency_ms': round(latency_ms, 2),
        'features': feat,
        'measurements_px': meas,
        'spec': spec,
        'spec_report': meas.get('spec_report', []),
        'outputs': {
            'measured_image': str(out_measured_stem),
            'panel_summary': str(out_panel_stem),
            'step_directory': str(step_dir)
        }
    }
    with open(sample_json_path, 'w', encoding='utf-8') as f:
        json.dump(to_python(sample_record), f, indent=2, ensure_ascii=False)

    # Atomically update consolidated predictions.json (Overwrite this product, preserve others)
    is_overwritten, total_count = update_predictions_json(pred_summary_file, sample_record, stem)

    # Atomically update evaluation.csv
    second_s = result['shortlist'][1]['shape_score'] if len(result.get('shortlist', [])) > 1 else None
    csv_row = {
        'filename': filename,
        'true_class': args.true_class or result['class'],
        'predicted_class': result['class'],
        'accepted': result['accepted'],
        'best_score': f"{result['best_score']:.6f}",
        'second_score': f"{second_s:.6f}" if second_s is not None else '',
        'margin': f"{result['margin']:.6f}",
        'template_class': template_result['predicted_class'],
        'template_similarity': f"{template_result['similarity']:.6f}",
        'template_margin': f"{template_result['margin']:.6f}",
        'template_defect_ratio': f"{template_result['defect_ratio']:.6f}",
        'template_missing_ratio': f"{template_result['missing_ratio']:.6f}",
        'template_pass': template_pass,
        'caliper_pass': caliper_pass,
        'measurement_pass': caliper_pass,
        'final_pass': final_pass,
        'latency_ms': f"{latency_ms:.2f}",
        'length_px': f"{meas['Length_L_px']:.2f}",
        'width_px': f"{meas['Width_W_px']:.2f}"
    }
    update_evaluation_csv(eval_csv_file, csv_row, stem)

    # Step 5 Console Report
    print('\n' + '=' * 75)
    print('[BƯỚC 5: XUẤT KẾT QUẢ ĐO ĐẠC & CƠ CHẾ GHI ĐÈ THEO SẢN PHẨM]')
    print('=' * 75)
    if is_overwritten:
        print(f'  >>> TRẠNG THÁI: [GHI ĐÈ KẾT QUẢ CŨ] <<<')
        print(f'  Sản phẩm "{stem}" đã từng được đo trước đó.')
        print(f'  Toàn bộ kết quả cũ của riêng sản phẩm này đã được GHI ĐÈ thành công!')
    else:
        print(f'  >>> TRẠNG THÁI: [TẠO MỚI KẾT QUẢ] <<<')
        print(f'  Sản phẩm "{stem}" lần đầu được đo. Đã thêm mới vào hệ thống.')

    print(f'  (LƯU Ý: Các sản phẩm khác cùng loại "{result["class"]}" và SKU khác vẫn được BẢO TOÀN NGUYÊN VẸN)')
    print('\n  CÁC FILE OUTPUT ĐƯỢC XUẤT RA TẠI:')
    print(f'  1. Ảnh đo đạc kích thước Caliper (High-Res):')
    print(f'     -> {out_measured_stem}')
    print(f'  2. Ảnh tổng hợp 6 panel kiểm tra chất lượng:')
    print(f'     -> {out_panel_stem}')
    print(f'  3. Thư mục 5 bước bóc tách trung gian:')
    print(f'     -> {step_dir}')
    print(f'        * step1_roi_crop.png')
    print(f'        * step2_edge_gradient_map.png')
    print(f'        * step3_workpiece_mask.png')
    print(f'        * step3_contours_extracted.png')
    print(f'        * step4_caliper_scanlines.png')
    print(f'        * step5_final_inspection.png')
    print(f'  4. Dữ liệu JSON chi tiết của riêng sản phẩm này:')
    print(f'     -> {sample_json_path}')
    print(f'  5. File tổng hợp dữ liệu hệ thống (Đã cập nhật):')
    print(f'     -> {pred_summary_file} (Tổng số sản phẩm đã lưu: {total_count})')
    print(f'  6. File CSV đánh giá toàn hệ thống (Đã cập nhật):')
    print(f'     -> {eval_csv_file}')
    print(f'  7. Ảnh xem nhanh kết quả mới nhất:')
    print(f'     -> {out_measured_latest}')
    print('=' * 75)


if __name__ == '__main__':
    main()
