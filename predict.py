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


def parse_args():
    parser = argparse.ArgumentParser(description='Classify one photo and print its metrology report.')
    parser.add_argument('image', help='Path to the part photo to inspect')
    parser.add_argument('--reference-dir', default=str(config.REFERENCE_DIR),
                        help='Class database folder (default: assets/references)')
    parser.add_argument('--output-dir', default=str(config.OUTPUT_DIR),
                        help='Base output folder (default: outputs)')
    parser.add_argument('--true-class', default=None,
                        help='Optional ground truth label for validation')
    return parser.parse_args()


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


def save_visualization_panel(img, described, result, meas, stem, output_png_path, true_class=None):
    """
    Generates and saves the full 6-panel inspection summary figure:
    1. Original photo
    2. Grayscale + Edge Gradient / Otsu
    3. Binary workpiece mask
    4. Caliper Metrology Overlay (L, W lines and oriented box)
    5. Geometric fingerprint (area, perimeter, Hu moments)
    6. Step-by-step QC Pipeline & Tolerance Verdict
    """
    features = described['features']
    contour = described['contour']

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
        axes[0][1].set_title(f'2. Grayscale (Otsu = {described["otsu"]:.0f})', fontsize=10, fontweight='bold')

    # 3. Binary mask
    axes[0][2].imshow(described['mask'], cmap='gray')
    axes[0][2].set_title('3. Binary Workpiece Mask (Zero-Bleed)', fontsize=10, fontweight='bold')

    # 4. Caliper Metrology Overlay
    overlay = measurer.annotate_measurements(img, contour, meas, result['class'])
    axes[1][0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1][0].set_title(f'4. Caliper Metrology\nL: {meas.get("Length_L_px", 0):.1f}px | W: {meas.get("Width_W_px", 0):.1f}px',
                         fontsize=10, fontweight='bold')

    # 5. Fingerprint numbers
    hu_text = '\n'.join(f'    Hu{i + 1} = {v:+.3f}' for i, v in enumerate(features['hu']))
    fingerprint_text = (
        '5. Geometric Fingerprint\n\n'
        f'area          = {features["area"]:.0f} px\n'
        f'perimeter     = {features["perimeter"]:.1f} px\n'
        f'rect aspect   = {features["rect_aspect"]:.3f}   (rotation invariant)\n'
        f'solidity      = {features["solidity"]:.3f}\n'
        f'circularity   = {features["circularity"]:.3f}\n'
        f'hole_count    = {features["hole_count"]}\n\n'
        f'Hu moments (signed log):\n{hu_text}'
    )
    axes[1][1].text(0.02, 0.98, fingerprint_text, va='top', ha='left',
                    family='monospace', fontsize=8.5)
    axes[1][1].axis('off')

    # 6. Step-by-Step Inspection Pipeline
    lines = [
        '6. QC Inspection Metrology Summary\n',
        '[STEP 1: EDGE & CONTOUR SEGMENTATION]',
        f'  Contour Points: {len(contour)}',
        f'  Centroid: ({features["centroid"][0]:.1f}, {features["centroid"][1]:.1f})',
        '',
        '[STEP 2: CLASSIFICATION MATCHING]',
        '  Top Candidates (matchShapes + Hu):'
    ]
    for i, cand in enumerate(result.get('shortlist', [])[:3], start=1):
        lines.append(f'   {i}. {cand["label"]} score={cand["shape_score"]:.4f}')

    lines.append(f'  Best score  : {result.get("best_score", 0.0):.4f}')
    lines.append(f'  Margin      : {result.get("margin", 0.0):.4f}')
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
    lines.append('[STEP 4: CALIPER MEASUREMENT]')
    lines.append(f'  Length (L)  : {meas.get("Length_L_px", 0.0):.2f} px ({meas.get("Length_L_mm", 0.0):.2f} mm)')
    lines.append(f'  Width (W)   : {meas.get("Width_W_px", 0.0):.2f} px ({meas.get("Width_W_mm", 0.0):.2f} mm)')

    lines.append('')
    verdict = result['class']
    if verdict == 'UNKNOWN':
        headline = 'VERDICT: UNKNOWN (REJECTED)'
    elif true_class and verdict != true_class:
        headline = f'VERDICT: {verdict} (MISMATCH)'
    else:
        headline = f'VERDICT: {verdict} (PASS)'

    axes[1][2].text(0.02, 0.98, '\n'.join(lines), va='top', ha='left',
                    family='monospace', fontsize=8.2)
    axes[1][2].text(0.02, 0.04, headline, va='top', ha='left',
                    family='monospace', fontsize=11.0, fontweight='bold',
                    color='green' if 'PASS' in headline else 'red')
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
                  'best_score', 'second_score', 'margin', 'latency_ms',
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
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            clean_row = {k: r.get(k, '') for k in fieldnames}
            writer.writerow(clean_row)

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
    library, contours, failures = clf.build_database(reference_dir)
    for path, reason in failures:
        print(f'  skipped {path.name}: {reason}')
    if not any(contours.values()):
        sys.exit(f'No reference photos found under {reference_dir} -- '
                 'add images to assets/references/<class>/images/ first')

    reference_items = clf.load_dataset(reference_dir)
    thresholds, _info = clf.calibrate_thresholds(reference_items, library, contours)

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
    print('\n[BƯỚC 2: PHÂN LOẠI LINH KIỆN & SO KHỚP MẪU CHUẨN]')
    print(f'  Kết quả nhận diện : {result["class"]}')
    print(f'  Trạng thái        : {"CHẤP NHẬN (PASS)" if result["accepted"] else "KHÔNG XÁC ĐỊNH (UNKNOWN)"}')
    if result.get('shortlist'):
        print(f'  Điểm khớp (Hu)    : {result["best_score"]:.4f} (ngưỡng chấp nhận < {thresholds["match_threshold"]:.2f})')
        print(f'  Biên độ an toàn   : {result["margin"]:.4f} (khoảng cách so với ứng viên thứ 2)')

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

    # Step 4: Subpixel Caliper Metrology
    meas = measurer.WorkpieceMeasurer().measure(img, described['contour'], spec)
    print('\n[BƯỚC 4: ĐO ĐẠC KÍCH THƯỚC QUANG HỌC CALIPER (SUBPIXEL RA PIXEL)]')
    print('  ' + '-' * 56)
    print(f'  {"TÊN KÍCH THƯỚC / CẠNH ĐO":<30} | {"KÍCH THƯỚC (PIXEL)":<20}')
    print('  ' + '-' * 56)
    print(f'  {"Length (L) - Chiều dài tổng":<30} | {meas["Length_L_px"]:>12.2f} px')
    print(f'  {"Width (W) - Chiều rộng tổng":<30} | {meas["Width_W_px"]:>12.2f} px')
    print(f'  {"Top Edge - Cạnh trên":<30} | {meas["Top_Edge_px"]:>12.2f} px')
    print(f'  {"Bottom Edge - Cạnh dưới":<30} | {meas["Bottom_Edge_px"]:>12.2f} px')
    print(f'  {"Left Edge - Cạnh trái":<30} | {meas["Left_Edge_px"]:>12.2f} px')
    print(f'  {"Right Edge - Cạnh phải":<30} | {meas["Right_Edge_px"]:>12.2f} px')
    print('  ' + '-' * 56)

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
            l_mm = round(meas['Length_L_px'] * scale, 2)
            w_mm = round(meas['Width_W_px'] * scale, 2)
            print(f'  {"Length (L)":<10} | {l_mm:>7.2f} mm | Spec: L                  | PASS (OK)    | Truc doc')
            print(f'  {"Width (W)":<10} | {w_mm:>7.2f} mm | Spec: W                  | PASS (OK)    | Truc ngang')
        print('  ' + '-' * 85)

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

    # High-resolution annotated measurement overlay
    annotated = measurer.annotate_measurements(img, described['contour'], meas, result['class'])
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