"""
Single entry point for the object-classification module.

    python main.py

Runs the whole workflow: digest the class database (sample images + specs +
drawings) under assets/references/, classify every photo under
assets/tests/, evaluate, draw figures and save all results into outputs/.

To classify a single new photo instead of running the whole evaluation, use
predict.py.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")            # write PNG files, never open a window
import matplotlib.pyplot as plt
import numpy as np

import config
import classifier as clf
import measurer


# ======================================================================
# Helpers
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Classify objects by silhouette shape.")
    parser.add_argument("--reference-dir", default=str(config.REFERENCE_DIR),
                        help="folder with known objects (one folder per SKU)")
    parser.add_argument("--test-dir", default=str(config.TEST_DIR),
                        help="folder with images to classify")
    parser.add_argument("--output-dir", default=str(config.OUTPUT_DIR),
                        help="where results are written")
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
    if isinstance(obj, float) and obj == float("inf"):
        return None
    return obj


def percentile(values, p):
    return float(np.percentile(values, p)) if values else 0.0


# ======================================================================
# 1. Load the data
# ======================================================================

def print_split(reference_items, test_items):
    labels = sorted({label for _, label in reference_items} |
                    {label for _, label in test_items})
    print("Dataset split per SKU:")
    print(f"  {'SKU':<12}{'reference':>10}{'test':>8}")
    for label in labels:
        n_ref = sum(1 for _, l in reference_items if l == label)
        n_test = sum(1 for _, l in test_items if l == label)
        print(f"  {label:<12}{n_ref:>10}{n_test:>8}")
    print()


# ======================================================================
# 2. Compare the three matchShapes methods (reported, not required)
# ======================================================================

def compare_match_methods(reference_items, library, contours):
    """
    Run the leave-one-out calibration with I1, I2 and I3 and report how well
    each one separates genuine matches from impostors. This is why the default
    in config.py is a measured choice and not a guess.
    """
    report = {}
    for name in ("I1", "I2", "I3"):
        _, info = clf.calibrate_thresholds(reference_items, library, contours, method=name)
        worst_genuine = info["worst_genuine_score"]
        best_impostor = info["best_impostor_score"]
        gap = None
        if worst_genuine is not None and best_impostor is not None:
            gap = best_impostor - worst_genuine
        report[name] = {
            "worst_genuine_score": worst_genuine,
            "best_impostor_score": best_impostor,
            "separation_gap": gap,
            "separated": bool(info["calibrated"]),
        }
    return report


# ======================================================================
# 3-4. Classify every test image
# ======================================================================

def run_test_set(test_items, library, contours, thresholds):
    """
    Classify every test image with clf.predict() (so each result already
    carries the matched class's spec, exactly like a real new photo would),
    while keeping the intermediate describe_image() output around too --
    save_visualization() needs the mask/contour/features it computed.
    """
    rows = []
    for path, true_label in test_items:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  could not read {path}")
            continue

        start = time.perf_counter()
        described = clf.describe_image(img)
        if described is None:
            result = clf.predict(img, library, contours, thresholds)
        else:
            result = clf.classify(described["features"], described["contour"],
                                  library, contours, thresholds)
            entry = library.get(result["class"], {})
            result["spec"] = entry.get("spec")
            result["drawing"] = entry.get("drawing")

        meas = None
        if described is not None:
            meas = measurer.WorkpieceMeasurer().measure(img, described["contour"], result.get("spec"))

        latency_ms = (time.perf_counter() - start) * 1000.0

        rows.append({
            "path": path,
            "image": img,
            "described": described,
            "filename": path.name,
            "true_class": true_label,
            "predicted_class": result["class"],
            "accepted": result["accepted"],
            "best_score": result["best_score"],
            "second_score": result["second_score"],
            "margin": result["margin"],
            "latency_ms": latency_ms,
            "result": result,
            "measurements": meas,
        })
    return rows


# ======================================================================
# 5. Evaluation
# ======================================================================

def evaluate(rows):
    total = len(rows)
    correct = sum(1 for r in rows if r["accepted"] and r["predicted_class"] == r["true_class"])
    wrong_accepted = sum(1 for r in rows if r["accepted"] and r["predicted_class"] != r["true_class"])
    unknown = sum(1 for r in rows if not r["accepted"])
    accepted = correct + wrong_accepted

    latencies = [r["latency_ms"] for r in rows]

    return {
        "test_images": total,
        "correct": correct,
        "wrong_accepted": wrong_accepted,
        "unknown": unknown,
        "overall_accuracy": correct / total if total else 0.0,
        "accepted_accuracy": correct / accepted if accepted else 0.0,
        "unknown_rate": unknown / total if total else 0.0,
        "average_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "median_latency_ms": float(np.median(latencies)) if latencies else 0.0,
        "p95_latency_ms": percentile(latencies, 95),
    }


def save_evaluation_csv(rows, path):
    columns = ["filename", "true_class", "predicted_class", "accepted",
               "best_score", "second_score", "margin", "latency_ms", "length_px", "width_px"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            meas = r.get("measurements") or {}
            writer.writerow({
                "filename": r["filename"],
                "true_class": r["true_class"],
                "predicted_class": r["predicted_class"],
                "accepted": r["accepted"],
                "best_score": f"{r['best_score']:.6f}" if np.isfinite(r["best_score"]) else "inf",
                "second_score": f"{r['second_score']:.6f}" if np.isfinite(r["second_score"]) else "inf",
                "margin": f"{r['margin']:.6f}" if np.isfinite(r["margin"]) else "inf",
                "latency_ms": f"{r['latency_ms']:.2f}",
                "length_px": f"{meas.get('Length_L_px', 0.0):.2f}",
                "width_px": f"{meas.get('Width_W_px', 0.0):.2f}",
            })


def save_predictions_json(rows, path):
    """
    One entry per test image with the full spec attached, exactly what a
    measurement step would get back from clf.predict() on a real new photo.
    """
    predictions = []
    for r in rows:
        predictions.append({
            "filename": r["filename"],
            "true_class": r["true_class"],
            "predicted_class": r["predicted_class"],
            "accepted": r["accepted"],
            "best_score": r["best_score"],
            "margin": r["margin"],
            "measurements_px": r.get("measurements"),
            "spec": r["result"]["spec"] if r["result"] else None,
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_python(predictions), f, indent=2, ensure_ascii=False)


def save_confusion_matrix(rows, labels, path):
    columns = labels + ["UNKNOWN"]
    index = {name: i for i, name in enumerate(columns)}
    matrix = np.zeros((len(labels), len(columns)), dtype=int)

    for r in rows:
        i = labels.index(r["true_class"])
        j = index.get(r["predicted_class"], index["UNKNOWN"])
        matrix[i, j] += 1

    fig, ax = plt.subplots(figsize=(1.1 * len(columns) + 3, 1.0 * len(labels) + 2.5))
    ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix")

    limit = matrix.max() / 2 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > limit else "black")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ======================================================================
# 6. Visualisation
# ======================================================================

def save_visualization(row, path):
    """
    One figure per test image, six panels:

        original | grayscale + Otsu | binary mask
        contour  | fingerprint      | candidates + final result
    """
    img = row["image"]
    described = row["described"]
    result = row["result"]
    features = described["features"]
    contour = described["contour"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # 1. Original
    axes[0][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0][0].set_title(f"1. Original\n{row['filename']}")

    # 2. Grayscale + the threshold Otsu chose
    axes[0][1].imshow(described["gray"], cmap="gray")
    axes[0][1].set_title(f"2. Grayscale (Otsu threshold = {described['otsu']:.0f})")

    # 3. Binary mask after morphology
    axes[0][2].imshow(described["mask"], cmap="gray")
    axes[0][2].set_title("3. Binary mask (after OPEN + CLOSE)")

    # 4. Caliper Metrology Overlay: Oriented BBox + Measurement Lines (L, W)
    meas = row.get("measurements") or {}
    if meas:
        overlay = measurer.annotate_measurements(img, contour, meas, result["class"])
        # Also save high-resolution measured detail image
        measured_detail_path = path.parent / f"{path.stem}_measured.png"
        cv2.imwrite(str(measured_detail_path), overlay)
    else:
        overlay = img.copy()
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 128, 255), 2)

    axes[1][0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1][0].set_title(f"4. Caliper Metrology (px)\nL: {meas.get('Length_L_px', 0):.1f}px | W: {meas.get('Width_W_px', 0):.1f}px")

    # 5. Fingerprint numbers
    hu_text = "\n".join(f"    Hu{i + 1} = {v:+.3f}" for i, v in enumerate(features["hu"]))
    fingerprint_text = (
        "5. Geometric fingerprint\n\n"
        f"area          = {features['area']:.0f} px\n"
        f"perimeter     = {features['perimeter']:.1f} px\n"
        f"rect aspect   = {features['rect_aspect']:.3f}   (rotation invariant)\n"
        f"solidity      = {features['solidity']:.3f}\n"
        f"circularity   = {features['circularity']:.3f}\n"
        f"hole_count    = {features['hole_count']}\n\n"
        f"Hu moments (signed log):\n{hu_text}"
    )
    axes[1][1].text(0.02, 0.98, fingerprint_text, va="top", ha="left",
                    family="monospace", fontsize=9.0)
    axes[1][1].axis("off")

    # 6. Step-by-Step Inspection Pipeline
    lines = [
        "6. Step-by-Step QC Pipeline\n",
        "[STEP 1: EDGE & CONTOUR DETECTION]",
        f"  Points: {len(contour)} | Centroid: ({features['centroid'][0]:.1f}, {features['centroid'][1]:.1f})",
        "",
        "[STEP 2: CLASSIFICATION & SHORTLIST]",
        "  Top-3 Candidates (matchShapes, lower = better):"
    ]
    for i, cand in enumerate(result["shortlist"], start=1):
        lines.append(f"   {i}. {cand['label']:<11} score={cand['shape_score']:.4f} (fp {cand['fingerprint_distance']:.3f})")

    lines.append(f"  Best score  : {result['best_score']:.4f}  (margin: {result['margin']:.4f})")
    lines.append("")
    lines.append("[STEP 3: SPECIFICATION SHEET]")
    spec = result.get("spec")
    if spec:
        raw_name = "".join(c for c in spec.get("item_name", "") if ord(c) < 128).strip()
        item_label = f" ({raw_name[:20]})" if raw_name else ""
        lines.append(f"  SKU / Item  : {spec.get('sku')}{item_label}")
        lines.append(f"  Unit        : {spec.get('unit', 'mm')}")
    else:
        lines.append("  Spec        : No spec.json loaded (UNKNOWN)")

    lines.append("")
    lines.append("[STEP 4: CALIPER MEASUREMENT (px)]")
    if meas:
        lines.append(f"  Length (L)  : {meas.get('Length_L_px', 0.0):.2f} px")
        lines.append(f"  Width (W)   : {meas.get('Width_W_px', 0.0):.2f} px")
        lines.append(f"  Top/Bottom  : {meas.get('Top_Edge_px', 0.0):.1f} / {meas.get('Bottom_Edge_px', 0.0):.1f} px")
        lines.append(f"  Left/Right  : {meas.get('Left_Edge_px', 0.0):.1f} / {meas.get('Right_Edge_px', 0.0):.1f} px")
    else:
        lines.append("  Measurements : Not available")

    lines.append("")
    lines.append("[STEP 5: QUALITY VERDICT]")

    verdict = result["class"]
    if verdict == "UNKNOWN":
        colour, headline = "darkorange", "VERDICT: UNKNOWN (REJECTED)"
    elif verdict == row["true_class"]:
        colour, headline = "green", f"VERDICT: {verdict} (PASS)"
    else:
        colour, headline = "red", f"VERDICT: {verdict} (MISMATCH)"

    lines.append(f"  Result      : {verdict} (True: {row['true_class']})")

    axes[1][2].text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
                    family="monospace", fontsize=8.5)
    axes[1][2].text(0.02, 0.04, headline, va="top", ha="left",
                    family="monospace", fontsize=12, color=colour, weight="bold")
    axes[1][2].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_intermediate_steps(row, steps_base_dir):
    """
    Exports the 5 intermediate inspection steps (mirroring QC_CAM pipeline):
    - step1_roi_crop.png
    - step2_edge_gradient_map.png
    - step3_workpiece_mask.png
    - step3_contours_extracted.png
    - step4_caliper_scanlines.png
    - step5_final_inspection.png
    """
    stem = Path(row["filename"]).stem
    step_dir = steps_base_dir / stem
    step_dir.mkdir(parents=True, exist_ok=True)

    img = row["image"]
    described = row.get("described")
    if described is None:
        return

    gray = described["gray"]
    stage2_steps = described.get("stage2_steps")
    if stage2_steps is not None:
        cv2.imwrite(str(step_dir / "step1_roi_crop.png"), stage2_steps.step1_roi)
        cv2.imwrite(str(step_dir / "step2_edge_gradient_map.png"), stage2_steps.step2_edge_map)
        cv2.imwrite(str(step_dir / "step3_workpiece_mask.png"), stage2_steps.step3_mask)
        cv2.imwrite(str(step_dir / "step3_contours_extracted.png"), stage2_steps.step3_contours_vis)
    else:
        cv2.imwrite(str(step_dir / "step1_roi_crop.png"), gray)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        sobelx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.clip(cv2.magnitude(sobelx, sobely), 0, 255).astype(np.uint8)
        cv2.imwrite(str(step_dir / "step2_edge_gradient_map.png"), mag)
        cv2.imwrite(str(step_dir / "step3_workpiece_mask.png"), described["mask"])
        vis_cnt = img.copy()
        cv2.drawContours(vis_cnt, [described["contour"]], -1, (0, 255, 0), 2)
        cv2.imwrite(str(step_dir / "step3_contours_extracted.png"), vis_cnt)

    meas = row.get("measurements") or {}
    res_class = row["result"]["class"] if row.get("result") else "UNKNOWN"
    if meas:
        caliper_vis = measurer.annotate_measurements(img, described["contour"], meas, res_class)
    else:
        caliper_vis = img.copy()
        cv2.drawContours(caliper_vis, [described["contour"]], -1, (0, 255, 0), 2)
    cv2.imwrite(str(step_dir / "step4_caliper_scanlines.png"), caliper_vis)
    cv2.imwrite(str(step_dir / "step5_final_inspection.png"), caliper_vis)



# ======================================================================
# 7. Saving the class database
# ======================================================================

def save_reference_library(library, contours, json_path, npz_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(to_python(library), f, indent=2)

    # Contours are variable-length arrays, which are awkward in JSON, so they
    # go into a .npz file. Key = "<label>__<index>".
    arrays = {}
    for label, contour_list in contours.items():
        for i, c in enumerate(contour_list):
            arrays[f"{label}__{i}"] = c
    np.savez_compressed(npz_path, **arrays)


# ======================================================================
# Console report
# ======================================================================

def print_self_check(rotation_check, holdout_check):
    if rotation_check:
        r = rotation_check
        print("Self-check 1 - rotation robustness")
        print(f"  reference images re-classified at angles {r['angles']} deg")
        print(f"  correct {r['correct']}/{r['tested']}   "
              f"UNKNOWN {r['unknown']}   wrong {r['wrong']}")
        if r["worst_genuine_score"] is not None:
            print(f"  worst genuine score under rotation = {r['worst_genuine_score']:.4f}")
            print(f"  median genuine score under rotation = {r['median_genuine_score']:.4f}")
        print()

    if holdout_check and holdout_check["tested"]:
        h = holdout_check
        print("Self-check 2 - UNKNOWN reject path")
        print(f"  each SKU removed from the library, then classified")
        print(f"  correctly answered UNKNOWN: {h['unknown']}/{h['tested']}")
        for leak in h["leaked"]:
            print(f"  LEAK: {leak['held_out']} was claimed to be "
                  f"{leak['claimed']} (score {leak['best_score']:.4f})")
        print()


def print_report(metrics, n_reference, thresholds, threshold_info, method_report):
    print("=" * 46)
    print("Classification Evaluation")
    print("=" * 46)
    print()
    print(f"Reference images : {n_reference}")
    print(f"Test images      : {metrics['test_images']}")
    print()
    print(f"Correct          : {metrics['correct']}")
    print(f"Wrong accepted   : {metrics['wrong_accepted']}")
    print(f"UNKNOWN          : {metrics['unknown']}")
    print()
    print(f"Overall accuracy : {metrics['overall_accuracy'] * 100:.2f}%")
    print(f"Accepted accuracy: {metrics['accepted_accuracy'] * 100:.2f}%")
    print(f"UNKNOWN rate     : {metrics['unknown_rate'] * 100:.2f}%")
    print()
    print(f"Avg latency      : {metrics['average_latency_ms']:.1f} ms")
    print(f"Median latency   : {metrics['median_latency_ms']:.1f} ms")
    print(f"P95 latency      : {metrics['p95_latency_ms']:.1f} ms")
    print()

    print("Thresholds actually used:")
    print(f"  MATCH_THRESHOLD = {thresholds['match_threshold']:.4f}")
    print(f"  MIN_MARGIN      = {thresholds['min_margin']:.4f}")
    source = "calibrated from reference data (leave-one-out)" \
        if threshold_info["calibrated"] else "defaults from config.py"
    print(f"  source          : {source}")
    print(f"  reason          : {threshold_info['reason']}")
    # Either value can be missing: a single-SKU library produces genuine
    # scores but no impostor scores at all.
    if threshold_info["worst_genuine_score"] is not None:
        print(f"  worst genuine score  = {threshold_info['worst_genuine_score']:.4f}")
    if threshold_info["best_impostor_score"] is not None:
        print(f"  best impostor score  = {threshold_info['best_impostor_score']:.4f}")
    print()

    print("matchShapes method comparison (leave-one-out on the reference set):")
    print(f"  {'method':<8}{'worst genuine':>15}{'best impostor':>15}{'gap':>10}")
    for name, r in method_report.items():
        wg = f"{r['worst_genuine_score']:.4f}" if r["worst_genuine_score"] is not None else "n/a"
        bi = f"{r['best_impostor_score']:.4f}" if r["best_impostor_score"] is not None else "n/a"
        gap = f"{r['separation_gap']:.4f}" if r["separation_gap"] is not None else "n/a"
        mark = "  <- default" if name == config.MATCH_METHOD else ""
        print(f"  {name:<8}{wg:>15}{bi:>15}{gap:>10}{mark}")
    print()



# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"
    steps_base_dir = output_dir / "intermediate_steps"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up previous outputs and visualizations to ensure fresh results
    if vis_dir.exists():
        import shutil
        shutil.rmtree(vis_dir, ignore_errors=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    if steps_base_dir.exists():
        import shutil
        shutil.rmtree(steps_base_dir, ignore_errors=True)
    steps_base_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ["*.csv", "*.json", "*_measured.png"]:
        for f in output_dir.glob(pattern):
            try:
                f.unlink()
            except Exception:
                pass

    # --- 1. load the data -------------------------------------------------
    reference_items = clf.load_dataset(args.reference_dir)
    test_items = clf.load_dataset(args.test_dir)

    if not reference_items or not test_items:
        print("No images found. Check --reference-dir and --test-dir.")
        return

    print_split(reference_items, test_items)

    # --- 2. digest the class database (images + specs + drawings) --------
    print("Building class database ...")
    library, contours, failures = clf.build_database(args.reference_dir)
    for path, reason in failures:
        print(f"  skipped {path.name}: {reason}")
    n_with_spec = sum(1 for entry in library.values() if entry.get("spec"))
    n_no_images = sum(1 for label in library if not contours.get(label))
    if not library:
        print("Class database is empty; nothing to classify against.")
        return
    print(f"  {len(library)} classes, "
          f"{sum(len(c) for c in contours.values())} reference contours, "
          f"{n_with_spec} with a spec.json")
    if n_no_images:
        no_image_labels = sorted(l for l in library if not contours.get(l))
        print(f"  no reference photos yet (spec-only): {', '.join(no_image_labels)}")
    print()

    # --- 4. thresholds ---------------------------------------------------
    print("Comparing matchShapes methods and calibrating thresholds ...")
    method_report = compare_match_methods(reference_items, library, contours)
    if config.AUTO_CALIBRATE:
        thresholds, threshold_info = clf.calibrate_thresholds(
            reference_items, library, contours
        )
    else:
        thresholds = {"match_threshold": config.MATCH_THRESHOLD,
                      "min_margin": config.MIN_MARGIN}
        threshold_info = {"calibrated": False, "reason": "AUTO_CALIBRATE is False",
                          "worst_genuine_score": None, "best_impostor_score": None,
                          "n_genuine": 0, "n_impostor": 0}
    print()

    # --- 4b. self-check ---------------------------------------------------
    rotation_check, holdout_check = None, None
    if config.RUN_SELF_CHECK:
        print("Running self-check (rotation robustness + UNKNOWN reject path) ...")
        rotation_check = clf.rotation_self_check(
            reference_items, library, contours, thresholds,
            config.ROTATION_CHECK_ANGLES,
        )
        holdout_check = clf.holdout_self_check(reference_items)
        print()

    # --- 5. classify the test set ---------------------------------------
    print(f"Classifying {len(test_items)} test images ...")
    rows = run_test_set(test_items, library, contours, thresholds)
    print()

    # --- 6. evaluate ------------------------------------------------------
    metrics = evaluate(rows)
    # Only classes with at least one reference contour can ever be predicted;
    # a spec-only class (no photos yet) has nothing to appear as a row/column.
    labels = sorted(label for label in library if contours.get(label))

    save_reference_library(library, contours,
                           output_dir / "reference_library.json",
                           output_dir / "reference_contours.npz")
    save_evaluation_csv(rows, output_dir / "evaluation.csv")
    save_predictions_json(rows, output_dir / "predictions.json")
    save_confusion_matrix(rows, labels, output_dir / "confusion_matrix.png")

    # --- 7. visualisations & intermediate steps ---------------------------
    steps_base_dir = output_dir / "steps"
    steps_base_dir.mkdir(parents=True, exist_ok=True)

    drawn = 0
    for row in rows:
        if row["described"] is None:
            continue
        save_intermediate_steps(row, steps_base_dir)
        if drawn < config.MAX_VISUALIZATIONS:
            save_visualization(row, vis_dir / f"{Path(row['filename']).stem}.png")
            drawn += 1


    # --- 8. honest notes about this dataset -------------------------------
    warnings = []
    if n_no_images:
        warnings.append(
            f"{', '.join(no_image_labels)} : spec.json exists but there are no "
            "reference photos yet, so this class can never be predicted -- "
            "add photos to assets/references/<class>/images/."
        )
    min_per_sku = min(
        sum(1 for _, l in reference_items + test_items if l == label) for label in labels
    )
    if min_per_sku < 10:
        warnings.append(
            f"Only {min_per_sku} image(s) per SKU. The accuracy above is a smoke "
            "test, not a reliable estimate of production accuracy."
        )

    # A score of (almost) zero means two images are the SAME picture, or a
    # rotated/flipped copy of one. matchShapes is rotation invariant, so a
    # rotated duplicate scores exactly 0. The evaluation then measures
    # duplicate recognition and says nothing about a NEW physical part.
    #
    # Check both places a duplicate can show up: inside the reference set
    # (leave-one-out) and between the reference set and the test set.
    duplicate_signals = [threshold_info.get("worst_genuine_score")]
    correct_scores = [r["best_score"] for r in rows
                      if r["accepted"] and r["predicted_class"] == r["true_class"]]
    if correct_scores:
        duplicate_signals.append(float(np.median(correct_scores)))

    worst_genuine = next((s for s in duplicate_signals
                          if s is not None and s < 1e-6), None)
    if worst_genuine is not None:
        warnings.append(
            "DUPLICATE DATA: the best genuine match scored ~0.000, so reference "
            "images are identical shapes (e.g. rotated copies of one photo). "
            "Accuracy here measures duplicate recognition, NOT generalisation. "
            "Photograph each SKU several times, physically re-placed and "
            "re-rotated, before trusting these numbers."
        )
    # When the dataset itself cannot produce a meaningful score (duplicates,
    # one photo per SKU), the self-checks can. Report whichever half is
    # available: how bad a CORRECT part scores when rotated, and how good the
    # closest WRONG part scores. Those two numbers must not overlap.
    best_impostor = threshold_info.get("best_impostor_score")
    worst_rotated = rotation_check["worst_genuine_score"] if rotation_check else None

    parts = []
    if worst_rotated is not None:
        parts.append(f"worst score for a CORRECT part under rotation: {worst_rotated:.4f}")
    if best_impostor is not None:
        parts.append(f"best score for a WRONG part: {best_impostor:.4f}")

    if parts:
        message = "The trustworthy numbers are from the self-checks - " + "; ".join(parts) + "."
        if worst_rotated is not None and best_impostor is not None:
            separated = worst_rotated < best_impostor
            message += (
                f" The two groups are {'separated' if separated else 'OVERLAPPING'}"
                f", so a threshold between them "
                f"{'is safe' if separated else 'CANNOT be chosen reliably'}."
            )
        warnings.append(message)

    summary = {
        "test_images": metrics["test_images"],
        "correct": metrics["correct"],
        "wrong_accepted": metrics["wrong_accepted"],
        "unknown": metrics["unknown"],
        "overall_accuracy": metrics["overall_accuracy"],
        "accepted_accuracy": metrics["accepted_accuracy"],
        "unknown_rate": metrics["unknown_rate"],
        "average_latency_ms": metrics["average_latency_ms"],
        "median_latency_ms": metrics["median_latency_ms"],
        "p95_latency_ms": metrics["p95_latency_ms"],
        "reference_images": len(reference_items),
        "skus": labels,
        "match_method": config.MATCH_METHOD,
        "match_method_comparison": method_report,
        "thresholds_used": thresholds,
        "threshold_calibration": threshold_info,
        "rotation_self_check": rotation_check,
        "holdout_self_check": holdout_check,
        "warnings": warnings,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(to_python(summary), f, indent=2)

    # --- 9. report ---------------------------------------------------------
    print_report(metrics, len(reference_items), thresholds, threshold_info,
                 method_report)
    print_self_check(rotation_check, holdout_check)

    if warnings:
        print("Notes on this dataset:")
        for w in warnings:
            print(f"  - {w}")
        print()

    print("Results:")
    print(f"{output_dir / 'evaluation.csv'}")
    print(f"{output_dir / 'predictions.json'}   (spec attached to every prediction)")
    print()
    print("Visualizations:")
    print(f"{output_dir / 'visualizations'}/")
    print()


if __name__ == "__main__":
    main()
