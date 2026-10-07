"""Production adapter for the dense rotation-invariant template matcher."""

from pathlib import Path
from typing import Any, Dict

import numpy as np

from tools import template_matching_experiment as engine


class TemplateMatcher:
    """Build the reference template bank once and score runtime images."""

    def __init__(self, reference_dir, min_similarity=0.60, min_margin=0.02,
                 max_defect_ratio=0.45, max_missing_ratio=0.35, max_extra_ratio=0.35):
        self.reference_dir = Path(reference_dir)
        self.min_similarity = float(min_similarity)
        self.min_margin = float(min_margin)
        self.max_defect_ratio = float(max_defect_ratio)
        self.max_missing_ratio = float(max_missing_ratio)
        self.max_extra_ratio = float(max_extra_ratio)
        references, skipped = engine.discover_references(self.reference_dir)
        self.bank, failures = engine.build_template_bank(references, output_dir=None)
        self.failures = list(skipped) + list(failures)
        if len(self.bank) < 2:
            raise RuntimeError("template matching requires at least two valid reference classes")
        # One contiguous matrix lets BLAS score every template in one call.
        # The former per-class matvec loop repeated Python/BLAS dispatch more
        # than fifty times for every camera frame.
        rows = []
        self._template_meta = []
        self._class_slices = {}
        offset = 0
        for label, entry in self.bank.items():
            count = len(entry["items"])
            rows.append(entry.pop("mask_norm"))
            self._template_meta.extend((label, item) for item in entry["items"])
            self._class_slices[label] = slice(offset, offset + count)
            offset += count
        self._mask_matrix = np.ascontiguousarray(np.concatenate(rows, axis=0))

    def match(self, image, object_mask, expected_label=None) -> Dict[str, Any]:
        """
        Dual-purpose template matching:
        1. Object recognition: Classifies query workpiece against the template bank.
        2. Quality inspection: Evaluates surface profile, missing material, and extra burr defects.
        """
        mask = (np.asarray(object_mask) > 0).astype(np.uint8) * 255
        query = engine.canonical_mask(mask)
        texture = engine.canonical_texture(image, mask)
        query_vector = query.reshape(-1).astype(np.float32)
        query_vector -= query_vector.mean()
        query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
        all_scores = self._mask_matrix @ query_vector
        class_scores = {
            label: float(np.max(all_scores[indexes]))
            for label, indexes in self._class_slices.items()
        }
        ranked = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
        top_label, top_similarity = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_similarity - second_score

        # If evaluating against a specific expected class (quality control stage),
        # use the best template for THAT class; otherwise use the top ranked class.
        target_label = expected_label if (expected_label and expected_label in self._class_slices) else top_label
        class_slice = self._class_slices[target_label]
        local_index = int(np.argmax(all_scores[class_slice]))
        global_index = class_slice.start + local_index
        similarity = float(all_scores[global_index])
        _, (template, _texture, name, angle) = self._template_meta[global_index]

        meta = {
            "reference": name,
            "angle": angle,
            "template": template,
            "silhouette_score": similarity,
            "texture_score": 0.0,
        }
        query_fg = query > 0
        template_fg = template > 0
        overlap = query_fg & template_fg
        missing = template_fg & ~query_fg
        extra = query_fg & ~template_fg
        reference_area = max(int(np.count_nonzero(template_fg)), 1)
        union_area = max(int(np.count_nonzero(query_fg | template_fg)), 1)
        defect_ratio = float(np.count_nonzero(missing | extra) / union_area)
        missing_ratio = float(np.count_nonzero(missing) / reference_area)
        extra_ratio = float(np.count_nonzero(extra) / reference_area)

        defect_overlay = np.zeros((query.shape[0], query.shape[1], 3), dtype=np.uint8)
        defect_overlay[overlap] = (70, 190, 70)       # green: matching material
        defect_overlay[missing] = (230, 55, 55)       # red: expected but missing
        defect_overlay[extra] = (65, 130, 255)        # blue: unexpected material

        # Quality inspection verdict:
        # Pass requires sufficient similarity, defect ratio below limit,
        # missing material below limit, and extra material below limit.
        fail_reasons = []
        if similarity < self.min_similarity:
            fail_reasons.append(f"Similarity {similarity:.3f} < {self.min_similarity:.2f}")
        if defect_ratio > self.max_defect_ratio:
            fail_reasons.append(f"Defect ratio {defect_ratio*100:.1f}% > {self.max_defect_ratio*100:.0f}%")
        if missing_ratio > self.max_missing_ratio:
            fail_reasons.append(f"Missing material {missing_ratio*100:.1f}% > {self.max_missing_ratio*100:.0f}%")
        if extra_ratio > self.max_extra_ratio:
            fail_reasons.append(f"Extra material {extra_ratio*100:.1f}% > {self.max_extra_ratio*100:.0f}%")

        template_quality_pass = len(fail_reasons) == 0

        score_pass = (
            top_similarity >= self.min_similarity
            and margin >= self.min_margin
            and defect_ratio <= self.max_defect_ratio
        )
        agreement = expected_label is None or top_label == expected_label

        return {
            "predicted_class": top_label,
            "target_class": target_label,
            "similarity": similarity,
            "margin": margin,
            "silhouette_score": meta["silhouette_score"],
            "texture_score": meta["texture_score"],
            "defect_ratio": defect_ratio,
            "missing_ratio": missing_ratio,
            "extra_ratio": extra_ratio,
            "defect_percent": defect_ratio * 100.0,
            "missing_percent": missing_ratio * 100.0,
            "extra_percent": extra_ratio * 100.0,
            "template_quality_pass": bool(template_quality_pass),
            "quality_status": "PASS" if template_quality_pass else "FAIL",
            "fail_reasons": fail_reasons,
            "reference": meta["reference"],
            "template_angle_deg": meta["angle"],
            "class_scores": class_scores,
            "min_similarity": self.min_similarity,
            "min_margin": self.min_margin,
            "max_defect_ratio": self.max_defect_ratio,
            "max_missing_ratio": self.max_missing_ratio,
            "max_extra_ratio": self.max_extra_ratio,
            "score_pass": bool(score_pass),
            "agrees_with_classification": bool(agreement),
            "accepted": bool(score_pass and agreement),
            "_query_mask": query,
            "_template_mask": template,
            "_defect_overlay_rgb": defect_overlay,
        }


def public_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Remove visualization-only arrays before JSON/CSV serialization."""
    return {key: value for key, value in result.items() if not key.startswith("_")}
