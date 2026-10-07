"""Summarise measurement repeatability across rotation-only variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np


ROTATION_SUFFIX = re.compile(r"_rot-?\d+(?:\.\d+)?$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "predictions", nargs="?", default="outputs/rotation_final/predictions.json"
    )
    return parser.parse_args()


def relative_span(values):
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    return float((values.max() - values.min()) / median) if median > 0 else float("inf")


def main():
    args = parse_args()
    rows = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    groups = {}
    for row in rows:
        # Filenames are not globally unique: two SKU folders may contain a
        # source photo with the same stem.  Include the ground-truth folder
        # label so unrelated workpieces are never merged into one rotation set.
        source_stem = ROTATION_SUFFIX.sub("", Path(row["filename"]).stem)
        key = f"{row['true_class']}/{source_stem}"
        groups.setdefault(key, []).append(row)

    spans = {"Length_L_px": [], "Width_W_px": []}
    worst = []
    for key, group in groups.items():
        for metric in spans:
            values = [row.get("measurements_px", {}).get(metric) for row in group]
            values = [float(value) for value in values if value is not None]
            if len(values) >= 2:
                span = relative_span(values)
                spans[metric].append(span)
                worst.append((span, key, metric, min(values), max(values)))

    print(f"rotation_groups={len(groups)} images={len(rows)}")
    for metric, values in spans.items():
        percentages = np.asarray(values) * 100.0
        print(
            f"{metric}: groups={len(values)} median_span={np.median(percentages):.2f}% "
            f"p95_span={np.percentile(percentages, 95):.2f}% "
            f"within_5pct={np.sum(percentages <= 5.0)}/{len(values)}"
        )
    print("worst_groups:")
    for span, key, metric, low, high in sorted(worst, reverse=True)[:10]:
        print(f"  {key} {metric}: {span * 100:.2f}% ({low:.2f}..{high:.2f}px)")


if __name__ == "__main__":
    main()
