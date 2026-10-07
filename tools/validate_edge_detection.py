"""Render a small representative edge/mask audit before a full pipeline run."""

from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import classifier as clf


SAMPLES = (
    "assets/tests/1/images/1_1_1_t01_a7_s0.70.jpg",
    "assets/tests/1/images/1_1_1_t04_a71_s1.30.jpg",
    "assets/tests/4/images/4_4_1_t01_a7_s0.70.jpg",
    "assets/tests/4/images/4_4_1_t04_a71_s1.30.jpg",
    "assets/tests/98003P/images/98003P_1_rot23.png",
    "assets/tests/98475/images/98475_1_rot149.png",
    "assets/tests/UH-004/images/UH-004_UH-004_1_t01_a7_s0.70.jpg",
    "assets/tests/UH-8715P/images/UH-8715P_UH-8715P_1_t04_a71_s1.30.jpg",
)


def main():
    output_dir = PROJECT_DIR / "outputs" / "edge_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    passed = 0
    for relative in SAMPLES:
        path = PROJECT_DIR / relative
        image = cv2.imread(str(path))
        if image is None:
            continue
        described = clf.describe_image(image)
        if described is None:
            continue
        passed += 1
        contour_view = image.copy()
        cv2.drawContours(contour_view, [described["contour"]], -1, (0, 255, 0), 3, cv2.LINE_AA)
        steps = described.get("stage2_steps")
        edge_view = steps.step2_edge_map if steps is not None else described["gray"]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Input")
        if edge_view.ndim == 3:
            axes[1].imshow(cv2.cvtColor(edge_view, cv2.COLOR_BGR2RGB))
        else:
            axes[1].imshow(edge_view, cmap="gray")
        axes[1].set_title("Adaptive edge cues")
        axes[2].imshow(cv2.cvtColor(contour_view, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Selected physical contour")
        for axis in axes:
            axis.axis("off")
        fig.suptitle(path.name)
        fig.tight_layout()
        fig.savefig(output_dir / f"{path.stem}_edge_audit.png", dpi=130)
        plt.close(fig)
    print(f"edge validation: {passed}/{len(SAMPLES)} segmented; output={output_dir}")


if __name__ == "__main__":
    main()
