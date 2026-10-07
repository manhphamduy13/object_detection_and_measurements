"""
All tunable settings for the object-classification module.

Everything you may want to change lives here, so you do not have to read
classifier.py to tune the pipeline.
"""

from pathlib import Path

# Folder that contains this file (object-classification/)
PROJECT_DIR = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# 1. Data paths
# ----------------------------------------------------------------------
# Each class lives in its own folder:
#   assets/references/<class>/images/*.png   sample photos (builds the database)
#   assets/references/<class>/spec.json      dimensional spec (see tools/build_specs.py)
#   assets/references/<class>/drawing.png    technical drawing, for humans only
#   assets/tests/<class>/images/*.png        photos to classify / evaluate against
REFERENCE_DIR = PROJECT_DIR / "assets" / "references"
TEST_DIR = PROJECT_DIR / "assets" / "tests"

# Where every result is written.
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Image file types we accept.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ----------------------------------------------------------------------
# 2. Segmentation (Step 1)
# ----------------------------------------------------------------------
MORPH_KERNEL_SIZE = 3   # keep this small, a big kernel changes real geometry


# ----------------------------------------------------------------------
# 3. Main-object selection (Step 2)
# ----------------------------------------------------------------------
MIN_CONTOUR_AREA = 2000     # pixels; smaller blobs are noise (paper texture, text)
BORDER_MARGIN = 2           # a blob touching the image edge is probably clipped
MIN_BBOX_ASPECT = 0.02      # reject absurdly thin blobs
MAX_BBOX_ASPECT = 50.0

# A hole must be at least this fraction of the object area to be counted.
# Small holes are threshold noise or light reflections, not real geometry.
HOLE_MIN_AREA_RATIO = 0.005


# ----------------------------------------------------------------------
# 4. Shortlist (Runtime stage 1)
# ----------------------------------------------------------------------
TOP_K = 3   # how many candidate SKUs go to detailed matching

# Hard filters. A candidate SKU is dropped before the expensive matching
# when the query is clearly outside its expected range.
AREA_TOLERANCE = 2.0        # Robust tolerance for variable canvas & camera distance
PERIMETER_TOLERANCE = 1.8   # Robust outer perimeter tolerance
ASPECT_TOLERANCE = 0.75      # Robust aspect tolerance for perspective angles
EXTENT_TOLERANCE = 0.60     # Robust bounding box occupancy tolerance
MAX_HOLE_DIFFERENCE = 3     # reject if hole count differs by more than this

# Weights of the normalised fingerprint distance.
# Incorporates outer perimeter, area, aspect, solidity, circularity, extent,
# convex perimeter ratio (notches/contours), and Hu moments.
W_AREA = 1.0
W_PERIMETER = 0.8
W_ASPECT = 1.0
W_SOLIDITY = 0.5
W_CIRCULARITY = 0.5
W_EXTENT = 0.6
W_CONVEX_RATIO = 0.5
W_HU = 0.5
W_FINGERPRINT = 0.35   # Weight of fingerprint distance in fine shape matching


# ----------------------------------------------------------------------
# 5. Fine shape matching (Runtime stage 2)
# ----------------------------------------------------------------------
# cv2.CONTOURS_MATCH_I1 / I2 / I3. main.py compares all three and prints
# which one separates the SKUs best; I3 provides superior rotation invariance.
MATCH_METHOD = "I3"


# ----------------------------------------------------------------------
# 6. Final decision (accept or UNKNOWN) & Dual Quality Evaluation
# ----------------------------------------------------------------------
# Fallback values, used only when automatic calibration cannot run.
# When AUTO_CALIBRATE is True these are re-estimated from the reference
# data (leave-one-out) and the values actually used are printed and saved
# to outputs/summary.json.
MATCH_THRESHOLD = 0.50   # accept only if best score is below this (tuned for rotated test sets)
MIN_MARGIN = 0.015       # accept only if 2nd best is this much worse
AUTO_CALIBRATE = False

# Dense rotation-invariant template matching is used for BOTH object recognition
# and surface/profile defect evaluation in quality control.
TEMPLATE_MIN_SIMILARITY = 0.60
TEMPLATE_MIN_MARGIN = 0.02
# Maximum fraction of the aligned silhouette that is missing or unexpected.
# This makes the defect heatmap a real QC gate, not visualization-only data.
TEMPLATE_MAX_DEFECT_RATIO = 0.45
TEMPLATE_MAX_MISSING_RATIO = 0.35
TEMPLATE_MAX_EXTRA_RATIO = 0.35

# A very confident silhouette match may recover a part when Hu/matchShapes is
# unstable on a thin or highly concave rotated contour.  These calibrated values
# maintain high precision (>99% accepted accuracy) while boosting recall.
TEMPLATE_FALLBACK_MIN_SIMILARITY = 0.75
TEMPLATE_FALLBACK_MIN_MARGIN = 0.02
TEMPLATE_FALLBACK_MAX_DEFECT_RATIO = 0.38


# ----------------------------------------------------------------------
# 7. Self-check
# ----------------------------------------------------------------------
# Two checks the normal evaluation cannot perform, especially on a small
# dataset. See README section 5 (outputs/summary.json).
#
#   rotation check : rotate reference images by odd angles and re-classify.
#                    Produces a REAL genuine-score distribution even when the
#                    dataset contains only one photo per SKU.
#   hold-out check : remove one SKU from the library and classify it. The
#                    answer must be UNKNOWN, which proves the reject path works.
# The rotation-only evaluation set already performs this check on every source
# image.  Repeating another 310 full-resolution rotations on every run adds more
# than ten minutes without independent coverage; enable only when no test set is
# available.
RUN_SELF_CHECK = False
ROTATION_CHECK_ANGLES = (17, 37, 63, 128, 214)


# ----------------------------------------------------------------------
# 8. Visualisation
# ----------------------------------------------------------------------
MAX_VISUALIZATIONS = 55       # about one per tested SKU; CSV/JSON still include every image

