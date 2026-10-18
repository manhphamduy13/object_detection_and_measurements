"""
Industrial Metrology Scanline Teaching Studio & Visual Recipe Manager.
Modern Dark-Themed GUI with:
- Real-time contour edge-snapped scanlines (snaps to actual detected physical edges, NOT points outside)
- Interactive drawing, selection, and 1-click deletion on GUI
- Direct CAD specification dropdown and nominal value editing
- Instant recipe saving to configs/caliper_recipes.json
- One-click live sample testing and tolerance verification

Usage:
  python tools/scanline_studio.py
  python tools/scanline_studio.py --sku 98661BBS-1
"""
import sys
import os
import json
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

import classifier as clf
import measurer
from src.specification import build_dimension, tolerance_magnitudes, upsert_dimension
from src.template_geometry import (
    profile_limits, golden_baseline, offset_contours, draw_profile,
)
from src.viewport import ImageViewport
RECIPE_FILE = PROJECT_DIR / "configs" / "caliper_recipes.json"

def get_wcs_info(cnt):
    """Calculates oriented bounding box and centroid offset for 180-deg disambiguation."""
    rect = cv2.minAreaRect(cnt)
    center, (rw, rh), angle = rect
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0
    M = cv2.moments(cnt)
    if M["m00"] > 0:
        cx_m, cy_m = M["m10"] / M["m00"], M["m01"] / M["m00"]
    else:
        cx_m, cy_m = center[0], center[1]
    
    centroid_offset = (cx_m - center[0], cy_m - center[1])
    return {
        "center": (float(center[0]), float(center[1])),
        "width": float(rw),
        "height": float(rh),
        "angle": float(angle),
        "centroid_offset": (float(centroid_offset[0]), float(centroid_offset[1]))
    }

def to_wcs(pt, wcs_info):
    """Maps absolute image coordinates (x, y) to normalized WCS (u, v)."""
    center = wcs_info["center"]
    angle = wcs_info["angle"]
    w = max(wcs_info["width"], 1.0)
    h = max(wcs_info["height"], 1.0)
    rad = np.radians(-angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    return (float(rx / w), float(ry / h))

def from_wcs(u, v, wcs_info):
    """Maps normalized WCS (u, v) back to target absolute image coordinates (x, y)."""
    center = wcs_info["center"]
    angle = wcs_info["angle"]
    w = wcs_info["width"]
    h = wcs_info["height"]
    rx = u * w
    ry = v * h
    rad = np.radians(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    dx = rx * cos_a - ry * sin_a
    dy = rx * sin_a + ry * cos_a
    return (float(center[0] + dx), float(center[1] + dy))

def get_proposed_calipers_for_sku(sku: str, wcs: dict, optical_scale: float) -> list:
    """Returns proposed default calipers for a SKU derived from CAD specification (spec.json)."""
    spec_path = PROJECT_DIR / "assets" / "references" / sku / "spec.json"
    spec_dims = []
    if spec_path.exists():
        try:
            with open(spec_path, "r", encoding="utf-8") as f:
                spec_dims = json.load(f).get("dimensions", [])
        except Exception:
            pass

    palette = [
        [0, 230, 255], [255, 160, 0], [0, 255, 180], [255, 0, 255],
        [0, 200, 255], [200, 255, 0], [255, 120, 120], [180, 180, 255]
    ]

    known_calipers = {
        "98661BBS-1": [
            {"label": "L", "desc": "Tong chieu dai (Overall length L)", "nominal_mm": 28.5, "p1_wcs": [-0.50, 0.0], "p2_wcs": [0.50, 0.0]},
            {"label": "W", "desc": "Chieu rong than (Body width W)", "nominal_mm": 4.3, "p1_wcs": [0.08, -0.40], "p2_wcs": [0.08, 0.40]},
            {"label": "F", "desc": "Chieu rong la tiep diem (Contact blade width F)", "nominal_mm": 4.3, "p1_wcs": [-0.35, -0.38], "p2_wcs": [-0.35, 0.38]},
            {"label": "J", "desc": "Chieu dai doan khoa ngam (Locking tab length J)", "nominal_mm": 6.3, "p1_wcs": [0.28, 0.0], "p2_wcs": [0.50, 0.0]},
            {"label": "K", "desc": "Khoang cach ngam kep (Contact gap K)", "nominal_mm": 3.0, "p1_wcs": [-0.35, -0.15], "p2_wcs": [-0.35, 0.15]},
            {"label": "d", "desc": "Duong kinh lo tron giua (Center hole diameter d)", "nominal_mm": 2.1, "p1_wcs": [0.08, -0.10], "p2_wcs": [0.08, 0.10]}
        ],
        "98475": [
            {"label": "L", "desc": "Tong chieu dai (Overall length L)", "nominal_mm": 32.5, "p1_wcs": [-0.50, 0.0], "p2_wcs": [0.50, 0.0]},
            {"label": "E", "desc": "Chieu dai than tiep diem (Body contact length E)", "nominal_mm": 24.5, "p1_wcs": [-0.25, 0.0], "p2_wcs": [0.50, 0.0]},
            {"label": "D", "desc": "Chieu rong hop bam cos (Crimp box width D)", "nominal_mm": 4.75, "p1_wcs": [0.35, -0.40], "p2_wcs": [0.35, 0.40]},
            {"label": "F", "desc": "Chieu rong dau co (Flag contact width F)", "nominal_mm": 4.0, "p1_wcs": [-0.42, -0.35], "p2_wcs": [-0.42, 0.35]},
            {"label": "W", "desc": "Chieu rong than (Body width W)", "nominal_mm": 4.9, "p1_wcs": [0.0, -0.42], "p2_wcs": [0.0, 0.42]}
        ],
        "UH-004": [
            {"label": "L", "desc": "Tong chieu dai (Overall length L)", "nominal_mm": 23.4, "p1_wcs": [-0.50, 0.0], "p2_wcs": [0.50, 0.0]},
            {"label": "W", "desc": "Chieu rong ngoai (Outer width W)", "nominal_mm": 12.0, "p1_wcs": [0.0, -0.45], "p2_wcs": [0.0, 0.45]},
            {"label": "d", "desc": "Duong kinh lo chot (Mounting hole diameter d)", "nominal_mm": 3.0, "p1_wcs": [0.30, -0.12], "p2_wcs": [0.30, 0.12]},
            {"label": "A", "desc": "Chieu rong khoang trong (Cavity opening A)", "nominal_mm": 8.0, "p1_wcs": [-0.20, -0.30], "p2_wcs": [-0.20, 0.30]},
            {"label": "B", "desc": "Chieu sau khoang (Cavity depth B)", "nominal_mm": 4.5, "p1_wcs": [-0.50, 0.0], "p2_wcs": [-0.30, 0.0]}
        ],
        "UH-8715P": [
            {"label": "E", "desc": "Chieu rong than (Body width E)", "nominal_mm": 16.0, "p1_wcs": [-0.10, -0.45], "p2_wcs": [-0.10, 0.45]},
            {"label": "L", "desc": "Chieu dai mieng khoang (Opening length L)", "nominal_mm": 7.9, "p1_wcs": [-0.50, 0.0], "p2_wcs": [-0.14, 0.0]},
            {"label": "K", "desc": "Khau do lo trong (Inner aperture K)", "nominal_mm": 5.8, "p1_wcs": [-0.30, -0.16], "p2_wcs": [-0.30, 0.16]},
            {"label": "G", "desc": "Khoang cach khe ranh (Slot gap width G)", "nominal_mm": 4.7, "p1_wcs": [0.15, -0.13], "p2_wcs": [0.15, 0.13]},
            {"label": "I", "desc": "Chieu rong ranh ngam (Locking notch width I)", "nominal_mm": 3.05, "p1_wcs": [0.35, -0.09], "p2_wcs": [0.35, 0.09]},
            {"label": "J", "desc": "Do day vach ben (Wall thickness J)", "nominal_mm": 1.6, "p1_wcs": [-0.20, -0.45], "p2_wcs": [-0.20, -0.38]}
        ],
        "98003P": [
            {"label": "L", "desc": "Tong chieu dai (Overall length L)", "nominal_mm": 32.0, "p1_wcs": [-0.50, 0.0], "p2_wcs": [0.50, 0.0]},
            {"label": "W", "desc": "Chieu rong than nhua (Body width W)", "nominal_mm": 26.0, "p1_wcs": [0.08, -0.45], "p2_wcs": [0.08, 0.45]},
            {"label": "D", "desc": "Duong kinh than pin (Pin shaft diameter D)", "nominal_mm": 4.0, "p1_wcs": [-0.30, 0.35], "p2_wcs": [-0.30, 0.20]},
            {"label": "D1", "desc": "Duong kinh dau pin (Pin tip diameter D1)", "nominal_mm": 3.95, "p1_wcs": [-0.30, 0.41], "p2_wcs": [-0.30, 0.27]},
            {"label": "D2", "desc": "Duong kinh go bac pin (Pin step diameter D2)", "nominal_mm": 3.5, "p1_wcs": [-0.30, -0.27], "p2_wcs": [-0.30, -0.42]}
        ]
    }

    proposals = []
    if sku in known_calipers:
        for idx, c in enumerate(known_calipers[sku]):
            p1_ref = from_wcs(c["p1_wcs"][0], c["p1_wcs"][1], wcs)
            p2_ref = from_wcs(c["p2_wcs"][0], c["p2_wcs"][1], wcs)
            dist_px = float(np.hypot(p2_ref[0] - p1_ref[0], p2_ref[1] - p1_ref[1]))
            proposals.append({
                "label": c["label"],
                "desc": c.get("desc", f"Caliper {c['label']}"),
                "nominal_mm": c["nominal_mm"],
                "axis": "2D_in_plane",
                "p1_wcs": c["p1_wcs"],
                "p2_wcs": c["p2_wcs"],
                "p1_ref": [round(p1_ref[0], 2), round(p1_ref[1], 2)],
                "p2_ref": [round(p2_ref[0], 2), round(p2_ref[1], 2)],
                "length_px": round(dist_px, 2),
                "length_mm": round(dist_px * optical_scale, 2),
                "measure_type": "cad_proposal",
                "color": palette[idx % len(palette)]
            })
    else:
        idx = 0
        for d in spec_dims:
            if d.get("axis") == "Z_height_side_view":
                continue
            lbl = d.get("label", f"DIM_{idx+1}")
            nom = d.get("nominal", 10.0)
            if lbl == "L":
                u1, v1, u2, v2 = -0.5, 0.0, 0.5, 0.0
            elif lbl == "W":
                u1, v1, u2, v2 = 0.0, -0.45, 0.0, 0.45
            else:
                step_y = 0.1 * ((idx % 4) - 1.5)
                u1, v1, u2, v2 = -0.2, step_y, 0.2, step_y

            p1_ref = from_wcs(u1, v1, wcs)
            p2_ref = from_wcs(u2, v2, wcs)
            dist_px = float(np.hypot(p2_ref[0] - p1_ref[0], p2_ref[1] - p1_ref[1]))
            proposals.append({
                "label": lbl,
                "desc": d.get("raw_spec", f"Dimension {lbl}"),
                "nominal_mm": nom,
                "axis": "2D_in_plane",
                "p1_wcs": [u1, v1],
                "p2_wcs": [u2, v2],
                "p1_ref": [round(p1_ref[0], 2), round(p1_ref[1], 2)],
                "p2_ref": [round(p2_ref[0], 2), round(p2_ref[1], 2)],
                "length_px": round(dist_px, 2),
                "length_mm": round(dist_px * optical_scale, 2),
                "measure_type": "cad_proposal",
                "color": palette[idx % len(palette)]
            })
            idx += 1
    return proposals

def load_products_dict():
    p_path = PROJECT_DIR / "configs" / "products.json"
    if p_path.exists():
        try:
            with open(p_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

# Color Theme
BG_DARK = "#181a20"
BG_CARD = "#222630"
BG_SIDEBAR = "#1e222b"
ACCENT_CYAN = "#00e5ff"
ACCENT_GREEN = "#00e676"
ACCENT_YELLOW = "#ffd600"
ACCENT_RED = "#ff5252"
TEXT_WHITE = "#ffffff"
TEXT_MUTED = "#90a4ae"

PALETTE = [
    (0, 230, 255), (255, 160, 0), (0, 255, 180), (255, 0, 255),
    (0, 200, 255), (200, 255, 0), (255, 120, 120), (180, 180, 255)
]

def snap_point_to_contour(pt, contour):
    """Snaps any point (x, y) to the nearest exact point on the physical contour polygon."""
    if contour is None or len(contour) < 3:
        return (float(pt[0]), float(pt[1])), 0.0
    pts = contour.reshape(-1, 2).astype(np.float64)
    p = np.asarray(pt, dtype=np.float64)
    n = len(pts)
    best_dist = float("inf")
    best_pt = (float(p[0]), float(p[1]))
    for i in range(n):
        v1 = pts[i]
        v2 = pts[(i + 1) % n]
        edge = v2 - v1
        len_sq = float(np.dot(edge, edge))
        if len_sq < 1e-8:
            t = 0.0
        else:
            t = max(0.0, min(1.0, float(np.dot(p - v1, edge) / len_sq)))
        proj = v1 + t * edge
        d = float(np.linalg.norm(p - proj))
        if d < best_dist:
            best_dist = d
            best_pt = (float(proj[0]), float(proj[1]))
    return best_pt, best_dist

class ZoomImageView(tk.Frame):
    """Zoom/pan viewer for inspection output, using the teaching viewport math."""

    def __init__(self, parent, image):
        super().__init__(parent, bg=BG_CARD)
        self.image = image
        self.view = ImageViewport()
        self.press = None
        self.label = tk.Label(self, bg=BG_CARD, fg=TEXT_MUTED,
                              text="Ctrl + wheel: zoom | Left drag: pan")
        self.label.pack(fill=tk.X)
        self.canvas = tk.Canvas(self, bg="#080b0f", height=280, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.render())
        self.canvas.bind("<Control-MouseWheel>", self.zoom)
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "press", None))

    def sizes(self):
        return (self.image.shape[1], self.image.shape[0]), (
            max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        )

    def zoom(self, event):
        if event.delta:
            self.view.zoom_at(1.25 if event.delta > 0 else .8,
                              event.x, event.y, *self.sizes())
            self.render()
        return "break"

    def start_pan(self, event):
        self.press = (event.x, event.y, self.view.pan_x, self.view.pan_y)

    def pan(self, event):
        if self.press and self.view.zoom > 1:
            x, y, px, py = self.press
            self.view.pan_x, self.view.pan_y = px + event.x - x, py + event.y - y
            self.render()

    def render(self):
        image_size, size = self.sizes()
        scale, x, y = self.view.transform(image_size, size)
        displayed = cv2.warpAffine(self.image, np.array([[scale, 0, x], [0, scale, y]]),
                                   size, borderValue=(8, 11, 15))
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(displayed, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.label.config(text=f"{self.view.zoom * 100:.0f}% | Ctrl + wheel: zoom | Left drag: pan")


class ScanlineStudioApp:
    def __init__(self, root, initial_sku="98661BBS-1"):
        self.root = root
        self.root.title("Industrial QC_CAM - Scanline Teaching Studio & Recipe Manager")
        self.root.configure(bg=BG_DARK)

        # Fit the initial window to the current monitor instead of assuming a
        # large engineering workstation.  The reserved vertical space keeps
        # the title bar and Windows taskbar visible on common laptop screens.
        screen_w = max(self.root.winfo_screenwidth(), 960)
        screen_h = max(self.root.winfo_screenheight(), 680)
        window_w = min(1400, int(screen_w * 0.92))
        window_h = min(860, max(620, screen_h - 110))
        pos_x = max(0, (screen_w - window_w) // 2)
        pos_y = max(0, (screen_h - window_h) // 2 - 10)
        self.root.geometry(f"{window_w}x{window_h}+{pos_x}+{pos_y}")
        self.root.minsize(min(900, window_w), min(620, window_h))

        self.products = load_products_dict()
        self.current_sku = initial_sku
        self.calipers = []
        self.selected_idx = -1
        self.drawing_points = []
        self.temp_mouse_pos = None
        self.active_mode = "select" # "select", "draw", or "draw_inner"
        self.pending_dimension = None
        self.viewport = ImageViewport()
        self._mouse_press = None
        self._dragging_image = False
        self._profile_preview_cache = None

        self.recipes = {}
        self.load_recipes()

        self.measurer_inst = measurer.WorkpieceMeasurer()

        self.setup_ui()
        self.load_sku(self.current_sku)

    def load_recipes(self):
        if RECIPE_FILE.exists():
            try:
                with open(RECIPE_FILE, "r", encoding="utf-8") as f:
                    self.recipes = json.load(f)
            except Exception as e:
                print(f"Error loading recipes: {e}")
                self.recipes = {}

    def _setup_legacy_ui(self):
        # Top Header Bar
        top_bar = tk.Frame(self.root, bg=BG_CARD, height=60, padx=15, pady=8)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        title_lbl = tk.Label(
            top_bar, text="QC_CAM METROLOGY STUDIO",
            font=("Segoe UI", 14, "bold"), fg=ACCENT_CYAN, bg=BG_CARD
        )
        title_lbl.pack(side=tk.LEFT, padx=(0, 25))

        # SKU Selector
        sku_lbl = tk.Label(top_bar, text="Linh kiện (SKU):", font=("Segoe UI", 10), fg=TEXT_WHITE, bg=BG_CARD)
        sku_lbl.pack(side=tk.LEFT, padx=(0, 8))

        skus = list(self.products.keys())
        self.sku_var = tk.StringVar(value=self.current_sku)
        self.sku_combo = ttk.Combobox(top_bar, textvariable=self.sku_var, values=skus, state="readonly", width=14, font=("Segoe UI", 10))
        self.sku_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.sku_combo.bind("<<ComboboxSelected>>", self.on_sku_changed)

        # Import PDF Button
        btn_import_pdf = tk.Button(
            top_bar, text="📥 Thêm SKU từ PDF", font=("Segoe UI", 9, "bold"),
            bg="#7c4dff", fg=TEXT_WHITE, activebackground="#651fff",
            relief=tk.FLAT, padx=10, pady=4, cursor="hand2", command=self.import_new_pdf
        )
        btn_import_pdf.pack(side=tk.LEFT, padx=(0, 15))

        # Toolbar Action Buttons
        self.btn_draw = tk.Button(
            top_bar, text="➕ Vẽ Tia Đo Mới", font=("Segoe UI", 9, "bold"),
            bg="#2979ff", fg=TEXT_WHITE, activebackground="#1565c0",
            relief=tk.FLAT, padx=12, pady=4, cursor="hand2", command=self.enable_draw_mode
        )
        self.btn_draw.pack(side=tk.LEFT, padx=5)

        self.btn_inner = tk.Button(
            top_bar, text="Do lo / khe trong", font=("Segoe UI", 9, "bold"),
            bg="#8e24aa", fg=TEXT_WHITE, activebackground="#6a1b9a",
            relief=tk.FLAT, padx=12, pady=4, cursor="hand2",
            command=self.enable_inner_draw_mode,
        )
        self.btn_inner.pack(side=tk.LEFT, padx=5)

        btn_save = tk.Button(
            top_bar, text="💾 Lưu Công Thức", font=("Segoe UI", 9, "bold"),
            bg=ACCENT_GREEN, fg="#000000", activebackground="#00c853",
            relief=tk.FLAT, padx=12, pady=4, cursor="hand2", command=self.save_recipe
        )
        btn_save.pack(side=tk.LEFT, padx=5)

        btn_reset = tk.Button(
            top_bar, text="🔄 Khôi Phục Đề Xuất CAD", font=("Segoe UI", 9),
            bg="#37474f", fg=TEXT_WHITE, activebackground="#455a64",
            relief=tk.FLAT, padx=10, pady=4, cursor="hand2", command=self.reset_proposals
        )
        btn_reset.pack(side=tk.LEFT, padx=5)

        btn_clear = tk.Button(
            top_bar, text="🧹 Xóa Hết", font=("Segoe UI", 9),
            bg="#424242", fg="#ff8a80", activebackground="#616161",
            relief=tk.FLAT, padx=10, pady=4, cursor="hand2", command=self.clear_all_calipers
        )
        btn_clear.pack(side=tk.LEFT, padx=5)

        btn_test = tk.Button(
            top_bar, text="▶️ Chạy Kiểm Tra Mẫu", font=("Segoe UI", 9, "bold"),
            bg=ACCENT_YELLOW, fg="#000000", activebackground="#ffab00",
            relief=tk.FLAT, padx=12, pady=4, cursor="hand2", command=self.run_test_sample
        )
        btn_test.pack(side=tk.RIGHT, padx=5)

        # Main Body (Split into Left Canvas and Right Control Panel)
        main_body = tk.Frame(self.root, bg=BG_DARK)
        main_body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Canvas Panel
        canvas_card = tk.Frame(main_body, bg=BG_CARD, bd=1, relief=tk.SOLID)
        canvas_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.canvas = tk.Canvas(canvas_card, bg="#0d0f12", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.canvas.bind("<Configure>", lambda e: self.redraw_canvas())

        # Canvas Status Overlay Banner at Bottom of Canvas Card
        self.canvas_status_lbl = tk.Label(
            canvas_card, text="Chế độ: Chọn & Xem tia đo | Click '➕ Vẽ Tia Đo Mới' để click 2 điểm qua cạnh cần đo",
            font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD, anchor="w", padx=10, pady=4
        )
        self.canvas_status_lbl.pack(side=tk.BOTTOM, fill=tk.X)

        # Right Sidebar Panel
        sidebar = tk.Frame(main_body, bg=BG_SIDEBAR, width=400)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Sidebar Title
        sb_title = tk.Label(
            sidebar, text="DANH SÁCH KÍCH THƯỚC ĐÃ CHỌN",
            font=("Segoe UI", 11, "bold"), fg=TEXT_WHITE, bg=BG_SIDEBAR
        )
        sb_title.pack(side=tk.TOP, fill=tk.X, padx=15, pady=(15, 5), anchor="w")

        sb_desc = tk.Label(
            sidebar, text="Hệ thống CHỈ đo & CHỈ vẽ các kích thước có trong danh sách này:",
            font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_SIDEBAR, justify=tk.LEFT
        )
        sb_desc.pack(side=tk.TOP, fill=tk.X, padx=15, pady=(0, 10), anchor="w")

        # Table of Calipers
        tree_frame = tk.Frame(sidebar, bg=BG_SIDEBAR)
        tree_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=15)

        columns = ("label", "nominal", "measured", "type")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=8, selectmode="browse")
        self.tree.heading("label", text="Nhãn")
        self.tree.heading("nominal", text="Tiêu chuẩn")
        self.tree.heading("measured", text="Đo thực")
        self.tree.heading("type", text="Loại đo")

        self.tree.column("label", width=65, anchor="center")
        self.tree.column("nominal", width=85, anchor="center")
        self.tree.column("measured", width=85, anchor="center")
        self.tree.column("type", width=75, anchor="center")
        self.tree.heading("type", text="Status")

        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        # Quick Delete / Action Buttons for Selected Caliper
        action_row = tk.Frame(sidebar, bg=BG_SIDEBAR)
        action_row.pack(side=tk.TOP, fill=tk.X, padx=15, pady=8)

        self.btn_del = tk.Button(
            action_row, text="🗑️ XÓA KÍCH THƯỚC NÀY", font=("Segoe UI", 9, "bold"),
            bg=ACCENT_RED, fg=TEXT_WHITE, activebackground="#d50000",
            relief=tk.FLAT, padx=10, pady=5, cursor="hand2", command=self.delete_selected_caliper
        )
        self.btn_del.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        # Caliper Detail / Edit Card
        edit_card = tk.LabelFrame(sidebar, text=" Gán / Chỉnh sửa Kích Thước ", font=("Segoe UI", 9, "bold"), fg=ACCENT_CYAN, bg=BG_CARD, padx=12, pady=10)
        edit_card.pack(side=tk.TOP, fill=tk.X, padx=15, pady=10)

        # Suggested CAD Specs Dropdown
        tk.Label(edit_card, text="Kích thước CAD từ PDF:", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD).grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.cad_combo_var = tk.StringVar()
        self.cad_combo = ttk.Combobox(edit_card, textvariable=self.cad_combo_var, state="readonly", width=30, font=("Segoe UI", 9))
        self.cad_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.cad_combo.bind("<<ComboboxSelected>>", self.on_cad_spec_selected)

        # Label Entry
        tk.Label(edit_card, text="Tên Nhãn (Label):", font=("Segoe UI", 8), fg=TEXT_WHITE, bg=BG_CARD).grid(row=2, column=0, sticky="w")
        self.entry_label = tk.Entry(edit_card, font=("Segoe UI", 9), bg="#2d323f", fg=TEXT_WHITE, insertbackground=TEXT_WHITE, bd=1, relief=tk.SOLID)
        self.entry_label.grid(row=3, column=0, sticky="ew", padx=(0, 5), pady=(0, 8))

        # Nominal Entry
        tk.Label(edit_card, text="Giá trị chuẩn (Nominal mm):", font=("Segoe UI", 8), fg=TEXT_WHITE, bg=BG_CARD).grid(row=2, column=1, sticky="w")
        self.entry_nom = tk.Entry(edit_card, font=("Segoe UI", 9), bg="#2d323f", fg=TEXT_WHITE, insertbackground=TEXT_WHITE, bd=1, relief=tk.SOLID)
        self.entry_nom.grid(row=3, column=1, sticky="ew", pady=(0, 8))

        # Description
        tk.Label(edit_card, text="Mô tả kỹ thuật:", font=("Segoe UI", 8), fg=TEXT_WHITE, bg=BG_CARD).grid(row=4, column=0, sticky="w")
        self.entry_desc = tk.Entry(edit_card, font=("Segoe UI", 9), bg="#2d323f", fg=TEXT_WHITE, insertbackground=TEXT_WHITE, bd=1, relief=tk.SOLID)
        self.entry_desc.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        btn_update = tk.Button(
            edit_card, text="✔ ÁP DỤNG / CẬP NHẬT KÍCH THƯỚC", font=("Segoe UI", 9, "bold"),
            bg="#0288d1", fg=TEXT_WHITE, activebackground="#0277bd",
            relief=tk.FLAT, pady=5, cursor="hand2", command=self.update_current_caliper
        )
        btn_update.grid(row=6, column=0, columnspan=2, sticky="ew")

        # Compact tolerance/spec section.  This is intentionally separated
        # from scanline geometry so operators only see the fields they need.
        spec_card = tk.LabelFrame(
            sidebar, text=" DUNG SAI & SPEC ", font=("Segoe UI", 9, "bold"),
            fg=ACCENT_GREEN, bg=BG_CARD, padx=12, pady=8
        )
        spec_card.pack(side=tk.TOP, fill=tk.X, padx=15, pady=(0, 8))
        tk.Label(spec_card, text="- tolerance (mm)", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD).grid(row=0, column=0, sticky="w")
        tk.Label(spec_card, text="+ tolerance (mm)", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD).grid(row=0, column=1, sticky="w")
        self.entry_tol_minus = tk.Entry(spec_card, font=("Segoe UI", 9), bg="#2d323f", fg=TEXT_WHITE, insertbackground=TEXT_WHITE, bd=1, relief=tk.SOLID)
        self.entry_tol_plus = tk.Entry(spec_card, font=("Segoe UI", 9), bg="#2d323f", fg=TEXT_WHITE, insertbackground=TEXT_WHITE, bd=1, relief=tk.SOLID)
        self.entry_tol_minus.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(2, 7))
        self.entry_tol_plus.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(2, 7))
        self.entry_tol_minus.insert(0, "0.10")
        self.entry_tol_plus.insert(0, "0.10")
        btn_add_dimension = tk.Button(
            spec_card, text="+ THÃŠM THÃ”NG Sá» & Váº¼ Cáº NH ÄO", font=("Segoe UI", 9, "bold"),
            bg=ACCENT_GREEN, fg="#101410", activebackground="#00c853",
            relief=tk.FLAT, pady=5, cursor="hand2", command=self.add_dimension_and_draw
        )
        btn_add_dimension.grid(row=2, column=0, columnspan=2, sticky="ew")
        btn_add_inner = tk.Button(
            spec_card, text="+ THEM THONG SO LO / KHE TRONG", font=("Segoe UI", 9, "bold"),
            bg="#8e24aa", fg=TEXT_WHITE, activebackground="#6a1b9a",
            relief=tk.FLAT, pady=5, cursor="hand2",
            command=self.add_dimension_and_draw_inner,
        )
        btn_add_inner.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        spec_card.columnconfigure(0, weight=1)
        spec_card.columnconfigure(1, weight=1)

        # Metrology Info Box at Bottom
        info_frame = tk.Frame(sidebar, bg=BG_CARD, padx=12, pady=10)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=15)

        self.info_scale_lbl = tk.Label(info_frame, text="Độ phân giải camera: -- um/px", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD, anchor="w")
        self.info_scale_lbl.pack(fill=tk.X)

        self.info_snapped_lbl = tk.Label(info_frame, text="Tọa độ mép cắt edge: Q1=(--, --), Q2=(--, --)", font=("Segoe UI", 8), fg=ACCENT_GREEN, bg=BG_CARD, anchor="w")
        self.info_snapped_lbl.pack(fill=tk.X)

        self.info_status_lbl = tk.Label(info_frame, text="Trạng thái: Đã đồng bộ với caliper_recipes.json", font=("Segoe UI", 8, "italic"), fg=TEXT_WHITE, bg=BG_CARD, anchor="w")
        self.info_status_lbl.pack(fill=tk.X, pady=(4, 0))

    def setup_ui(self):
        """Build the operator-facing industrial recipe editor.

        The legacy layout mixed machine actions, recipe actions and edit
        fields in one dense row.  This layout establishes a clear hierarchy:
        machine context, commands, live image, then the dimension inspector.
        Widget attribute names remain stable so the existing recipe logic is
        unchanged.
        """
        self.root.option_add("*Font", ("Segoe UI", 9))
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Industrial.Treeview", background="#161c24", fieldbackground="#161c24",
            foreground="#dce6f2", rowheight=30, borderwidth=0,
        )
        style.configure(
            "Industrial.Treeview.Heading", background="#273241", foreground="#9fb1c5",
            relief="flat", font=("Segoe UI Semibold", 9), padding=(8, 8),
        )
        style.map(
            "Industrial.Treeview", background=[("selected", "#164e63")],
            foreground=[("selected", "#ffffff")],
        )
        style.configure(
            "Industrial.TCombobox", fieldbackground="#151b23", background="#273241",
            foreground="#e7eef7", arrowcolor="#54d6e8", padding=5,
        )

        def button(parent, text, command, bg="#273241", fg="#e7eef7", width=None):
            widget = tk.Button(
                parent, text=text, command=command, bg=bg, fg=fg,
                activebackground=bg, activeforeground=fg, relief=tk.FLAT,
                bd=0, padx=14, pady=8, cursor="hand2",
                font=("Segoe UI Semibold", 9), width=width,
                highlightthickness=0,
            )
            return widget

        def section_title(parent, code, title, description=None):
            row = tk.Frame(parent, bg=BG_SIDEBAR)
            row.pack(fill=tk.X, padx=16, pady=(14, 7))
            tk.Label(
                row, text=code, bg="#164e63", fg="#67e8f9",
                font=("Consolas", 8, "bold"), padx=7, pady=3,
            ).pack(side=tk.LEFT)
            tk.Label(
                row, text=title, bg=BG_SIDEBAR, fg=TEXT_WHITE,
                font=("Segoe UI Semibold", 10),
            ).pack(side=tk.LEFT, padx=8)
            if description:
                tk.Label(
                    row, text=description, bg=BG_SIDEBAR, fg=TEXT_MUTED,
                    font=("Segoe UI", 8),
                ).pack(side=tk.RIGHT)

        def field(parent, label, row, column, colspan=1):
            tk.Label(
                parent, text=label.upper(), bg=BG_CARD, fg="#8294a8",
                font=("Segoe UI Semibold", 7), anchor="w",
            ).grid(row=row, column=column, columnspan=colspan, sticky="ew", pady=(0, 4))
            entry = tk.Entry(
                parent, bg="#151b23", fg="#eef5fb", insertbackground="#67e8f9",
                selectbackground="#0e7490", relief=tk.FLAT, bd=0,
                highlightthickness=1, highlightbackground="#344253",
                highlightcolor="#22d3ee", font=("Segoe UI", 10),
            )
            entry.grid(
                row=row + 1, column=column, columnspan=colspan, sticky="ew",
                padx=(0, 7 if colspan == 1 and column == 0 else 0), ipady=7, pady=(0, 10),
            )
            return entry

        # Application header -------------------------------------------------
        header = tk.Frame(self.root, bg="#11161d", height=70)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        identity = tk.Frame(header, bg="#11161d")
        identity.pack(side=tk.LEFT, padx=20, pady=12)
        tk.Label(
            identity, text="QC  /  METROLOGY", bg="#11161d", fg="#67e8f9",
            font=("Consolas", 9, "bold"), anchor="w",
        ).pack(anchor="w")
        tk.Label(
            identity, text="Recipe Engineering Station", bg="#11161d", fg="#f2f6fa",
            font=("Segoe UI Semibold", 17), anchor="w",
        ).pack(anchor="w")
        status = tk.Frame(header, bg="#11161d")
        status.pack(side=tk.RIGHT, padx=20)
        tk.Label(
            status, text="VISION ONLINE", bg="#12372f", fg="#6ee7b7",
            font=("Consolas", 8, "bold"), padx=10, pady=5,
        ).pack(side=tk.RIGHT)
        tk.Label(
            status, text="ENGINE 2D  |  LIVE RECIPE", bg="#11161d", fg="#718297",
            font=("Consolas", 8), padx=12,
        ).pack(side=tk.RIGHT)

        # Command strip ------------------------------------------------------
        toolbar = tk.Frame(self.root, bg="#1b232d", padx=18, pady=10)
        toolbar.pack(fill=tk.X)
        tk.Label(
            toolbar, text="PART NUMBER", bg="#1b232d", fg="#8294a8",
            font=("Segoe UI Semibold", 7),
        ).pack(side=tk.LEFT, padx=(0, 7))
        self.sku_var = tk.StringVar(value=self.current_sku)
        self.sku_combo = ttk.Combobox(
            toolbar, textvariable=self.sku_var, values=list(self.products.keys()),
            state="readonly", width=17, style="Industrial.TCombobox",
        )
        self.sku_combo.pack(side=tk.LEFT, padx=(0, 16))
        self.sku_combo.bind("<<ComboboxSelected>>", self.on_sku_changed)
        button(toolbar, "TẠO SKU TỪ ẢNH MẪU", self.create_new_sku_from_image, "#1e3a8a", "#93c5fd").pack(side=tk.LEFT, padx=(0, 6))
        button(toolbar, "NHẬP PDF (PHỤ)", self.import_new_pdf, "#374151", "#9ca3af").pack(side=tk.LEFT, padx=(0, 10))
        button(toolbar, "▶ CHẠY THỬ MẪU", self.run_test_sample, "#b7791f", "#ffffff").pack(side=tk.RIGHT, padx=4)
        button(toolbar, "💾 LƯU SPEC & CÔNG THỨC", self.save_recipe, "#059669", "#ffffff").pack(side=tk.RIGHT, padx=4)
        toolbar = tk.Frame(self.root, bg="#1b232d", padx=18, pady=5)
        toolbar.pack(fill=tk.X)
        tk.Frame(toolbar, bg="#344253", width=1, height=30).pack(side=tk.LEFT, padx=(0, 8))
        button(toolbar, "↔ CHIỀU DÀI L", self.add_quick_length_caliper, "#0e7490", "#ffffff").pack(side=tk.LEFT, padx=3)
        button(toolbar, "↕ BỀ RỘNG W", self.add_quick_width_caliper, "#0284c7", "#ffffff").pack(side=tk.LEFT, padx=3)
        self.btn_draw = button(toolbar, "📐 CALIPER TỰ DO (2 ĐIỂM)", self.enable_draw_mode, "#2563eb")
        self.btn_draw.pack(side=tk.LEFT, padx=3)
        self.btn_inner = button(toolbar, "⭕ LỖ / RÃNH", self.enable_inner_draw_mode, "#6d28d9")
        self.btn_inner.pack(side=tk.LEFT, padx=3)
        self.ortho_var = tk.BooleanVar(value=False)
        self.chk_ortho = tk.Checkbutton(
            toolbar, text="Khóa vuông góc (Ortho)", variable=self.ortho_var,
            bg="#1b232d", fg="#67e8f9", selectcolor="#11161d",
            activebackground="#1b232d", activeforeground="#67e8f9",
            font=("Segoe UI", 8, "bold")
        )
        self.chk_ortho.pack(side=tk.LEFT, padx=(5, 6))
        button(toolbar, "XÓA HẾT", self.clear_all_calipers, "#3b2730", "#fda4af").pack(side=tk.LEFT, padx=3)

        # Work area ----------------------------------------------------------
        body = tk.PanedWindow(
            self.root, orient=tk.HORIZONTAL, bg=BG_DARK, sashwidth=6,
            sashrelief=tk.FLAT, bd=0,
        )
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(12, 8))

        viewport = tk.Frame(body, bg=BG_CARD, highlightthickness=1, highlightbackground="#303c4b")
        body.add(viewport, stretch="always", minsize=520)
        viewport_header = tk.Frame(viewport, bg=BG_CARD, height=44)
        viewport_header.pack(fill=tk.X)
        tk.Label(
            viewport_header, text="LIVE EDGE VIEW", bg=BG_CARD, fg="#dce6f2",
            font=("Segoe UI Semibold", 10), padx=12,
        ).pack(side=tk.LEFT, pady=10)
        tk.Label(
            viewport_header, text="OUTER LIMIT", bg=BG_CARD, fg="#ff9100",
            font=("Consolas", 8, "bold"), padx=8,
        ).pack(side=tk.RIGHT)
        tk.Label(
            viewport_header, text="INNER LIMIT", bg=BG_CARD, fg="#2896ff",
            font=("Consolas", 8, "bold"), padx=8,
        ).pack(side=tk.RIGHT)
        self.canvas = tk.Canvas(
            viewport, bg="#080b0f", highlightthickness=0, cursor="crosshair",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=6)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Control-MouseWheel>", self.on_canvas_zoom)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.canvas.bind("<Configure>", lambda e: self.redraw_canvas())
        self.canvas_status_lbl = tk.Label(
            viewport,
            text="SELECT MODE  |  Choose a caliper tool, then place two points across the target edges",
            bg="#11161d", fg="#91a3b7", anchor="w", padx=12, pady=8,
            font=("Consolas", 8),
        )
        self.canvas_status_lbl.pack(fill=tk.X, padx=6, pady=(0, 6))
        self.zoom_label = tk.Label(
            viewport, text="100% | Ctrl + wheel: zoom | Left drag: pan",
            bg=BG_CARD, fg=TEXT_MUTED, anchor="w", padx=12,
        )
        self.zoom_label.pack(fill=tk.X)

        # The inspector is taller than many laptop screens.  Host it in a
        # vertical canvas so every acquisition control remains reachable
        # instead of being clipped below the bottom edge of the window.
        sidebar_host = tk.Frame(body, bg=BG_SIDEBAR, width=420)
        sidebar_host.pack_propagate(False)
        body.add(sidebar_host, minsize=380, width=420)

        sidebar_canvas = tk.Canvas(
            sidebar_host, bg=BG_SIDEBAR, highlightthickness=0, bd=0,
        )
        sidebar_scrollbar = ttk.Scrollbar(
            sidebar_host, orient=tk.VERTICAL, command=sidebar_canvas.yview,
        )
        sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)
        sidebar_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(sidebar_canvas, bg=BG_SIDEBAR)
        sidebar_window = sidebar_canvas.create_window(
            (0, 0), window=sidebar, anchor="nw"
        )

        def update_sidebar_scrollregion(_event=None):
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))

        def fit_sidebar_width(event):
            sidebar_canvas.itemconfigure(sidebar_window, width=event.width)

        def scroll_sidebar(event):
            units = -1 if event.delta > 0 else 1
            sidebar_canvas.yview_scroll(units * 3, "units")
            return "break"

        sidebar.bind("<Configure>", update_sidebar_scrollregion)
        sidebar_canvas.bind("<Configure>", fit_sidebar_width)

        section_title(sidebar, "01", "ACTIVE DIMENSIONS", "recipe output")
        tree_frame = tk.Frame(sidebar, bg=BG_SIDEBAR)
        tree_frame.pack(fill=tk.X, padx=16)
        columns = ("label", "nominal", "measured", "type", "status")
        self.tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings", height=5,
            selectmode="browse", style="Industrial.Treeview",
        )
        headings = {
            "label": "ID", "nominal": "NOMINAL", "measured": "GOLDEN", "type": "MODE", "status": "STATUS"
        }
        widths = {"label": 46, "nominal": 74, "measured": 78, "type": 64, "status": 64}
        for name in columns:
            self.tree.heading(name, text=headings[name])
            self.tree.column(name, width=widths[name], anchor="center", stretch=True)
        scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.btn_del = button(sidebar, "DELETE SELECTED DIMENSION", self.delete_selected_caliper, "#4c2630", "#fda4af")
        self.btn_del.pack(fill=tk.X, padx=16, pady=(7, 0))

        section_title(sidebar, "02", "DIMENSION INSPECTOR", "mm")
        inspector = tk.Frame(sidebar, bg=BG_CARD, padx=12, pady=11)
        inspector.pack(fill=tk.X, padx=16)
        tk.Label(
            inspector, text="CAD DIMENSION", bg=BG_CARD, fg="#8294a8",
            font=("Segoe UI Semibold", 7), anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self.cad_combo_var = tk.StringVar()
        self.cad_combo = ttk.Combobox(
            inspector, textvariable=self.cad_combo_var, state="readonly",
            style="Industrial.TCombobox",
        )
        self.cad_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.cad_combo.bind("<<ComboboxSelected>>", self.on_cad_spec_selected)
        self.entry_label = field(inspector, "Dimension ID", 2, 0)
        self.entry_nom = field(inspector, "Nominal", 2, 1)
        self.entry_desc = field(inspector, "Engineering description", 4, 0, 2)
        self.entry_golden_actual = field(inspector, "TPL Golden actual mm (blank = image)", 6, 0, 2)
        button(
            inspector, "APPLY DIMENSION", self.update_current_caliper, "#0e7490"
        ).grid(row=8, column=0, columnspan=2, sticky="ew")
        inspector.columnconfigure(0, weight=1)
        inspector.columnconfigure(1, weight=1)

        section_title(sidebar, "03", "DIMENSION TOLERANCE", "per scanline")
        tolerance = tk.Frame(sidebar, bg=BG_CARD, padx=12, pady=11)
        tolerance.pack(fill=tk.X, padx=16)
        self.entry_tol_minus = field(tolerance, "Dimension minus (mm)", 0, 0)
        self.entry_tol_plus = field(tolerance, "Dimension plus (mm)", 0, 1)
        self.entry_tol_minus.insert(0, "0.10")
        self.entry_tol_plus.insert(0, "0.10")
        button(
            tolerance, "ADD OUTER DIMENSION", self.add_dimension_and_draw, "#145b70"
        ).grid(row=2, column=0, sticky="ew", padx=(0, 4))
        button(
            tolerance, "ADD INNER DIMENSION", self.add_dimension_and_draw_inner, "#5b21b6"
        ).grid(row=2, column=1, sticky="ew", padx=(4, 0))
        tolerance.columnconfigure(0, weight=1)
        tolerance.columnconfigure(1, weight=1)
        section_title(sidebar, "04", "TEMPLATE CONTOUR", "whole outline")
        profile_card = tk.Frame(sidebar, bg=BG_CARD, padx=12, pady=11)
        profile_card.pack(fill=tk.X, padx=16)
        self.entry_profile_lower = field(profile_card, "Inward / Lower (mm)", 0, 0)
        self.entry_profile_upper = field(profile_card, "Outward / Upper (mm)", 0, 1)
        for entry in (self.entry_profile_lower, self.entry_profile_upper):
            entry.insert(0, "0.10")
            entry.bind("<KeyRelease>", lambda e: self.redraw_canvas())
        tk.Label(profile_card, text="BLUE: inner  |  ORANGE: outer  |  GRAY: golden",
                 bg=BG_CARD, fg="#aabbd0", font=("Consolas", 8)).grid(
                     row=2, column=0, columnspan=2, sticky="w", pady=5)
        button(
            profile_card, "ADD TEMPLATE-CONTOUR DIMENSION",
            self.add_dimension_and_draw_template, "#9a5b13", "#ffffff"
        ).grid(row=3, column=0, columnspan=2, sticky="ew")
        self.profile_state_label = tk.Label(
            profile_card, text="Golden contour inspection enabled",
            bg=BG_CARD, fg=ACCENT_YELLOW, font=("Segoe UI", 8), anchor="w",
        )
        self.profile_state_label.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.profile_enabled_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            profile_card, text="CHECK GOLDEN CONTOUR", variable=self.profile_enabled_var,
            command=self.refresh_table, bg=BG_CARD, fg=TEXT_WHITE,
            selectcolor=BG_DARK, activebackground=BG_CARD, activeforeground=TEXT_WHITE,
        ).grid(row=5, column=0, columnspan=2, sticky="w")
        profile_card.columnconfigure(0, weight=1)
        profile_card.columnconfigure(1, weight=1)

        info = tk.Frame(sidebar, bg="#11161d", padx=14, pady=10)
        info.pack(fill=tk.X, padx=16, pady=14)
        self.info_scale_lbl = tk.Label(
            info, text="SCALE  -- um/px", bg="#11161d", fg="#91a3b7",
            anchor="w", font=("Consolas", 8),
        )
        self.info_scale_lbl.pack(fill=tk.X)
        self.info_snapped_lbl = tk.Label(
            info, text="EDGE  Q1=(--, --)  Q2=(--, --)", bg="#11161d",
            fg="#6ee7b7", anchor="w", font=("Consolas", 8),
        )
        self.info_snapped_lbl.pack(fill=tk.X, pady=3)
        self.info_status_lbl = tk.Label(
            info, text="READY  Recipe store synchronized", bg="#11161d",
            fg="#dce6f2", anchor="w", font=("Consolas", 8),
        )
        self.info_status_lbl.pack(fill=tk.X)

        # Bind the wheel on every sidebar child.  The dimensions table keeps
        # its dedicated scrollbar; the mouse wheel moves the whole inspector.
        def bind_sidebar_wheel(widget):
            widget.bind("<MouseWheel>", scroll_sidebar, add="+")
            for child in widget.winfo_children():
                bind_sidebar_wheel(child)

        bind_sidebar_wheel(sidebar_canvas)
        self.sidebar_canvas = sidebar_canvas

    def load_sku(self, sku):
        self.current_sku = sku
        self.sku_var.set(sku)
        self.selected_idx = -1
        self.drawing_points = []
        self.active_mode = "select"
        self.viewport.reset()
        self._profile_preview_cache = None

        # Load reference image
        ref_dir = PROJECT_DIR / "assets" / "references" / sku / "images"
        ref_candidates = []
        if ref_dir.is_dir():
            for pat in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG", "*.bmp"):
                ref_candidates.extend(ref_dir.glob(pat))
        if not ref_candidates:
            test_dir = PROJECT_DIR / "assets" / "tests" / sku
            if test_dir.is_dir():
                for pat in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG", "*.bmp"):
                    ref_candidates.extend(test_dir.rglob(pat))
        if not ref_candidates:
            ref_root = PROJECT_DIR / "assets" / "references" / sku
            if ref_root.is_dir():
                for pat in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG", "*.bmp"):
                    ref_candidates.extend(ref_root.glob(pat))
        if not ref_candidates:
            messagebox.showerror("Lỗi", f"Không tìm thấy ảnh mẫu chuẩn cho SKU: {sku}")
            return

        taught_image = self.recipes.get(sku, {}).get("reference_image")
        taught_path = PROJECT_DIR / taught_image if taught_image else None
        self.ref_img_path = taught_path if taught_path and taught_path.is_file() else ref_candidates[0]
        self.ref_img = cv2.imread(str(self.ref_img_path))
        if self.ref_img is None:
            messagebox.showerror("Lỗi", f"Không thể đọc file ảnh: {self.ref_img_path}")
            return
        self.desc = clf.describe_image(self.ref_img)
        if self.desc is None:
            messagebox.showerror("Lỗi", f"Không nhận diện được vật thể trong ảnh mẫu của SKU: {sku}")
            return
        self.cnt = self.desc["contour"]
        self.gray = self.desc["gray"]
        self.wcs = get_wcs_info(self.cnt)
        self.detected_inner_contours = self.measurer_inst.detect_inner_contours(
            self.gray, self.cnt, image_bgr=self.ref_img
        )

        # Load specs from spec.json
        self.spec_dims = []
        self.spec_path = PROJECT_DIR / "assets" / "references" / sku / "spec.json"
        self.spec_data = {"sku": sku, "unit": "mm", "dimensions": []}
        if self.spec_path.exists():
            try:
                with open(self.spec_path, "r", encoding="utf-8") as f:
                    self.spec_data = json.load(f)
                    self.spec_dims = self.spec_data.get("dimensions", [])
            except Exception:
                pass

        self.optical_scale = self.spec_data.get(
            "optical_scale_mm_per_px",
            self.products.get(sku, {}).get("optical_scale_mm_per_px", 0.044),
        )
        self.info_scale_lbl.config(text=f"Độ phân giải camera: {self.optical_scale*1000.0:.2f} um/px ({self.optical_scale:.5f} mm/px)")

        # Update CAD combo values
        cad_choices = []
        for d in self.spec_dims:
            lbl = d.get("label", "")
            raw = d.get("raw_spec", "")
            axis = d.get("axis", "")
            if axis != "Z_height_side_view":
                cad_choices.append(f"{lbl}: {raw}")
        self.cad_combo["values"] = cad_choices
        if cad_choices:
            self.cad_combo.current(0)

        # Load calipers from recipe, or if empty propose defaults
        sku_recipe = self.recipes.get(sku, {})
        self.profile_enabled_var.set(sku_recipe.get("contour_profile_enabled", True))
        lower, upper = profile_limits(sku_recipe)
        self._replace_entry(self.entry_profile_lower, f"{lower:g}")
        self._replace_entry(self.entry_profile_upper, f"{upper:g}")
        calipers = sku_recipe.get("calipers", [])
        if not calipers:
            calipers = get_proposed_calipers_for_sku(sku, self.wcs, self.optical_scale)
            self.recipes[sku] = {
                "sku": sku,
                "reference_image": str(self.ref_img_path.relative_to(PROJECT_DIR)),
                "optical_scale_mm_per_px": self.optical_scale,
                "wcs": self.wcs,
                "calipers": calipers
            }

        # Snap loaded calipers to either the external silhouette or a taught
        # internal hole/slot edge pair.
        for cal in calipers:
            # Pixel coordinates belong to the exact image on which the recipe
            # was taught. Rebuild them from WCS whenever possible so opening a
            # different crop/resolution of the same SKU cannot displace the
            # displayed caliper away from the current contour.
            p1_wcs = cal.get("p1_wcs")
            p2_wcs = cal.get("p2_wcs")
            if p1_wcs and p2_wcs:
                p1_ref = from_wcs(p1_wcs[0], p1_wcs[1], self.wcs)
                p2_ref = from_wcs(p2_wcs[0], p2_wcs[1], self.wcs)
                cal["p1_ref"] = [round(p1_ref[0], 2), round(p1_ref[1], 2)]
                cal["p2_ref"] = [round(p2_ref[0], 2), round(p2_ref[1], 2)]
            else:
                p1_ref = cal.get("p1_ref")
                p2_ref = cal.get("p2_ref")
            if p1_ref and p2_ref:
                if cal.get("measure_type") in {"inner_contour", "hole", "slot"}:
                    q1, q2, inner_contour = self.measurer_inst.snap_scanline_to_inner_contour(
                        self.gray, self.cnt, p1_ref, p2_ref, image_bgr=self.ref_img
                    )
                    cal["inner_contour"] = inner_contour
                else:
                    q1, q2 = self.measurer_inst.snap_scanline_to_edges(
                        self.gray, self.cnt, p1_ref, p2_ref
                    )
                cal["p1_edge"] = q1
                cal["p2_edge"] = q2
                d_px = float(np.hypot(q2[0] - q1[0], q2[1] - q1[1]))
                cal["measured_mm"] = round(d_px * self.optical_scale, 2)
                if cal.get("measure_type") == "template_contour":
                    cal["p1_ref"], cal["p2_ref"] = list(q1), list(q2)
                    cal["golden_length_mm"], cal["golden_source"] = golden_baseline(
                        cal, d_px, self.optical_scale
                    )
                    cal["measured_mm"] = cal["golden_length_mm"]

        self.calipers = calipers
        self.refresh_table()
        self.redraw_canvas()

    def create_new_sku_from_image(self):
        """Create a new SKU directly from Golden Sample photo(s) without needing any PDF."""
        from tkinter import filedialog, simpledialog
        import shutil

        sku_name = simpledialog.askstring("Tạo SKU Mới", "Nhập mã sản phẩm mới (SKU ID):", parent=self.root)
        if not sku_name or not sku_name.strip():
            return
        sku = sku_name.strip()

        img_path = filedialog.askopenfilename(
            title=f"Chọn ảnh chụp mẫu chuẩn (Golden Sample) cho {sku}",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")]
        )
        if not img_path:
            return

        sku_ref_dir = PROJECT_DIR / "assets" / "references" / sku
        sku_img_dir = sku_ref_dir / "images"
        sku_img_dir.mkdir(parents=True, exist_ok=True)

        dest_file = sku_img_dir / f"{sku}_1.png"
        shutil.copy(img_path, dest_file)

        img = cv2.imread(str(dest_file))
        if img is None:
            messagebox.showerror("Lỗi", "Không thể đọc file ảnh vừa chọn!")
            return

        described = clf.describe_image(img)
        if described is None:
            messagebox.showerror("Lỗi", "Không tìm thấy vật thể trong ảnh mẫu! Vui lòng chọn ảnh chụp rõ nét.")
            return

        cnt = described["contour"]
        wcs = get_wcs_info(cnt)

        default_scale = 0.044255
        scale_str = simpledialog.askstring(
            "Độ Phân Giải Camera",
            f"Nhập độ phân giải camera (mm/pixel) cho SKU '{sku}':\n(Mặc định 0.044255 mm/px ~ 22.6 px/mm)",
            initialvalue=f"{default_scale:.6f}",
            parent=self.root
        )
        try:
            scale = float(scale_str) if scale_str else default_scale
        except ValueError:
            scale = default_scale

        spec_data = {
            "sku": sku,
            "unit": "mm",
            "optical_scale_mm_per_px": round(scale, 6),
            "golden_sample": {
                "image": str(dest_file.relative_to(PROJECT_DIR)),
                "wcs": wcs,
            },
            "dimensions": []
        }
        spec_path = sku_ref_dir / "spec.json"
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(spec_data, f, indent=2, ensure_ascii=False)

        self.products = load_products_dict()
        self.products[sku] = spec_data
        with open(PROJECT_DIR / "configs" / "products.json", "w", encoding="utf-8") as f:
            json.dump(self.products, f, indent=2, ensure_ascii=False)

        all_skus = list(self.products.keys())
        self.sku_combo["values"] = all_skus
        self.sku_var.set(sku)
        self.load_sku(sku)

        messagebox.showinfo(
            "Tạo SKU Thành Công",
            f"Đã tạo thành công SKU '{sku}' từ ảnh mẫu!\n\n"
            f"- Đã nhận diện đường bao contour và hệ toạ độ WCS ({wcs['width']:.1f} x {wcs['height']:.1f} px).\n"
            f"- Hãy dùng các công cụ (↔ Chiều dài, ↕ Bề rộng, Caliper) để dạy các kích thước đo, "
            f"nhập giá trị tiêu chuẩn & sai số, sau đó bấm 'LƯU SPEC & CÔNG THỨC' (chỉ làm 1 lần)."
        )

    def add_quick_length_caliper(self):
        """Immediately adds a longitudinal overall length caliper along the central axis."""
        p1_wcs = [-0.5, 0.0]
        p2_wcs = [0.5, 0.0]
        p1_raw = from_wcs(p1_wcs[0], p1_wcs[1], self.wcs)
        p2_raw = from_wcs(p2_wcs[0], p2_wcs[1], self.wcs)

        direction = np.asarray(p2_raw, dtype=np.float64) - np.asarray(p1_raw, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm
            pts = self.cnt.reshape(-1, 2).astype(np.float64)
            projections = pts @ direction
            c_raw = (np.asarray(p1_raw, dtype=np.float64) + np.asarray(p2_raw, dtype=np.float64)) / 2.0
            c_proj = float(c_raw @ direction)
            q1_edge = tuple(float(v) for v in (c_raw + (np.min(projections) - c_proj) * direction))
            q2_edge = tuple(float(v) for v in (c_raw + (np.max(projections) - c_proj) * direction))
        else:
            q1_edge, q2_edge = p1_raw, p2_raw

        d_px = float(np.hypot(q2_edge[0] - q1_edge[0], q2_edge[1] - q1_edge[1]))
        d_mm = round(d_px * self.optical_scale, 2)

        u1, v1 = to_wcs(q1_edge, self.wcs)
        u2, v2 = to_wcs(q2_edge, self.wcs)
        v1 = 0.0
        v2 = 0.0

        label = "L"
        nom = d_mm
        desc = "Chiều dài tổng thể dọc trục L"
        if self.entry_label.get().strip():
            label = self.entry_label.get().strip()
        try:
            if self.entry_nom.get().strip():
                nom = float(self.entry_nom.get().strip())
        except ValueError:
            pass

        tol_minus = 0.15
        tol_plus = 0.15
        try:
            tol_minus = float(self.entry_tol_minus.get().strip() or "0.15")
            tol_plus = float(self.entry_tol_plus.get().strip() or "0.15")
        except ValueError:
            pass

        new_cal = {
            "label": label,
            "desc": desc,
            "nominal_mm": round(nom, 2),
            "axis": "2D_in_plane",
            "p1_wcs": [round(u1, 4), round(v1, 4)],
            "p2_wcs": [round(u2, 4), round(v2, 4)],
            "p1_ref": [round(q1_edge[0], 2), round(q1_edge[1], 2)],
            "p2_ref": [round(q2_edge[0], 2), round(q2_edge[1], 2)],
            "p1_edge": q1_edge,
            "p2_edge": q2_edge,
            "measured_mm": d_mm,
            "length_px": round(d_px, 2),
            "measure_type": "length",
            "tolerance_minus_mm": tol_minus,
            "tolerance_plus_mm": tol_plus,
            "color": [0, 230, 255]
        }
        self.calipers.append(new_cal)
        self.selected_idx = len(self.calipers) - 1
        self.refresh_table()
        self.on_tree_select()
        self.redraw_canvas()
        self.canvas_status_lbl.config(
            text=f"✔ Đã thêm tia Chiều Dài L thẳng hàng 100%: Đo được {d_mm:.2f} mm.",
            fg=ACCENT_GREEN
        )

    def add_quick_width_caliper(self):
        """Enters 1-click transverse slice mode: clicking any point on the part creates a perpendicular width scanline."""
        self.active_mode = "slice_width"
        self.drawing_points = []
        self.canvas_status_lbl.config(
            text="CHẾ ĐỘ ĐO BỀ RỘNG W: Hãy CLICK 1 điểm bất kỳ trên thân vật thể để cắt ngang vuông góc 90°.",
            fg=ACCENT_CYAN
        )

    def import_new_pdf(self):
        from tkinter import filedialog
        import shutil
        import tools.auto_setup_from_pdf as auto_pdf

        pdf_path = filedialog.askopenfilename(
            title="Chọn file bản vẽ kỹ thuật PDF mới",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if not pdf_path:
            return

        sku = Path(pdf_path).stem
        try:
            # 1. Bóc tách động thông số từ PDF
            auto_pdf.process_all_pdfs(single_pdf=pdf_path)

            # 2. Cập nhật danh bạ sản phẩm
            self.products = load_products_dict()
            all_skus = list(self.products.keys())
            self.sku_combo["values"] = all_skus

            # 3. Kiểm tra ảnh chụp camera mẫu chuẩn (Golden Sample)
            ref_candidates = list((PROJECT_DIR / "assets" / "references" / sku / "images").glob("*.png"))
            if not ref_candidates:
                ref_candidates = list((PROJECT_DIR / "assets" / "tests" / sku).glob("**/*.png"))

            if not ref_candidates:
                ask_img = messagebox.askyesno(
                    "Cần ảnh mẫu chụp thực tế",
                    f"Đã trích xuất xong thông số CAD cho SKU '{sku}'.\n\nBạn có muốn nạp ảnh chụp camera mẫu chuẩn (Golden Sample) cho chi tiết này ngay bây giờ không?"
                )
                if ask_img:
                    img_path = filedialog.askopenfilename(
                        title=f"Chọn ảnh chụp mẫu camera cho {sku}",
                        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")]
                    )
                    if img_path:
                        dest_dir = PROJECT_DIR / "assets" / "references" / sku / "images"
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest_file = dest_dir / f"{sku}_1.png"
                        shutil.copy(img_path, dest_file)

                        # Hiệu chuẩn lại optical scale với ảnh thực tế
                        spec_file = PROJECT_DIR / "assets" / "references" / sku / "spec.json"
                        if spec_file.exists():
                            with open(spec_file, "r", encoding="utf-8") as f:
                                spec = json.load(f)
                            scale, g_info = auto_pdf.calibrate_optical_scale(sku, spec)
                            spec["optical_scale_mm_per_px"] = round(scale, 6)
                            if g_info:
                                spec["golden_sample"] = g_info
                            with open(spec_file, "w", encoding="utf-8") as f:
                                json.dump(spec, f, indent=2, ensure_ascii=False)
                            self.products[sku] = spec
                            with open(PROJECT_DIR / "configs" / "products.json", "w", encoding="utf-8") as f:
                                json.dump(self.products, f, indent=2, ensure_ascii=False)

            self.sku_var.set(sku)
            self.load_sku(sku)
            messagebox.showinfo(
                "Nạp Thành Công",
                f"Đã cập nhật hệ thống thành công!\n- SKU mới: {sku}\n- Bản vẽ CAD và các thông số kích thước đã sẵn sàng trên màn hình."
            )
        except Exception as e:
            messagebox.showerror("Lỗi khi nạp PDF", f"Không thể xử lý file PDF: {e}")

    def on_sku_changed(self, event=None):
        self.products = load_products_dict()
        sku = self.sku_var.get()
        if sku != self.current_sku:
            self.load_sku(sku)

    @staticmethod
    def _replace_entry(entry, value):
        entry.delete(0, tk.END)
        entry.insert(0, str(value))

    def _spec_for_label(self, label):
        key = str(label).strip().casefold()
        return next((d for d in self.spec_dims if str(d.get("label", "")).strip().casefold() == key), None)

    def _dimension_from_form(self):
        label = self.entry_label.get().strip()
        if not label:
            raise ValueError("Vui long nhap ten thong so.")
        try:
            nominal = float(self.entry_nom.get().strip())
            tol_minus = float(self.entry_tol_minus.get().strip() or "0")
            tol_plus = float(self.entry_tol_plus.get().strip() or "0")
        except ValueError as exc:
            raise ValueError("Gia tri chuan va sai so phai la so hop le.") from exc
        return build_dimension(
            label, nominal, tol_minus, tol_plus,
            description=self.entry_desc.get().strip(),
            diameter=label.lower().startswith(("d", "phi", "φ")),
        )

    def _save_spec(self):
        self.spec_data["sku"] = self.current_sku
        self.spec_data.setdefault("unit", "mm")
        self.spec_data["optical_scale_mm_per_px"] = self.optical_scale
        self.spec_data["dimensions"] = self.spec_dims
        self.spec_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.spec_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.spec_data, f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.spec_path)

    def _upsert_form_dimension(self):
        dimension = self._dimension_from_form()
        self.spec_dims = upsert_dimension(self.spec_dims, dimension)
        self.spec_data["dimensions"] = self.spec_dims
        return dimension

    def add_dimension_and_draw(self):
        """Register a spec dimension, then let the operator teach its edge."""
        try:
            dimension = self._upsert_form_dimension()
        except ValueError as exc:
            messagebox.showerror("Thong so khong hop le", str(exc))
            return
        self.pending_dimension = dimension
        self.enable_draw_mode(keep_pending=True)
        self.canvas_status_lbl.config(
            text=f"Dang them [{dimension['label']}]. Click 2 diem cat qua hai canh can do.",
            fg=ACCENT_CYAN,
        )

    def add_dimension_and_draw_inner(self):
        """Register a spec dimension and teach two opposing inner edges."""
        try:
            dimension = self._upsert_form_dimension()
        except ValueError as exc:
            messagebox.showerror("Thong so khong hop le", str(exc))
            return
        self.pending_dimension = dimension
        self.enable_inner_draw_mode(keep_pending=True)
        self.canvas_status_lbl.config(
            text=f"Dang them INNER [{dimension['label']}]. Click 2 mep doi dien cua lo/khe.",
            fg="#ce93d8",
        )

    def add_dimension_and_draw_template(self):
        """Register a dimension measured as signed deviation from the golden contour."""
        try:
            dimension = self._upsert_form_dimension()
            lower, upper = self._profile_offsets()
            self._golden_actual_from_form()
        except ValueError as exc:
            messagebox.showerror("Thong so khong hop le", str(exc))
            return
        self.pending_dimension = dimension
        self.enable_template_draw_mode(keep_pending=True)
        self.canvas_status_lbl.config(
            text=(f"TEMPLATE [{dimension['label']}]: click 2 moc tren contour golden. "
                  f"Inward {lower:g} / Outward {upper:g} mm."),
            fg="#fbbf24",
        )

    def _profile_offsets(self):
        return profile_limits({
            "contour_lower_mm": self.entry_profile_lower.get().strip() or "0",
            "contour_upper_mm": self.entry_profile_upper.get().strip() or "0",
        })

    def _golden_actual_from_form(self):
        text = self.entry_golden_actual.get().strip()
        if not text:
            return None
        value = float(text)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Golden actual must be greater than zero, or leave blank for image")
        return value

    def refresh_table(self):
        if hasattr(self, "profile_state_label"):
            enabled = self.profile_enabled_var.get()
            self.profile_state_label.config(
                text="Profile QC enabled: whole outline (SCAN / TPL)" if enabled else
                     "Profile QC disabled: preview only",
                fg=ACCENT_GREEN if enabled else ACCENT_YELLOW,
            )
        for item in self.tree.get_children():
            self.tree.delete(item)

        for idx, cal in enumerate(self.calipers):
            lbl = cal.get("label", "DIM")
            nom = f"{cal.get('nominal_mm', 0.0):.2f} mm"
            meas_v = f"{cal.get('measured_mm', 0.0):.2f} mm"
            spec_dim = self._spec_for_label(lbl)
            if spec_dim:
                lo = float(spec_dim.get("min", spec_dim.get("nominal", 0.0)))
                hi = float(spec_dim.get("max", spec_dim.get("nominal", 0.0)))
                measured = float(cal.get("measured_mm", 0.0))
                status = "PASS" if lo <= measured <= hi else "FAIL"
            else:
                status = "--"
            mode = "TPL" if cal.get("measure_type") == "template_contour" else "SCAN"
            values = (lbl, nom, meas_v, mode, status)
            self.tree.insert("", tk.END, iid=str(idx), values=values)

        if 0 <= self.selected_idx < len(self.calipers):
            self.tree.selection_set(str(self.selected_idx))

    def on_tree_select(self, event=None):
        selected = self.tree.selection()
        if selected:
            self.selected_idx = int(selected[0])
            cal = self.calipers[self.selected_idx]
            self.entry_label.delete(0, tk.END)
            self.entry_label.insert(0, cal.get("label", ""))
            self.entry_nom.delete(0, tk.END)
            self.entry_nom.insert(0, str(cal.get("nominal_mm", "")))
            self.entry_desc.delete(0, tk.END)
            self.entry_desc.insert(0, cal.get("desc", ""))
            actual = cal.get("golden_actual_mm")
            self._replace_entry(self.entry_golden_actual, "" if actual is None else f"{actual:g}")
            spec_dim = self._spec_for_label(cal.get("label", "")) or cal
            tol_minus, tol_plus = tolerance_magnitudes(spec_dim)
            self._replace_entry(self.entry_tol_minus, f"{tol_minus:g}")
            self._replace_entry(self.entry_tol_plus, f"{tol_plus:g}")

            q1 = cal.get("p1_edge", cal.get("p1_ref", (0, 0)))
            q2 = cal.get("p2_edge", cal.get("p2_ref", (0, 0)))
            self.info_snapped_lbl.config(text=f"Mép cắt: Q1=({q1[0]:.1f}, {q1[1]:.1f}) -> Q2=({q2[0]:.1f}, {q2[1]:.1f})")
            self.redraw_canvas()

    def on_cad_spec_selected(self, event=None):
        selected_text = self.cad_combo_var.get()
        if not selected_text:
            return
        lbl = selected_text.split(":")[0].strip()
        for d in self.spec_dims:
            if d.get("label") == lbl:
                self.entry_label.delete(0, tk.END)
                self.entry_label.insert(0, lbl)
                self.entry_nom.delete(0, tk.END)
                self.entry_nom.insert(0, str(d.get("nominal", "")))
                self.entry_desc.delete(0, tk.END)
                self.entry_desc.insert(0, d.get("raw_spec", ""))
                tol_minus, tol_plus = tolerance_magnitudes(d)
                self._replace_entry(self.entry_tol_minus, f"{tol_minus:g}")
                self._replace_entry(self.entry_tol_plus, f"{tol_plus:g}")
                break

    def update_current_caliper(self):
        if self.selected_idx < 0 or self.selected_idx >= len(self.calipers):
            messagebox.showinfo("Thông báo", "Vui lòng chọn một kích thước từ danh sách để cập nhật.")
            return

        lbl = self.entry_label.get().strip().upper()
        if not lbl:
            messagebox.showerror("Lỗi", "Tên nhãn không được để trống!")
            return

        try:
            nom = float(self.entry_nom.get().strip())
        except ValueError:
            messagebox.showerror("Lỗi", "Giá trị tiêu chuẩn (Nominal) phải là số hợp lệ!")
            return

        desc = self.entry_desc.get().strip()

        cal = self.calipers[self.selected_idx]
        cal["label"] = lbl
        cal["nominal_mm"] = nom
        cal["desc"] = desc

        self.info_status_lbl.config(text="Trạng thái: Đã cập nhật (nhấn Lưu Công Thức để áp dụng)", fg=ACCENT_YELLOW)
        self.refresh_table()
        self.redraw_canvas()

    # This second definition intentionally supersedes the legacy nominal-only
    # editor above and keeps spec.json and the scanline recipe synchronized.
    def update_current_caliper(self):
        if self.selected_idx < 0 or self.selected_idx >= len(self.calipers):
            messagebox.showinfo("Chua chon", "Hay chon mot thong so trong danh sach.")
            return
        try:
            dimension = self._upsert_form_dimension()
            actual = self._golden_actual_from_form()
        except ValueError as exc:
            messagebox.showerror("Thong so khong hop le", str(exc))
            return

        self.calipers[self.selected_idx].update({
            "label": dimension["label"],
            "nominal_mm": dimension["nominal"],
            "desc": self.entry_desc.get().strip(),
            "min_mm": dimension["min"],
            "max_mm": dimension["max"],
            "tolerance_minus_mm": abs(dimension["lower_tol"]),
            "tolerance_plus_mm": dimension["upper_tol"],
            "golden_actual_mm": actual,
        })
        cal = self.calipers[self.selected_idx]
        if cal.get("measure_type") == "template_contour":
            p1, p2 = cal["p1_edge"], cal["p2_edge"]
            px = float(np.linalg.norm(np.asarray(p2) - p1))
            cal["golden_length_mm"], cal["golden_source"] = golden_baseline(cal, px, self.optical_scale)
            cal["measured_mm"] = cal["golden_length_mm"]
        self.info_status_lbl.config(text="Da cap nhat. Nhan Luu de ghi spec va recipe.", fg=ACCENT_YELLOW)
        self.refresh_table()
        self.redraw_canvas()

    def delete_selected_caliper(self):
        if self.selected_idx < 0 or self.selected_idx >= len(self.calipers):
            messagebox.showinfo("Thông báo", "Vui lòng chọn kích thước cần xóa từ bảng bên trên.")
            return

        removed = self.calipers.pop(self.selected_idx)
        self.selected_idx = -1
        self.refresh_table()
        self.redraw_canvas()
        self.info_status_lbl.config(
            text=f"Đã xóa kích thước [{removed['label']}]. Nhấn Lưu để cập nhật!", fg=ACCENT_YELLOW
        )

    def snap_outer_endpoints(self, p1, p2):
        """Snaps p1 and p2 to physical outer edges.
        If ortho lock is enabled, forces orientation along primary axes.
        Otherwise, if clicked near edges, snaps directly to the nearest edge points at any arbitrary angle.
        If drawn across the body, intersects the scanline ray with the contour.
        """
        if getattr(self, "ortho_var", None) and self.ortho_var.get():
            u1, v1 = to_wcs(p1, self.wcs)
            u2, v2 = to_wcs(p2, self.wcs)
            if abs(u2 - u1) >= abs(v2 - v1):
                v2 = v1
                axis_type = "length"
            else:
                u2 = u1
                axis_type = "width"
            p1_adj = from_wcs(u1, v1, self.wcs)
            p2_adj = from_wcs(u2, v2, self.wcs)
            q1, q2 = self.measurer_inst.snap_scanline_to_edges(self.gray, self.cnt, p1_adj, p2_adj)
            return q1, q2, axis_type

        # Free arbitrary angle: snap each point to the closest physical contour edge
        q1_near, d1 = snap_point_to_contour(p1, self.cnt)
        q2_near, d2 = snap_point_to_contour(p2, self.cnt)
        if d1 <= 60.0 and d2 <= 60.0:
            return q1_near, q2_near, "caliper"
        else:
            q1, q2 = self.measurer_inst.snap_scanline_to_edges(self.gray, self.cnt, p1, p2)
            return q1, q2, "caliper"

    def enable_draw_mode(self, keep_pending=False):
        if not keep_pending:
            self.pending_dimension = None
        self.active_mode = "draw"
        self.drawing_points = []
        self.canvas_status_lbl.config(
            text="👉 CHẾ ĐỘ CALIPER TỰ DO: Click chuột trái điểm 1 (P1) vào bất kỳ vị trí nào trên mép, sau đó click điểm 2 (P2)...",
            fg=ACCENT_CYAN
        )

    def enable_inner_draw_mode(self, keep_pending=False):
        """Teach a diameter/width/length across an internal hole or slot."""
        if not keep_pending:
            self.pending_dimension = None
        self.active_mode = "draw_inner"
        self.drawing_points = []
        self.canvas_status_lbl.config(
            text="INNER CONTOUR: click gan mep trong thu nhat, sau do click mep trong doi dien.",
            fg="#ce93d8",
        )

    def enable_template_draw_mode(self, keep_pending=False):
        """Teach a scanline whose value is derived from golden-edge deviations."""
        if not keep_pending:
            self.pending_dimension = None
        self.active_mode = "draw_template"
        self.drawing_points = []
        self.canvas_status_lbl.config(
            text="TEMPLATE CONTOUR: click P1 and P2 on opposite golden edges.",
            fg="#fbbf24",
        )

    def _viewport_sizes(self):
        return (self.ref_img.shape[1], self.ref_img.shape[0]), (
            max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        )

    def on_canvas_zoom(self, event):
        if not hasattr(self, "ref_img") or self.ref_img is None or not event.delta:
            return "break"
        image_size, canvas_size = self._viewport_sizes()
        steps = max(1, abs(event.delta) / 120)
        factor = 1.25 ** (steps if event.delta > 0 else -steps)
        self.viewport.zoom_at(factor, event.x, event.y, image_size, canvas_size)
        self.redraw_canvas()
        return "break"

    def on_canvas_press(self, event):
        self._mouse_press = (event.x, event.y, self.viewport.pan_x, self.viewport.pan_y)
        self._dragging_image = False

    def on_canvas_drag(self, event):
        if self._mouse_press is None:
            return
        x, y, pan_x, pan_y = self._mouse_press
        dx, dy = event.x - x, event.y - y
        if np.hypot(dx, dy) > 4:
            self._dragging_image = True
        if self._dragging_image and self.viewport.zoom > 1:
            self.viewport.pan_x, self.viewport.pan_y = pan_x + dx, pan_y + dy
            self.canvas.configure(cursor="fleur")
            self.redraw_canvas()

    def on_canvas_release(self, event):
        click = self._mouse_press is not None and not self._dragging_image
        self._mouse_press = None
        self._dragging_image = False
        self.canvas.configure(cursor="crosshair")
        if click:
            self.on_canvas_click(event)

    def on_canvas_click(self, event):
        # Convert canvas coords to image coords
        if not hasattr(self, "img_scale") or self.img_scale <= 0:
            return

        ix = (event.x - self.img_offset_x) / self.img_scale
        iy = (event.y - self.img_offset_y) / self.img_scale
        if not (0 <= ix < self.ref_img.shape[1] and 0 <= iy < self.ref_img.shape[0]):
            return

        if self.active_mode == "slice_width":
            u_pt, _ = to_wcs((ix, iy), self.wcs)
            # Create transverse scanline perpendicular to axis (90 degrees):
            p1_raw = from_wcs(u_pt, -0.5, self.wcs)
            p2_raw = from_wcs(u_pt, 0.5, self.wcs)
            q1_edge, q2_edge = self.measurer_inst.snap_scanline_to_edges(
                self.gray, self.cnt, p1_raw, p2_raw
            )
            d_px = float(np.hypot(q2_edge[0] - q1_edge[0], q2_edge[1] - q1_edge[1]))
            d_mm = round(d_px * self.optical_scale, 2)
            u1, v1 = to_wcs(q1_edge, self.wcs)
            u2, v2 = to_wcs(q2_edge, self.wcs)
            u1 = round(u_pt, 4)
            u2 = round(u_pt, 4)

            label = f"W_{len(self.calipers)+1}" if any(c["label"] == "W" for c in self.calipers) else "W"
            if self.entry_label.get().strip():
                label = self.entry_label.get().strip()

            nom = d_mm
            try:
                if self.entry_nom.get().strip():
                    nom = float(self.entry_nom.get().strip())
            except ValueError:
                pass

            tol_minus = 0.10
            tol_plus = 0.10
            try:
                tol_minus = float(self.entry_tol_minus.get().strip() or "0.10")
                tol_plus = float(self.entry_tol_plus.get().strip() or "0.10")
            except ValueError:
                pass

            new_cal = {
                "label": label,
                "desc": f"Bề rộng cắt ngang {label} (u={u_pt:.2f})",
                "nominal_mm": round(nom, 2),
                "axis": "2D_in_plane",
                "p1_wcs": [u1, round(v1, 4)],
                "p2_wcs": [u2, round(v2, 4)],
                "p1_ref": [round(q1_edge[0], 2), round(q1_edge[1], 2)],
                "p2_ref": [round(q2_edge[0], 2), round(q2_edge[1], 2)],
                "p1_edge": q1_edge,
                "p2_edge": q2_edge,
                "measured_mm": d_mm,
                "length_px": round(d_px, 2),
                "measure_type": "width",
                "tolerance_minus_mm": tol_minus,
                "tolerance_plus_mm": tol_plus,
                "color": PALETTE[len(self.calipers) % len(PALETTE)]
            }
            self.calipers.append(new_cal)
            self.selected_idx = len(self.calipers) - 1
            self.active_mode = "select"
            self.refresh_table()
            self.on_tree_select()
            self.redraw_canvas()
            self.canvas_status_lbl.config(
                text=f"✔ Đã tạo mặt cắt ngang [{label}] vuông góc 90°: Đo được {d_mm:.2f} mm.",
                fg=ACCENT_GREEN
            )
            return

        if self.active_mode in ("draw", "draw_inner", "draw_template"):
            self.drawing_points.append((ix, iy))
            if len(self.drawing_points) == 1:
                self.canvas_status_lbl.config(
                    text=f"Đã chọn P1=({ix:.1f}, {iy:.1f}). Bây giờ hãy Click điểm P2 qua cạnh đối diện...",
                    fg=ACCENT_YELLOW
                )
                self.redraw_canvas()
            elif len(self.drawing_points) >= 2:
                p1_raw = self.drawing_points[0]
                p2_raw = self.drawing_points[1]

                is_inner = self.active_mode == "draw_inner"
                is_template = self.active_mode == "draw_template"
                try:
                    actual = self._golden_actual_from_form() if is_template else None
                except ValueError as exc:
                    self.drawing_points = []
                    messagebox.showerror("Golden actual", str(exc))
                    return
                if is_inner:
                    q1_edge, q2_edge, selected_inner_contour = self.measurer_inst.snap_scanline_to_inner_contour(
                        self.gray, self.cnt, p1_raw, p2_raw, image_bgr=self.ref_img
                    )
                    axis_type = "inner"
                else:
                    selected_inner_contour = None
                    q1_edge, q2_edge, axis_type = self.snap_outer_endpoints(p1_raw, p2_raw)

                d_px = float(np.hypot(q2_edge[0] - q1_edge[0], q2_edge[1] - q1_edge[1]))
                d_mm = round(d_px * self.optical_scale, 2)

                u1, v1 = to_wcs(q1_edge, self.wcs)
                u2, v2 = to_wcs(q2_edge, self.wcs)
                if axis_type == "length":
                    v1 = (v1 + v2) / 2.0
                    v2 = v1
                elif axis_type == "width":
                    u1 = (u1 + u2) / 2.0
                    u2 = u1

                # Determine suggested label from CAD specs
                existing_labels = set(c["label"] for c in self.calipers)
                avail_labels = [d["label"] for d in self.spec_dims if d.get("axis") != "Z_height_side_view" and d["label"] not in existing_labels]
                default_lbl = avail_labels[0] if avail_labels else f"DIM_{len(self.calipers)+1}"

                # Match nominal
                nom_val = d_mm
                desc_val = f"Custom caliper {default_lbl}"
                for d in self.spec_dims:
                    if d["label"] == default_lbl:
                        nom_val = d.get("nominal", d_mm)
                        desc_val = d.get("raw_spec", desc_val)
                        break

                new_cal = {
                    "label": default_lbl,
                    "desc": desc_val,
                    "nominal_mm": round(nom_val, 2),
                    "axis": "2D_in_plane",
                    "p1_wcs": [round(u1, 4), round(v1, 4)],
                    "p2_wcs": [round(u2, 4), round(v2, 4)],
                    "p1_ref": [round(q1_edge[0], 2), round(q1_edge[1], 2)],
                    "p2_ref": [round(q2_edge[0], 2), round(q2_edge[1], 2)],
                    "p1_edge": q1_edge,
                    "p2_edge": q2_edge,
                    "measured_mm": d_mm,
                    "length_px": round(d_px, 2),
                    "measure_type": (
                        "inner_contour" if is_inner else
                        "template_contour" if is_template else
                        "taught_caliper"
                    ),
                    "golden_length_mm": d_mm if is_template else None,
                    "golden_actual_mm": actual,
                    "golden_length_px": round(d_px, 4) if is_template else None,
                    "inner_contour": selected_inner_contour,
                    "color": [210, 80, 230] if is_inner else PALETTE[len(self.calipers) % len(PALETTE)]
                }

                if is_template:
                    new_cal["golden_length_mm"], new_cal["golden_source"] = golden_baseline(
                        new_cal, d_px, self.optical_scale
                    )
                    new_cal["measured_mm"] = new_cal["golden_length_mm"]

                # A dimension entered in the compact spec form takes priority
                # over the next automatic CAD suggestion.
                if self.pending_dimension is not None:
                    dim = self.pending_dimension
                    new_cal.update({
                        "label": dim["label"],
                        "desc": dim["raw_spec"],
                        "nominal_mm": dim["nominal"],
                        "min_mm": dim["min"],
                        "max_mm": dim["max"],
                        "tolerance_minus_mm": abs(dim["lower_tol"]),
                        "tolerance_plus_mm": dim["upper_tol"],
                    })
                    default_lbl = dim["label"]

                self.calipers.append(new_cal)
                self.pending_dimension = None
                self.selected_idx = len(self.calipers) - 1
                self.drawing_points = []
                self.active_mode = "select"

                self.canvas_status_lbl.config(
                    text=f"✔ Đã tạo tia đo [{default_lbl}] bắt đúng mép cạnh! Đo được: {d_mm:.2f} mm. Nhấn Lưu để cập nhật.",
                    fg=ACCENT_GREEN
                )
                self.refresh_table()
                self.on_tree_select()
                self.redraw_canvas()
        else:
            # Mode "select": find closest caliper to click
            best_dist = 25.0 / max(self.img_scale, 1e-9)
            best_idx = -1
            for idx, cal in enumerate(self.calipers):
                p1 = cal.get("p1_edge", cal.get("p1_ref", (0, 0)))
                p2 = cal.get("p2_edge", cal.get("p2_ref", (0, 0)))
                # Distance from point to line segment
                dist = self.point_to_segment_dist((ix, iy), p1, p2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx >= 0:
                self.selected_idx = best_idx
                self.tree.selection_set(str(self.selected_idx))
                self.on_tree_select()
                self.redraw_canvas()

    def on_canvas_motion(self, event):
        if self._mouse_press is not None:
            return
        if not hasattr(self, "img_scale") or self.img_scale <= 0:
            return

        ix = (event.x - self.img_offset_x) / self.img_scale
        iy = (event.y - self.img_offset_y) / self.img_scale
        self.temp_mouse_pos = (ix, iy)

        if self.active_mode == "slice_width":
            u_pt, _ = to_wcs((ix, iy), self.wcs)
            p1 = from_wcs(u_pt, -0.5, self.wcs)
            p2 = from_wcs(u_pt, 0.5, self.wcs)
            q1, q2 = self.measurer_inst.snap_scanline_to_edges(self.gray, self.cnt, p1, p2)
            dist_px = np.hypot(q2[0] - q1[0], q2[1] - q1[1])
            dist_mm = dist_px * self.optical_scale
            self.canvas_status_lbl.config(
                text=f"MẶT CẮT NGANG (u={u_pt:.2f}): Bề rộng = {dist_mm:.2f} mm ({dist_px:.1f} px). CLICK để chốt tia đo này!",
                fg=ACCENT_CYAN
            )
            self.redraw_canvas()
            return

        if self.active_mode in ("draw", "draw_inner", "draw_template") and len(self.drawing_points) == 1:
            p1 = self.drawing_points[0]
            p2 = (ix, iy)
            if self.active_mode == "draw_inner":
                q1, q2, _ = self.measurer_inst.snap_scanline_to_inner_contour(
                    self.gray, self.cnt, p1, p2, image_bgr=self.ref_img
                )
            else:
                q1, q2, _ = self.snap_outer_endpoints(p1, p2)
            dist_px = np.hypot(q2[0] - q1[0], q2[1] - q1[1])
            dist_mm = dist_px * self.optical_scale
            self.canvas_status_lbl.config(
                text=f"Bắt mép cắt thực: Q1=({q1[0]:.1f}, {q1[1]:.1f}) -> Q2=({q2[0]:.1f}, {q2[1]:.1f}) | Khoảng cách: {dist_px:.1f}px ({dist_mm:.2f}mm)",
                fg=ACCENT_GREEN
            )
            self.redraw_canvas()

    def on_canvas_right_click(self, event):
        # Cancel drawing
        if self.active_mode in ("draw", "draw_inner", "draw_template", "slice_width"):
            self.drawing_points = []
            self.active_mode = "select"
            self.pending_dimension = None
            self.canvas_status_lbl.config(text="Đã hủy chế độ vẽ.", fg=TEXT_MUTED)
            self.redraw_canvas()

    @staticmethod
    def point_to_segment_dist(pt, p1, p2):
        p = np.array(pt)
        a = np.array(p1)
        b = np.array(p2)
        ab = b - a
        d_ab = np.linalg.norm(ab)
        if d_ab < 1e-6:
            return float(np.linalg.norm(p - a))
        t = max(0.0, min(1.0, float(np.dot(p - a, ab) / (d_ab ** 2))))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def redraw_canvas(self):
        if not hasattr(self, "ref_img") or self.ref_img is None:
            return

        c_w = self.canvas.winfo_width()
        c_h = self.canvas.winfo_height()
        if c_w <= 10 or c_h <= 10:
            c_w, c_h = 800, 700

        vis = self.ref_img.copy()

        # Preview uses the same offset geometry/colors as inspection output.
        try:
            lower, upper = self._profile_offsets()
            key = (id(self.cnt), lower, upper, self.optical_scale)
            if self._profile_preview_cache is None or self._profile_preview_cache[0] != key:
                inner, outer = offset_contours(
                    self.cnt, lower / self.optical_scale, upper / self.optical_scale
                )
                profile = {
                    "aligned_golden_contour": self.cnt,
                    "inner_contours": inner, "outer_contours": outer,
                    "lower_px": lower / self.optical_scale,
                    "upper_px": upper / self.optical_scale,
                }
                self._profile_preview_cache = (key, profile)
            draw_profile(vis, self._profile_preview_cache[1])
        except (ValueError, ZeroDivisionError):
            cv2.drawContours(vis, [self.cnt], -1, (160, 160, 160), 1, cv2.LINE_AA)

        # Closed inner edges detected strictly inside the workpiece. They are
        # shown independently from caliper lines so the operator can verify
        # the actual hole/slot contour before teaching a measurement.
        if self.active_mode == "draw_inner":
            for inner_idx, inner_contour in enumerate(getattr(self, "detected_inner_contours", []), start=1):
                cv2.drawContours(vis, [inner_contour], -1, (255, 210, 0), 1, cv2.LINE_AA)
                moments = cv2.moments(inner_contour)
                if abs(moments["m00"]) > 1e-6:
                    cx = int(round(moments["m10"] / moments["m00"]))
                    cy = int(round(moments["m01"] / moments["m00"]))
                    cv2.putText(vis, f"I{inner_idx}", (cx + 5, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 210, 0), 1, cv2.LINE_AA)

        # 2. Draw all calipers snapped to physical edge intersections
        for idx, cal in enumerate(self.calipers):
            is_sel = (idx == self.selected_idx)
            q1 = cal.get("p1_edge", cal.get("p1_ref", (0, 0)))
            q2 = cal.get("p2_edge", cal.get("p2_ref", (0, 0)))

            p1_i = (int(round(q1[0])), int(round(q1[1])))
            p2_i = (int(round(q2[0])), int(round(q2[1])))
            lbl = cal.get("label", "DIM")
            nom = cal.get("nominal_mm", 0.0)
            meas_v = cal.get("measured_mm", 0.0)

            col = (0, 255, 255) if is_sel else tuple(cal.get("color", PALETTE[idx % len(PALETTE)]))
            thick = 3 if is_sel else 2

            inner_contour = cal.get("inner_contour")
            if inner_contour is not None:
                cv2.drawContours(vis, [np.asarray(inner_contour, dtype=np.int32)], -1,
                                 col, thick + 1, cv2.LINE_AA)

            # Caliper line
            cv2.line(vis, p1_i, p2_i, col, thick, cv2.LINE_AA)
            cv2.circle(vis, p1_i, 5 if is_sel else 4, (0, 0, 255) if is_sel else col, -1, cv2.LINE_AA)
            cv2.circle(vis, p2_i, 5 if is_sel else 4, (0, 0, 255) if is_sel else col, -1, cv2.LINE_AA)

            # Ticks at real physical edge intersections
            dx = q2[0] - q1[0]
            dy = q2[1] - q1[1]
            dist = np.hypot(dx, dy)
            if dist > 1.0:
                nx = int(round(-dy / dist * (10 if is_sel else 8)))
                ny = int(round(dx / dist * (10 if is_sel else 8)))
                cv2.line(vis, (p1_i[0] + nx, p1_i[1] + ny), (p1_i[0] - nx, p1_i[1] - ny), col, thick, cv2.LINE_AA)
                cv2.line(vis, (p2_i[0] + nx, p2_i[1] + ny), (p2_i[0] - nx, p2_i[1] - ny), col, thick, cv2.LINE_AA)

            # Label tag
            mx = (p1_i[0] + p2_i[0]) // 2
            my = (p1_i[1] + p2_i[1]) // 2
            tag = f"[{idx+1}] {lbl}: {meas_v:.2f}mm"
            spec_dim = self._spec_for_label(lbl)
            if spec_dim:
                lo = float(spec_dim.get("min", nom))
                hi = float(spec_dim.get("max", nom))
                status = "PASS" if lo <= float(meas_v) <= hi else "FAIL"
                tag = f"[{idx+1}] {lbl} {meas_v:.2f}mm | {nom:.2f} [{lo:.2f}..{hi:.2f}] {status}"
            cv2.putText(vis, tag, (mx + 8, my - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)

        # 3. Draw active drawing line preview (snapped to physical edge)
        if self.active_mode == "slice_width" and self.temp_mouse_pos:
            u_pt, _ = to_wcs(self.temp_mouse_pos, self.wcs)
            p1 = from_wcs(u_pt, -0.5, self.wcs)
            p2 = from_wcs(u_pt, 0.5, self.wcs)
            q1, q2 = self.measurer_inst.snap_scanline_to_edges(self.gray, self.cnt, p1, p2)
            q1_i = (int(round(q1[0])), int(round(q1[1])))
            q2_i = (int(round(q2[0])), int(round(q2[1])))
            cv2.line(vis, q1_i, q2_i, (0, 230, 255), 2, cv2.LINE_AA)
            cv2.circle(vis, q1_i, 6, (0, 230, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, q2_i, 6, (0, 230, 255), -1, cv2.LINE_AA)

        elif self.active_mode in ("draw", "draw_inner", "draw_template") and len(self.drawing_points) == 1 and self.temp_mouse_pos:
            p1 = self.drawing_points[0]
            p2 = self.temp_mouse_pos
            if self.active_mode == "draw_inner":
                q1, q2, preview_inner = self.measurer_inst.snap_scanline_to_inner_contour(
                    self.gray, self.cnt, p1, p2, image_bgr=self.ref_img
                )
                if preview_inner is not None:
                    cv2.drawContours(vis, [preview_inner], -1, (210, 80, 230), 3, cv2.LINE_AA)
            else:
                q1, q2, _ = self.snap_outer_endpoints(p1, p2)
            cv2.line(vis, (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1]))), (100, 100, 100), 1, cv2.LINE_AA)
            # Snapped edge points
            q1_i = (int(round(q1[0])), int(round(q1[1])))
            q2_i = (int(round(q2[0])), int(round(q2[1])))
            cv2.line(vis, q1_i, q2_i, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(vis, q1_i, 6, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(vis, q2_i, 6, (0, 255, 0), -1, cv2.LINE_AA)

        # Render only canvas-sized pixels, even at 32x, to avoid allocating a
        # gigantic resized image. Measurements always use source coordinates.
        img_h, img_w = vis.shape[:2]
        scale, offset_x, offset_y = self.viewport.transform((img_w, img_h), (c_w, c_h))
        self.img_scale = scale
        self.img_offset_x, self.img_offset_y = offset_x, offset_y
        resized = cv2.warpAffine(
            vis, np.array([[scale, 0, offset_x], [0, scale, offset_y]], dtype=np.float64),
            (c_w, c_h), flags=cv2.INTER_LINEAR, borderValue=(8, 11, 15),
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.pil_img = Image.fromarray(rgb)
        self.tk_img = ImageTk.PhotoImage(self.pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.zoom_label.config(text=(
            f"{self.viewport.zoom * 100:.0f}% | Ctrl + wheel: zoom | Left drag: pan"
        ))

    def _recipe_snapshot(self):
        """Build one complete in-memory recipe shared by Save and Test."""
        lower, upper = self._profile_offsets()
        if not np.isfinite(self.optical_scale) or self.optical_scale <= 0:
            raise ValueError("mm/pixel calibration must be greater than zero")
        calipers = []
        labels = [str(c.get("label", "")).strip().casefold() for c in self.calipers]
        if len(labels) != len(set(labels)):
            raise ValueError("Dimension IDs must be unique. Delete the duplicate or change its ID.")
        for original in self.calipers:
            cal = dict(original)
            if cal.get("measure_type") == "template_contour":
                p1 = cal.get("p1_edge", cal.get("p1_ref"))
                p2 = cal.get("p2_edge", cal.get("p2_ref"))
                if p1 is None or p2 is None:
                    raise ValueError("Template scanline is missing its golden landmarks")
                px = float(np.linalg.norm(np.asarray(p2) - p1))
                baseline, source = golden_baseline(cal, px, self.optical_scale)
                cal.update({
                    "p1_ref": list(p1), "p2_ref": list(p2),
                    "p1_wcs": list(to_wcs(p1, self.wcs)),
                    "p2_wcs": list(to_wcs(p2, self.wcs)),
                    "golden_length_px": px, "golden_length_mm": baseline,
                    "golden_source": source,
                })
            calipers.append(cal)
        # Keep every vertex so narrow defects in the golden are not simplified away.
        local = [self.measurer_inst._point_to_frame_local(p, self.wcs).tolist()
                 for p in self.cnt.reshape(-1, 2)]
        return {
            "sku": self.current_sku,
            "reference_image": str(self.ref_img_path.relative_to(PROJECT_DIR)),
            "optical_scale_mm_per_px": self.optical_scale,
            "wcs": self.wcs, "golden_contour_local_px": local,
            "contour_lower_mm": lower, "contour_upper_mm": upper,
            "contour_profile_enabled": bool(self.profile_enabled_var.get()),
            "calipers": calipers,
        }

    @staticmethod
    def _json_value(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    def save_recipe(self):
        sku = self.current_sku
        try:
            recipe = self._recipe_snapshot()
        except ValueError as exc:
            messagebox.showerror(
                "Thong so khong hop le", str(exc),
            )
            return

        # Build dimensions list for spec.json from active calipers
        new_dims = []
        for cal in self.calipers:
            lbl = cal.get("label", "DIM")
            nom = float(cal.get("nominal_mm", cal.get("measured_mm", 0.0)))
            tol_minus = abs(float(cal.get("tolerance_minus_mm", 0.10)))
            tol_plus = abs(float(cal.get("tolerance_plus_mm", 0.10)))
            lo = nom - tol_minus
            hi = nom + tol_plus
            desc = cal.get("desc", f"Kích thước {lbl}")
            new_dims.append({
                "label": lbl,
                "nominal": nom,
                "min": lo,
                "max": hi,
                "lower_tol": -tol_minus,
                "upper_tol": tol_plus,
                "axis": cal.get("axis", "2D_in_plane"),
                "raw_spec": desc
            })
        self.spec_dims = new_dims
        self.spec_data["contour_profile"] = {
            "enabled": recipe["contour_profile_enabled"],
            "lower_mm": recipe["contour_lower_mm"],
            "upper_mm": recipe["contour_upper_mm"],
            "rule": "signed_offset_from_golden",
        }
        self._save_spec()

        self.recipes[sku] = recipe
        with open(RECIPE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.recipes, f, indent=2, ensure_ascii=False, default=self._json_value)

        # Update products.json as well
        self.products[sku] = self.spec_data
        with open(PROJECT_DIR / "configs" / "products.json", "w", encoding="utf-8") as f:
            json.dump(self.products, f, indent=2, ensure_ascii=False)

        self.info_status_lbl.config(
            text=f"✔ ĐÃ LƯU SPEC ({len(self.calipers)} KÍCH THƯỚC) VÀO spec.json & configs/caliper_recipes.json!",
            fg=ACCENT_GREEN
        )
        messagebox.showinfo(
            "Đã Lưu Thành Công",
            f"Đã lưu thành công bộ thông số Spec & công thức đo ({len(self.calipers)} kích thước) cho SKU '{sku}'!\n\n"
            f"- File Spec CAD: assets/references/{sku}/spec.json\n"
            f"- File Caliper Recipe: configs/caliper_recipes.json\n\n"
            f"Chỉ cần làm 1 lần! Kể từ bây giờ mọi ảnh kiểm tra của '{sku}' sẽ được tự động đo và so sánh theo Spec này."
        )

    def reset_proposals(self):
        if messagebox.askyesno("Xác nhận", f"Khôi phục toàn bộ các kích thước đề xuất ban đầu từ bản vẽ CAD cho SKU {self.current_sku}?"):
            props = get_proposed_calipers_for_sku(self.current_sku, self.wcs, self.optical_scale)
            for cal in props:
                p1_ref = cal.get("p1_ref")
                p2_ref = cal.get("p2_ref")
                if p1_ref and p2_ref:
                    q1, q2 = self.measurer_inst.snap_scanline_to_edges(self.gray, self.cnt, p1_ref, p2_ref)
                    cal["p1_edge"] = q1
                    cal["p2_edge"] = q2
                    d_px = float(np.hypot(q2[0] - q1[0], q2[1] - q1[1]))
                    cal["measured_mm"] = round(d_px * self.optical_scale, 2)
            self.calipers = props
            self.selected_idx = -1
            self.refresh_table()
            self.redraw_canvas()
            self.info_status_lbl.config(text="Đã nạp lại đề xuất CAD ban đầu. Nhấn Lưu để áp dụng.", fg=ACCENT_YELLOW)

    def clear_all_calipers(self):
        if messagebox.askyesno("Xác nhận", f"Bạn có chắc chắn muốn xóa toàn bộ calipers của SKU {self.current_sku} để tự vẽ lại từ đầu?"):
            self.calipers.clear()
            self.selected_idx = -1
            self.refresh_table()
            self.redraw_canvas()
            self.info_status_lbl.config(text="Đã xóa trắng. Nhấn 'Thêm Tia Đo' để vẽ mới.", fg=ACCENT_YELLOW)

    def run_test_sample(self):
        from tkinter import filedialog
        # Ask operator to pick an image or default
        init_dir = PROJECT_DIR / "assets" / "tests" / self.current_sku
        test_path_str = filedialog.askopenfilename(
            title=f"Chọn ảnh cần kiểm tra đo lường cho SKU {self.current_sku} (hoặc Cancel để dùng mẫu test mặc định)",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")],
            initialdir=str(init_dir) if init_dir.exists() else str(PROJECT_DIR)
        )
        if test_path_str:
            test_img_path = Path(test_path_str)
        else:
            test_candidates = list((PROJECT_DIR / "assets" / "tests" / self.current_sku).rglob("*.png"))
            if not test_candidates:
                test_candidates = [self.ref_img_path]
            test_img_path = test_candidates[0]

        test_img = cv2.imread(str(test_img_path))
        if test_img is None:
            messagebox.showerror("Lỗi", "Không thể đọc ảnh test!")
            return

        desc = clf.describe_image(test_img)
        if desc is None:
            messagebox.showerror("Lỗi", "Không tìm thấy vật thể trong ảnh test!")
            return

        cnt = desc["contour"]
        spec = dict(self.spec_data)
        spec["optical_scale_mm_per_px"] = self.optical_scale
        spec["dimensions"] = [build_dimension(
            c["label"], c.get("nominal_mm", 0),
            c.get("tolerance_minus_mm", .1), c.get("tolerance_plus_mm", .1),
            axis=c.get("axis", "2D_in_plane"),
        ) for c in self.calipers]

        try:
            from predict import inspect_known_sample
            meas, report_paths = inspect_known_sample(
                test_img, self._recipe_snapshot(), spec, PROJECT_DIR / "outputs", test_img_path.stem,
                described=desc, golden_image=self.ref_img,
            )
        except (ValueError, OSError) as exc:
            messagebox.showerror("Template configuration", str(exc))
            return

        # Generate annotated inspection image with HUD table
        annotated_vis = measurer.annotate_measurements(test_img, cnt, meas, self.current_sku)

        # Show results dialog
        win = tk.Toplevel(self.root)
        win.title(f"Kết quả kiểm tra đo lường QC - {test_img_path.name}")
        win.geometry("1100x800")
        win.configure(bg=BG_CARD)

        lbl_hdr = tk.Label(
            win, text=f"BÁO CÁO ĐO ĐẠC & KIỂM TRA DUNG SAI ({self.current_sku}) - {test_img_path.name}",
            font=("Segoe UI", 12, "bold"), fg=ACCENT_CYAN, bg=BG_CARD
        )
        lbl_hdr.pack(pady=10)

        # Visual preview frame
        body_f = tk.Frame(win, bg=BG_CARD)
        body_f.pack(fill=tk.BOTH, expand=True, padx=12, pady=5)
        tabs = ttk.Notebook(body_f)
        tabs.pack(fill=tk.BOTH, expand=True)
        overlay_tab = ZoomImageView(tabs, annotated_vis)
        tabs.add(overlay_tab, text="Contour / Scanline")
        panel = cv2.imread(str(report_paths["panel_summary"]))
        if panel is not None:
            tabs.add(ZoomImageView(tabs, panel), text="Báo cáo 6 khung")
        tk.Label(body_f, text=f"Đã lưu báo cáo: {report_paths['panel_summary']}",
                 bg=BG_CARD, fg=TEXT_MUTED, anchor="w").pack(fill=tk.X)
        profile = meas.get("contour_profile")
        if profile:
            text = (f"PROFILE {profile['status']} | Inward {profile.get('lower_mm', 0):.3f} mm | "
                    f"Outward {profile.get('upper_mm', 0):.3f} mm | "
                    f"Max inward {profile.get('max_inward_mm', 0):.3f} mm | "
                    f"Max outward {profile.get('max_outward_mm', 0):.3f} mm")
            if profile.get("error"):
                text = "PROFILE FAIL: " + profile["error"]
            tk.Label(body_f, text=text, bg=BG_CARD,
                     fg=ACCENT_GREEN if profile["pass"] else ACCENT_RED).pack(fill=tk.X)

        # Table frame
        tree_f = tk.Frame(body_f, bg=BG_CARD)
        tree_f.pack(fill=tk.X, pady=5)

        cols = ("label", "golden", "measured", "delta", "edges", "status")
        tree = ttk.Treeview(tree_f, columns=cols, show="headings", height=4, style="Industrial.Treeview")
        tree.heading("label", text="Kích thước")
        tree.heading("golden", text="Golden (mm)")
        tree.heading("measured", text="Đo được (mm)")
        tree.heading("delta", text="Chênh lệch (mm)")
        tree.heading("edges", text="Δ P1 / Δ P2 (mm)")
        tree.heading("status", text="Kết luận")

        tree.column("label", width=90, anchor="center")
        tree.column("measured", width=120, anchor="center")
        tree.column("golden", width=120, anchor="center")
        tree.column("delta", width=130, anchor="center")
        tree.column("edges", width=200, anchor="center")
        tree.column("status", width=90, anchor="center")

        report = meas.get("spec_report", [])
        for r in report:
            tree.insert("", tk.END, values=(
                r["label"],
                f"{r['golden_length_mm']:.3f}" if r.get('golden_length_mm') is not None else "--",
                f"{r['measured_mm']:.3f}" if r['measured_mm'] is not None else "N/A",
                f"{r['total_delta_mm']:+.3f}" if r.get('total_delta_mm') is not None else "--",
                (f"{r['delta_p1_mm']:+.3f} / {r['delta_p2_mm']:+.3f}"
                 if r.get('delta_p1_mm') is not None else "--"),
                r.get("status", ""),
            ))

        results_scroll = ttk.Scrollbar(tree_f, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=results_scroll.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        is_pass = meas.get("overall_pass", False)
        verdict_str = "TỔNG KẾT: PASS (ĐẠT CHUẨN XUẤT XƯỞNG)" if is_pass else "TỔNG KẾT: FAIL (VƯỢT QUÁ SAI SỐ CHO PHÉP)"
        verdict_col = ACCENT_GREEN if is_pass else ACCENT_RED
        lbl_v = tk.Label(win, text=verdict_str, font=("Segoe UI", 12, "bold"), fg=verdict_col, bg=BG_CARD)
        lbl_v.pack(pady=10)


def launch_studio_gui(sku="98661BBS-1"):
    root = tk.Tk()
    app = ScanlineStudioApp(root, initial_sku=sku)
    root.mainloop()

def main():
    parser = argparse.ArgumentParser(description="Industrial Metrology Scanline Teaching Studio")
    parser.add_argument("--sku", type=str, default="98661BBS-1", help="Mã SKU để mở giao diện cấu hình")
    args = parser.parse_args()
    launch_studio_gui(args.sku)

if __name__ == "__main__":
    main()
