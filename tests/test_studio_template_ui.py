"""Hidden Tk integration checks: teaching, Save/Test parity and drag-vs-click."""

import json
from types import SimpleNamespace
import tkinter as tk

import cv2
import numpy as np
import pytest

from tools import scanline_studio as studio


@pytest.fixture(scope="module")
def tk_root():
    # Reuse one interpreter: Tk 9 on Windows can fail to reload Tcl scripts
    # when several independent Tk roots are created/destroyed in one process.
    root = tk.Tk()
    root.withdraw()
    root.attributes("-alpha", 0)
    yield root
    root.destroy()


@pytest.fixture
def app(monkeypatch, tmp_path, tk_root):
    # All persistence goes to the pytest sandbox, never real SKU recipes.
    (tmp_path / "configs").mkdir()
    monkeypatch.setattr(studio, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(studio, "RECIPE_FILE", tmp_path / "configs" / "caliper_recipes.json")
    original_toplevel = tk.Toplevel

    class InvisibleToplevel(original_toplevel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.attributes("-alpha", 0)

    monkeypatch.setattr(studio.tk, "Toplevel", InvisibleToplevel)

    def load_fixture(self, sku):
        self.ref_img = np.full((600, 800, 3), 235, dtype=np.uint8)
        cv2.rectangle(self.ref_img, (100, 120), (700, 480), (60, 60, 60), -1)
        self.gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)
        self.cnt = np.array([[[100, 120]], [[700, 120]], [[700, 480]], [[100, 480]]], dtype=np.int32)
        self.wcs = self.measurer_inst._contour_frame(self.cnt)
        self.optical_scale = .01
        self.ref_img_path = tmp_path / "golden.png"
        self.spec_path = tmp_path / "spec.json"
        self.spec_data = {"sku": sku, "optical_scale_mm_per_px": .01}
        self.spec_dims = []
        self.detected_inner_contours = []
        self.calipers = [{
            "label": "W", "nominal_mm": 6, "measure_type": "template_contour",
            "p1_ref": [100., 300.], "p2_ref": [700., 300.],
            "p1_edge": [100., 300.], "p2_edge": [700., 300.],
            "p1_wcs": studio.to_wcs((100, 300), self.wcs),
            "p2_wcs": studio.to_wcs((700, 300), self.wcs),
            "tolerance_minus_mm": .5, "tolerance_plus_mm": .5,
            "measured_mm": 6, "inner_contour": None,
        }]
        self.refresh_table()
        self.redraw_canvas()

    monkeypatch.setattr(studio.ScanlineStudioApp, "load_sku", load_fixture)
    root = tk_root
    root.withdraw()
    root.attributes("-alpha", 0)  # Invisible, but mapped so Tk computes real layouts.
    instance = studio.ScanlineStudioApp(root, initial_sku="SYNTHETIC_UI")
    root.geometry("1100x740")
    root.deiconify()
    root.update()
    yield instance
    for child in root.winfo_children():
        child.destroy()
    root.withdraw()


def test_template_controls_scroll_save_actual_and_clear_to_image(app, monkeypatch):
    app.tree.selection_set("0")
    app.on_tree_select()
    app._replace_entry(app.entry_profile_lower, ".05")
    app._replace_entry(app.entry_profile_upper, ".1")
    app._replace_entry(app.entry_golden_actual, "6.25")
    app.update_current_caliper()
    app.sidebar_canvas.yview_moveto(1)
    app.root.update()
    assert app.entry_profile_upper.winfo_ismapped()
    # Upper/Lower controls must actually intersect the visible scroll viewport.
    top = app.sidebar_canvas.winfo_rooty()
    bottom = top + app.sidebar_canvas.winfo_height()
    assert top <= app.entry_profile_upper.winfo_rooty() <= bottom
    monkeypatch.setattr(studio.messagebox, "showinfo", lambda *a, **k: None)
    app.save_recipe()
    saved = json.loads(studio.RECIPE_FILE.read_text(encoding="utf-8"))[app.current_sku]
    assert saved["contour_lower_mm"] == .05
    assert saved["contour_upper_mm"] == .1
    assert saved["calipers"][0]["golden_actual_mm"] == 6.25
    assert saved["calipers"][0]["golden_length_mm"] == 6.25
    app._replace_entry(app.entry_golden_actual, "")
    app.update_current_caliper()
    app.save_recipe()
    saved = json.loads(studio.RECIPE_FILE.read_text(encoding="utf-8"))[app.current_sku]
    assert saved["calipers"][0]["golden_source"] == "image"
    assert saved["calipers"][0]["golden_length_mm"] == 6


def test_zoom_drag_does_not_create_landmarks_and_click_uses_image_coordinates(app):
    x, y = app.canvas.winfo_width() // 2, app.canvas.winfo_height() // 2
    event = SimpleNamespace(x=x, y=y, delta=120)
    for _ in range(5):
        app.on_canvas_zoom(event)
    assert app.viewport.zoom > 1
    before = app._recipe_snapshot()
    app.active_mode = "draw_template"
    app.drawing_points = []
    app.on_canvas_press(event)
    dragged = SimpleNamespace(x=x + 30, y=y + 20)
    app.on_canvas_drag(dragged)
    app.on_canvas_release(dragged)
    assert app.drawing_points == []
    scale, ox, oy = app.img_scale, app.img_offset_x, app.img_offset_y
    point = (400., 300.)
    click = SimpleNamespace(x=point[0] * scale + ox, y=point[1] * scale + oy)
    app.on_canvas_press(click)
    app.on_canvas_release(click)
    assert np.allclose(app.drawing_points[0], point)
    assert app._recipe_snapshot() == before
    event.delta = -120
    for _ in range(30):
        app.on_canvas_zoom(event)
    assert app.viewport.zoom == 1


def test_test_sample_uses_unsaved_offsets_and_displays_profile_and_image(app, monkeypatch):
    from tkinter import filedialog
    app._replace_entry(app.entry_profile_lower, ".07")
    app._replace_entry(app.entry_profile_upper, ".13")
    monkeypatch.setattr(filedialog, "askopenfilename", lambda **kw: "synthetic_test.png")
    monkeypatch.setattr(studio.cv2, "imread", lambda path: app.ref_img.copy())
    monkeypatch.setattr(studio.clf, "describe_image", lambda image: {
        "contour": app.cnt, "gray": app.gray, "features": {}, "otsu": 100,
    })
    # Reproduce the user's legacy recipe: no TPL caliper, but whole-contour
    # inspection must still run and export the same six-panel report.
    app.calipers[0]["measure_type"] = "taught_caliper"
    app.run_test_sample()
    app.root.update()
    windows = [c for c in app.root.winfo_children() if isinstance(c, tk.Toplevel)]
    assert len(windows) == 1

    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)

    widgets = list(descendants(windows[0]))
    assert any(isinstance(w, studio.ZoomImageView) for w in widgets)
    texts = [str(w.cget("text")) for w in widgets if isinstance(w, tk.Label)]
    assert any("PROFILE PASS" in s and "0.070" in s and "0.130" in s for s in texts)
    report_path = studio.PROJECT_DIR / "outputs" / "visualizations" / "synthetic_test.png"
    prediction = studio.PROJECT_DIR / "outputs" / "predictions" / "synthetic_test.json"
    assert report_path.is_file()
    record = json.loads(prediction.read_text(encoding="utf-8"))
    assert record["profile_pass"] is True
    assert record["classification_performed"] is False
    assert record["spec_report"][0]["golden_length_mm"] == 6
    assert record["spec_report"][0]["total_delta_mm"] == 0
    tabs = [w for w in widgets if isinstance(w, studio.ttk.Notebook)]
    assert any("Báo cáo 6 khung" in [w.tab(tab, "text") for tab in w.tabs()] for w in tabs)
