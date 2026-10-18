"""Image/canvas coordinate mapping independent of Tk and measurement data."""

from dataclasses import dataclass


@dataclass
class ImageViewport:
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    max_zoom: float = 32.0

    def reset(self):
        self.zoom, self.pan_x, self.pan_y = 1.0, 0.0, 0.0

    def transform(self, image_size, canvas_size):
        iw, ih = image_size
        cw, ch = canvas_size
        scale = min(cw / iw, ch / ih) * self.zoom
        # Clamp movement so the image cannot be lost outside the viewport.
        self.pan_x = max(-max(0, (iw * scale - cw) / 2),
                         min(max(0, (iw * scale - cw) / 2), self.pan_x))
        self.pan_y = max(-max(0, (ih * scale - ch) / 2),
                         min(max(0, (ih * scale - ch) / 2), self.pan_y))
        return scale, (cw - iw * scale) / 2 + self.pan_x, (ch - ih * scale) / 2 + self.pan_y

    def zoom_at(self, factor, x, y, image_size, canvas_size):
        old_scale, ox, oy = self.transform(image_size, canvas_size)
        ix, iy = (x - ox) / old_scale, (y - oy) / old_scale
        self.zoom = min(self.max_zoom, max(1.0, self.zoom * factor))
        if self.zoom < 1.000001:
            self.reset()
            return
        iw, ih = image_size
        cw, ch = canvas_size
        scale = min(cw / iw, ch / ih) * self.zoom
        self.pan_x = x - ix * scale - (cw - iw * scale) / 2
        self.pan_y = y - iy * scale - (ch - ih * scale) / 2
        self.transform(image_size, canvas_size)
