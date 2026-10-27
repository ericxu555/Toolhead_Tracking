"""Frame sources. The window talks only to FrameSource, so live, offline and demo are interchangeable.

DemoSource renders a synthetic episode so the layout can be developed and
screenshotted with no ROS, models or recordings.
"""
import cv2
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from .data import BASELINE, ESTIMATE, FINAL, FrameData

TOOLHEAD, MARGIN = "Toolhead-following", "Margin"


class FrameSource(QObject):
    """Base class. A source with a timeline (offline, demo) starts paused on frame 0."""
    frame = Signal(object)           # FrameData
    playing_changed = Signal(bool)
    label_changed = Signal(str)      # emit when the header should change, e.g. a new episode
    has_timeline = False
    label = ""                       # the header line, e.g. "Live · Episode 3 · Calibration 2026-10-20"

    def start(self):
        """Emit the first frame."""

    def play(self):
        pass

    def pause(self):
        pass

    def seek(self, index: int):
        pass

    def set_view(self, view: str):
        """'toolhead' or 'both'. A source may skip the margin model while it is not shown."""

    def close(self):
        pass


class DemoSource(FrameSource):
    has_timeline = True
    label = "Demo · Synthetic episode"
    SIZE = 540
    N = 240
    BASELINE_END, CUT_END = 10, 180
    MM_PER_PX = 0.05
    CENTER, AXES = (270, 250), (110, 85)       # tumor ellipse: center, half-axes in px

    def __init__(self, fps: float = 10.0):
        super().__init__()
        self._index = 0
        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / fps))
        self._timer.timeout.connect(self._tick)
        rng = np.random.RandomState(0)
        noise = cv2.GaussianBlur(rng.rand(self.SIZE, self.SIZE).astype(np.float32), (0, 0), 14)
        tissue = np.empty((self.SIZE, self.SIZE, 3), np.float32)
        for c, base in enumerate((125, 140, 205)):             # BGR, pinkish tissue
            tissue[..., c] = base + (noise - noise.mean()) * 90
        self._tissue = np.clip(tissue, 0, 255).astype(np.uint8)
        yy, xx = np.mgrid[0:self.SIZE, 0:self.SIZE]
        a, b = self.AXES
        self._r = np.sqrt(((xx - self.CENTER[0]) / a) ** 2 + ((yy - self.CENTER[1]) / b) ** 2)
        self._tumor = self._r <= 1.0

    def start(self):
        self._emit()

    def play(self):
        if self._index >= self.N - 1:
            self._index = 0
        self._timer.start()
        self.playing_changed.emit(True)

    def pause(self):
        self._timer.stop()
        self.playing_changed.emit(False)

    def seek(self, index: int):
        self._index = max(0, min(self.N - 1, int(index)))
        self._emit()

    def _tick(self):
        if self._index >= self.N - 1:
            self.pause()
            return
        self._index += 1
        self._emit()

    def _tip(self, i: int):
        """The tool enters from the bottom during the cut, then leaves."""
        if i < self.BASELINE_END:
            return None
        t = min(1.0, (i - self.BASELINE_END) / (self.CUT_END - self.BASELINE_END))
        if i >= self.CUT_END:
            t = max(0.0, 1.0 - (i - self.CUT_END) / 20.0) * 1.0
        return (270 + 14 * np.sin(6 * t), 520 - 225 * t) if t > 0 else None

    def _swept(self, i: int):
        t = min(1.0, (min(i, self.CUT_END) - self.BASELINE_END) / (self.CUT_END - self.BASELINE_END))
        mask = np.zeros((self.SIZE, self.SIZE), np.uint8)
        if t > 0:
            cv2.line(mask, (270, 520), (int(270 + 14 * np.sin(6 * t)), int(520 - 225 * t)), 1, 56)
        return (mask > 0) & self._tumor

    def _emit(self):
        i = self._index
        tip = self._tip(i)
        swept = self._swept(i)
        tool_in = tip is not None
        tool = self._tissue.copy()
        margin = self._tissue.copy()
        tumor_px = float(self._tumor.sum())

        tint = lambda img, m, bgr, a: img.__setitem__(m, (a * np.array(bgr) + (1 - a) * img[m]).astype(np.uint8))
        for img in (tool, margin):
            img[self._tumor] = (0.55 * np.array((95, 105, 165)) + 0.45 * img[self._tumor]).astype(np.uint8)
        tint(tool, swept, (0, 0, 255), 0.5)
        tint(margin, self._tumor & ~swept, (90, 200, 90), 0.5)
        if tool_in:
            cv2.line(tool, (400, 540), (int(tip[0]), int(tip[1])), (190, 190, 190), 16, cv2.LINE_AA)
            cv2.circle(tool, (int(tip[0]), int(tip[1])), 5, (255, 0, 255), -1, cv2.LINE_AA)
            r = np.sqrt(((tip[0] - self.CENTER[0]) / self.AXES[0]) ** 2 + ((tip[1] - self.CENTER[1]) / self.AXES[1]) ** 2)
            d_mm = abs(1.0 - r) * min(self.AXES) * self.MM_PER_PX
        else:
            d_mm = None

        if i < self.BASELINE_END:
            state, fraction = BASELINE, None
        elif i < self.CUT_END:
            state, fraction = ESTIMATE, swept.sum() / tumor_px * 0.95
        else:
            state, fraction = FINAL, swept.sum() / tumor_px
            d_mm = None if not tool_in else d_mm
        self.frame.emit(FrameData(streams=[(TOOLHEAD, tool), (MARGIN, margin)], d_mm=d_mm, warn_mm=2.0,
                                  warn=d_mm is not None and d_mm < 2.0,
                                  fraction=fraction, fraction_state=state, index=i, total=self.N))
