"""Custom widgets: hover lift, buttons, view selector, square stream, readout cards.

Sizes, colors and fonts all come from theme; nothing here picks its own.
"""
from typing import Optional

import numpy as np
from PySide6.QtCore import (QEasingCurve, QEvent, QObject, QPointF, QPropertyAnimation, QRectF, Qt,
                            QVariantAnimation, Signal)
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QPushButton, QWidget

from . import theme as T
from .data import BASELINE, BLOCKED, ESTIMATE, FINAL, READY

W = QFont.Weight


class HoverLift(QObject):
    """A soft shadow that appears and drops under the widget while the pointer is over it.

    Invisible at rest. Qt widgets cannot animate scale, so the lift is the shadow alone.
    """
    REST = (0.0, 0.0)      # blur radius, y offset
    HOVER = (18.0, 4.0)

    def __init__(self, widget: QWidget):
        super().__init__(widget)
        self.fx = QGraphicsDropShadowEffect(widget)
        self.fx.setColor(QColor(0, 0, 0, 46))
        self.fx.setBlurRadius(self.REST[0])
        self.fx.setOffset(0, self.REST[1])
        widget.setGraphicsEffect(self.fx)
        self._anims = []
        for name in ("blurRadius", "yOffset"):
            anim = QPropertyAnimation(self.fx, name.encode(), self)
            anim.setDuration(160)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            self._anims.append((anim, name))
        widget.installEventFilter(self)

    def _go(self, target):
        for (anim, name), end in zip(self._anims, target):
            anim.stop()
            anim.setStartValue(self.fx.property(name))
            anim.setEndValue(end)
            anim.start()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Enter:
            self._go(self.HOVER)
        elif event.type() == QEvent.Type.Leave:
            self._go(self.REST)
        return False


def _interactive(widget: QWidget) -> QWidget:
    widget.setCursor(Qt.PointingHandCursor)
    widget.setFocusPolicy(Qt.NoFocus)
    return widget


def primary_button(text: str, width: Optional[int] = None) -> QPushButton:
    button = _interactive(QPushButton(text))
    button.setObjectName("primary")
    if width:
        button.setFixedWidth(width)
    HoverLift(button)
    return button


class SegmentedControl(QWidget):
    """Exclusive options in one control, painted so its height, radius and type match a button.

    The selected segment is teal with white text; the others are white with black text,
    all inside one hairline outline with RADIUS corners.
    """
    changed = Signal(str)
    SEGMENT_W = 104

    def __init__(self, options):
        super().__init__()
        self._options = list(options)
        self._selected = self._options[0][0]
        self.setFixedSize(self.SEGMENT_W * len(self._options), T.CONTROL_H)
        _interactive(self)
        HoverLift(self)

    def select(self, key: str):
        if key != self._selected:
            self._selected = key
            self.update()
            self.changed.emit(key)

    def mousePressEvent(self, event):
        i = int(event.position().x() // self.SEGMENT_W)
        if 0 <= i < len(self._options):
            self.select(self._options[i][0])

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        outline = QRectF(self.rect()).adjusted(.5, .5, -.5, -.5)
        clip = QPainterPath()
        clip.addRoundedRect(outline, T.RADIUS, T.RADIUS)
        p.fillPath(clip, T.WHITE)
        p.setFont(T.font(T.TEXT, W.DemiBold))
        for i, (key, label) in enumerate(self._options):
            seg = QRectF(i * self.SEGMENT_W, 0, self.SEGMENT_W, self.height())
            if key == self._selected:
                p.save()
                p.setClipPath(clip)
                p.fillRect(seg, T.TEAL)
                p.restore()
                p.setPen(T.WHITE)
            else:
                p.setPen(T.BLACK)
            p.drawText(seg, Qt.AlignCenter, label)
        p.setPen(QPen(T.HAIRLINE, 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(outline, T.RADIUS, T.RADIUS)
        for i in range(1, len(self._options)):        # dividers between unselected neighbours only
            left, right = self._options[i - 1][0], self._options[i][0]
            if self._selected not in (left, right):
                x = i * self.SEGMENT_W + .5
                p.drawLine(QPointF(x, 0), QPointF(x, self.height()))


def _to_qimage(bgr: np.ndarray) -> QImage:
    bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    return QImage(bgr.data, w, h, w * 3, QImage.Format_BGR888).copy()   # copy: detach from numpy's buffer


class StreamView(QWidget):
    """A square video frame with a caption underneath. The window sizes it to side x (side + CAPTION_H)."""

    def __init__(self, placeholder: str = "Waiting for video"):
        super().__init__()
        self._image: Optional[QImage] = None
        self._caption = ""
        self._placeholder = placeholder

    def set_frame(self, caption: str, bgr: np.ndarray):
        self._caption = caption
        self._image = _to_qimage(bgr)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        side = self.width()
        square = QRectF(0, 0, side, side)
        if self._image is None:
            p.fillRect(square, T.PLACEHOLDER)
            p.setFont(T.font(T.TEXT))
            p.setPen(T.BLACK)
            p.drawText(square, Qt.AlignCenter, self._placeholder)
        else:
            p.drawImage(square, self._image)
        if self._caption:
            p.setFont(T.font(T.TEXT, W.Medium))
            p.setPen(T.BLACK)
            p.drawText(QRectF(0, side, side, T.CAPTION_H), Qt.AlignCenter, self._caption)


def _mix(a: QColor, b: QColor, t: float) -> QColor:
    return QColor(*(int(x + (y - x) * t) for x, y in ((a.red(), b.red()), (a.green(), b.green()),
                                                       (a.blue(), b.blue()))))


class _Card(QWidget):
    """White card, hairline border, RADIUS corners, S3 padding, everything centered.

    Rows, measured inside the padding: the title at the top, the footer (note or
    bar) at the bottom, and the readout centered between them. Both cards have
    the same height, so their readouts line up.
    """
    PAD = T.S3
    TITLE_ROW = 28
    FOOTER_H = 16
    READOUT_H = 64
    TAG_H = 28

    def _readout_rect(self, inner: QRectF) -> QRectF:
        top = inner.top() + self.TITLE_ROW
        bottom = inner.bottom() - self.FOOTER_H
        return QRectF(inner.left(), top + (bottom - top - self.READOUT_H) / 2, inner.width(), self.READOUT_H)

    def _frame(self, p: QPainter, border: QColor, width: float = 1.0) -> QRectF:
        p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        inset = width / 2
        p.setPen(QPen(border, width))
        p.setBrush(T.WHITE)
        p.drawRoundedRect(QRectF(self.rect()).adjusted(inset, inset, -inset, -inset), T.RADIUS, T.RADIUS)
        return QRectF(self.rect()).adjusted(self.PAD, self.PAD, -self.PAD, -self.PAD)

    def _title(self, p: QPainter, inner: QRectF, text: str):
        p.setFont(T.font(T.TITLE, W.DemiBold))
        p.setPen(T.BLACK)
        p.drawText(QRectF(inner.left(), inner.top(), inner.width(), self.TITLE_ROW), Qt.AlignCenter, text)

    def _readout(self, p: QPainter, inner: QRectF, number: str, unit: str, color: QColor):
        """Number and unit on one baseline, centered as a group."""
        big = T.font(T.DISPLAY, W.Bold, tabular=True)
        small = T.font(T.TITLE, W.Medium)
        fm, fs = QFontMetricsF(big), QFontMetricsF(small)
        row = self._readout_rect(inner)
        baseline = row.top() + (row.height() + fm.capHeight()) / 2
        uw = T.S1 + fs.horizontalAdvance(unit) if unit else 0.0
        x = row.center().x() - (fm.horizontalAdvance(number) + uw) / 2
        p.setPen(color)
        p.setFont(big)
        p.drawText(QPointF(x, baseline), number)
        if unit:
            p.setFont(small)
            p.drawText(QPointF(x + fm.horizontalAdvance(number) + T.S1, baseline), unit)

    def _message(self, p: QPainter, inner: QRectF, text: str):
        p.setFont(T.font(T.TITLE, W.Medium))
        p.setPen(T.BLACK)
        p.drawText(self._readout_rect(inner), Qt.AlignCenter, text)


class DistanceCard(_Card):
    """Tip-to-boundary distance in mm. Below the threshold the number and border ease to red."""

    def __init__(self):
        super().__init__()
        self._d: Optional[float] = None
        self._d_px: Optional[float] = None
        self._warn_mm = 2.0
        self._mix = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(180)
        self._anim.setEndValue(0.0)
        self._anim.valueChanged.connect(self._set_mix)

    def _set_mix(self, v):
        self._mix = float(v)
        self.update()

    def set_frame(self, d_mm: Optional[float], d_px: Optional[float], warn_mm: float, warn: bool):
        """mm when the source has a depth (visual-servoing tip), else pixels."""
        self._d, self._d_px, self._warn_mm = d_mm, d_px, warn_mm
        target = 1.0 if warn else 0.0
        if target != self._anim.endValue():
            self._anim.stop()
            self._anim.setStartValue(self._mix)
            self._anim.setEndValue(target)
            self._anim.start()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        t = self._mix
        inner = self._frame(p, _mix(T.HAIRLINE, T.RED, t), 1.0 + t)
        self._title(p, inner, "Distance to Boundary")
        if self._d is not None:
            self._readout(p, inner, f"{self._d:.1f}", "mm", _mix(T.BLACK, T.RED, t))
        elif self._d_px is not None:          # no depth (vision-only tip): pixels, no mm warning
            self._readout(p, inner, f"{self._d_px:.0f}", "px", T.BLACK)
        else:
            self._message(p, inner, "No boundary in view")
        if self._d_px is None or self._d is not None:
            p.setFont(T.font(T.TEXT))
            p.setPen(T.BLACK)
            p.drawText(QRectF(inner.left(), inner.bottom() - self.FOOTER_H, inner.width(), self.FOOTER_H),
                       Qt.AlignCenter, f"Warning below {self._warn_mm:g} mm")


class FractionCard(_Card):
    """Resected fraction. Estimate: an outlined tag. Final: a filled tag.

    The tag sits centered just under the number.
    """
    BAR_H = 8

    def __init__(self):
        super().__init__()
        self._fraction: Optional[float] = None
        self._state = BASELINE

    def set_frame(self, fraction: Optional[float], state: str):
        self._fraction, self._state = fraction, state
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        inner = self._frame(p, T.HAIRLINE)
        self._title(p, inner, "Resected Area")
        shown = self._state in (ESTIMATE, FINAL) and self._fraction is not None
        if shown:
            final = self._state == FINAL
            self._readout(p, inner, f"{round(self._fraction * 100):d}", "%", T.BLACK)
            self._tag(p, inner, "Final" if final else "Estimate", filled=final)
        else:
            self._message(p, inner, {BASELINE: "Capturing baseline",
                                     BLOCKED: "Move the tool out of view",
                                     READY: "Baseline ready"}.get(self._state, "—"))

        bar = QRectF(inner.left(), inner.bottom() - (self.FOOTER_H + self.BAR_H) / 2, inner.width(), self.BAR_H)
        r = self.BAR_H / 2
        p.setPen(Qt.NoPen)
        p.setBrush(T.TRACK)
        p.drawRoundedRect(bar, r, r)
        if shown and self._fraction > 0:
            fill = QRectF(bar.left(), bar.top(), max(bar.height(), bar.width() * min(1.0, self._fraction)), bar.height())
            p.setBrush(T.BLACK)
            p.drawRoundedRect(fill, r, r)

    def _tag(self, p: QPainter, inner: QRectF, text: str, filled: bool):
        p.setFont(T.font(T.TEXT, W.DemiBold))
        w = QFontMetricsF(p.font()).horizontalAdvance(text) + 2 * T.S2
        below = self._readout_rect(inner).bottom() + T.S1
        rect = QRectF(inner.center().x() - w / 2, below, w, self.TAG_H)
        if filled:
            p.setPen(Qt.NoPen)
            p.setBrush(T.BLACK)
            p.drawRoundedRect(rect, T.RADIUS, T.RADIUS)
            p.setPen(T.WHITE)
        else:
            p.setPen(QPen(T.BLACK, 1))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(rect.adjusted(.5, .5, -.5, -.5), T.RADIUS, T.RADIUS)
            p.setPen(T.BLACK)
        p.drawText(rect, Qt.AlignCenter, text)
