"""The main window: a header, then controls above the stream(s) and the readout cards beside them.

The body is placed by computed geometry rather than stretch layouts, so the
proportions hold at any window size:

    header                                                   (HEADER_H, hairline below)
    ┌ controls row (CONTROL_H) ┐
    │ S2                       │
    │ stream(s), square         │ S3 │ distance card │
    │                          │    │ S2            │
    │ caption                  │    │ fraction card │
    └──────────────────────────┘

The streams and the card column share top and bottom edges, the whole block is
centered in the body, and everything sits on the 8 px grid.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from . import theme as T
from .data import FrameData
from .sources import FrameSource
from .widgets import DistanceCard, FractionCard, SegmentedControl, StreamView, primary_button

APP_NAME = "Resection Monitor"      # window title only; the header shows the mode and source


class Header(QWidget):
    """One line: the mode and source, e.g. "Live · Episode 3 · Calibration 2026-10-20"."""

    def __init__(self, text: str):
        super().__init__()
        self.setFixedHeight(T.HEADER_H)
        self.label = QLabel(text)
        self.label.setObjectName("header")
        row = QHBoxLayout()
        row.setContentsMargins(T.S3, 0, T.S3, 0)
        row.addWidget(self.label)
        row.addStretch(1)
        divider = QWidget()
        divider.setObjectName("divider")
        divider.setAttribute(Qt.WA_StyledBackground, True)
        divider.setFixedHeight(1)
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        col.addLayout(row, 1)
        col.addWidget(divider)


class Body(QWidget):
    """Places the controls, streams and cards by hand; see the module docstring."""
    MARGIN = T.S3
    GAP = T.S3             # streams to cards
    STREAM_GAP = T.S2      # between two streams
    CARD_GAP = T.S2        # between the two cards
    MIN_SIDE = 480

    def __init__(self, controls: QWidget, streams, cards):
        super().__init__()
        self.controls, self.streams, self.cards = controls, streams, cards
        for w in (controls, *streams, *cards):
            w.setParent(self)
        self.n_streams = 1

    def set_streams(self, n: int):
        self.n_streams = n
        self.streams[1].setVisible(n == 2)
        self._place()

    def minimum_body_size(self):
        n = self.n_streams
        w = n * self.MIN_SIDE + (n - 1) * self.STREAM_GAP + self.GAP + T.PANEL_W + 2 * self.MARGIN
        h = T.CONTROL_H + T.S2 + self.MIN_SIDE + T.CAPTION_H + 2 * self.MARGIN
        return w, h

    def resizeEvent(self, _):
        self._place()

    def _place(self):
        n = self.n_streams
        avail_w = self.width() - 2 * self.MARGIN - self.GAP - T.PANEL_W - (n - 1) * self.STREAM_GAP
        avail_h = self.height() - 2 * self.MARGIN - T.CONTROL_H - T.S2 - T.CAPTION_H
        side = int(max(1, min(avail_w / n, avail_h)) // T.S1 * T.S1)       # snap to the grid
        streams_w = n * side + (n - 1) * self.STREAM_GAP
        block_w = streams_w + self.GAP + T.PANEL_W
        block_h = T.CONTROL_H + T.S2 + side + T.CAPTION_H
        x0 = (self.width() - block_w) // 2
        y0 = (self.height() - block_h) // 2

        self.controls.setGeometry(x0, y0, streams_w, T.CONTROL_H)
        top = y0 + T.CONTROL_H + T.S2
        for i, s in enumerate(self.streams[:n]):
            s.setGeometry(x0 + i * (side + self.STREAM_GAP), top, side, side + T.CAPTION_H)
        card_h = (side - self.CARD_GAP) // 2
        px = x0 + streams_w + self.GAP
        self.cards[0].setGeometry(px, top, T.PANEL_W, card_h)
        self.cards[1].setGeometry(px, top + card_h + self.CARD_GAP, T.PANEL_W, side - card_h - self.CARD_GAP)


class MainWindow(QWidget):

    def __init__(self, source: FrameSource):
        super().__init__()
        self.setObjectName("root")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setWindowTitle(APP_NAME)
        self.source = source
        self._seeking = False

        self.view_control = SegmentedControl([("toolhead", "Toolhead"), ("both", "Both")])
        self.view_control.changed.connect(self._on_view)
        self.play_button = primary_button("Play", width=88)
        self.play_button.clicked.connect(self._toggle_play)
        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setRange(0, 0)
        self.scrubber.setCursor(Qt.PointingHandCursor)
        self.scrubber.valueChanged.connect(self._on_scrub)
        self.position = QLabel("")
        self.position.setFont(T.font(T.TEXT, tabular=True))
        self.position.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.position.setFixedWidth(80)

        controls = QWidget()
        row = QHBoxLayout(controls)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(T.S2)
        row.addWidget(self.view_control)
        if source.has_timeline:
            row.addWidget(self.play_button)
            row.addWidget(self.scrubber, 1)
            row.addWidget(self.position)
        else:
            row.addStretch(1)

        self.streams = [StreamView(), StreamView()]
        self.distance = DistanceCard()
        self.fraction = FractionCard()
        self.body = Body(controls, self.streams, [self.distance, self.fraction])
        self.body.set_streams(1)

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        self.header = Header(source.label)
        source.label_changed.connect(self.header.label.setText)
        col.addWidget(self.header)
        col.addWidget(self.body, 1)
        self._apply_minimum()
        self.resize(1280, 800)

        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._toggle_play)
        source.frame.connect(self.show_frame)
        source.playing_changed.connect(self._on_playing)
        source.start()

    def _apply_minimum(self):
        w, h = self.body.minimum_body_size()
        self.setMinimumSize(w, h + T.HEADER_H)

    # ------------------------------------------------------------ display
    def show_frame(self, data: FrameData):
        for view, (caption, image) in zip(self.streams, data.streams):
            if image is not None:                 # None: keep the last frame (or the placeholder)
                view.set_frame(caption, image)
        self.distance.set_frame(data.d_mm, data.d_px, data.warn_mm, data.warn)
        self.fraction.set_frame(data.fraction, data.fraction_state)
        if data.total:
            self._seeking = True
            self.scrubber.setRange(0, data.total - 1)
            self.scrubber.setValue(data.index)
            self._seeking = False
            self.position.setText(f"{data.index + 1} / {data.total}")

    # ------------------------------------------------------------ controls
    def _on_view(self, view: str):
        self.body.set_streams(2 if view == "both" else 1)
        self._apply_minimum()
        self.source.set_view(view)

    def _toggle_play(self):
        if not self.source.has_timeline:
            return
        if self.play_button.text() == "Play":
            self.source.play()
        else:
            self.source.pause()

    def _on_playing(self, playing: bool):
        self.play_button.setText("Pause" if playing else "Play")

    def _on_scrub(self, value: int):
        if not self._seeking:
            self.source.seek(value)

    def closeEvent(self, event):
        self.source.close()
        super().closeEvent(event)
