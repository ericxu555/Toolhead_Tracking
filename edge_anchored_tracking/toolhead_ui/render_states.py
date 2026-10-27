"""Render the window in each display state to PNGs, headless. For design checks and paper figures.

    python -m toolhead_ui.render_states OUT_DIR
"""
import sys
from pathlib import Path

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from . import theme
from .data import BLOCKED, ESTIMATE, FINAL
from .sources import DemoSource
from .window import MainWindow


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv[:1])
    theme.load_fonts(app)
    app.setStyleSheet(theme.STYLESHEET)
    source = DemoSource()
    window = MainWindow(source)
    window.resize(1280, 800)
    window.show()

    def snap(name):
        loop = QEventLoop()
        QTimer.singleShot(300, loop.quit)     # let the warning color transition finish
        loop.exec()
        window.grab().save(str(out / f"{name}.png"))

    source.seek(3);   snap("1_baseline")
    source.seek(60);  snap("2_estimate_far")
    source.seek(150); snap("3_estimate_warning")
    source.seek(215); snap("4_final")
    window.view_control.select("both"); window.resize(1440, 800)
    source.seek(150); snap("5_both")
    # blocked: no source frame has it, so drive the card directly
    window.fraction.set_frame(None, BLOCKED); snap("6_blocked")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
