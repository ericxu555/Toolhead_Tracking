"""Render the live UI from messages a node replay published, without ROS.

tools/replay_ui_flow.py --dump PKL records every /toolhead/overlay and
/toolhead/status message; this feeds them, in order, to LiveSource's callbacks
and saves the window whenever the fraction state changes, plus the first
warning frame.

    python -m toolhead_ui.render_live PKL OUT_DIR
"""
import pickle
import sys
import types
from pathlib import Path

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from . import theme
from .live_source import LiveSource
from .window import MainWindow


def main(pkl: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv[:1])
    theme.load_fonts(app)
    app.setStyleSheet(theme.STYLESHEET)
    LiveSource.start = lambda self: None          # no ROS here; messages are fed directly
    source = LiveSource()
    window = MainWindow(source)
    window.resize(1280, 800)
    window.show()

    def snap(name):
        loop = QEventLoop()
        QTimer.singleShot(300, loop.quit)        # queued signals, then the warning transition
        loop.exec()
        window.grab().save(str(out / f"{name}.png"))
        print("saved", name)

    last_state, warned, n = None, False, 0
    for topic, payload in pickle.load(open(pkl, "rb")):
        if topic == "/toolhead/status":
            source.on_status(types.SimpleNamespace(data=payload))
            continue
        h, w, enc, data = payload
        source.on_overlay(types.SimpleNamespace(height=h, width=w, encoding=enc, data=data))
        app.processEvents()
        s = source._status
        n += 1
        if s.get("fraction_state") != last_state:
            last_state = s.get("fraction_state")
            snap(f"{n:04d}_{last_state}")
        if s.get("warn") and not warned:
            warned = True
            snap(f"{n:04d}_warning")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
