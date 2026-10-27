#!/usr/bin/env python3
"""Operator display for the toolhead-following and margin models.

    python run_ui.py --mode demo                       # synthetic episode, no ROS or models
    python run_ui.py --mode live                       # the running toolhead (and margin) nodes
    python run_ui.py --mode offline --source FRAMES    # recorded frames (not wired yet)

Live mode reads the toolhead node's topics; start that node with
-p show_distance_overlay:=false so the distance appears only on the card.

The mode is chosen here, not in the window: live needs ROS 2, offline and demo do not,
so offline and demo run on Windows, Linux and macOS with only `pip install PySide6`.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import QApplication

from toolhead_ui import theme
from toolhead_ui.sources import DemoSource
from toolhead_ui.window import MainWindow


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("demo", "offline", "live"), default="demo")
    ap.add_argument("--source", type=Path, help="folder of frames (offline mode)")
    args = ap.parse_args(argv)
    if args.mode == "offline":
        ap.error("--mode offline is not wired to the pipeline yet")

    app = QApplication(sys.argv[:1])
    theme.load_fonts(app)
    app.setStyleSheet(theme.STYLESHEET)
    if args.mode == "live":
        from toolhead_ui.live_source import LiveSource   # imports ROS 2 only when asked for
        source = LiveSource()
    else:
        source = DemoSource()
    window = MainWindow(source)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
