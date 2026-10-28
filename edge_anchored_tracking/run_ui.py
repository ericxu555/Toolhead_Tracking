#!/usr/bin/env python3
"""Operator display for the toolhead-following and margin models.

    python run_ui.py --mode demo                       # synthetic episode, no ROS or models
    python run_ui.py --mode live                       # the running toolhead (and margin) nodes
    python run_ui.py --mode offline --source FRAMES --tips tip_track.csv [--checkpoint DIR]

Live mode reads the toolhead node's topics; start that node with
-p show_distance_overlay:=false so the distance appears only on the card.

Offline mode plays a recorded frame folder through the live nodes' own code (ROS
stubbed): the toolhead node, with the tip from a tip track CSV (frame,x,y in 540 px,
e.g. from the vision-only tip pipeline), and the margin node from the margin
repository. It needs both repositories' Python dependencies, the model checkpoints
and a GPU; the distance is in pixels because recordings have no camera calibration.

The mode is chosen here, not in the window: live needs ROS 2, offline and demo do not,
so they run on Windows, Linux and macOS; demo needs only `pip install PySide6`.
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
    ap.add_argument("--source", type=Path, help="folder of .png frames (offline mode)")
    ap.add_argument("--tips", type=Path, help="tip track CSV with frame,x,y (offline mode)")
    ap.add_argument("--checkpoint", type=Path,
                    help="tumor model folder with best_model_epoch_swa.pth (offline mode; "
                         "default: the SPATIAL_MODEL/SPATIAL_META environment variables)")
    ap.add_argument("--margin-repo", type=Path,
                    help="margin model repository (offline mode; default: $MARGIN_REPO or a sibling "
                         "Margin_Model_Github / Margin_Segmentation_Model folder)")
    ap.add_argument("--out", type=Path, help="where the nodes write their episode outputs (offline mode)")
    args = ap.parse_args(argv)
    if args.mode == "offline":
        if args.source is None or args.tips is None:
            ap.error("--mode offline needs --source FRAMES and --tips tip_track.csv")
        for p in (args.source, args.tips):
            if not p.exists():
                ap.error(f"not found: {p}")

    app = QApplication(sys.argv[:1])
    theme.load_fonts(app)
    app.setStyleSheet(theme.STYLESHEET)
    if args.mode == "live":
        from toolhead_ui.live_source import LiveSource   # imports ROS 2 only when asked for
        source = LiveSource()
    elif args.mode == "offline":
        from toolhead_ui.offline_source import OfflineSource   # imports the node only when asked for
        source = OfflineSource(str(args.source), str(args.tips),
                               checkpoint_dir=str(args.checkpoint) if args.checkpoint else None,
                               out_dir=str(args.out) if args.out else None,
                               margin_repo=str(args.margin_repo) if args.margin_repo else None)
    else:
        source = DemoSource()
    window = MainWindow(source)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
