"""Offline source: a recorded frame folder played through the live toolhead and margin nodes.

The nodes' own code (ros_toolhead_pose_node.py, and ros_margin_node.py from the
margin repository) runs with rclpy stubbed, as in tools/offline_replay_test.py,
so the operator sees exactly what the live system would have shown for this video.
Every frame goes through both nodes; the toolhead node's episode is:

    warm-up   the first frames without a tip pose, so the node captures its baseline
    press     every frame with its tip from the tip track (pose_only tip source)
    release   the last frame repeated with the tip out of view until the final fraction

The node is stateful (the mask accumulates), so a worker thread processes the
whole video once and caches every published overlay and status; play and the
scrubber move freely over what is already processed. Recordings have no camera
calibration, so the distance is shown in pixels, as in the paper.
"""
import csv
import glob
import importlib.util
import json
import os
import sys
import threading
import types

import cv2
import numpy as np
from PySide6.QtCore import QTimer

from .data import FrameData
from .sources import MARGIN, TOOLHEAD, FrameSource

HERE = os.path.dirname(os.path.abspath(__file__))
NODE = os.path.normpath(os.path.join(HERE, "..", "ros_toolhead_pose_node.py"))
TOOLS = os.path.normpath(os.path.join(HERE, "..", "..", "tools"))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))

WARMUP_FRAMES = 5     # the node needs 5 tool-free frames for its baseline
RELEASE_MAX = 10      # frames after the release; the node reports final after 5
MARGIN_REPO_NAMES = ("Margin_Model_Github", "Margin_Segmentation_Model")


def find_margin_repo(given=None):
    """The margin repository: as given, $MARGIN_REPO, or a sibling of this repository."""
    for d in [given, os.environ.get("MARGIN_REPO")] + [
            os.path.join(os.path.dirname(REPO), n) for n in MARGIN_REPO_NAMES]:
        if d and os.path.isfile(os.path.join(d, "ros_margin_node.py")):
            return d
    return None


def recording_name(frames_dir: str) -> str:
    """'Full_EXP2 · 20260615-164758-188519' for .../Full_EXP2/20260615-164758-188519/endoscope."""
    parts = os.path.normpath(frames_dir).split(os.sep)
    if parts[-1].lower() == "endoscope" and len(parts) >= 3:
        return f"{parts[-3]} · {parts[-2]}"
    return parts[-1]


class OfflineSource(FrameSource):
    has_timeline = True

    def __init__(self, frames_dir, tips_csv, checkpoint_dir=None, out_dir=None, margin_repo=None,
                 fps: float = 30.0):
        super().__init__()
        self._margin_repo = find_margin_repo(margin_repo)
        self._margin_note = None          # shown in the header when the margin model cannot run
        self._files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        if not self._files:
            raise FileNotFoundError(f"No .png frames in {frames_dir}")
        self._tips = {int(r["frame"]): (float(r["x"]), float(r["y"])) for r in csv.DictReader(open(tips_csv))}
        self._checkpoint = checkpoint_dir
        self._out = out_dir or os.path.join(REPO, "Output", "offline_ui")
        self._name = recording_name(frames_dir)
        self.label = f"Offline · {self._name} · Loading the models"

        self._cache = []                  # (toolhead jpeg, margin jpeg or None, status dict), in timeline order
        self._done = False
        self._error = None
        self._stop = False
        self._lock = threading.Lock()
        self._index = 0
        self._shown = -1
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(int(1000 / fps))
        self._play_timer.timeout.connect(self._tick)
        self._poll = QTimer(self)         # progress in the header; the first frame once it exists
        self._poll.setInterval(250)
        self._poll.timeout.connect(self._on_poll)
        self._total = len(self._files)    # grows by the release frames when processing ends

    # ---------------------------------------------------------------- worker
    def _run(self):
        try:
            self._process()
        except Exception as exc:          # shown in the header; the window stays usable
            self._error = f"{type(exc).__name__}: {exc}"
        self._done = True

    def _process(self):
        if self._checkpoint:
            os.environ["SPATIAL_MODEL"] = os.path.join(self._checkpoint, "best_model_epoch_swa.pth")
            os.environ["SPATIAL_META"] = os.path.join(self._checkpoint, "best_model_epoch_swa_metadata.json")
        sys.path.insert(0, TOOLS)
        import offline_replay_test as R

        os.makedirs(self._out, exist_ok=True)
        calib = os.path.join(self._out, "synthetic_calib.yaml")
        R.write_calib(calib)              # needed to back-project the tip; its mm values are not shown
        R.OVERRIDES.clear()
        R.OVERRIDES.update({
            "tip_source": "pose_only", "camera_model": "rect", "image_topic": "/ves_camera/image_rect",
            "calib_file": calib, "output_root": self._out, "trigger_mode": "service",
            "show_distance_overlay": False,
        })
        R.install_stubs()
        spec = importlib.util.spec_from_file_location("toolhead_node_offline", NODE)
        nodemod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nodemod)
        node = nodemod.ToolheadNode()
        pose_cb = node._subs[node.pose_topic][0]
        img_cb = node._subs[node.image_topic][0]

        margin = None
        if self._margin_repo:
            # Both repositories ship the same src/ package (identical files), so
            # the margin node's `from src...` import is satisfied by either copy.
            R.OVERRIDES.clear()
            R.OVERRIDES.update({"image_topic": "/ves_camera/image", "trigger_mode": "off",
                                "output_root": os.path.join(self._out, "margin")})
            spec = importlib.util.spec_from_file_location(
                "margin_node_offline", os.path.join(self._margin_repo, "ros_margin_node.py"))
            mmod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mmod)
            try:
                margin = mmod.MarginNode()
                margin_cb = margin._subs[margin.image_topic][0]
            except SystemExit:            # the node exits when its weights are missing
                self._margin_note = "Margin model weights not found"
        else:
            self._margin_note = "Margin model not found (--margin-repo)"
        t_ns = [1_700_000_000_000_000_000]

        def last_image(n, topic):
            img = None
            for t, m in n.published:
                if t == topic:
                    img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
            return img

        def jpeg(img):
            return None if img is None else cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()

        def feed(k, tip=None, keep=True):
            t_ns[0] += 33_333_333
            if tip == "off":
                pose_cb(R.pose_msg(-400.0, -400.0, t_ns[0], 0.0005))
            elif tip is not None:
                pose_cb(R.pose_msg(tip[0], tip[1], t_ns[0], 0.0005))
            native = cv2.resize(cv2.imread(self._files[k]), (R.NATIVE, R.NATIVE), interpolation=cv2.INTER_NEAREST)
            msg = R.image_msg(native, t_ns[0])
            img_cb(msg)
            overlay = last_image(node, "/toolhead/overlay")
            status = next((json.loads(m.data) for t, m in reversed(node.published) if t == "/toolhead/status"), {})
            del node.published[:]         # the stub keeps every message otherwise
            margin_overlay = None
            if margin is not None:
                margin_cb(msg)
                margin_overlay = last_image(margin, "/margin/overlay")
                del margin.published[:]
            if keep and overlay is not None:
                with self._lock:
                    self._cache.append((jpeg(overlay), jpeg(margin_overlay), status))
            return status

        def call(name):
            node._services[name](None, types.SimpleNamespace(success=None, message=None))

        for k in range(min(WARMUP_FRAMES, len(self._files))):
            if self._stop:
                return
            feed(k, keep=False)
        call("/toolhead/start")
        for k in range(len(self._files)):
            if self._stop:
                return
            feed(k, self._tips.get(k))
        call("/toolhead/stop")
        self._total = len(self._files) + RELEASE_MAX
        for _ in range(RELEASE_MAX):
            if self._stop:
                return
            if feed(len(self._files) - 1, "off").get("fraction_state") == "final":
                break
        with self._lock:
            self._total = len(self._cache)

    # ------------------------------------------------------------- playback
    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        self._poll.start()

    def close(self):
        self._stop = True
        self._play_timer.stop()
        self._poll.stop()

    def play(self):
        with self._lock:
            n = len(self._cache)
        if self._done and self._index >= n - 1:
            self._index = 0
        self._play_timer.start()
        self.playing_changed.emit(True)

    def pause(self):
        self._play_timer.stop()
        self.playing_changed.emit(False)

    def seek(self, index: int):
        with self._lock:
            n = len(self._cache)
        if n:
            self._index = max(0, min(n - 1, int(index)))
            self._emit()

    def _tick(self):
        with self._lock:
            n = len(self._cache)
        if self._index + 1 < n:
            self._index += 1
            self._emit()
        elif self._done:
            self.pause()
        # else: still processing; wait for the next frame

    def _on_poll(self):
        with self._lock:
            n = len(self._cache)
        if self._error:
            line = f"Offline · {self._name} · Error: {self._error}"
        elif self._done:
            line = f"Offline · {self._name}"
        elif n:
            line = f"Offline · {self._name} · Processing {min(n, len(self._files))} / {len(self._files)}"
        else:
            line = f"Offline · {self._name} · Loading the models"
        if self._margin_note and not self._error:
            line += f" · {self._margin_note}"
        if line != self.label:
            self.label = line
            self.label_changed.emit(line)
        if n and self._shown < 0:
            self._emit()                  # first frame, paused
        elif n and not self._play_timer.isActive():
            self._emit()                  # keeps the scrubber range growing while paused
        if self._done and not self._error:
            self._poll.stop()
            self._emit()

    def _emit(self):
        with self._lock:
            if not self._cache:
                return
            jpg, mjpg, s = self._cache[self._index]
            total = self._total
        decode = lambda b: None if b is None else cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        self._shown = self._index
        self.frame.emit(FrameData(
            streams=[(TOOLHEAD, decode(jpg)), (MARGIN, decode(mjpg))],
            d_mm=None, d_px=s.get("d_px"), warn=False,
            fraction=s.get("fraction"), fraction_state=s.get("fraction_state", "baseline"),
            index=self._index, total=total))
