"""Live source: the toolhead node's topics, read with rclpy on a background thread.

    /toolhead/overlay   sensor_msgs/Image    the frame with the swept-region tint and tip dot
    /toolhead/status    std_msgs/String      JSON readouts (distance, fraction, episode, calibration)
    /margin/overlay     sensor_msgs/Image    the margin model's frame, shown in the Both view

Run the toolhead node with -p show_distance_overlay:=false, so the distance is
not also burned into the video; the card shows it. Kept apart from sources.py
so demo and offline never import ROS.
"""
import json
import threading

import cv2
import numpy as np

from .data import FrameData
from .sources import MARGIN, TOOLHEAD, FrameSource

OVERLAY_TOPIC = "/toolhead/overlay"
STATUS_TOPIC = "/toolhead/status"
MARGIN_TOPIC = "/margin/overlay"


def image_to_bgr(msg) -> np.ndarray:
    """sensor_msgs/Image to BGR without cv_bridge, as the node does."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc in ("bgr8", "rgb8"):
        img = buf.reshape(msg.height, msg.width, 3)
        return img[:, :, ::-1].copy() if enc == "rgb8" else img.copy()
    if enc == "mono8":
        return cv2.cvtColor(buf.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
    if enc in ("bgra8", "rgba8"):
        img = buf.reshape(msg.height, msg.width, 4)[:, :, :3]
        return img[:, :, ::-1].copy() if enc == "rgba8" else img.copy()
    raise ValueError(f"Unsupported encoding: {msg.encoding}")


def header_line(status: dict) -> str:
    parts = ["Live"]
    n = status.get("episode", 0)
    parts.append(f"Episode {n}" if n else "Waiting for the first cut")
    if status.get("calibration"):
        parts.append(f"Calibration {status['calibration']}")
    return " · ".join(parts)


class LiveSource(FrameSource):
    """Emits one FrameData per toolhead overlay, with the newest status and margin frame."""
    has_timeline = False
    label = "Live · Waiting for the toolhead node"

    def __init__(self, overlay_topic=OVERLAY_TOPIC, status_topic=STATUS_TOPIC, margin_topic=MARGIN_TOPIC):
        super().__init__()
        self._topics = (overlay_topic, status_topic, margin_topic)
        self._status = {}
        self._margin = None             # newest margin frame, only kept in the Both view
        self._want_margin = False
        self._lock = threading.Lock()
        self._node = self._executor = self._thread = None

    # ---- ROS plumbing (not exercised without ROS 2)
    def start(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image
        from std_msgs.msg import String

        rclpy.init()
        self._node = rclpy.create_node("toolhead_ui")
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        overlay, status, margin = self._topics
        self._node.create_subscription(Image, overlay, self.on_overlay, qos)
        self._node.create_subscription(String, status, self.on_status, qos)
        self._node.create_subscription(Image, margin, self.on_margin, qos)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def close(self):
        if self._executor is not None:
            import rclpy
            self._executor.shutdown()
            self._node.destroy_node()
            rclpy.try_shutdown()
            self._executor = None

    # ---- callbacks (ROS thread). Signals cross into the Qt thread queued.
    def set_view(self, view: str):
        with self._lock:
            self._want_margin = view == "both"
            if not self._want_margin:
                self._margin = None

    def on_status(self, msg):
        status = json.loads(msg.data)
        line = header_line(status)
        with self._lock:
            changed = line != self.label
            self._status, self.label = status, line
        if changed:
            self.label_changed.emit(line)

    def on_margin(self, msg):
        with self._lock:
            if self._want_margin:
                self._margin = image_to_bgr(msg)

    def on_overlay(self, msg):
        frame = image_to_bgr(msg)
        with self._lock:
            s, margin = dict(self._status), self._margin
        self.frame.emit(FrameData(
            streams=[(TOOLHEAD, frame), (MARGIN, margin)],
            d_mm=s.get("d_mm"), d_px=s.get("d_px"),
            warn_mm=s.get("warn_mm") or 2.0, warn=bool(s.get("warn")),
            fraction=s.get("fraction"), fraction_state=s.get("fraction_state", "baseline")))
