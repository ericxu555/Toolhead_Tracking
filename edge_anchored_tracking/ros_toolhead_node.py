#!/usr/bin/env python3
"""Real-time cutline-sweep resection mask as a ROS 2 node.

Same pipeline as run_autonomous_toolhead.py, driven by a live camera topic
instead of a folder of frames. The algorithm is unchanged: it was already
causal (TAPNext++ carries state frame to frame, and the mask accumulates
per frame with no lookahead), so nothing needed re-deriving for streaming.

The tumor segmentation, perimeter, tangent and perpendicular-segment geometry
are IMPORTED from cutline_sweep_mask_walk.py and called directly, so this node
runs the same code as the offline pipeline rather than a reimplementation. No
existing file in the repository is modified by adding this node.

    Subscribes  /ves_camera/image            sensor_msgs/Image
    Publishes   /toolhead/overlay            sensor_msgs/Image   mask on the live frame
                /toolhead/mask               sensor_msgs/Image   mono8, 255 = resected
                /toolhead/tip                geometry_msgs/PointStamped
    Services    /toolhead/start              std_srvs/srv/Trigger
                /toolhead/stop               std_srvs/srv/Trigger

One start/stop cycle is one cut. On stop, the episode's final images are
written and the node resets, ready for the next cut.

    ros2 run ... ros_toolhead_node.py          # or: python3 ros_toolhead_node.py
    ros2 service call /toolhead/start std_srvs/srv/Trigger
    ...perform the cut...
    ros2 service call /toolhead/stop  std_srvs/srv/Trigger

Written for ROS 2 Jazzy; uses only stable rclpy APIs.

Episode lifecycle
-----------------
IDLE     -> nothing is processed; frames are ignored.
SEEDING  -> after start. Buffers frames until the tip detector commits to a
            location (auto_seed_tip's plurality vote needs a few frames),
            then refines that point once with SAM2 and begins tracking.
            Typically well under a second at 30 fps.
TRACKING -> steady state. Every processed frame updates the tip, rebuilds the
            frame polygon, accumulates the mask, and publishes.

Realtime policy
---------------
Frames arriving while a frame is still being processed are dropped, so the
overlay stays aligned with what the camera sees rather than falling behind.
The node reports the drop rate on stop.
"""
import os
import sys
import time
import json
import tempfile
import threading
from datetime import datetime

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PointStamped
from std_srvs.srv import Trigger

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for p in (REPO_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_module_functions(path, module_name):
    """Load a script module's functions without running its script preamble.

    Several files in this repository are written as command-line tools: at
    module level they read sys.argv, create directories, or open model
    sessions. Executing those statements here would fail (there is no argv) or
    have side effects (a stray output directory, a duplicate ONNX session).

    Only the top-level statements with those side effects are dropped. Every
    function, class and constant is compiled and executed normally, so the
    logic still lives in its original file and any change there is picked up
    automatically -- nothing is copied into this node.
    """
    import ast
    import types

    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=path)
    kept = []
    dropped_names = set()
    for node in tree.body:
        seg = ast.get_source_segment(source, node) or ""
        drop = False
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and (
                "sys.argv" in seg or "ort.InferenceSession" in seg):
            drop = True
        elif isinstance(node, ast.Expr) and "os.makedirs" in seg:
            drop = True
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            # Also drop assignments that depend on an already-dropped name,
            # e.g. `DEBUG_DUMP_DIR = out_dir` where out_dir came from argv.
            used = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            if used & dropped_names:
                drop = True
        if drop:
            for t in getattr(node, "targets", []) or ([node.target] if
                                                      getattr(node, "target", None) else []):
                if isinstance(t, ast.Name):
                    dropped_names.add(t.id)
            continue
        kept.append(node)
    tree.body = kept

    mod = types.ModuleType(module_name)
    mod.__file__ = path
    exec(compile(tree, path, "exec"), mod.__dict__)
    return mod


# Tumor segmentation, perimeter, tangent and perpendicular-segment geometry,
# taken from the offline pipeline so both run the same code.
G = load_module_functions(
    os.path.join(HERE, "cutline_sweep_mask_walk.py"), "cutline_geometry")

# Mask-construction parameters. These mirror the offline defaults; see the
# README's "Mask parameters" table.
TIP_SMOOTH = 15          # frames of moving average on the tip trajectory
MASK_EMA = 0.5           # tumor-probability smoothing; 1.0 disables
DILATE = 0               # optional dilation of the displayed mask, in pixels

# Seeding parameters, matching auto_seed_tip.py.
SEED_MAX_FRAMES = 30     # give up and report failure after this many frames
SEED_MIN_VOTES = 5       # plurality-vote window

DEFAULT_IMAGE_TOPIC = "/ves_camera/image"

# Tool-button trigger. The Virtuoso publishes its joint state continuously; the
# entry named TOOL_JOINT_NAME reads 1.0 while the cut button is held and 0.0
# when it is released, so the button itself can bracket an episode instead of
# the start/stop services.
DEFAULT_TOOL_TOPIC = "/ves/right/joint/measured_jp"
TOOL_JOINT_NAME = "tool"
TOOL_PRESSED = 0.5        # midpoint of the 0.0/1.0 signal
TOOL_RELEASE_HOLD = 0.5   # seconds the button must read released before ending
DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "Output")

# The models were trained on 540x540 frames, while /ves_camera/image publishes
# 1080x1080 (exactly 2x). Every frame is downscaled to PROC_SIZE before any
# processing, so the input distribution, the tip-smoothing window, the SAM2
# skeleton walk and the mask-shape gates all see the scale they were tuned at.
# Overlays are published back at the camera's native resolution.
PROC_SIZE = 540

IDLE, SEEDING, TRACKING = "idle", "seeding", "tracking"


class ToolheadNode(Node):

    def __init__(self):
        super().__init__("toolhead_tracking")

        self.declare_parameter("image_topic", DEFAULT_IMAGE_TOPIC)
        self.declare_parameter("output_root", DEFAULT_OUTPUT_ROOT)
        self.declare_parameter("resect_side", "right")
        self.declare_parameter("refine", True)
        self.declare_parameter("publish_mask", True)
        self.declare_parameter("publish_tip", True)
        # "service": episodes are bracketed by /toolhead/start and
        # /toolhead/stop, which is convenient for testing. "tool": episodes
        # follow the Virtuoso's cut button, so a resection is captured without
        # anyone touching the terminal.
        self.declare_parameter("trigger_mode", "service")
        self.declare_parameter("tool_topic", DEFAULT_TOOL_TOPIC)

        self.image_topic = self.get_parameter("image_topic").value
        self.output_root = self.get_parameter("output_root").value
        self.resect_side = self.get_parameter("resect_side").value
        self.use_refine = bool(self.get_parameter("refine").value)
        self.trigger_mode = str(self.get_parameter("trigger_mode").value).lower()
        self.tool_topic = self.get_parameter("tool_topic").value

        self.declare_parameter("proc_size", PROC_SIZE)
        self.proc_size = int(self.get_parameter("proc_size").value)

        # Keep only the newest frame, so a slow processing pass cannot build a
        # backlog that delays the overlay. RELIABLE matches the autocropper
        # publisher; a BEST_EFFORT subscriber would still connect to a RELIABLE
        # publisher, but matching avoids any ambiguity.
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)

        self.pub_overlay = self.create_publisher(Image, "/toolhead/overlay", 10)
        self.pub_mask = (self.create_publisher(Image, "/toolhead/mask", 10)
                         if self.get_parameter("publish_mask").value else None)
        self.pub_tip = (self.create_publisher(PointStamped, "/toolhead/tip", 10)
                        if self.get_parameter("publish_tip").value else None)

        # The services stay available in both modes: they remain the way to
        # test without the robot, and a stop is still useful as an override.
        self.create_service(Trigger, "/toolhead/start", self._on_start)
        self.create_service(Trigger, "/toolhead/stop", self._on_stop)

        self._tool_held = False        # last debounced button state
        self._tool_release_t = None    # when the button first read released
        self._tool_seen = False        # has any joint message arrived
        self._tool_ignore = False      # ignore a press held at startup
        if self.trigger_mode == "tool":
            # Joint state is a continuous stream where only the newest value
            # matters, so a depth-1 BEST_EFFORT queue is enough and cannot
            # build a backlog.
            jqos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(JointState, self.tool_topic,
                                     self._on_joint, jqos)
            self.get_logger().info(
                f"Trigger: tool button on {self.tool_topic} "
                f"(joint '{TOOL_JOINT_NAME}'). Hold to cut; release to save.")
        else:
            self.get_logger().info(
                "Trigger: /toolhead/start and /toolhead/stop services.")

        self.state = IDLE
        self.lock = threading.Lock()
        self.busy = False
        self.n_received = 0
        self._noframe_timer = None
        self._refine_mod = None      # SAM2 refinement module, loaded on first use
        self.n_processed = 0
        self.n_dropped = 0

        self._reset_episode_state()

        self.get_logger().info("Loading models...")
        self._load_models()
        self.get_logger().info("Models loaded.")

        self.create_subscription(Image, self.image_topic, self._on_image, qos)
        self.get_logger().info(
            f"Listening on {self.image_topic}. Waiting for /toolhead/start.")

    # ---------------------------------------------------------------- models

    def _load_models(self):
        """Load the tumor segmentation engine, tip detector and tracker once,
        so starting an episode costs nothing but seeding."""
        from src.inference_engine import InferenceEngine
        self.engine = InferenceEngine(
            model_path=G.SPATIAL_MODEL, metadata_path=G.SPATIAL_META, device="cuda")

        import onnxruntime as ort
        kp = os.environ.get("KEYPOINT_MODEL",
                            os.path.join(REPO_ROOT, "checkpoints_keypoint", "keypoint_rcnn.onnx"))
        if not os.path.isfile(kp):
            raise SystemExit(
                f"keypoint_rcnn.onnx not found at {kp}. Download it (README step 4b) "
                f"or set KEYPOINT_MODEL.")
        # Prefer CUDA, but fall back cleanly. onnxruntime-gpu builds are tied to
        # a specific CUDA major version, so a package built for CUDA 13 will not
        # load against a CUDA 12 toolkit -- it reports a missing libcublasLt and
        # silently drops to CPU. The detector only runs during seeding, not per
        # frame, so CPU is workable; the log line says which is in use.
        avail = ort.get_available_providers()
        want = [p for p in ("CUDAExecutionProvider",) if p in avail] + ["CPUExecutionProvider"]
        try:
            self.kp_sess = ort.InferenceSession(kp, providers=want)
        except Exception:
            self.kp_sess = ort.InferenceSession(kp, providers=["CPUExecutionProvider"])
        active = self.kp_sess.get_providers()[0]
        if active != "CUDAExecutionProvider":
            self.get_logger().warn(
                "Tip detector running on CPU. Seeding will take a few seconds "
                "longer per episode. To use the GPU, install an onnxruntime-gpu "
                "build matching your CUDA version (torch reports it via "
                "torch.version.cuda).")
        else:
            self.get_logger().info("Tip detector on CUDA.")

        # The tip-selection logic (clustering, strong-partner override,
        # plurality vote), loaded without auto_seed_tip.py's script preamble --
        # that would otherwise open a second copy of the ONNX detector.
        self.seed_mod = load_module_functions(
            os.path.join(HERE, "auto_seed_tip.py"), "auto_seed_functions")

        # Import TAPNextPP from its own package rather than from
        # tapnextpp_hybrid.py: that file runs a full tracking pass at import
        # time, since it is written as a script.
        import torch
        import pathlib
        import urllib.request
        from tapnet.tapnextpp.votsp2026.model import TAPNextPP

        cache = pathlib.Path.home() / ".cache" / "tapnextpp"
        cache.mkdir(parents=True, exist_ok=True)
        ckpt = cache / "tapnextpp_512.ckpt"
        if not ckpt.exists():
            self.get_logger().info("Downloading TAPNext++ checkpoint (first run only)...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt",
                str(ckpt))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Built once and reused across episodes. _reset_episode_state() clears
        # self.tracker between cuts, so the loaded model is kept separately
        # here; rebuilding it per episode would reload the checkpoint.
        self.tracker_model = TAPNextPP.from_checkpoint(
            str(ckpt), device=device, input_resolution=512)
        self.get_logger().info(f"TAPNext++ loaded on {device}")

    # ------------------------------------------------------------- lifecycle

    def _reset_episode_state(self):
        self.H = self.W = None
        self.seed_buffer = []          # (frame, labels, scores, kps) while seeding
        self.seed_votes = []
        self.tracker = None
        self.track_state = None
        self.tip_history = []          # raw tips, for the smoothing window
        self.cutline_x = None
        # Ordered trace of tip positions. cutline_x holds one x per image
        # row, which cannot represent sideways motion (successive samples
        # land on the same row and overwrite each other) and never joins
        # rows written on different passes, leaving a hard horizontal step
        # where two passes meet. The polygon's edge follows this trace
        # instead; cutline_x is still kept and still backs every other use.
        self.cut_path = []
        self.prev_tip_y = None
        self.keep_side = None      # which side of the cut is resected
        self.keep_seed = None      # a point known to be on that side
        self.start_stub = None     # start-end barrier anchor, set once
        self.frontier_y = -1
        self.accum_mask = None
        self.ema_mask_prob = None
        self.last_angle_deg = 0.0
        self.case_decided = False
        self.cut_start_y = None
        self.last_frame = None
        self.last_overlay = None
        self.episode_dir = None
        self.t_start = None

    def _on_start(self, request, response):
        ok, msg = self._begin_episode()
        response.success = ok
        response.message = msg
        return response

    def _begin_episode(self):
        """Start an episode. Shared by the /toolhead/start service and by the
        tool-button trigger, so both paths run identical logic."""
        with self.lock:
            if self.state != IDLE:
                return False, f"Already running (state={self.state}). Call stop first."
            self._reset_episode_state()
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            # Count only existing episode directories. The conditional must
            # guard the listdir alone: written as a trailing if/else on the
            # whole expression it binds to `1 + len(...)`, which is wrong.
            existing = ([d for d in os.listdir(self.output_root)
                         if d.startswith("episode_")]
                        if os.path.isdir(self.output_root) else [])
            n = len(existing) + 1
            self.episode_dir = os.path.join(self.output_root, f"episode_{n:03d}_{stamp}")
            os.makedirs(self.episode_dir, exist_ok=True)
            self.state = SEEDING
            self.n_received = self.n_processed = self.n_dropped = 0
            self.t_start = time.time()
        self.get_logger().info(f"Episode started -> {self.episode_dir}")
        # Seeding is driven entirely by the image callback, so if the camera is
        # not publishing the node waits in silence with nothing on the overlay.
        # This one-shot check says so rather than leaving it ambiguous.
        self._noframe_timer = self.create_timer(3.0, self._warn_if_no_frames)
        return True, f"Started. Output: {self.episode_dir}"

    def _on_joint(self, msg):
        """Bracket episodes with the Virtuoso's cut button.

        The joint state carries several entries; only TOOL_JOINT_NAME reports
        the button, reading 1.0 while it is held. Pressing starts an episode
        (seeding and SAM2 refinement run then, so the first moment of the cut
        overlaps seeding) and releasing ends it and writes the mask. Each press
        is a separate episode with its own directory.

        A press acts immediately, but a release must persist for
        TOOL_RELEASE_HOLD before the episode ends: the signal can dip for a
        sample or two mid-cut, and acting on that would split one resection
        into several partial masks.
        """
        try:
            idx = list(msg.name).index(TOOL_JOINT_NAME)
        except ValueError:
            if not self._tool_seen:
                self._tool_seen = True
                self.get_logger().error(
                    f"No joint named '{TOOL_JOINT_NAME}' on {self.tool_topic}. "
                    f"Found: {list(msg.name)}. The tool trigger is inactive.")
            return
        if idx >= len(msg.position):
            return

        held = float(msg.position[idx]) >= TOOL_PRESSED

        if not self._tool_seen:
            self._tool_seen = True
            self.get_logger().info(
                f"Tool button detected on {self.tool_topic} "
                f"(currently {'held' if held else 'released'}).")
            if held:
                # Already held at startup: how much has been cut is unknown, so
                # wait for a fresh press rather than begin mid-resection. The
                # held flag stays False, so the eventual release ends nothing.
                self._tool_ignore = True
                self.get_logger().warn(
                    "Button already held at startup; waiting for it to be "
                    "released before the first episode.")
                return

        if self._tool_ignore:
            # Still waiting out the press that was active at startup.
            if held:
                return
            self._tool_ignore = False
            self.get_logger().info("Button released; ready for the next cut.")
            return

        if held:
            self._tool_release_t = None
            if not self._tool_held:
                self._tool_held = True
                ok, message = self._begin_episode()
                self.get_logger().info(f"Tool pressed -> {message}")
            return

        # Released. Wait out the hold before ending, so a brief dropout in the
        # signal does not cut the episode short.
        if self._tool_held:
            now = time.time()
            if self._tool_release_t is None:
                self._tool_release_t = now
                return
            if now - self._tool_release_t < TOOL_RELEASE_HOLD:
                return
            self._tool_held = False
            self._tool_release_t = None
            ok, message = self._end_episode()
            self.get_logger().info(f"Tool released -> {message}")

    def _warn_if_no_frames(self):
        """Fires 3s after start; warns if the camera has delivered nothing."""
        if self._noframe_timer is not None:
            self._noframe_timer.cancel()
            self._noframe_timer = None
        with self.lock:
            state, received = self.state, self.n_received
        if state != IDLE and received == 0:
            self.get_logger().warn(
                f"No frames received on {self.image_topic} since start. "
                f"Is the camera publishing? Check with: ros2 topic hz {self.image_topic}")

    def _on_stop(self, request, response):
        ok, msg = self._end_episode()
        response.success = ok
        response.message = msg
        return response

    def _end_episode(self):
        """End an episode and write its outputs. Shared by the /toolhead/stop
        service and by the tool-button trigger."""
        with self.lock:
            if self.state == IDLE:
                return False, "Not running."
            state_was = self.state
            self.state = IDLE
        if self._noframe_timer is not None:
            self._noframe_timer.cancel()
            self._noframe_timer = None

        if state_was == SEEDING or self.accum_mask is None:
            msg = "Stopped during seeding; no mask was produced."
            self.get_logger().warn(msg)
            self._reset_episode_state()
            return True, msg

        paths = self._write_episode_outputs()
        dt = time.time() - self.t_start
        summary = (f"Episode ended. {self.n_processed} frames processed, "
                   f"{self.n_dropped} dropped, {dt:.1f}s. Wrote: {paths}")
        self.get_logger().info(summary)
        self._reset_episode_state()
        return True, summary

    def _write_episode_outputs(self):
        """Write the episode's final images, matching the offline pipeline."""
        d = self.episode_dir
        out = []

        mask_u8 = (self.accum_mask.astype(np.uint8) * 255)
        p = os.path.join(d, "final_mask.png")
        cv2.imwrite(p, mask_u8); out.append("final_mask.png")

        if self.last_overlay is not None:
            p = os.path.join(d, "final_overlay.png")
            cv2.imwrite(p, self.last_overlay); out.append("final_overlay.png")

        # Cutline-only view: the accumulated tip trajectory drawn on the last
        # frame, with no mask fill. Same purpose as cutline_only.mp4 offline --
        # the quickest way to see whether tracking stayed on the right tool.
        if self.last_frame is not None:
            vis = self.last_frame.copy()
            pts = [(int(round(x)), int(round(y))) for (x, y) in self.tip_history]
            for a, b in zip(pts, pts[1:]):
                cv2.line(vis, a, b, (255, 0, 255), 2, cv2.LINE_AA)
            if pts:
                cv2.circle(vis, pts[-1], 5, (255, 0, 255), -1)
            p = os.path.join(d, "cutline_overlay.png")
            cv2.imwrite(p, vis); out.append("cutline_overlay.png")

        with open(os.path.join(d, "episode_info.json"), "w") as f:
            json.dump({
                "frames_processed": self.n_processed,
                "frames_dropped": self.n_dropped,
                "duration_sec": round(time.time() - self.t_start, 2),
                "resect_side": self.resect_side,
                "refined_seed_used": self.use_refine,
                "final_mask_px": int(self.accum_mask.sum()),
            }, f, indent=2)
        out.append("episode_info.json")
        return ", ".join(out)

    # ------------------------------------------------------------- callbacks

    def _on_image(self, msg):
        self.n_received += 1
        with self.lock:
            if self.state == IDLE:
                return
            if self.busy:
                # Stay live rather than queueing: the newest frame matters more
                # than completeness for an overlay the surgeon is watching.
                self.n_dropped += 1
                return
            self.busy = True
        try:
            native = self._to_bgr(msg)
            self.native_hw = native.shape[:2]
            # Downscale to the resolution the models were trained at. All
            # processing, geometry and stored masks live in this space.
            if native.shape[0] != self.proc_size or native.shape[1] != self.proc_size:
                frame = cv2.resize(native, (self.proc_size, self.proc_size),
                                   interpolation=cv2.INTER_AREA)
            else:
                frame = native
            if self.state == SEEDING:
                self._step_seeding(frame, msg.header)
            elif self.state == TRACKING:
                self._step_tracking(frame, msg.header)
        except Exception as e:  # never let one bad frame kill the node
            self.get_logger().error(f"Frame failed: {e}")
        finally:
            self.busy = False

    def _to_bgr(self, msg):
        """Convert sensor_msgs/Image to BGR without requiring cv_bridge."""
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

    # --------------------------------------------------------------- seeding

    def _publish_status(self, frame, header, text, candidate=None, seed=None):
        """Publish the live frame with a status banner during seeding.

        Nothing is published between /toolhead/start and the first tracked
        frame otherwise, so a blank viewer cannot be told apart from a camera
        that is not publishing, or from a node stuck in SAM2 refinement. This
        keeps the overlay live and says what the node is waiting for.
        """
        if self.pub_overlay is None:
            return
        vis = frame.copy()
        # Candidate detections are drawn hollow, the committed seed filled, so
        # the moment of commitment is visible.
        if candidate is not None:
            cv2.circle(vis, (int(candidate[0]), int(candidate[1])), 7, (0, 255, 255), 2)
        if seed is not None:
            cv2.circle(vis, (int(seed[0]), int(seed[1])), 8, (0, 255, 0), -1)
            cv2.circle(vis, (int(seed[0]), int(seed[1])), 8, (0, 0, 0), 2)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(vis, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        nh, nw = getattr(self, "native_hw", vis.shape[:2])
        if (nh, nw) != vis.shape[:2]:
            vis = cv2.resize(vis, (nw, nh), interpolation=cv2.INTER_LINEAR)
        self.pub_overlay.publish(self._to_msg(vis, header, "bgr8"))

    def _step_seeding(self, frame, header):
        """Run the tip detector until the plurality vote commits, mirroring
        auto_seed_tip.py's logic on a live stream."""
        A = self.seed_mod
        if self.H is None:
            self.H, self.W = frame.shape[:2]

        labels, scores, kps = self._detect(frame)
        best = A.find_tip(labels, scores, kps) if len(scores) else None

        chosen = None
        if best is not None:
            score, xy = best
            if score >= A.CONFIDENCE_THRESHOLD:
                chosen = xy
            else:
                self.seed_votes.append((len(self.seed_votes), float(score), xy))
                if len(self.seed_votes) >= SEED_MIN_VOTES:
                    win = A._plurality_vote(self.seed_votes)
                    if win is not None:
                        chosen = win[2]

        if chosen is None:
            # Keep the overlay live while seeding, so it is obvious the node is
            # receiving frames and what the detector is currently proposing.
            self._publish_status(
                frame, header,
                f"SEEDING  {len(self.seed_votes)}/{SEED_MAX_FRAMES}  hold the tool still",
                candidate=(best[1] if best is not None else None))
            if len(self.seed_votes) >= SEED_MAX_FRAMES:
                self.get_logger().error(
                    "No confident tip detection in the first "
                    f"{SEED_MAX_FRAMES} frames. Stopping episode; reposition and retry.")
                with self.lock:
                    self.state = IDLE
            return

        seed_xy = np.array(chosen, dtype=np.float32)
        self.get_logger().info(f"Tip seeded at ({seed_xy[0]:.1f}, {seed_xy[1]:.1f})")
        # Show the committed seed before refinement begins, which takes a few
        # seconds and would otherwise look like the node had frozen.
        self._publish_status(frame, header, "SEED FOUND  refining, keep still",
                             seed=seed_xy)

        if self.use_refine:
            refined = self._refine_seed(frame, seed_xy)
            if refined is not None:
                self.get_logger().info(
                    f"Refined to ({refined[0]:.1f}, {refined[1]:.1f}) "
                    f"[moved {np.linalg.norm(refined - seed_xy):.1f}px]")
                seed_xy = refined
            else:
                self.get_logger().warn("Refinement failed its shape check; using raw detection.")

        self._begin_tracking(frame, seed_xy, header)

    def _detect(self, frame_bgr):
        """keypoint_rcnn.onnx, with the same crop/resize recipe as auto_seed_tip."""
        H, W = frame_bgr.shape[:2]
        side = min(H, W)
        cy, cx = (H - side) // 2, (W - side) // 2
        crop = frame_bgr[cy:cy + side, cx:cx + side]
        resized = cv2.resize(crop, (256, 256))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.expand_dims(np.rollaxis(rgb, 2, 0), 0)
        out = self.kp_sess.run(None, {self.kp_sess.get_inputs()[0].name: inp})
        labels, scores, kps = out[1], out[2], out[3]
        xy = kps[:, 0, 0:2] * (side / 256.0)
        xy[:, 0] += cx
        xy[:, 1] += cy
        return labels, scores, xy

    def _refine_seed(self, frame_bgr, seed_xy):
        """SAM2 two-pass refinement. Runs once per episode, on one frame, so
        it costs a few seconds at start and nothing thereafter."""
        if self._refine_mod is None:
            try:
                # Same treatment as the other script modules: this file reads
                # sys.argv and creates its output directory at import time.
                R = load_module_functions(
                    os.path.join(HERE, "refine_tip_with_sam2_v2.py"), "sam2_refine")
                # WALK_PX is defined by one of the stripped argv lines, so
                # restore the default the offline pipeline uses.
                R.WALK_PX = 5
                self._refine_mod = R
            except Exception as e:
                self.get_logger().warn(
                    f"SAM2 refinement unavailable ({e}); using raw detection.")
                self.use_refine = False
                return None
        R = self._refine_mod
        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "frame000001.png")
            cv2.imwrite(fp, frame_bgr)
            seed_json = os.path.join(td, "seed.json")
            with open(seed_json, "w") as f:
                json.dump({"points": [[float(seed_xy[0]), float(seed_xy[1])]],
                           "W": int(self.W), "H": int(self.H), "frame": 0}, f)
            try:
                res = R.refine(td, json.load(open(seed_json)))
            except Exception as e:
                self.get_logger().warn(f"Refinement error ({e}); using raw detection.")
                return None
        if not str(res.get("status", "")).startswith("ok"):
            return None
        return np.array(res["refined_xy"], dtype=np.float32)

    def _begin_tracking(self, frame, seed_xy, header):
        # The model is loaded once at startup; each episode just starts a fresh
        # track from the new seed, carried in self.track_state.
        self.tracker = self.tracker_model
        self.track_state = None
        self.cutline_x = np.full(self.H, np.nan, dtype=np.float32)
        self.cut_path = []
        self.prev_tip_y = None
        self.keep_side = None
        self.keep_seed = None
        self.start_stub = None
        self.accum_mask = np.zeros((self.H, self.W), dtype=bool)
        self.seed_xy = seed_xy
        with self.lock:
            self.state = TRACKING
        self.get_logger().info("Tracking started.")
        self._step_tracking(frame, header, first=True)

    # -------------------------------------------------------------- tracking

    def _step_tracking(self, frame, header, first=False):
        if first:
            pts0 = np.array([self.seed_xy], dtype=np.float32)
            positions, visible, self.track_state = self.tracker.track_frame(
                frame, query_points_xy=pts0)
        else:
            positions, visible, self.track_state = self.tracker.track_frame(
                frame, state=self.track_state)

        tip = np.array(positions[0], dtype=np.float32)
        self.tip_history.append((float(tip[0]), float(tip[1])))

        # Moving average over the last TIP_SMOOTH frames. The offline pipeline
        # centres this window; a live stream has no future, so it trails.
        #
        # This is the ONLY place the two pipelines differ. Every mask-
        # construction step -- the ordered cut path, latest-pass-wins, the arc
        # side constraint, the no-cross rule, notch protection, the hull clip
        # and the component filter -- is the same code or the same logic, and
        # the geometry itself is imported from cutline_sweep_mask_walk.py.
        # Centring the window here would mean holding the mask back by half a
        # window (~7 frames, 0.23 s at 30 Hz) to wait for frames that have not
        # arrived; the trailing average is preferred so the overlay stays live,
        # at the cost of the tip lagging slightly during fast motion. Measured
        # against the offline result on three recorded episodes, the final
        # masks agree at IoU 0.976-0.997.
        w = self.tip_history[-TIP_SMOOTH:]
        tx = float(np.mean([p[0] for p in w]))
        ty = float(np.mean([p[1] for p in w]))

        self._update_mask(frame, (tx, ty))
        self._publish(frame, header, (tx, ty))
        self.n_processed += 1

    def _update_mask(self, frame_bgr, tip_xy):
        """One frame of cutline-sweep accumulation, using the same geometry
        functions as the offline pipeline."""
        H, W = self.H, self.W
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        raw_prob = G.raw_tumor_mask(self.engine, rgb, H, W)
        self.ema_mask_prob = raw_prob if self.ema_mask_prob is None else \
            MASK_EMA * raw_prob + (1 - MASK_EMA) * self.ema_mask_prob
        T = G.largest_component(self.ema_mask_prob > 0.5)
        perim = G.outer_perimeter(T)

        tx, ty = tip_xy
        tyi = int(round(np.clip(ty, 0, H - 1)))
        if not self.cut_path or (abs(self.cut_path[-1][0] - tx) > 1e-6 or
                                 abs(self.cut_path[-1][1] - ty) > 1e-6):
            self.cut_path.append((float(tx), float(ty)))

        # LATEST PASS OWNS THE ROW. Blending the two visits 50/50 strands the
        # row halfway between them: a row crossed on the way out and again on
        # the way back can hold positions over 100 px apart, and the polygon
        # spans cutline -> boundary, so a stale value drags the sweep sideways.
        # The tool's current position is the better description of the cut.
        self.cutline_x[tyi] = tx
        # Fill rows between the previous tip row and this one in BOTH
        # directions, so the path stays continuous when the tool moves back up
        # as well as down. Only downward motion advances the frontier.
        if self.prev_tip_y is not None and self.prev_tip_y != tyi:
            prev_y = self.prev_tip_y
            prev_x = self.cutline_x[prev_y] if not np.isnan(self.cutline_x[prev_y]) else tx
            step = 1 if tyi > prev_y else -1
            for yy in range(prev_y + step, tyi + step, step):
                frac = (yy - prev_y) / float(tyi - prev_y)
                self.cutline_x[yy] = prev_x + frac * (tx - prev_x)
        elif tyi > self.frontier_y:
            prev_y = self.frontier_y if self.frontier_y >= 0 else tyi
            prev_x = self.cutline_x[prev_y] if not np.isnan(self.cutline_x[prev_y]) else tx
            for yy in range(prev_y + 1, tyi + 1):
                frac = (yy - prev_y) / max(tyi - prev_y, 1)
                self.cutline_x[yy] = prev_x + frac * (tx - prev_x)
        if tyi > self.frontier_y:
            self.frontier_y = tyi
        self.prev_tip_y = tyi

        if not self.case_decided and perim is not None and len(perim) >= 3:
            if cv2.pointPolygonTest(perim.reshape(-1, 1, 2), (float(tx), float(ty)), False) >= 0:
                self.case_decided = True
                self.cut_start_y = tyi

        perp_dir = None
        if perim is not None and len(perim) >= 40:
            self.last_angle_deg, perp_dir, _, _ = G.perp_direction_at_tip(perim, (tx, ty))

        frame_mask = np.zeros((H, W), dtype=bool)
        if self.frontier_y >= 0 and perim is not None and perp_dir is not None:
            # Same gate as the offline pipeline: the cut needs at least two
            # visited rows before it defines a polygon edge.
            visited = np.where(~np.isnan(self.cutline_x))[0]
            visited = visited[visited <= tyi]
            if len(visited) >= 2 and len(self.cut_path) >= 2:
                # Which of the cut's two sides is the resection, from
                # connectivity rather than from a direction convention. A
                # continuous curve has exactly two sides however it bends, so
                # this stays correct where a row-indexed or tangent-based test
                # does not -- the horizontal stretch of an inverted-C path, and
                # the Z it becomes when the tool reverses.
                if (self.start_stub is None and len(self.cut_path) >= 2
                        and perim is not None and len(perim) >= 3):
                    # Leave the tumor PERPENDICULAR to its boundary. Using the
                    # cut's own reverse heading sends this stub wherever the
                    # tool happened to be moving when it entered, which can run
                    # parallel to the edge instead of out through it; the
                    # barrier then never crosses the outline at that end and
                    # the two sides stay connected around the cut's start.
                    p0 = np.asarray(self.cut_path[0], dtype=np.float64)
                    pp = np.asarray(perim, dtype=np.float64)
                    kk = int(np.argmin(np.hypot(pp[:, 0] - p0[0],
                                                pp[:, 1] - p0[1])))
                    mm = len(pp)
                    ww = max(int(mm * 0.05), 5)
                    ta = pp[(kk - ww) % mm]
                    tb = pp[(kk + ww) % mm]
                    tv = tb - ta
                    tn = float(np.hypot(tv[0], tv[1]))
                    if tn > 1e-6:
                        nv = np.array([-tv[1], tv[0]]) / tn
                        # Point it OUT of the tumor.
                        probe = p0 + nv * 6.0
                        if cv2.pointPolygonTest(
                                np.asarray(perim, np.int32),
                                (float(probe[0]), float(probe[1])),
                                False) >= 0:
                            nv = -nv
                        self.start_stub = G._ray_to_boundary(
                            p0, nv, perim, float(H + W))
                side_lab = G._side_partition(self.cut_path, (tx, ty), H, W,
                                             perim=perim,
                                             start_stub=self.start_stub)
                if side_lab is not None:
                    # Seeded once from a probe placed on the resection side of
                    # the tip, then re-identified by that stored point every
                    # frame: connectedComponents renumbers labels each frame, so
                    # the seed is what says which label is which.
                    if self.keep_side is None:
                        off = 8.0 if self.resect_side != "left" else -8.0
                        for d in (off, off * 2, off * 4):
                            c = G._side_of(side_lab, (tx + d, ty), H, W)
                            if c:
                                self.keep_side = c
                                self.keep_seed = np.array([tx + d, ty])
                                break
                    elif self.keep_seed is not None:
                        c = G._side_of(side_lab, self.keep_seed, H, W)
                        if c:
                            self.keep_side = c
                seg_end = G.segment_end_on_perimeter(
                    perim, (tx, ty), perp_dir, self.cutline_x,
                    side_lab=side_lab, keep_side=self.keep_side,
                    resect_side=self.resect_side)
                if seg_end is not None:
                    # Contiguous PREFIX of the trace, never a filter: dropping
                    # scattered points splices non-neighbours together and puts
                    # a long straight jump back into the edge.
                    stop = len(self.cut_path)
                    for k in range(len(self.cut_path) - 1, -1, -1):
                        if self.cut_path[k][1] <= ty + 0.5:
                            stop = k + 1
                            break
                    cut_pts = self.cut_path[:stop]
                    if len(cut_pts) < 2:
                        cut_pts = list(self.cut_path)
                    # Close the path on the tool tip itself, so the polygon has
                    # no gap at the cut's leading end.
                    if (abs(cut_pts[-1][0] - tx) > 1e-6 or
                            abs(cut_pts[-1][1] - ty) > 1e-6):
                        cut_pts = cut_pts + [(float(tx), float(ty))]
                    i_end = int(np.argmin(np.linalg.norm(
                        perim - np.array(seg_end, dtype=np.float32), axis=1)))
                    i_start = int(np.argmin(np.linalg.norm(
                        perim - np.array(cut_pts[0], dtype=np.float32), axis=1)))
                    n = len(perim)
                    fwd = (i_start - i_end) % n
                    arc_f = [perim[(i_end + k) % n] for k in range(fwd + 1)]
                    arc_b = [perim[(i_end - k) % n] for k in range(n - fwd + 1)]

                    # ARC SIDE CONSTRAINT. The two candidate arcs wind opposite
                    # ways round the perimeter, and the winding decides which
                    # side of the cut the polygon fills. Picking by proximity
                    # alone let the mask flip to the wrong side of the tool
                    # whenever the tip ran nearer the far boundary. Choose the
                    # arc that keeps the resection on the requested side, with
                    # the sample count in hand so a near-empty arc cannot win
                    # on a noisy fraction.
                    def frac_wrong(arc_pts):
                        wrong = total = 0
                        for ax, ay in arc_pts:
                            r = int(round(min(max(ay, 0), H - 1)))
                            cx = self.cutline_x[r]
                            if np.isnan(cx):
                                continue
                            total += 1
                            if self.resect_side == "left":
                                if ax > cx + 1.0:
                                    wrong += 1
                            elif ax < cx - 1.0:
                                wrong += 1
                        return wrong / max(total, 1)

                    def n_samples(arc_pts):
                        t = 0
                        for _ax, ay in arc_pts:
                            r = int(round(min(max(ay, 0), H - 1)))
                            if not np.isnan(self.cutline_x[r]):
                                t += 1
                        return t

                    short, long_ = (arc_f, arc_b) if len(arc_f) <= len(arc_b)                         else (arc_b, arc_f)
                    ws, wl = frac_wrong(short), frac_wrong(long_)
                    ns, nl = n_samples(short), n_samples(long_)
                    if ns < G.ARC_MIN_SAMPLES or nl < G.ARC_MIN_SAMPLES:
                        arc = short
                    elif ws - wl > G.ARC_SIDE_MARGIN:
                        arc = long_
                    else:
                        arc = short

                    poly = cut_pts + [(float(seg_end[0]), float(seg_end[1]))] +                            [(float(p[0]), float(p[1])) for p in arc]
                    filled = np.zeros((H, W), dtype=np.uint8)
                    cv2.fillPoly(filled, [np.array(poly, np.int32).reshape(-1, 1, 2)], 1)
                    frame_mask = T & filled.astype(bool)
                    # Vetting only the sweep endpoint is not enough: the arc can
                    # wrap the far side of the tumor and fill the unresected
                    # half however the endpoint was chosen.
                    if side_lab is not None and self.keep_side:
                        frame_mask &= ~((side_lab > 0)
                                        & (side_lab != self.keep_side))

        # Monotonic union, re-clipped to the tumor every frame. The clip is
        # load-bearing: it lets a wrongly-included pixel drop back out once
        # the segmenter stops calling it tumor.
        #
        # But T is also false wherever a TOOL occludes tissue -- the segmenter
        # does not label metal as tumor -- and that dropout is temporary while
        # the deletion is permanent, so each pass of a shaft over resected
        # ground ratchets the mask down. The convex hull already bridges the
        # tool notches, so `hull & ~T` is the tool region with no extra
        # detector; the part of it above the tip is tissue the cut has already
        # passed and is exempt from retraction. Below the tip the tool sits on
        # tissue the cut has not reached, which must still be free to clear.
        occluded = np.zeros((H, W), dtype=bool)
        bound = None
        if perim is not None and len(perim) >= 3:
            hull_f = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(hull_f, [perim.astype(np.int32).reshape(-1, 1, 2)], 1)
            bound = hull_f.astype(bool)
            occluded = bound & ~T
            occluded[tyi:, :] = False
        self.accum_mask = (self.accum_mask | frame_mask) & (T | occluded)

        # Never hold mask outside the tumor's own outer boundary. The
        # occlusion exemption is what lets a pixel sit outside T at all, and
        # it keeps re-qualifying while the shaft covers it, so without this
        # the exempt region can drift past the tumor edge as T fluctuates.
        if bound is not None:
            self.accum_mask &= bound

        # KEEP ONLY THE REGION CONNECTED TO THE CUT. The resection grows from
        # the cut path and stays attached to it, so a component touching no
        # part of the cutline came from a bad frame. Accumulation is monotonic,
        # so without this such a blob survives to the end of the episode.
        n_lab, lab = cv2.connectedComponents(
            self.accum_mask.astype(np.uint8), connectivity=8)
        if n_lab > 2:
            keep = set()
            for r in np.where(~np.isnan(self.cutline_x))[0]:
                c = int(round(self.cutline_x[r]))
                for off in (0, -2, 2, -5, 5):
                    cc = c + off
                    if 0 <= cc < W and lab[r, cc]:
                        keep.add(int(lab[r, cc]))
            if keep:
                self.accum_mask = np.isin(lab, list(keep))

    def _publish(self, frame_bgr, header, tip_xy):
        disp = self.accum_mask
        if DILATE > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(DILATE | 1, 3),) * 2)
            disp = cv2.dilate(disp.astype(np.uint8), k).astype(bool) & self.accum_mask

        vis = frame_bgr.copy()
        vis[disp] = (0.5 * np.array([0, 0, 255]) + 0.5 * vis[disp]).astype(np.uint8)
        tx, ty = int(round(tip_xy[0])), int(round(tip_xy[1]))
        cv2.circle(vis, (tx, ty), 5, (255, 0, 255), -1)
        self.last_frame = frame_bgr
        self.last_overlay = vis

        # Publish at the camera's native resolution so the overlay lines up
        # with the raw stream in RViz and for any downstream consumer.
        nh, nw = getattr(self, "native_hw", (vis.shape[0], vis.shape[1]))
        scale = nw / float(self.W)
        if (nh, nw) != vis.shape[:2]:
            pub_vis = cv2.resize(vis, (nw, nh), interpolation=cv2.INTER_LINEAR)
            pub_mask = cv2.resize(disp.astype(np.uint8) * 255, (nw, nh),
                                  interpolation=cv2.INTER_NEAREST)
        else:
            pub_vis, pub_mask = vis, disp.astype(np.uint8) * 255

        self.pub_overlay.publish(self._to_msg(pub_vis, header, "bgr8"))
        if self.pub_mask is not None:
            self.pub_mask.publish(self._to_msg(pub_mask, header, "mono8"))
        if self.pub_tip is not None:
            pt = PointStamped()
            pt.header = header
            # Report the tip in the camera's native pixel coordinates.
            pt.point.x = float(tip_xy[0]) * scale
            pt.point.y = float(tip_xy[1]) * scale
            pt.point.z = 0.0
            self.pub_tip.publish(pt)

    def _to_msg(self, img, header, encoding):
        msg = Image()
        msg.header = header
        msg.height, msg.width = img.shape[0], img.shape[1]
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = img.shape[1] * (3 if encoding == "bgr8" else 1)
        msg.data = img.tobytes()
        return msg


def main():
    rclpy.init()
    node = ToolheadNode()
    os.makedirs(node.output_root, exist_ok=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
