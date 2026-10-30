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
                /toolhead/status             std_msgs/String     JSON readouts for the operator UI
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

Distance display
----------------
The published overlay carries "d = NN px" at the top left: the distance from
the smoothed tip to the nearest point of the tumor's convex hull, the same
point the sweep's edge is oriented from. Units are processing-frame pixels
(proc_size, 540 by default). With a visual-servoing tip it reads "d = N.N mm"
instead: the pixel distance times the tip depth over the focal length, so it is
exact only at the tip's depth. "--" when there is no hull to measure against.

Between cuts
------------
IDLE frames are no longer ignored: the live frame is published on the overlay,
the tumor is segmented to keep a baseline for the resected fraction, and after
a release the final fraction is measured once the tip has left the tumor (see
BASELINE_FRAMES). The calibration is re-read at every press.
Drawn on the published copy only -- never on final_overlay.png, the mask, or
any saved raw frame. -p show_distance_overlay:=false turns it off.

Experiment logging
------------------
-p experiment:=true writes a session folder of per-frame logs, raw frames and
the displayed video for offline analysis; see experiment_log.py. Off by
default, and it changes nothing the node computes.
"""
import os
import sys
import time
import json
import tempfile
import threading
from collections import deque
from datetime import datetime

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import (PointStamped, PoseStamped,
                               PoseWithCovarianceStamped)
from std_msgs.msg import String
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
MASK_EMA = 0.3           # tumor-probability smoothing; 1.0 disables
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

# Visual-servoing tip pose. The smoother publishes the tool tip already
# expressed in the CAMERA frame, in metres, so it only has to be projected
# through the camera's intrinsics to land on a pixel -- no hand-eye transform
# is involved.
DEFAULT_POSE_TOPIC = "/ves_smoother/right/tip_pose"

# Calibration is redone periodically and each run writes its own folder, so the
# intrinsics must be read at startup rather than baked in. A stale matrix would
# silently put the seed on the wrong tissue.
DEFAULT_CALIB_ROOT = os.path.expanduser("~/aliss_runs")
CALIB_RELPATH = os.path.join("calibration", "camera", "ves_camera.yaml")
TOOL_JOINT_NAME = "tool"
TOOL_PRESSED = 0.5        # midpoint of the 0.0/1.0 signal
TOOL_RELEASE_HOLD = 0.5   # seconds the button must read released before ending

# trigger_mode:=autonomous. The action-chunking policy asserts the tool signal
# in bursts with the signal low between chunks, so the release hold that works
# for a human press (a sample or two of dropout) would end the episode at every
# chunk boundary. This is the same idea with a horizon long enough to bridge a
# whole gap: set it above the longest pause the policy leaves, and below the
# time you would wait before deciding the run is over.
AUTONOMOUS_IDLE_TIMEOUT = 10.0
DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "Output")

# The models were trained on 540x540 frames, while /ves_camera/image publishes
# 1080x1080 (exactly 2x). Every frame is downscaled to PROC_SIZE before any
# processing, so the input distribution, the tip-smoothing window, the SAM2
# skeleton walk and the mask-shape gates all see the scale they were tuned at.
# Overlays are published back at the camera's native resolution.
PROC_SIZE = 540

IDLE, SEEDING, TRACKING = "idle", "seeding", "tracking"

# Resected-fraction display (published on /toolhead/status for the operator UI).
# Between cuts the node keeps segmenting the tumor: a rolling average of the
# last BASELINE_FRAMES frames with the tool out of view is the baseline T0,
# frozen when the button is pressed. During the cut the fraction is an
# ESTIMATE, |A ∩ T0| / |T0|, approximate because the tumor deforms. After the
# release, once the tip has left the tumor, FINAL_FRAMES frames are averaged
# into T_end and the FINAL fraction is |A| / (|A| + |T_end \ A|), measured with
# nothing blocking the view. If the tip never leaves within FINAL_TIMEOUT the
# estimate stands. Display and logging only: nothing here feeds the mask.
BASELINE_FRAMES = 5
FINAL_FRAMES = 5
FINAL_TIMEOUT = 5.0       # seconds after release
FRAC_BASELINE, FRAC_BLOCKED, FRAC_READY = "baseline", "blocked", "ready"
FRAC_ESTIMATE, FRAC_FINAL, FRAC_NONE = "estimate", "final", "none"


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
        self.declare_parameter("autonomous_idle_timeout",
                               AUTONOMOUS_IDLE_TIMEOUT)
        self.declare_parameter("tool_topic", DEFAULT_TOOL_TOPIC)
        # Where the per-frame tool tip comes from.
        #   detector  -- keypoint R-CNN + SAM2 seed, then TAPNext++ (default)
        #   pose_seed -- tip_pose seeds the track, TAPNext++ still tracks
        #   pose_only -- tip_pose drives the cutline directly, no tracker
        self.declare_parameter("tip_source", "detector")
        self.declare_parameter("pose_topic", DEFAULT_POSE_TOPIC)
        self.declare_parameter("calib_root", DEFAULT_CALIB_ROOT)
        self.declare_parameter("calib_file", "")
        # "raw" uses camera_matrix + distortion, matching /ves_camera/image.
        # "rect" uses projection_matrix with no distortion, for a rectified
        # stream. The two focal lengths differ by ~23% here, so this must
        # match whatever the image topic actually carries.
        self.declare_parameter("camera_model", "raw")
        # Live tip-to-boundary distance on the published overlay. Read again at
        # each episode start, so a study can switch it between trials with
        # `ros2 param set`. warn_distance_px > 0 turns the readout red below
        # that distance; 0 leaves it white.
        self.declare_parameter("show_distance_overlay", True)
        self.declare_parameter("warn_distance_px", 0.0)
        # With a visual-servoing tip the distance is also reported in mm (tip
        # depth from the pose, focal length from the calibration), and this
        # threshold replaces warn_distance_px. 0 disables the warning.
        self.declare_parameter("warn_distance_mm", 2.0)
        # Frames in the trailing moving average on the tip. The vision tip
        # needs it; the visual-servoing pose is already filtered by the smoother,
        # where a 15-frame window at ~6 Hz only adds ~1.2 s of lag (1 = none).
        self.declare_parameter("tip_smooth", TIP_SMOOTH)
        # Experiment logging (experiment_log.py). Off by default: a normal run
        # writes exactly what it always has.
        self.declare_parameter("experiment", False)
        self.declare_parameter("snapshot_every", 15)
        self.declare_parameter("raw_png_compression", 1)
        # Study metadata for the episode's meta.json, read at each episode
        # start so it can be set per trial. trial < 0 numbers trials by
        # episode order within the session.
        self.declare_parameter("tumor_id", "")
        self.declare_parameter("condition", "")
        self.declare_parameter("operator", "")
        self.declare_parameter("trial", -1)

        self.image_topic = self.get_parameter("image_topic").value
        self.tip_smooth = max(1, int(self.get_parameter("tip_smooth").value))
        self.output_root = self.get_parameter("output_root").value
        self.resect_side = self.get_parameter("resect_side").value
        self.use_refine = bool(self.get_parameter("refine").value)
        self.trigger_mode = str(self.get_parameter("trigger_mode").value).lower()
        self.autonomous_idle = float(
            self.get_parameter("autonomous_idle_timeout").value)
        if self.trigger_mode not in ("service", "tool", "autonomous"):
            raise SystemExit(
                f"trigger_mode must be service, tool or autonomous "
                f"(got {self.trigger_mode!r})")
        self.tool_topic = self.get_parameter("tool_topic").value
        self.tip_source = str(self.get_parameter("tip_source").value).lower()
        self.pose_topic = self.get_parameter("pose_topic").value
        self.calib_root = self.get_parameter("calib_root").value
        self.calib_file = self.get_parameter("calib_file").value
        self.camera_model = str(self.get_parameter("camera_model").value).lower()
        if self.tip_source not in ("detector", "pose_seed", "pose_only"):
            raise SystemExit(f"tip_source must be detector, pose_seed or "
                             f"pose_only (got {self.tip_source!r})")
        if self.camera_model not in ("raw", "rect"):
            raise SystemExit(f"camera_model must be raw or rect "
                             f"(got {self.camera_model!r})")

        self.declare_parameter("proc_size", PROC_SIZE)
        self.proc_size = int(self.get_parameter("proc_size").value)

        self.show_distance_overlay = bool(
            self.get_parameter("show_distance_overlay").value)
        self.warn_distance_px = float(
            self.get_parameter("warn_distance_px").value)
        self.warn_distance_mm = float(
            self.get_parameter("warn_distance_mm").value)
        # Display and logging only: written by _update_mask, never read back
        # by any tracking or mask step.
        self.last_hull_pt = None       # nearest hull point to the tip
        self.last_hull_dist = None     # its distance, processing px
        self.last_hull_dist_mm = None  # the same in mm, when a pose gives the depth
        # Resected fraction (see BASELINE_FRAMES). Session state: survives
        # episodes, so the baseline captured between cuts is there at a press.
        self.episode_n = 0
        self.baseline_probs = deque(maxlen=BASELINE_FRAMES)
        self.T0 = None                 # baseline tumor mask, frozen at the press
        self.fraction = None
        self.fraction_state = FRAC_BASELINE
        self.post = None               # pending final measurement after a release
        self.last_seg_end = None       # where this frame's sweep edge ended
        self.last_T = None             # this frame's tumor mask
        self.last_perim = None         # this frame's hull perimeter
        self.last_d_text = ""
        self.last_warn_state = ""
        self.last_pose_cov = None      # experiment log only
        self._xrec = None              # experiment-log record of this frame
        self._xlog_res_done = False

        # LOG ONLY. The projection model must match the stream: the smoother
        # projects through camera_info.P (rectified), so /ves_camera/image_rect
        # pairs with camera_model:=rect. The topic name is the only hint
        # available here, so this warns and changes nothing.
        if self.tip_source != "detector":
            topic_rect = "rect" in str(self.image_topic).lower()
            if topic_rect != (self.camera_model == "rect"):
                self.get_logger().warn(
                    f"camera_model={self.camera_model} but image_topic="
                    f"{self.image_topic} looks "
                    f"{'rectified' if topic_rect else 'unrectified'}. The "
                    f"projected tip will be offset if these disagree. "
                    f"(Warning only; nothing is changed.)")

        self.xlog = None
        if bool(self.get_parameter("experiment").value):
            from experiment_log import ExperimentLog
            self.xlog = ExperimentLog(
                self.output_root,
                lambda: self.get_clock().now().nanoseconds,
                self.get_logger(), G.contour_of,
                snapshot_every=int(self.get_parameter("snapshot_every").value),
                png_level=int(self.get_parameter("raw_png_compression").value))
            self.get_logger().info(
                f"Experiment logging ON -> {self.xlog.session_dir}")

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
        # One JSON object per processed frame (IDLE included) with the readouts
        # the operator UI shows; see _publish_readouts for the fields.
        self.pub_status = self.create_publisher(String, "/toolhead/status", 10)

        # The services stay available in both modes: they remain the way to
        # test without the robot, and a stop is still useful as an override.
        self.create_service(Trigger, "/toolhead/start", self._on_start)
        self.create_service(Trigger, "/toolhead/stop", self._on_stop)

        # Newest tip pose, and the camera intrinsics it is projected through.
        self.last_pose = None          # (x, y, z) metres, camera frame
        self.last_pose_stamp = None    # rclpy time of that pose
        self.K = None                  # 3x3 intrinsics at calibration size
        self.dist = None               # distortion coefficients
        self.calib_wh = None           # (width, height) the calibration used
        self.calib_path = None         # which file was loaded
        self.calib_stamp = None        # its modification time, for the header
        self._pose_warned = False
        self.n_pose = 0                # tip poses received this run

        if self.tip_source in ("pose_seed", "pose_only"):
            self._load_calibration()
            msg_type = self._resolve_pose_type()
            # RELIABLE matches a reliable publisher exactly and still accepts
            # a best-effort one, so one subscription covers both. (A second
            # subscription to the same topic is not an option: ROS 2 refuses
            # two with different types, and a duplicate adds nothing here.)
            self.create_subscription(
                PoseWithCovarianceStamped if msg_type == "cov" else PoseStamped,
                self.pose_topic, self._on_pose,
                QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                           history=HistoryPolicy.KEEP_LAST, depth=1))
            self.get_logger().info(
                f"Tip source: {self.tip_source} from {self.pose_topic} "
                f"({'PoseWithCovarianceStamped' if msg_type == 'cov' else 'PoseStamped'})")

        self._tool_held = False        # last debounced button state
        self._tool_release_t = None    # when the button first read released
        self._tool_seen = False        # has any joint message arrived
        self._tool_ignore = False      # ignore a press held at startup

        if self.trigger_mode in ("tool", "autonomous"):
            # Joint state is a continuous stream where only the newest value
            # matters, so a depth-1 BEST_EFFORT queue is enough and cannot
            # build a backlog.
            jqos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(JointState, self.tool_topic,
                                     self._on_joint, jqos)
            self.get_logger().info(
                f"Trigger: tool button on {self.tool_topic} "
                f"(joint '{TOOL_JOINT_NAME}'). "
                + ("Hold to cut; release to save."
                   if self.trigger_mode == "tool" else
                   f"Autonomous: one episode per run, ending after "
                   f"{self.autonomous_idle:.0f}s with no signal "
                   f"(or Ctrl-C)."))
        else:
            self.get_logger().info(
                "Trigger: /toolhead/start and /toolhead/stop services.")
            if self.xlog is not None:
                # Record the tool signal even when it does not drive episodes.
                # This callback only logs; it cannot start or stop anything.
                self.create_subscription(
                    JointState, self.tool_topic, self._xlog_on_joint,
                    QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                               history=HistoryPolicy.KEEP_LAST, depth=1))

        self.state = IDLE
        self.lock = threading.Lock()
        self.busy = False
        self.n_received = 0
        self._noframe_timer = None
        self._refine_mod = None      # SAM2 refinement module, loaded on first use
        self.n_processed = 0
        self.n_dropped = 0
        self.last_episode_mask = None   # the last cut's mask, shown between cuts until the next press

        self._reset_episode_state()

        self.get_logger().info("Loading models...")
        self._load_models()
        self.get_logger().info("Models loaded.")
        if self.xlog is not None:
            self._xlog_config()

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

        if self.tip_source != "detector":
            self.kp_sess = None
            self.seed_mod = None
            self.get_logger().info(
                f"tip_source={self.tip_source}: skipping the keypoint detector"
                + ("" if self.tip_source == "pose_seed"
                   else " and TAPNext++"))

        if self.tip_source == "detector":
            self._load_detector()

        # pose_only drives the cut path straight from the pose, so the tracker
        # is never stepped and its checkpoint need not be downloaded.
        if self.tip_source != "pose_only":
            self._load_tracker()

    def _load_detector(self):
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

    def _load_tracker(self):
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
        # Arm the projection warnings again for each episode: they fire once
        # per run, and a press that fails after an earlier one would otherwise
        # report the bare error with none of the explanation.
        self._pose_warned = False
        self.seed_buffer = []          # (frame, labels, scores, kps) while seeding
        self.seed_votes = []
        self.tracker = None
        self.track_state = None
        self.tip_history = []          # raw tips, for the smoothing window
        self.mask_tips = []            # smoothed tips the mask was built from
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
        self.prev_keep = None      # last frame's kept region, to carry the side
        self.prev_arc_frac = None  # last frame's arc, as a fraction of the perimeter
        self.prev_seg_len = None   # last frame's sweep length, in px
        self.seg_vetoed = 0        # consecutive frames the sweep leapt
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
        # A new episode must not show the previous one's last distance.
        self.last_hull_dist = self.last_hull_dist_mm = None
        self.last_warn_state = ""

    def _on_start(self, request, response):
        ok, msg = self._begin_episode(source="service")
        response.success = ok
        response.message = msg
        return response

    def _begin_episode(self, source="service"):
        """Start an episode. Shared by the /toolhead/start service and by the
        tool-button trigger, so both paths run identical logic."""
        # An experiment session keeps its episodes inside its own folder.
        root = self.xlog.session_dir if self.xlog is not None else self.output_root
        with self.lock:
            if self.state != IDLE:
                return False, f"Already running (state={self.state}). Call stop first."
            self._reset_episode_state()
            self.last_episode_mask = None
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            # Count only existing episode directories. The conditional must
            # guard the listdir alone: written as a trailing if/else on the
            # whole expression it binds to `1 + len(...)`, which is wrong.
            existing = ([d for d in os.listdir(root)
                         if d.startswith("episode_")]
                        if os.path.isdir(root) else [])
            n = len(existing) + 1
            self.episode_dir = os.path.join(root, f"episode_{n:03d}_{stamp}")
            os.makedirs(self.episode_dir, exist_ok=True)
            self.state = SEEDING
            self.n_received = self.n_processed = self.n_dropped = 0
            self.t_start = time.time()
        self.episode_n += 1
        self.show_distance_overlay = bool(
            self.get_parameter("show_distance_overlay").value)
        self.warn_distance_px = float(
            self.get_parameter("warn_distance_px").value)
        self.warn_distance_mm = float(
            self.get_parameter("warn_distance_mm").value)
        # The camera is recalibrated between sessions and sometimes between
        # cuts, so the newest calibration is read again at every press. A
        # failure keeps the previous one rather than stopping a running session.
        if self.tip_source != "detector":
            try:
                self._load_calibration()
            except SystemExit as e:
                self.get_logger().error(
                    f"Calibration reload failed ({e}); keeping {self.calib_path}.")
        # A release whose final measurement is still pending keeps its estimate.
        self._finish_final(None, "next_press")
        self.T0 = self._baseline_mask()
        self.fraction = 0.0 if self.T0 is not None else None
        # No baseline at the press (the tool never left the view): no estimate
        # for this cut, though the final can still be measured after it.
        self.fraction_state = (FRAC_ESTIMATE if self.T0 is not None
                               else FRAC_NONE)
        if self.xlog is not None:
            trial = int(self.get_parameter("trial").value)
            self.xlog.episode_begin(self.episode_dir, {
                "tumor_id": str(self.get_parameter("tumor_id").value),
                "condition": str(self.get_parameter("condition").value),
                "operator": str(self.get_parameter("operator").value),
                "trial": trial if trial >= 0 else n,
                "trial_from_param": trial >= 0,
                "overlay_on": self.show_distance_overlay,
                "show_distance_overlay": self.show_distance_overlay,
                "warn_distance_px": self.warn_distance_px,
                "trigger_mode": self.trigger_mode,
                "trigger_source": source,
                "tip_source": self.tip_source,
                "tip_smooth": self.tip_smooth,
                "episode_dir": self.episode_dir,
                # Re-read at this press; a copy goes into the episode folder.
                "calib_path": self.calib_path,
            })
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
        if self.xlog is not None:
            self.xlog.tool_signal(held, float(msg.position[idx]),
                                  msg.header.stamp, "trigger")

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
                ok, message = self._begin_episode(source=self.trigger_mode)
                self.get_logger().info(f"Tool pressed -> {message}")
            return

        # Released. Wait out the hold before ending, so a brief dropout in the
        # signal does not cut the episode short. In autonomous mode that hold
        # spans a whole inter-chunk gap, so the policy's bursts read as one
        # continuous episode instead of one episode per chunk.
        if self._tool_held:
            now = time.time()
            hold = (self.autonomous_idle if self.trigger_mode == "autonomous"
                    else TOOL_RELEASE_HOLD)
            if self._tool_release_t is None:
                self._tool_release_t = now
                return
            if now - self._tool_release_t < hold:
                return
            self._tool_held = False
            self._tool_release_t = None
            ok, message = self._end_episode(source=self.trigger_mode)
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
        ok, msg = self._end_episode(source="service")
        response.success = ok
        response.message = msg
        return response

    def _end_episode(self, source="service"):
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
            self.fraction, self.fraction_state = None, FRAC_BASELINE
            if self.xlog is not None:
                self.xlog.episode_end(f"stopped_during_seeding ({source})")
            self._reset_episode_state()
            return True, msg

        paths = self._write_episode_outputs()
        # The final fraction is measured on the IDLE frames that follow, once
        # the tip has left the tumor (_step_idle).
        self.post = {"A": self.accum_mask.copy(), "dir": self.episode_dir,
                     "t0": time.time(), "probs": []}
        self.last_episode_mask = self.accum_mask.copy()
        if self.xlog is not None:
            self.xlog.episode_end(source, outputs={
                "accum": self.accum_mask, "T": self.last_T,
                "cutline_x": (None if self.cutline_x is None
                              else self.cutline_x.copy()),
                "cut_path": list(self.cut_path),
                "tip_history": list(self.tip_history),
                "mask_tips": list(self.mask_tips)})
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
                "calibration": self.calib_path,
                "calibration_mtime": (None if not self.calib_path else
                                      datetime.fromtimestamp(os.path.getmtime(
                                          self.calib_path)).isoformat(timespec="seconds")),
                "baseline_px": None if self.T0 is None else int(self.T0.sum()),
                "fraction_estimate": self.fraction,
                "fraction_final": None,          # filled in by _finish_final
            }, f, indent=2)
        out.append("episode_info.json")
        return ", ".join(out)

    # ------------------------------------------------------------- callbacks

    def _on_image(self, msg):
        self.n_received += 1
        # Experiment log: every received frame gets a frames.csv row, IDLE
        # included. Only enqueues; the writing happens on other threads.
        xl = self.xlog
        rec = xl.frame_begin(msg.header) if xl is not None else None
        with self.lock:
            if self.state == IDLE:
                if rec is not None:
                    xl.frame_end(rec, IDLE)
                if self.busy:
                    return
                self.busy = True
                idle = True
            else:
                idle = False
        if idle:
            # Between cuts: keep the operator's view live and maintain the
            # baseline and final fraction. Nothing here touches the mask.
            try:
                self._step_idle(self._proc_frame(self._to_bgr(msg)), msg.header)
            except Exception as e:
                self.get_logger().error(f"Idle frame failed: {e}",
                                        throttle_duration_sec=5.0)
            finally:
                self.busy = False
            return
        with self.lock:
            if self.busy:
                # Stay live rather than queueing: the newest frame matters more
                # than completeness for an overlay the surgeon is watching.
                self.n_dropped += 1
                if rec is not None:
                    xl.raw_frame(rec, msg, convert=self._xlog_convert)
                    xl.frame_end(rec, self.state, dropped="busy")
                return
            self.busy = True
            state_in = self.state
        self._xrec = rec
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
            if rec is not None:
                xl.raw_frame(rec, frame)
                if not self._xlog_res_done:
                    self._xlog_res_done = True
                    xl.write_config({"resize": self._xlog_resize_info(native)})
            if self.state == SEEDING:
                self._step_seeding(frame, msg.header)
            elif self.state == TRACKING:
                self._step_tracking(frame, msg.header)
        except Exception as e:  # never let one bad frame kill the node
            self.get_logger().error(f"Frame failed: {e}")
            if rec is not None:
                xl.event("frame_failed", str(e))
        finally:
            self.busy = False
            if rec is not None:
                self._xlog_frame_done(rec, state_in)

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
        if self.xlog is not None:
            self.xlog.overlay_frame(self._xrec, vis, "status")

        nh, nw = getattr(self, "native_hw", vis.shape[:2])
        if (nh, nw) != vis.shape[:2]:
            vis = cv2.resize(vis, (nw, nh), interpolation=cv2.INTER_LINEAR)
        self.pub_overlay.publish(self._to_msg(vis, header, "bgr8"))
        self._publish_readouts(header)

    # ------------------------------------------------------- visual servoing

    def _load_calibration(self):
        """Read the camera intrinsics from the newest calibration run.

        Calibration is redone periodically and each run writes its own folder
        under `calib_root`, so the matrix is not a constant -- reading it at
        startup is what keeps a recalibration from silently invalidating the
        projection. The resolved path and its timestamp are logged so it is
        always visible which calibration a recording used.

        Refuses to start if the file is missing or malformed. A guessed
        projection would place the seed on the wrong tissue, which is worse
        than not starting.
        """
        path = self.calib_file
        if not path:
            if not os.path.isdir(self.calib_root):
                raise SystemExit(
                    f"Calibration root not found: {self.calib_root}. "
                    f"Pass -p calib_file:=/path/to/ves_camera.yaml")
            runs = []
            for d in os.listdir(self.calib_root):
                cand = os.path.join(self.calib_root, d, CALIB_RELPATH)
                if os.path.isfile(cand):
                    runs.append((os.path.getmtime(cand), cand))
            if not runs:
                raise SystemExit(
                    f"No {CALIB_RELPATH} under any run in {self.calib_root}. "
                    f"Pass -p calib_file:=/path/to/ves_camera.yaml")
            runs.sort()
            path = runs[-1][1]
            if len(runs) > 1:
                self.get_logger().info(
                    f"{len(runs)} calibrations found; using the newest.")

        # Parsed as plain YAML, NOT with cv2.FileStorage. ROS
        # camera_calibration writes ordinary YAML, while FileStorage demands
        # OpenCV's own dialect (a %YAML:1.0 directive and !!opencv-matrix
        # tags) and rejects the file outright -- verified against a copy of
        # this calibration.
        try:
            import yaml
            with open(path) as f:
                y = yaml.safe_load(f)
        except Exception as e:
            raise SystemExit(f"Could not read calibration {path}: {e}")
        if not isinstance(y, dict):
            raise SystemExit(f"Calibration {path} is not a YAML mapping.")

        def block(key, rows, cols):
            """One ROS matrix block as an array, or None if absent."""
            node = y.get(key)
            if node is None:
                return None
            if not isinstance(node, dict) or "data" not in node:
                raise SystemExit(f"Calibration {path}: {key} has no 'data'.")
            vals = np.asarray(node["data"], dtype=np.float64).ravel()
            r = int(node.get("rows", rows))
            c = int(node.get("cols", cols))
            if vals.size != r * c:
                raise SystemExit(
                    f"Calibration {path}: {key} claims {r}x{c} but carries "
                    f"{vals.size} values.")
            return vals.reshape(r, c)

        try:
            width = int(y["image_width"])
            height = int(y["image_height"])
        except (KeyError, TypeError, ValueError):
            raise SystemExit(f"Calibration {path} is missing image_width / "
                             f"image_height.")
        cam = block("camera_matrix", 3, 3)
        dist = block("distortion_coefficients", 1, 5)
        proj = block("projection_matrix", 3, 4)

        if cam is None or width <= 0 or height <= 0:
            raise SystemExit(f"Calibration {path} is missing camera_matrix "
                             f"or image size.")

        model = str(y.get("distortion_model", "plumb_bob"))
        if model != "plumb_bob":
            # projectPoints implements the plumb_bob/rational model. A fisheye
            # calibration needs cv2.fisheye and would be silently wrong here.
            self.get_logger().warn(
                f"Calibration uses distortion_model '{model}', not plumb_bob. "
                f"The projection assumes plumb_bob and may be inaccurate; "
                f"consider camera_model:=rect.")

        if self.camera_model == "rect":
            # A rectified stream is already undistorted, and its intrinsics
            # live in the projection matrix's left 3x3 block.
            if proj is None:
                raise SystemExit(f"camera_model:=rect but {path} has no "
                                 f"projection_matrix.")
            self.K = np.asarray(proj, dtype=np.float64)[:3, :3].copy()
            self.dist = np.zeros((5, 1), dtype=np.float64)
        else:
            self.K = np.asarray(cam, dtype=np.float64).reshape(3, 3).copy()
            self.dist = (np.zeros((5, 1), dtype=np.float64) if dist is None
                         else np.asarray(dist, dtype=np.float64).reshape(-1, 1))

        self.calib_wh = (width, height)
        self.calib_path = path
        stamp = datetime.fromtimestamp(os.path.getmtime(path)).strftime(
            "%Y-%m-%d %H:%M")
        self.calib_stamp = stamp
        self.get_logger().info(
            f"Calibration [{self.camera_model}] {path} ({stamp})  "
            f"{width}x{height}  fx={self.K[0, 0]:.1f} fy={self.K[1, 1]:.1f} "
            f"cx={self.K[0, 2]:.1f} cy={self.K[1, 2]:.1f}")

    def _resolve_pose_type(self, timeout_s=5.0):
        """Which message type the pose publisher offers: "cov" or "plain".

        The type cannot be guessed. A subscription created with the wrong one
        never connects and never logs, and ROS 2 will not allow a second
        subscription to the same topic with the other type -- so the choice
        has to be right the first time. The graph knows the answer; this is
        the same information `ros2 topic info <topic>` prints.

        The publisher may not have appeared yet when this node starts, so
        wait briefly for it. If it never appears, assume the
        covariance-stamped form the visual-servoing smoother publishes and
        say so: the episode will then fail with the "no tip pose received"
        message, which names the topic to check.
        """
        deadline = time.time() + timeout_s
        want = self.pose_topic.lstrip("/")
        while time.time() < deadline:
            for name, types_ in self.get_topic_names_and_types():
                if name.lstrip("/") != want:
                    continue
                for t in types_:
                    if "PoseWithCovarianceStamped" in t:
                        return "cov"
                    if "PoseStamped" in t:
                        return "plain"
            time.sleep(0.1)
        self.get_logger().warn(
            f"{self.pose_topic} has no publisher after {timeout_s:.0f}s; "
            f"assuming PoseWithCovarianceStamped. If the tip never seeds, "
            f"check: ros2 topic info {self.pose_topic} --verbose")
        return "cov"

    def _on_pose(self, msg):
        """Keep the newest tip pose. Projection happens per image frame.

        Accepts either message type: PoseWithCovarianceStamped nests the pose
        one level deeper (msg.pose.pose) than PoseStamped (msg.pose). The
        covariance is not used -- the smoother's own uncertainty does not
        change where the tip is drawn.
        """
        inner = msg.pose
        pos = getattr(inner, "pose", inner).position
        self.last_pose = (float(pos.x), float(pos.y), float(pos.z))
        self.last_pose_stamp = msg.header.stamp
        if self.xlog is not None:
            # Logged, still not used: 6x6, GTSAM order (rotation block first,
            # then position), camera frame.
            cov = getattr(inner, "covariance", None)
            self.last_pose_cov = (None if cov is None
                                  else np.array(cov, dtype=np.float64))
        self.n_pose += 1
        if self.n_pose == 1:
            self.get_logger().info(
                f"First tip pose received: "
                f"({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f}) m "
                f"frame_id={msg.header.frame_id or '(empty)'}")

    def _project_tip(self):
        """The newest tip pose as a pixel in the PROCESSING frame, or None.

        The pose is already in the camera frame, so this is a pure intrinsic
        projection. It is computed at the calibration's resolution and then
        scaled to the processing size, which keeps it correct if either the
        camera resolution or proc_size changes.

        A point at or behind the image plane cannot be projected, and a tip
        that lands outside the frame is refused rather than clamped -- a
        clamped point would sit on the border and silently corrupt the cut
        path.
        """
        if self.last_pose is None or self.K is None or self.H is None:
            return None
        x, y, z = self.last_pose
        if z <= 1e-6:
            # z is depth along the optical axis; <=0 puts the tip at or behind
            # the image plane, which no intrinsic projection can place. Almost
            # always means the pose is in some other frame than the camera's.
            if not self._pose_warned:
                self._pose_warned = True
                self.get_logger().warn(
                    f"Tip pose has z={z:.4f} m (must be > 0). The pose is "
                    f"probably not expressed in the camera optical frame.")
            return None

        pts = np.array([[[x, y, z]]], dtype=np.float64)
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        uv, _ = cv2.projectPoints(pts, rvec, tvec, self.K, self.dist)
        u, v = float(uv[0, 0, 0]), float(uv[0, 0, 1])

        cw, ch = self.calib_wh
        u *= self.W / float(cw)
        v *= self.H / float(ch)

        if not (0 <= u < self.W and 0 <= v < self.H):
            if not self._pose_warned:
                self._pose_warned = True
                self.get_logger().warn(
                    f"Tip projects outside the image: ({u:.0f},{v:.0f}) not "
                    f"within {self.W}x{self.H}. camera_model="
                    f"{self.camera_model}; try the other one "
                    f"(raw <-> rect). Pose was "
                    f"({x:.4f}, {y:.4f}, {z:.4f}) m.")
            return None
        return np.array([u, v], dtype=np.float32)

    def _step_seeding(self, frame, header):
        """Establish the first tip position.

        detector: run the keypoint R-CNN until the plurality vote commits,
        mirroring auto_seed_tip.py's logic on a live stream, then optionally
        refine with SAM2. pose_seed/pose_only: take the tip straight from the
        visual-servoing pose, which needs neither vote nor refinement."""
        if self.H is None:
            self.H, self.W = frame.shape[:2]

        # pose_seed / pose_only: the visual servoing smoother already knows
        # where the tip is, so there is nothing to detect or refine. The wait
        # is only for the first pose message to arrive.
        if self.tip_source in ("pose_seed", "pose_only"):
            tip = self._project_tip()
            if tip is None:
                self.seed_votes.append(None)
                self._publish_status(
                    frame, header,
                    f"SEEDING  waiting for {self.pose_topic}")
                if len(self.seed_votes) >= SEED_MAX_FRAMES:
                    # Distinguish "nothing arrived" from "arrived but could
                    # not be projected" -- the fixes are completely different.
                    if self.n_pose == 0:
                        self.get_logger().error(
                            f"No tip pose received at all on "
                            f"{self.pose_topic} during {SEED_MAX_FRAMES} "
                            f"frames. The topic is not publishing, the name "
                            f"is wrong, or the type is not PoseStamped. "
                            f"Check: ros2 topic info {self.pose_topic} "
                            f"--verbose")
                    else:
                        x, y, z = self.last_pose
                        self.get_logger().error(
                            f"{self.n_pose} tip poses received but none could "
                            f"be projected into the image. Last pose "
                            f"({x:.4f}, {y:.4f}, {z:.4f}) m, camera_model="
                            f"{self.camera_model}, calibration "
                            f"{self.calib_wh[0]}x{self.calib_wh[1]} -> "
                            f"processing {self.W}x{self.H}. "
                            f"A z<=0 means the pose is not in the camera "
                            f"frame; a large u/v means camera_model or the "
                            f"calibration is wrong.")
                    with self.lock:
                        self.state = IDLE
                return
            self.get_logger().info(
                f"Tip seeded from pose at ({tip[0]:.1f}, {tip[1]:.1f})")
            self._begin_tracking(frame, tip, header)
            return

        A = self.seed_mod
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
        self.tracker = getattr(self, "tracker_model", None)
        self.track_state = None
        self.cutline_x = np.full(self.H, np.nan, dtype=np.float32)
        self.cut_path = []
        self.prev_tip_y = None
        self.keep_side = None
        self.prev_keep = None
        self.prev_arc_frac = None
        self.prev_seg_len = None
        self.seg_vetoed = 0
        self.start_stub = None
        self.accum_mask = np.zeros((self.H, self.W), dtype=bool)
        self.seed_xy = seed_xy
        with self.lock:
            self.state = TRACKING
        self.get_logger().info("Tracking started.")
        self._step_tracking(frame, header, first=True)

    # -------------------------------------------------------------- tracking

    def _step_tracking(self, frame, header, first=False):
        # pose_only: the pose drives the cut path directly and TAPNext++ never
        # runs. There is no image-based fallback here by design -- if the pose
        # is wrong the cut path is wrong, with nothing to catch it -- so every
        # frame logs what it used and how stale it was.
        if self.tip_source == "pose_only":
            tip = self._project_tip()
            if tip is None:
                self._publish_status(frame, header, "TRACKING  no tip pose")
                if self.xlog is not None:
                    self._xlog_tip(None, None,
                                   reject=self._xlog_reject_reason())
                return
            dt = float("nan")
            if self.last_pose_stamp is not None and header.stamp is not None:
                dt = ((header.stamp.sec - self.last_pose_stamp.sec)
                      + 1e-9 * (header.stamp.nanosec
                                - self.last_pose_stamp.nanosec))
            self.get_logger().info(
                f"pose_only tip=({tip[0]:.1f},{tip[1]:.1f}) "
                f"image-pose dt={dt * 1e3:+.0f}ms",
                throttle_duration_sec=1.0)
            self.tip_history.append((float(tip[0]), float(tip[1])))
            w = self.tip_history[-self.tip_smooth:]
            tx = float(np.mean([q[0] for q in w]))
            ty = float(np.mean([q[1] for q in w]))
            self.mask_tips.append((tx, ty))
            self._update_mask(frame, (tx, ty))
            self._publish(frame, header, (tx, ty))
            if self.xlog is not None:
                self._xlog_tip(tip, (tx, ty))
            self.n_processed += 1
            return

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
        w = self.tip_history[-self.tip_smooth:]
        tx = float(np.mean([p[0] for p in w]))
        ty = float(np.mean([p[1] for p in w]))

        self.mask_tips.append((tx, ty))
        self._update_mask(frame, (tx, ty))
        self._publish(frame, header, (tx, ty))
        if self.xlog is not None:
            self._xlog_tip(tip, (tx, ty), visible=visible)
        self.n_processed += 1

    def _update_mask(self, frame_bgr, tip_xy):
        """One frame of cutline-sweep accumulation, using the same geometry
        functions as the offline pipeline."""
        H, W = self.H, self.W
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        raw_prob = G.raw_tumor_mask(self.engine, rgb, H, W)
        self.ema_mask_prob = raw_prob if self.ema_mask_prob is None else \
            MASK_EMA * raw_prob + (1 - MASK_EMA) * self.ema_mask_prob
        T = G.tumor_mask_from_prob(self.ema_mask_prob)
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
        # The nearest hull point and its distance have always come back from
        # this call; they are now kept for the distance readout and the
        # experiment log. Nothing in the mask construction reads them.
        hull_pt = hull_dist = None
        if perim is not None and len(perim) >= 40:
            self.last_angle_deg, perp_dir, hull_pt, hull_dist = \
                G.perp_direction_at_tip(perim, (tx, ty))
        self.last_hull_pt, self.last_hull_dist = hull_pt, hull_dist
        mm_per_px = self._mm_per_px()
        self.last_hull_dist_mm = (None if hull_dist is None or mm_per_px is None
                                  else float(hull_dist) * mm_per_px)
        self.last_T, self.last_perim = T, perim
        self.last_seg_end = None

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
                        # Point it OUT of the tumor: whichever way lies
                        # further outside (signed distance). "Flip if the probe
                        # is inside" never fires when the cut starts beyond the
                        # probe length, and the stub can then split the tumor.
                        pc = np.asarray(perim, np.int32)
                        dp = cv2.pointPolygonTest(
                            pc, (float(p0[0] + nv[0] * 6.0), float(p0[1] + nv[1] * 6.0)), True)
                        dm = cv2.pointPolygonTest(
                            pc, (float(p0[0] - nv[0] * 6.0), float(p0[1] - nv[1] * 6.0)), True)
                        if dp > dm:
                            nv = -nv
                        self.start_stub = G._ray_to_boundary(
                            p0, nv, perim, float(H + W))
                side_lab = G._side_partition(self.cut_path, (tx, ty), H, W,
                                             perim=perim,
                                             start_stub=self.start_stub)
                if side_lab is not None:
                    # Which region is the resection, as offline: chosen once by
                    # the two regions' centroids (a probe beside the tip is a
                    # coin flip, since the tip sits where the regions meet),
                    # then CARRIED by overlap with the region kept last frame.
                    # Re-choosing by centroid every frame can swap the regions
                    # once an angled cut stacks them; labels are renumbered
                    # every frame, but the region keeps its identity.
                    m1 = (side_lab == 1)
                    m2 = (side_lab == 2)
                    # Until a separating frame has fixed the side as a region,
                    # choose it afresh: a carried label number names nothing.
                    if self.prev_keep is None:
                        self.keep_side = None
                    if self.prev_keep is not None and (m1.any() or m2.any()):
                        o1 = int((self.prev_keep & m1).sum())
                        o2 = int((self.prev_keep & m2).sum())
                        # No overlap: the partition changed shape completely
                        # (a new cut), so choose the side afresh.
                        self.keep_side = (1 if o1 >= o2 else 2) if (o1 or o2) else None
                    if self.keep_side is None and m1.any() and m2.any():
                        x1 = float(np.nonzero(m1)[1].mean())
                        x2 = float(np.nonzero(m2)[1].mean())
                        if self.resect_side == "left":
                            self.keep_side = 1 if x1 <= x2 else 2
                        else:
                            self.keep_side = 1 if x1 >= x2 else 2
                # Only clip when the partition genuinely separates the tumor:
                # a region that degenerates to a sliver would otherwise let
                # the whole polygon through in one frame. The minor region's
                # share is bimodal (sound 41-42%, failed 0-0.2%); 5% sits in
                # the gap. A stored keep_side that names a vanished label
                # would delete the whole sweep, so it is dropped too.
                separates = False
                if side_lab is not None:
                    n1 = int((side_lab == 1).sum())
                    n2 = int((side_lab == 2).sum())
                    separates = (n1 + n2 > 0 and min(n1, n2) / (n1 + n2) >= 0.05)
                keep_now = self.keep_side
                if side_lab is not None and not separates:
                    keep_now = None
                elif (keep_now is not None and side_lab is not None
                        and not (side_lab == keep_now).any()):
                    keep_now = None
                seg_end = G.segment_end_on_perimeter(
                    perim, (tx, ty), perp_dir, self.cutline_x,
                    side_lab=side_lab, keep_side=keep_now,
                    resect_side=self.resect_side)
                # THE SWEEP IS CONTINUOUS: A ONE-FRAME LEAP IS NOT A SWEEP.
                #
                # segment_end_on_perimeter walks its ranked boundary hits
                # rejecting any whose segment would cross the cut path. When
                # every near candidate crosses, the search keeps walking until
                # something far away happens to pass -- landing on the opposite
                # side of the tumor and sweeping in tens of thousands of px for
                # a single frame. Measured offline: the nearest boundary sat
                # 20 px from the tip while the accepted endpoint was 269 px
                # away, with the frames either side using 21 px and 20 px.
                #
                # Scale comes from the previous frame, not a constant. A leap
                # that persists is real growth and is let through after a couple
                # of frames, so the sweep can never be frozen.
                if seg_end is not None and self.prev_seg_len is not None:
                    sl_now = float(np.hypot(seg_end[0] - tx, seg_end[1] - ty))
                    if sl_now > 4.0 * max(self.prev_seg_len, 8.0):
                        self.seg_vetoed += 1
                        if self.seg_vetoed <= 2:
                            seg_end = None
                        else:
                            self.prev_seg_len = sl_now
                    else:
                        self.seg_vetoed = 0
                self.last_seg_end = seg_end      # logging only
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

                    if self.prev_arc_frac is not None:
                        chosen_fr = len(arc) / max(len(perim), 1)
                        alt = long_ if arc is short else short
                        alt_fr = len(alt) / max(len(perim), 1)
                        # A sweep cannot leap from a sliver to swallowing the
                        # boundary in one frame and back out the next. Veto
                        # only a near-total wrap that the alternative does not
                        # require; every ordinary frame is left to the side
                        # test above.
                        if (chosen_fr - self.prev_arc_frac > 0.5
                                and chosen_fr > 0.9
                                and alt_fr < chosen_fr):
                            arc = alt
                    self.prev_arc_frac = len(arc) / max(len(perim), 1)
                    self.prev_seg_len = float(np.hypot(seg_end[0] - tx,
                                                       seg_end[1] - ty))
                    poly = cut_pts + [(float(seg_end[0]), float(seg_end[1]))] +                            [(float(p[0]), float(p[1])) for p in arc]
                    filled = np.zeros((H, W), dtype=np.uint8)
                    cv2.fillPoly(filled, [np.array(poly, np.int32).reshape(-1, 1, 2)], 1)
                    frame_mask = T & filled.astype(bool)
                    # Vetting only the sweep endpoint is not enough: the arc can
                    # wrap the far side of the tumor and fill the unresected
                    # half however the endpoint was chosen.
                    if side_lab is not None and keep_now:
                        frame_mask &= ~((side_lab > 0)
                                        & (side_lab != keep_now))
                    # Carry the kept region to the next frame, only from a frame
                    # whose partition separates the tumor (a degenerate frame's
                    # region spans the whole tumor and would flip the side).
                    if side_lab is not None and keep_now and separates:
                        self.prev_keep = (side_lab == keep_now)

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

        # KEEP ONLY WHAT THE RESECTION PRODUCED. Accumulation is monotonic, so
        # a region planted by a bad frame survives to the end unless it is
        # removed.
        #
        # Which components the resection actually made is answered by the TIP'S
        # OWN HISTORY: a resected region is one the tool was inside, a leftover
        # is one the tool merely passed beside. Measured offline on
        # failure_case2 -- main region 286 tips inside, second region 19, and
        # every leftover sliver 0 tips at 10-39 px away.
        #
        # This replaces a probe taken at one x per image row from cutline_x,
        # which failed two ways. cutline_x is overwritten latest-pass-wins, so a
        # later cut crossing the same rows moved the probes off an earlier
        # region and deleted it whole; and being per-row, what survived was a
        # comb of one-column-per-row fragments. It also assumed the resection
        # stays attached to the cut path, which is false once the tool leaves
        # the tumor and re-enters elsewhere -- the second cut's region is
        # legitimately a separate component, and connectivity is not tested
        # here at all.
        #
        # Tip history is append-only, so nothing can move the anchor off a
        # region already earned.
        n_lab, lab = cv2.connectedComponents(
            self.accum_mask.astype(np.uint8), connectivity=8)
        if n_lab > 2:
            keep = set()
            for _tp in self.mask_tips:
                _px = int(round(min(max(_tp[0], 0), W - 1)))
                _py = int(round(min(max(_tp[1], 0), H - 1)))
                _l = int(lab[_py, _px])
                if _l:
                    keep.add(_l)
            if keep:
                self.accum_mask = np.isin(lab, list(keep))

    def _publish(self, frame_bgr, header, tip_xy):
        if self.T0 is not None and self.accum_mask is not None:
            self.fraction = float((self.accum_mask & self.T0).sum()) / float(self.T0.sum())
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

        # Distance readout, drawn on a COPY: final_overlay.png (last_overlay
        # above), the mask and every saved frame stay exactly as before.
        self.last_d_text, self.last_warn_state = self._distance_label()
        shown = vis
        if self.show_distance_overlay:
            shown = vis.copy()
            self._draw_distance(shown, self.last_d_text,
                                self.last_warn_state == "warn")
        if self.xlog is not None:
            self.xlog.overlay_frame(self._xrec, shown, "tracking")

        # Publish at the camera's native resolution so the overlay lines up
        # with the raw stream in RViz and for any downstream consumer.
        nh, nw = getattr(self, "native_hw", (vis.shape[0], vis.shape[1]))
        scale = nw / float(self.W)
        if (nh, nw) != vis.shape[:2]:
            pub_vis = cv2.resize(shown, (nw, nh), interpolation=cv2.INTER_LINEAR)
            pub_mask = cv2.resize(disp.astype(np.uint8) * 255, (nw, nh),
                                  interpolation=cv2.INTER_NEAREST)
        else:
            pub_vis, pub_mask = shown, disp.astype(np.uint8) * 255

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
        self._publish_readouts(header)

    # ----------------------------------------- between cuts and the readouts

    def _proc_frame(self, native):
        """Native camera frame -> processing frame, as _on_image does."""
        self.native_hw = native.shape[:2]
        if native.shape[0] != self.proc_size or native.shape[1] != self.proc_size:
            return cv2.resize(native, (self.proc_size, self.proc_size),
                              interpolation=cv2.INTER_AREA)
        return native

    def _tool_tip_now(self, frame):
        """Where the tool tip is in this frame (processing px), or None when the
        tool is out of view. Pose modes project the pose; detector mode runs the
        keypoint detector."""
        if self.tip_source != "detector":
            return self._project_tip()
        labels, scores, kps = self._detect(frame)
        best = self.seed_mod.find_tip(labels, scores, kps) if len(scores) else None
        return None if best is None else best[1]

    def _step_idle(self, frame, header):
        """One frame between cuts: publish the live view, keep the baseline
        tumor mask current while the tool is out of view, and take the final
        fraction of the last cut once the tip has left the tumor."""
        self.H, self.W = frame.shape[:2]
        prob = G.raw_tumor_mask(self.engine, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                self.H, self.W)
        tip = self._tool_tip_now(frame)

        if self.post is not None:
            T = G.tumor_mask_from_prob(prob)
            in_tumor = tip is not None and bool(
                T[int(np.clip(round(tip[1]), 0, self.H - 1)),
                  int(np.clip(round(tip[0]), 0, self.W - 1))])
            if not in_tumor:
                self.post["probs"].append(prob)
            if len(self.post["probs"]) >= FINAL_FRAMES:
                self._finish_final(G.tumor_mask_from_prob(
                    np.mean(self.post["probs"], axis=0)), "measured")
            elif time.time() - self.post["t0"] > FINAL_TIMEOUT:
                self._finish_final(None, "timeout_tip_in_tumor")

        if tip is None:
            self.baseline_probs.append(prob)
        # The last cut's result stays on screen until the next press.
        if self.fraction_state not in (FRAC_ESTIMATE, FRAC_FINAL):
            self.fraction = None
            self.fraction_state = (
                FRAC_READY if len(self.baseline_probs) >= BASELINE_FRAMES
                else FRAC_BLOCKED if tip is not None else FRAC_BASELINE)

        if self.pub_overlay is not None:
            vis = frame
            m = self.last_episode_mask
            if m is not None and m.shape == frame.shape[:2] and m.any():
                # Same tint as during the cut, so the result does not vanish at the release.
                vis = frame.copy()
                vis[m] = (0.5 * np.array([0, 0, 255]) + 0.5 * vis[m]).astype(np.uint8)
            nh, nw = self.native_hw
            if (nh, nw) != vis.shape[:2]:
                vis = cv2.resize(vis, (nw, nh), interpolation=cv2.INTER_LINEAR)
            self.pub_overlay.publish(self._to_msg(vis, header, "bgr8"))
        self._publish_readouts(header)

    def _baseline_mask(self):
        """The baseline tumor mask from the last BASELINE_FRAMES tool-free
        frames, or None if there are not enough of them yet."""
        if len(self.baseline_probs) < BASELINE_FRAMES:
            return None
        T = G.tumor_mask_from_prob(np.mean(self.baseline_probs, axis=0))
        return T if T.any() else None

    def _finish_final(self, T_end, reason):
        """Close the pending final measurement: the fraction is
        |A| / (|A| + |T_end minus A|), written into the episode's
        episode_info.json. T_end None keeps the estimate."""
        post, self.post = self.post, None
        if post is None:
            return
        final = None
        if T_end is not None:
            a = int(post["A"].sum())
            rest = int((T_end & ~post["A"]).sum())
            final = a / float(a + rest) if a + rest > 0 else None
        if final is not None:
            self.fraction, self.fraction_state = final, FRAC_FINAL
        path = os.path.join(post["dir"], "episode_info.json")
        try:
            with open(path) as f:
                info = json.load(f)
            info.update({"fraction_final": final, "fraction_final_reason": reason,
                         "final_tumor_px": None if T_end is None else int(T_end.sum())})
            with open(path, "w") as f:
                json.dump(info, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"Could not add the final fraction to {path}: {e}")
        self.get_logger().info(
            f"Final resected fraction: "
            f"{'--' if final is None else f'{100 * final:.1f}%'} ({reason})")

    def _mm_per_px(self):
        """Millimetres per processing-frame pixel at the tip's depth, or None
        without a pose. Exact only at that depth: the boundary point the
        distance is measured to is assumed to lie at the same depth."""
        if (self.tip_source == "detector" or self.last_pose is None
                or self.K is None or self.W is None):
            return None
        z = self.last_pose[2]
        if z <= 1e-6:
            return None
        fx_proc = self.K[0, 0] * self.W / float(self.calib_wh[0])
        return z * 1000.0 / fx_proc

    def _publish_readouts(self, header):
        """/toolhead/status: what the operator UI shows, one JSON object per
        processed frame. d_mm is null without a pose; fraction_state is one of
        baseline, blocked, ready, estimate, final."""
        status = {
            "stamp": [int(header.stamp.sec), int(header.stamp.nanosec)],
            "state": self.state,
            "episode": self.episode_n,
            "d_px": (None if self.last_hull_dist is None
                     else round(float(self.last_hull_dist), 2)),
            "d_mm": (None if self.last_hull_dist_mm is None
                     else round(self.last_hull_dist_mm, 3)),
            "warn_mm": self.warn_distance_mm,
            "warn": self.last_warn_state == "warn",
            "fraction": None if self.fraction is None else round(self.fraction, 4),
            "fraction_state": self.fraction_state,
            "calibration": self.calib_stamp,
            "tip_source": self.tip_source,
        }
        msg = String()
        msg.data = json.dumps(status)
        self.pub_status.publish(msg)

    # ------------------------------------------------- distance readout

    def _distance_label(self):
        """("d = NN px" or "d = --", warn state) for this frame's hull
        distance. The number is round(last_hull_dist), the same value the
        experiment log records as hull_dist_px."""
        d = self.last_hull_dist
        if d is None or not np.isfinite(d):
            return "d = --", "invalid"
        mm = self.last_hull_dist_mm
        if mm is not None:
            text = f"d = {mm:.1f} mm"
            if self.warn_distance_mm > 0:
                return text, ("warn" if mm < self.warn_distance_mm else "ok")
            return text, "off"
        text = f"d = {int(round(d))} px"
        if self.warn_distance_px > 0:
            return text, ("warn" if d < self.warn_distance_px else "ok")
        return text, "off"

    @staticmethod
    def _draw_distance(img, text, warn):
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        (tw, tht), base = cv2.getTextSize(text, font, fs, th)
        cv2.rectangle(img, (4, 4), (4 + tw + 10, 4 + tht + base + 10),
                      (0, 0, 0), -1)
        cv2.putText(img, text, (9, 9 + tht), font, fs,
                    (0, 0, 255) if warn else (255, 255, 255), th, cv2.LINE_AA)

    # ------------------------------------------------- experiment logging
    # Everything below runs only with -p experiment:=true. Each method reads
    # node state and hands it to experiment_log.ExperimentLog; none of them
    # writes back to anything tracking or the mask uses.

    def _xlog_on_joint(self, msg):
        """Service mode: record the tool signal without acting on it."""
        try:
            idx = list(msg.name).index(TOOL_JOINT_NAME)
        except ValueError:
            return
        if idx < len(msg.position):
            v = float(msg.position[idx])
            self.xlog.tool_signal(v >= TOOL_PRESSED, v, msg.header.stamp,
                                  "log_only")

    def _xlog_convert(self, msg):
        """A raw message as the processing frame, for a frame that was
        dropped before conversion. Runs on a writer thread."""
        native = self._to_bgr(msg)
        if native.shape[0] != self.proc_size or native.shape[1] != self.proc_size:
            return cv2.resize(native, (self.proc_size, self.proc_size),
                              interpolation=cv2.INTER_AREA)
        return native

    def _xlog_resize_info(self, native):
        nh, nw = native.shape[:2]
        return {"native_hw": [int(nh), int(nw)],
                "proc_hw": [self.proc_size, self.proc_size],
                "scale_native_per_proc": [nh / float(self.proc_size),
                                          nw / float(self.proc_size)],
                "crop": "none (full frame)",
                "offset_px": [0, 0],
                "method": "cv2.resize INTER_AREA",
                "native_is_proc": bool(nh == self.proc_size
                                       and nw == self.proc_size)}

    def _xlog_frame_done(self, rec, state_in):
        xl = self.xlog
        state_after = self.state
        xl.frame_end(rec, state_in, state_after, processed=True)
        if state_after != state_in:
            xl.event("state", f"{state_in}->{state_after}", rec["stamp"])
            # Seeding gave up and set IDLE itself, without _end_episode.
            if state_after == IDLE and xl.episode is not None:
                xl.episode_end("seeding_failed")
        self._xrec = None

    def _xlog_reject_reason(self):
        """Why _project_tip returned None, by the same tests it applies."""
        if self.last_pose is None:
            return "no_pose"
        if self.K is None:
            return "no_calibration"
        if self.last_pose[2] <= 1e-6:
            return "z_nonpositive"
        return "off_frame"

    def _xlog_tip(self, tip_raw, tip_smooth, reject="", visible=None):
        ok = tip_smooth is not None
        vis_flag = None
        if visible is not None:
            try:
                vis_flag = bool(np.asarray(visible).ravel()[0])
            except Exception:
                vis_flag = None
        pose_sub = self.tip_source in ("pose_seed", "pose_only")
        self.xlog.tip(self._xrec, {
            "tip_source": self.tip_source,
            "tip_raw": (None if tip_raw is None
                        else (float(tip_raw[0]), float(tip_raw[1]))),
            "tip_smooth": tip_smooth,
            "reject_reason": reject,
            "tracker_visible": vis_flag,
            "pose_subscribed": pose_sub,
            "pose": self.last_pose if pose_sub else None,
            "pose_stamp": self.last_pose_stamp if pose_sub else None,
            "pose_msgs_total": self.n_pose if pose_sub else None,
            "cov": self.last_pose_cov if pose_sub else None,
            "K": self.K, "dist": self.dist, "calib_wh": self.calib_wh,
            "WH": (self.W, self.H),
            "hull_dist": self.last_hull_dist if ok else None,
            "hull_pt": self.last_hull_pt if ok else None,
            "perim": self.last_perim if ok else None,
            "T": self.last_T if ok else None,
            "seg_end": self.last_seg_end if ok else None,
            "angle_deg": self.last_angle_deg if ok else None,
            "accum": self.accum_mask if ok else None,
            "accum_px": int(self.accum_mask.sum()) if ok else None,
            "overlay_on": self.show_distance_overlay,
            "d_shown": (self.last_d_text
                        if ok and self.show_distance_overlay else ""),
            "hull_dist_mm": self.last_hull_dist_mm if ok else None,
            "warn_px": self.warn_distance_px,
            "warn_mm": self.warn_distance_mm,
            "warn_state": self.last_warn_state if ok else "",
        })

    def _xlog_config(self):
        """run_config.json: everything needed to reproduce the session."""
        import socket
        xl = self.xlog
        try:
            params = {n: p.value for n, p in self._parameters.items()}
        except Exception:
            params = {}
        calib = None
        if self.calib_path:
            try:
                calib = xl.copy_calibration(self.calib_path)
            except Exception as e:
                calib = {"original_path": self.calib_path, "error": str(e)}
        xl.write_config({
            "session": xl.session_name,
            "created": datetime.now().isoformat(),
            "host": socket.gethostname(),
            "node_file": os.path.abspath(__file__),
            "argv": list(sys.argv),
            "params": params,
            "constants": {
                "TIP_SMOOTH": self.tip_smooth, "MASK_EMA": MASK_EMA,
                "DILATE": DILATE, "PROC_SIZE": PROC_SIZE,
                "SEED_MAX_FRAMES": SEED_MAX_FRAMES,
                "SEED_MIN_VOTES": SEED_MIN_VOTES,
                "MASK_THRESHOLD": getattr(G, "MASK_THRESHOLD", None),
                "ARC_MIN_SAMPLES": getattr(G, "ARC_MIN_SAMPLES", None),
                "ARC_SIDE_MARGIN": getattr(G, "ARC_SIDE_MARGIN", None),
                "ARC_FRAC": getattr(G, "ARC_FRAC", None),
                "hull_dist_min_perimeter_pts": 40,
            },
            "image": {
                "topic": self.image_topic,
                "qos": "RELIABLE, KEEP_LAST, depth 1",
                "executor": "rclpy.spin (single-threaded)",
                "note": ("Frames arriving while one is processed are "
                         "overwritten in the depth-1 queue before this node "
                         "sees them, so they never reach frames.csv. "
                         "Upstream drops show only as stamp gaps; the bag "
                         "has every frame."),
            },
            "resize": {"proc_hw": [self.proc_size, self.proc_size],
                       "native_hw": "filled in on the first frame"},
            "pose": {
                "tip_source": self.tip_source, "topic": self.pose_topic,
                "camera_model": self.camera_model,
                "K_used": None if self.K is None else self.K.tolist(),
                "dist_used": (None if self.dist is None
                              else self.dist.ravel().tolist()),
                "calib_wh": self.calib_wh,
                "calibration": calib,
                "covariance_layout": ("6x6 row-major, GTSAM order: rows/cols "
                                      "0-2 rotation (rad^2), 3-5 position "
                                      "(m^2), camera frame"),
            },
            "trigger": {
                "mode": self.trigger_mode, "topic": self.tool_topic,
                "joint": TOOL_JOINT_NAME, "pressed_threshold": TOOL_PRESSED,
                "release_hold_s": TOOL_RELEASE_HOLD,
                "autonomous_idle_timeout_s": self.autonomous_idle,
                "note": ("events.csv tool_on/tool_off are raw edges of this "
                         "joint. It is the Virtuoso's cut button; whether it "
                         "is the cautery pedal itself is not established by "
                         "this code."),
            },
            "distance": {
                "definition": ("tip_smooth to nearest point of the tumor's "
                               "convex hull (G.outer_perimeter), from "
                               "G.perp_direction_at_tip -- the point the "
                               "sweep edge is oriented from; unsigned; "
                               "processing px"),
                "contour_definition": ("tip_smooth to nearest point of the "
                                       "largest external contour of the tumor "
                                       "mask (G.contour_of), processing px"),
            },
            "margin_model": "not loaded by this node",
            "logging": {
                "raw": "PNG, processing frame, compression level "
                       f"{int(self.get_parameter('raw_png_compression').value)}",
                "snapshot_every": int(self.get_parameter("snapshot_every").value),
            },
        })
        tap = os.path.join(os.path.expanduser("~"), ".cache", "tapnextpp",
                           "tapnextpp_512.ckpt")
        kp = os.environ.get("KEYPOINT_MODEL",
                            os.path.join(REPO_ROOT, "checkpoints_keypoint",
                                         "keypoint_rcnn.onnx"))
        xl.gather_background({
            "stage1_tumor_model": G.SPATIAL_MODEL,
            "stage1_tumor_meta": G.SPATIAL_META,
            "keypoint_rcnn": kp if self.tip_source == "detector" else None,
            "tapnextpp": tap if self.tip_source != "pose_only" else None,
        }, REPO_ROOT)
        xl.event("session_start",
                 f"tip_source={self.tip_source} trigger={self.trigger_mode} "
                 f"camera_model={self.camera_model}")

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
        # An episode in progress holds the whole accumulated mask in memory and
        # writes it only when it ends, so shutting down without ending it threw
        # the run away -- exactly the run someone is most likely to interrupt.
        try:
            ok, message = node._end_episode(source="ctrl_c")
            if ok:
                node.get_logger().info(f"Interrupted -> {message}")
        except Exception as e:
            node.get_logger().error(f"Could not save on interrupt: {e}")
    finally:
        if getattr(node, "xlog", None) is not None:
            node.xlog.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
