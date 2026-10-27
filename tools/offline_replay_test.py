#!/usr/bin/env python3
"""Replay recorded frames through ros_toolhead_pose_node.py with rclpy STUBBED.

No ROS, no robot, no camera: the node's real callbacks are called directly
with synthetic messages built from a folder of frames and a per-frame tip CSV.
Used to show that experiment logging and the distance readout change nothing
the node computes.

    # one run (writes <out>/run.json plus the node's normal outputs)
    python tools/offline_replay_test.py run --node edge_anchored_tracking/ros_toolhead_pose_node.py \
        --frames E:/frames --tips tip_track.csv --out OUT --params '{"experiment": true}'

    # compare runs
    python tools/offline_replay_test.py compare OUT_A OUT_B ...

    # logging overhead, paired frame by frame in one process
    python tools/offline_replay_test.py paired --node ... --frames ... --tips ... --out OUT

The tip CSV needs columns frame,x,y in processing-frame (540) pixels, row k
being the k-th frame of the sorted folder. Each tip is back-projected through
a synthetic rectified calibration into a PoseWithCovarianceStamped, so the
node runs tip_source:=pose_only and projects it back to the same pixel.

Each run plays the same session: IDLE frames, episode 1 over all frames,
IDLE frames, then a short episode 2 with different study parameters (and the
distance overlay off), so parameter switching between trials is exercised.
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time
import types

import numpy as np

FX = FY = 500.0
CX = CY = 270.0
CALIB_WH = 540
Z = 0.05
NATIVE = 1080
EP2_FRAMES = 60
IDLE_FRAMES = 3


# =================================================================== stubs

class _Stamp:
    def __init__(self, ns):
        self.sec = int(ns // 1_000_000_000)
        self.nanosec = int(ns % 1_000_000_000)


class _Header:
    def __init__(self, ns=0, frame_id=""):
        self.stamp = _Stamp(ns)
        self.frame_id = frame_id


class _Msg:
    def __init__(self, *a, **k):
        self.header = _Header()


class Image(_Msg):
    pass


class JointState(_Msg):
    pass


class PointStamped(_Msg):
    def __init__(self):
        super().__init__()
        self.point = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)


class PoseStamped(_Msg):
    pass


class PoseWithCovarianceStamped(_Msg):
    pass


class Trigger:
    pass


class String:
    def __init__(self):
        self.data = ""


class _Param:
    def __init__(self, v):
        self.value = v


class _Logger:
    def __init__(self, sink):
        self.sink = sink

    def _log(self, lvl, m, **k):
        self.sink.append((lvl, str(m)))

    def info(self, m, **k):
        self._log("I", m)

    def warn(self, m, **k):
        self._log("W", m)

    warning = warn

    def error(self, m, **k):
        self._log("E", m)


class _Clock:
    def now(self):
        return types.SimpleNamespace(nanoseconds=time.time_ns())


class _Pub:
    def __init__(self, topic, store):
        self.topic = topic
        self.store = store

    def publish(self, msg):
        self.store.append((self.topic, msg))


class _Timer:
    def cancel(self):
        pass


OVERRIDES = {}
POSE_TOPIC_TYPE = "geometry_msgs/msg/PoseWithCovarianceStamped"


class Node:
    def __init__(self, name):
        self._parameters = {}
        self._subs = {}
        self._services = {}
        self.published = []
        self.logs = []
        self._logger = _Logger(self.logs)

    def declare_parameter(self, name, default):
        self._parameters[name] = _Param(OVERRIDES.get(name, default))

    def get_parameter(self, name):
        return self._parameters[name]

    def set_param(self, name, value):          # harness only
        if name in self._parameters:
            self._parameters[name] = _Param(value)

    def get_logger(self):
        return self._logger

    def get_clock(self):
        return _Clock()

    def create_publisher(self, typ, topic, qos):
        return _Pub(topic, self.published)

    def create_subscription(self, typ, topic, cb, qos):
        self._subs.setdefault(topic, []).append(cb)

    def create_service(self, typ, name, cb):
        self._services[name] = cb

    def create_timer(self, period, cb):
        return _Timer()

    def get_topic_names_and_types(self):
        return [(OVERRIDES.get("pose_topic", "/ves_smoother/right/tip_pose"),
                 [POSE_TOPIC_TYPE])]

    def destroy_node(self):
        pass


def install_stubs():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    class _QoS:
        def __init__(self, **k):
            pass

    rclpy = mod("rclpy", init=lambda *a, **k: None, spin=lambda n: None,
                shutdown=lambda: None)
    rclpy.node = mod("rclpy.node", Node=Node)
    rclpy.qos = mod("rclpy.qos", QoSProfile=_QoS,
                    ReliabilityPolicy=types.SimpleNamespace(RELIABLE=1, BEST_EFFORT=2),
                    HistoryPolicy=types.SimpleNamespace(KEEP_LAST=1))
    mod("sensor_msgs")
    mod("sensor_msgs.msg", Image=Image, JointState=JointState)
    mod("geometry_msgs")
    mod("geometry_msgs.msg", PointStamped=PointStamped, PoseStamped=PoseStamped,
        PoseWithCovarianceStamped=PoseWithCovarianceStamped)
    mod("std_msgs")
    mod("std_msgs.msg", String=String)
    mod("std_srvs")
    mod("std_srvs.srv", Trigger=Trigger)


# ================================================================= messages

def image_msg(bgr, ns):
    m = Image()
    m.header = _Header(ns, "ves_camera")
    m.height, m.width = bgr.shape[:2]
    m.encoding = "bgr8"
    m.data = bgr.tobytes()
    return m


def pose_msg(u, v, ns, sigma_m):
    x = (u - CX) / FX * Z
    y = (v - CY) / FY * Z
    m = PoseWithCovarianceStamped()
    m.header = _Header(ns, "ves_camera")
    pos = types.SimpleNamespace(x=x, y=y, z=Z)
    cov = np.zeros(36)
    for i in range(3):
        cov[i * 7] = 1e-4                       # rotation, rad^2
    for i in range(3, 6):
        cov[i * 7] = sigma_m ** 2               # position, m^2
    m.pose = types.SimpleNamespace(pose=types.SimpleNamespace(position=pos),
                                   covariance=cov)
    return m


def joint_msg(value, ns):
    m = JointState()
    m.header = _Header(ns)
    m.name = ["j0", "tool"]
    m.position = [0.0, float(value)]
    return m


def write_calib(path):
    with open(path, "w") as f:
        f.write(
            f"image_width: {CALIB_WH}\nimage_height: {CALIB_WH}\n"
            "camera_name: synthetic\n"
            "camera_matrix:\n  rows: 3\n  cols: 3\n"
            f"  data: [{FX}, 0, {CX}, 0, {FY}, {CY}, 0, 0, 1]\n"
            "distortion_model: plumb_bob\n"
            "distortion_coefficients:\n  rows: 1\n  cols: 5\n"
            "  data: [0, 0, 0, 0, 0]\n"
            "projection_matrix:\n  rows: 3\n  cols: 4\n"
            f"  data: [{FX}, 0, {CX}, 0, 0, {FY}, {CY}, 0, 0, 0, 1, 0]\n")


# ====================================================================== run

def run(args):
    import cv2
    params = json.loads(args.params)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    calib = os.path.join(out, "synthetic_calib.yaml")
    write_calib(calib)

    OVERRIDES.update({
        "tip_source": "pose_only", "camera_model": "rect",
        "image_topic": "/ves_camera/image_rect", "calib_file": calib,
        "output_root": os.path.join(out, "Output"), "trigger_mode": "service",
    })
    OVERRIDES.update(params)
    install_stubs()

    node_path = os.path.abspath(args.node)
    spec = importlib.util.spec_from_file_location("node_under_test", node_path)
    nodemod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nodemod)

    node = nodemod.ToolheadNode()
    os.makedirs(node.output_root, exist_ok=True)

    # Spy on the distance text actually drawn; it records, it changes nothing.
    drawn = []
    if hasattr(node, "_draw_distance"):
        orig_draw = node._draw_distance

        def spy(img, text, warn):
            drawn.append(text)
            return orig_draw(img, text, warn)
        node._draw_distance = spy

    files = sorted(glob.glob(os.path.join(args.frames, "*.png")))
    tips = {}
    import csv
    for r in csv.DictReader(open(args.tips)):
        tips[int(r["frame"])] = (float(r["x"]), float(r["y"]))
    n = min(len(files), len(tips)) if args.max_frames <= 0 else \
        min(len(files), len(tips), args.max_frames)

    pose_cb = node._subs.get(node.pose_topic, [None])[0]
    img_cb = node._subs[node.image_topic][0]
    joint_cbs = node._subs.get(node.tool_topic, [])

    t_ns = 1_700_000_000_000_000_000
    period = 33_333_333
    timings = []
    pub_hash = []           # per published overlay: sha1 of the image bytes

    def frame(k, publish_pose=True):
        nonlocal t_ns
        t_ns += period
        img = cv2.imread(files[k])
        native = cv2.resize(img, (NATIVE, NATIVE), interpolation=cv2.INTER_NEAREST)
        u, v = tips[k]
        # Covariance grows over frames 100-130 to stand in for a keypoint
        # dropout; only the log should see it.
        sigma = 0.0005 if not (100 <= k < 130) else 0.0005 + 0.0002 * (k - 100)
        if pose_cb is not None and publish_pose:
            pose_cb(pose_msg(u, v, t_ns - 70_000_000, sigma))
        before = len(node.published)
        t0 = time.perf_counter()
        img_cb(image_msg(native, t_ns))
        timings.append((time.perf_counter() - t0) * 1e3)
        for topic, m in node.published[before:]:
            if topic == "/toolhead/overlay":
                pub_hash.append(hashlib.sha1(bytes(m.data)).hexdigest())

    def joints(value):
        for cb in joint_cbs:
            cb(joint_msg(value, t_ns))

    def call(name):
        resp = types.SimpleNamespace(success=None, message=None)
        return node._services[name](None, resp)

    for k in range(IDLE_FRAMES):
        frame(k)
    joints(1.0)
    call("/toolhead/start")
    for k in range(n):
        frame(k)
    call("/toolhead/stop")
    joints(0.0)
    for k in range(IDLE_FRAMES):
        frame(k)

    # Trial 2: study parameters changed between episodes.
    for name, val in (("tumor_id", "T02"), ("condition", "no_overlay"),
                      ("operator", "op_b"), ("trial", 7),
                      ("show_distance_overlay", False)):
        node.set_param(name, val)
    joints(1.0)
    call("/toolhead/start")
    for k in range(min(EP2_FRAMES, n)):
        frame(k)
    call("/toolhead/stop")
    joints(0.0)

    if getattr(node, "xlog", None) is not None:
        node.xlog.close()

    eps = sorted(glob.glob(os.path.join(node.output_root, "**", "episode_*"),
                           recursive=True))
    eps = [e for e in eps if os.path.isdir(e)]
    res = {
        "node": node_path, "params": params, "frames": n,
        "episodes": eps,
        "timing_ms": {"mean": float(np.mean(timings)),
                      "median": float(np.median(timings)),
                      "p95": float(np.percentile(timings, 95)),
                      "n": len(timings)},
        "pub_overlay_sha1": pub_hash,
        "drawn_texts": drawn,
        "errors": [m for lvl, m in node.logs if lvl == "E"],
        "warnings": [m for lvl, m in node.logs if lvl == "W"],
        "session_dir": getattr(getattr(node, "xlog", None), "session_dir", None),
    }
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("frames", "timing_ms", "episodes",
                                          "session_dir")}, indent=1))
    print("errors:", res["errors"][:5])


# ================================================================== compare

OUTPUTS = ("final_mask.png", "final_overlay.png", "cutline_overlay.png")


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def compare(args):
    runs = []
    for d in args.dirs:
        with open(os.path.join(d, "run.json")) as f:
            runs.append((d, json.load(f)))
    ref_dir, ref = runs[0]
    ok = True
    print(f"reference: {ref_dir}  ({ref['params']})")
    for d, r in runs[1:]:
        print(f"\n== {d}  ({r['params']})")
        if len(r["episodes"]) != len(ref["episodes"]):
            print("  episode count differs"); ok = False; continue
        for ea, eb in zip(ref["episodes"], r["episodes"]):
            for name in OUTPUTS:
                pa, pb = os.path.join(ea, name), os.path.join(eb, name)
                same = os.path.isfile(pa) and os.path.isfile(pb) and sha(pa) == sha(pb)
                ok &= same
                print(f"  {os.path.basename(eb)[:11]} {name:20s} "
                      f"{'IDENTICAL' if same else 'DIFFERENT'}")
            ia = json.load(open(os.path.join(ea, "episode_info.json")))
            ib = json.load(open(os.path.join(eb, "episode_info.json")))
            ia.pop("duration_sec", None); ib.pop("duration_sec", None)
            same = ia == ib
            ok &= same
            print(f"  {os.path.basename(eb)[:11]} episode_info (ex. duration) "
                  f"{'IDENTICAL' if same else f'DIFFERENT {ia} vs {ib}'}")
        ha, hb = ref["pub_overlay_sha1"], r["pub_overlay_sha1"]
        n_same = sum(a == b for a, b in zip(ha, hb))
        print(f"  published overlays: {n_same}/{len(ha)} identical "
              f"(count {len(ha)} vs {len(hb)})")
        print(f"  timing ms  ref mean {ref['timing_ms']['mean']:.2f} "
              f"median {ref['timing_ms']['median']:.2f} p95 {ref['timing_ms']['p95']:.2f}"
              f"  |  this mean {r['timing_ms']['mean']:.2f} "
              f"median {r['timing_ms']['median']:.2f} p95 {r['timing_ms']['p95']:.2f}")
    print("\nALL OUTPUTS BIT-IDENTICAL" if ok else "\nDIFFERENCES FOUND")
    return 0 if ok else 1


def paired(args):
    """Two nodes in ONE process -- logging/readout off vs on -- fed the same
    frames alternately. Process-to-process timing varies by tens of ms on a
    shared GPU; pairing frame by frame cancels that, so the difference is the
    cost of the logging and the readout themselves. Also checks that every
    published overlay differs from the plain one only inside the readout box,
    and that the two final masks match."""
    import csv
    import cv2
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    calib = os.path.join(out, "synthetic_calib.yaml")
    write_calib(calib)
    install_stubs()
    node_path = os.path.abspath(args.node)
    spec = importlib.util.spec_from_file_location("node_under_test", node_path)
    nodemod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nodemod)

    base = {"tip_source": "pose_only", "camera_model": "rect",
            "image_topic": "/ves_camera/image_rect", "calib_file": calib,
            "trigger_mode": "service"}
    nodes = {}
    for tag, extra in (("off", {"experiment": False,
                                "show_distance_overlay": False}),
                       ("on", {"experiment": True,
                               "show_distance_overlay": True})):
        OVERRIDES.clear()
        OVERRIDES.update(base, output_root=os.path.join(out, tag), **extra)
        n = nodemod.ToolheadNode()
        os.makedirs(n.output_root, exist_ok=True)
        nodes[tag] = n

    files = sorted(glob.glob(os.path.join(args.frames, "*.png")))
    tips = {int(r["frame"]): (float(r["x"]), float(r["y"]))
            for r in csv.DictReader(open(args.tips))}
    n = min(len(files), len(tips))
    if args.max_frames > 0:
        n = min(n, args.max_frames)

    for nd in nodes.values():
        nd._services["/toolhead/start"](None, types.SimpleNamespace())
    t_ns = 1_700_000_000_000_000_000
    times = {"off": [], "on": []}
    bbox_bad = 0
    bbox_union = None
    for k in range(n):
        t_ns += 33_333_333
        img = cv2.imread(files[k])
        native = cv2.resize(img, (NATIVE, NATIVE), interpolation=cv2.INTER_NEAREST)
        u, v = tips[k]
        pubs = {}
        order = ("off", "on") if k % 2 == 0 else ("on", "off")
        for tag in order:
            nd = nodes[tag]
            nd._subs[nd.pose_topic][0](pose_msg(u, v, t_ns - 70_000_000, 0.0005))
            before = len(nd.published)
            t0 = time.perf_counter()
            nd._subs[nd.image_topic][0](image_msg(native, t_ns))
            times[tag].append((time.perf_counter() - t0) * 1e3)
            for topic, m in nd.published[before:]:
                if topic == "/toolhead/overlay":
                    pubs[tag] = np.frombuffer(bytes(m.data), np.uint8).reshape(
                        m.height, m.width, 3)
        if "off" in pubs and "on" in pubs:
            d = np.any(pubs["off"] != pubs["on"], axis=2)
            if d.any():
                ys, xs = np.nonzero(d)
                bb = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                bbox_union = bb if bbox_union is None else (
                    min(bbox_union[0], bb[0]), min(bbox_union[1], bb[1]),
                    max(bbox_union[2], bb[2]), max(bbox_union[3], bb[3]))
                # Readout box at 540 is at most (4,4)-(~140,~32); at 1080
                # that is (8,8)-(~280,~64), plus a pixel of interpolation.
                if bb[2] > 300 or bb[3] > 80:
                    bbox_bad += 1
    for nd in nodes.values():
        nd._services["/toolhead/stop"](None, types.SimpleNamespace())
    nodes["on"].xlog.close()

    a, b = np.array(times["off"]), np.array(times["on"])
    d = b - a
    masks = []
    for tag in ("off", "on"):
        m = glob.glob(os.path.join(out, tag, "**", "final_mask.png"), recursive=True)
        masks.append(open(m[0], "rb").read() if m else None)
    res = {
        "frames": n,
        "off_ms": {"mean": a.mean(), "median": float(np.median(a)),
                   "p95": float(np.percentile(a, 95))},
        "on_ms": {"mean": b.mean(), "median": float(np.median(b)),
                  "p95": float(np.percentile(b, 95))},
        "paired_diff_ms": {"mean": d.mean(), "median": float(np.median(d)),
                           "p05": float(np.percentile(d, 5)),
                           "p95": float(np.percentile(d, 95))},
        "overlay_diff_bbox_union_native_px": bbox_union,
        "overlay_frames_with_diff_outside_box": bbox_bad,
        "final_masks_identical": masks[0] is not None and masks[0] == masks[1],
    }
    print(json.dumps(res, indent=1, default=float))
    with open(os.path.join(out, "paired.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("paired")
    p.add_argument("--node", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--tips", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-frames", type=int, default=0)
    a = sub.add_parser("run")
    a.add_argument("--node", required=True)
    a.add_argument("--frames", required=True)
    a.add_argument("--tips", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--params", default="{}")
    a.add_argument("--max-frames", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
    elif args.cmd == "paired":
        paired(args)
    else:
        sys.exit(compare(args))


if __name__ == "__main__":
    main()
