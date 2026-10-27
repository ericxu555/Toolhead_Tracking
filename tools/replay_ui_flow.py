#!/usr/bin/env python3
"""Check the readouts the operator UI depends on, with rclpy STUBBED.

Plays one session through ros_toolhead_pose_node.py and checks /toolhead/status:

    1. IDLE, no pose (tool out of view)   -> baseline captured, state "ready"
    2. press, frames with the tip pose    -> "estimate", d_mm == d_px * z / fx
    3. release, tip pose off the image    -> "final", written to episode_info.json
    4. recalibrate (fx changes), press    -> the new calibration is used at once

    python tools/replay_ui_flow.py --frames FRAMES --tips tip_track.csv --out OUT
"""
import argparse
import glob
import importlib.util
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_replay_test as R          # noqa: E402  (stubs, message builders)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", default=os.path.join(HERE, "..", "edge_anchored_tracking",
                                                   "ros_toolhead_pose_node.py"))
    ap.add_argument("--frames", required=True)
    ap.add_argument("--tips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cut-frames", type=int, default=120)
    ap.add_argument("--dump", help="pickle every published overlay and status message here, in order")
    args = ap.parse_args()

    import csv
    import cv2
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    calib = os.path.join(out, "calib.yaml")
    R.write_calib(calib)
    R.OVERRIDES.update({
        "tip_source": "pose_only", "camera_model": "rect",
        "image_topic": "/ves_camera/image_rect", "calib_file": calib,
        "output_root": os.path.join(out, "Output"), "trigger_mode": "service",
    })
    R.install_stubs()
    spec = importlib.util.spec_from_file_location("node_under_test", os.path.abspath(args.node))
    nodemod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nodemod)
    node = nodemod.ToolheadNode()
    os.makedirs(node.output_root, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.frames, "*.png")))
    tips = {int(r["frame"]): (float(r["x"]), float(r["y"])) for r in csv.DictReader(open(args.tips))}
    pose_cb = node._subs[node.pose_topic][0]
    img_cb = node._subs[node.image_topic][0]
    t_ns = [1_700_000_000_000_000_000]

    def frame(k, tip=None):
        """Feed frame k; tip=(u, v) also sends a pose, tip='off' one that projects off the image."""
        t_ns[0] += 33_333_333
        if tip == "off":
            pose_cb(R.pose_msg(-400.0, -400.0, t_ns[0], 0.0005))
        elif tip is not None:
            pose_cb(R.pose_msg(tip[0], tip[1], t_ns[0], 0.0005))
        native = cv2.resize(cv2.imread(files[k]), (R.NATIVE, R.NATIVE), interpolation=cv2.INTER_NEAREST)
        before = len(node.published)
        img_cb(R.image_msg(native, t_ns[0]))
        return [json.loads(m.data) for topic, m in node.published[before:] if topic == "/toolhead/status"]

    def call(name):
        return node._services[name](None, types.SimpleNamespace(success=None, message=None))

    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # 1. Before any cut, tool out of view.
    idle = [s for k in range(8) for s in frame(k)]
    check("idle publishes status every frame", len(idle) == 8, f"{len(idle)} of 8")
    check("idle overlay published", sum(1 for t, _ in node.published if t == "/toolhead/overlay") >= 8)
    states = [s["fraction_state"] for s in idle]
    check("baseline -> ready after 5 tool-free frames",
          states[:4] == ["baseline"] * 4 and states[4:] == ["ready"] * 4, str(states))

    # 2. The cut.
    call("/toolhead/start")
    cut = [s for k in range(8, 8 + args.cut_frames) for s in frame(k, tips[k])]
    est = [s for s in cut if s["fraction_state"] == "estimate" and s["fraction"] is not None]
    check("estimate published during the cut", len(est) > 0.8 * len(cut), f"{len(est)}/{len(cut)}")
    fr = [s["fraction"] for s in est]
    check("estimate within [0, 1] and non-decreasing", all(0 <= f <= 1 for f in fr) and
          all(b >= a - 1e-9 for a, b in zip(fr, fr[1:])), f"first {fr[:1]}, last {fr[-1:]}")
    pairs = [(s["d_px"], s["d_mm"]) for s in cut if s["d_px"] is not None]
    scale = R.Z * 1000.0 / R.FX                         # mm per px: synthetic calibration at 540 px
    err = max((abs(mm - px * scale) for px, mm in pairs), default=np.nan)
    check("d_mm == d_px * z / fx", pairs and err < 1e-2, f"{len(pairs)} frames, max err {err:.4f} mm")
    warn_ok = all(s["warn"] == (s["d_mm"] is not None and s["d_mm"] < s["warn_mm"]) for s in cut)
    check("warn exactly when d_mm < warn_mm", warn_ok)
    call("/toolhead/stop")
    ep1 = sorted(glob.glob(os.path.join(node.output_root, "episode_*")))[-1]

    # 3. After the release, the tool leaves.
    post = [s for k in range(8 + args.cut_frames, 8 + args.cut_frames + 8) for s in frame(k, "off")]
    states = [s["fraction_state"] for s in post]
    check("final after 5 frames with the tip out", "final" in states and states.index("final") == 4, str(states))
    info = json.load(open(os.path.join(ep1, "episode_info.json")))
    check("episode_info has calibration and both fractions",
          info.get("calibration") and info.get("fraction_estimate") is not None
          and info.get("fraction_final") is not None, json.dumps({k: info.get(k) for k in (
              "calibration", "baseline_px", "fraction_estimate", "fraction_final",
              "fraction_final_reason", "final_tumor_px")}))
    check("final stays until the next press", post[-1]["fraction_state"] == "final")

    # 4. Recalibrate between cuts: fx 500 -> 400 must change the mm scale at the next press.
    txt = open(calib).read().replace(f"{R.FX}", "400.0")
    open(calib, "w").write(txt)
    call("/toolhead/start")
    cut2 = [s for k in range(200, 260) for s in frame(k, tips[k])]
    call("/toolhead/stop")
    pairs2 = [(s["d_px"], s["d_mm"]) for s in cut2 if s["d_px"] is not None]
    scale2 = R.Z * 1000.0 / 400.0
    err2 = max((abs(mm - px * scale2) for px, mm in pairs2), default=np.nan)
    check("recalibration picked up at the next press", pairs2 and err2 < 1e-2,
          f"{len(pairs2)} frames, max err {err2:.4f} mm at the new scale")
    reloads = [m for lvl, m in node.logs if lvl == "I" and m.startswith("Calibration [")]
    check("calibration read at startup and at each press", len(reloads) == 3, f"{len(reloads)} reads")
    check("no errors logged", not [m for lvl, m in node.logs if lvl == "E"],
          str([m for lvl, m in node.logs if lvl == "E"][:3]))

    if args.dump:
        import pickle
        keep = []
        for topic, m in node.published:
            if topic == "/toolhead/status":
                keep.append((topic, m.data))
            elif topic == "/toolhead/overlay":
                keep.append((topic, (m.height, m.width, m.encoding, bytes(m.data))))
        with open(args.dump, "wb") as f:
            pickle.dump(keep, f)

    width = max(len(n) for n, _, _ in checks)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print(f"\n{sum(ok for _, ok, _ in checks)}/{len(checks)} passed")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
