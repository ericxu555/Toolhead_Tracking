#!/usr/bin/env python3
"""Validate an experiment session written by ros_toolhead_pose_node.py
(-p experiment:=true).

    python3 tools/check_session_logs.py Output/session_20261006-164230

Checks that every file is present and parses, that the files agree with each
other, and reports what the logs say about the run itself:

  * frames.csv is contiguous, and stamp gaps estimate how many frames the
    camera published that never reached the node (the node subscribes with
    a depth-1 queue, so frames overwritten while one is processed leave no
    row -- only a gap in the stamps)
  * every raw PNG is accounted for by a frames.csv row and vice versa
  * the distance drawn on the overlay ("d = NN px") equals round() of the
    logged hull distance on every frame it was shown
  * overlay.mp4 has exactly as many frames as overlay_frames.csv has rows
  * snapshots/final_accum.png equals the episode's final_mask.png
  * log drops, queue high-water marks and storage rate

Exit status is 0 when there are no errors (warnings allowed), 1 otherwise.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:            # file checks still run; image checks are skipped
    cv2 = None


class Report:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def err(self, m):
        self.errors.append(m)
        print(f"  ERROR  {m}")

    def warn(self, m):
        self.warnings.append(m)
        print(f"  warn   {m}")

    def ok(self, m):
        print(f"  ok     {m}")

    def info(self, m):
        print(f"         {m}")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def stamp_of(row):
    try:
        return int(row["stamp_sec"]) + int(row["stamp_nsec"]) * 1e-9
    except (KeyError, ValueError):
        return None


def check_session(sdir, R):
    print(f"\n=== session {sdir}")
    cfg_p = os.path.join(sdir, "run_config.json")
    if not os.path.isfile(cfg_p):
        R.err("run_config.json missing")
        cfg = {}
    else:
        cfg = json.load(open(cfg_p))
        for k in ("params", "constants", "image", "resize", "pose", "trigger",
                  "checkpoints", "git"):
            if k not in cfg:
                R.err(f"run_config.json has no '{k}'")
        ck = cfg.get("checkpoints", {})
        for name, v in ck.items():
            if v is None:
                continue
            if not v.get("exists", False):
                R.warn(f"checkpoint {name} not found at {v.get('path')}")
            elif "sha256" not in v:
                R.err(f"checkpoint {name} has no sha256")
        if ck:
            R.ok("checkpoints: " + ", ".join(
                f"{n}={v['sha256'][:12]}" for n, v in ck.items()
                if v and "sha256" in v))
        rz = cfg.get("resize", {})
        if not isinstance(rz.get("native_hw"), list):
            R.warn("resize.native_hw never filled (no frame received?)")
        else:
            R.ok(f"resize {rz['native_hw']} -> {rz.get('proc_hw')} "
                 f"({rz.get('method')}, crop {rz.get('crop')})")
        cal = (cfg.get("pose") or {}).get("calibration")
        if cal:
            p = os.path.join(sdir, cal.get("copied_to", ""))
            if os.path.isfile(p):
                R.ok(f"calibration copy {cal.get('copied_to')} "
                     f"(from {cal.get('original_path')}, mtime {cal.get('mtime')})")
            else:
                R.err("calibration copy missing")
        vs = cfg.get("ves_smoother", {})
        if "params" in vs:
            p = vs["params"]
            R.ok("ves_smoother params: " + ", ".join(
                f"{k}={p[k]}" for k in ("camera_lag_seconds",
                                        "ignore_keypoints_mode",
                                        "right.tip_offset.z") if k in p))
        else:
            R.warn(f"ves_smoother params unavailable: {vs.get('unavailable', '?')[:120]}")
        drops = cfg.get("log_drops_total", {})
        if drops:
            R.warn(f"logging drops in session: {drops}")

    # ---------------------------------------------------------------- frames
    fp = os.path.join(sdir, "frames.csv")
    frames = read_csv(fp) if os.path.isfile(fp) else []
    if not frames:
        R.err("frames.csv missing or empty")
    else:
        idx = [int(r["frame_idx"]) for r in frames]
        if idx != list(range(idx[0], idx[0] + len(idx))):
            R.err("frames.csv frame_idx not contiguous")
        else:
            R.ok(f"frames.csv {len(frames)} rows, frame_idx contiguous")
        st = np.array([stamp_of(r) for r in frames], dtype=np.float64)
        d = np.diff(st)
        if (d < 0).any():
            R.warn(f"{int((d < 0).sum())} image stamps go backwards")
        dpos = d[d > 0]
        if len(dpos):
            med = float(np.median(dpos))
            gaps = np.round(d / med) - 1
            gaps[gaps < 0] = 0
            R.ok(f"camera period ~{med * 1e3:.1f} ms ({1 / med:.1f} Hz)")
            frames_by_ep = {}
            for k, r in enumerate(frames[1:], start=1):
                frames_by_ep.setdefault(r["episode"], []).append(gaps[k - 1])
            for ep, g in frames_by_ep.items():
                if not ep:
                    continue
                miss = int(np.sum(g))
                got = len(g)
                R.info(f"{ep}: {got} frames reached the node, ~{miss} more were "
                       f"published but overwritten upstream "
                       f"({100.0 * miss / max(miss + got, 1):.1f}%)")
        drop = [r for r in frames if r["dropped"] == "1"]
        if drop:
            R.warn(f"{len(drop)} frames dropped in-node (reasons: "
                   f"{sorted(set(r['drop_reason'] for r in drop))})")
        pm = np.array([float(r["proc_ms"]) for r in frames
                       if r["processed"] == "1"] or [np.nan])
        if np.isfinite(pm).any():
            R.ok(f"callback time (processed frames) median {np.nanmedian(pm):.1f} ms, "
                 f"p95 {np.nanpercentile(pm, 95):.1f} ms, max {np.nanmax(pm):.1f} ms")

    # ---------------------------------------------------------------- events
    ep_ = os.path.join(sdir, "events.csv")
    events = read_csv(ep_) if os.path.isfile(ep_) else []
    if not events:
        R.err("events.csv missing or empty")
    else:
        names = [e["event"] for e in events]
        ns, ne = names.count("episode_start"), names.count("episode_end")
        if ns != ne:
            R.err(f"episode_start x{ns} but episode_end x{ne}")
        R.ok(f"events.csv {len(events)} rows: " + ", ".join(
            f"{n}x{names.count(n)}" for n in sorted(set(names))))
        if names.count("log_drop") or names.count("log_drop_raw"):
            R.warn("logging drops recorded in events.csv")
        if "frame_failed" in names:
            R.warn(f"{names.count('frame_failed')} frames raised in processing")

    md = os.path.join(sdir, "metadata.csv")
    meta_rows = read_csv(md) if os.path.isfile(md) else []

    eps = sorted(d for d in glob.glob(os.path.join(sdir, "episode_*"))
                 if os.path.isdir(d))
    if len(meta_rows) != len(eps):
        R.err(f"metadata.csv has {len(meta_rows)} rows for {len(eps)} episodes")
    else:
        R.ok(f"metadata.csv {len(meta_rows)} rows")
    for e in eps:
        check_episode(e, frames, R)
    return eps


def check_episode(edir, frames, R):
    name = os.path.basename(edir)
    print(f"\n--- {name}")
    meta_p = os.path.join(edir, "meta.json")
    if os.path.isfile(meta_p):
        m = json.load(open(meta_p))
        miss = [k for k in ("tumor_id", "condition", "operator", "trial",
                            "overlay_on") if k not in m]
        if miss:
            R.err(f"meta.json missing {miss}")
        else:
            R.ok(f"meta: tumor_id={m['tumor_id']!r} condition={m['condition']!r} "
                 f"operator={m['operator']!r} trial={m['trial']} "
                 f"overlay_on={m['overlay_on']}")
            if not m["tumor_id"] or not m["condition"] or not m["operator"]:
                R.warn("meta has empty tumor_id/condition/operator")
    else:
        R.err("meta.json missing")

    ep_frames = [r for r in frames if r["episode"] == name]
    saved = {int(r["frame_idx"]) for r in ep_frames if r["raw_saved"] == "1"}
    raws = glob.glob(os.path.join(edir, "raw", "*.png"))
    on_disk = {int(os.path.splitext(os.path.basename(p))[0]) for p in raws}
    if saved != on_disk:
        R.err(f"raw/: {len(on_disk)} files vs {len(saved)} rows marked saved "
              f"(missing {len(saved - on_disk)}, extra {len(on_disk - saved)})")
    else:
        R.ok(f"raw/: {len(on_disk)} PNGs, one per received frame "
             f"({len(ep_frames) - len(saved)} not saved)")
    if cv2 is not None and raws:
        shapes = {cv2.imread(p).shape for p in (min(raws), max(raws))}
        R.ok(f"raw frame shape {sorted(shapes)}")

    tp = os.path.join(edir, "tips.csv")
    tips = read_csv(tp) if os.path.isfile(tp) else []
    if not tips:
        R.warn("tips.csv missing or empty (episode never tracked?)")
    else:
        valid = [t for t in tips if t["tip_valid"] == "1"]
        rej = [t["reject_reason"] for t in tips if t["tip_valid"] != "1"]
        R.ok(f"tips.csv {len(tips)} rows, {len(valid)} with a tip"
             + (f", rejected: { {r: rej.count(r) for r in set(rej)} }" if rej else ""))
        fidx = {int(r["frame_idx"]) for r in ep_frames}
        if not all(int(t["frame_idx"]) in fidx for t in tips):
            R.err("tips.csv frame_idx not all in frames.csv for this episode")
        # Displayed distance == logged hull distance.
        shown = [t for t in tips if t["overlay_on"] == "1" and t["tip_valid"] == "1"]
        bad = 0
        for t in shown:
            if t["hull_valid"] == "1" and t.get("hull_dist_mm", "") != "":
                want = f"d = {float(t['hull_dist_mm']):.1f} mm"
            elif t["hull_valid"] == "1" and t["hull_dist_px"] != "":
                want = f"d = {int(round(float(t['hull_dist_px'])))} px"
            else:
                want = "d = --"
            if t["d_shown"] != want:
                bad += 1
        if shown:
            if bad:
                R.err(f"d_shown disagrees with the logged hull distance on {bad}/{len(shown)} frames")
            else:
                R.ok(f"d_shown == the logged hull distance (mm, else px) on all {len(shown)} displayed frames")
        hv = np.array([float(t["hull_dist_px"]) for t in valid
                       if t["hull_dist_px"] != ""] or [np.nan])
        cv = np.array([float(t["contour_dist_px"]) for t in valid
                       if t["contour_dist_px"] != ""] or [np.nan])
        if np.isfinite(hv).any():
            R.info(f"hull distance px: median {np.nanmedian(hv):.1f}, "
                   f"min {np.nanmin(hv):.1f}, max {np.nanmax(hv):.1f}; "
                   f"contour distance median {np.nanmedian(cv):.1f}")
        sig = np.array([float(t["pos_sigma_mm"]) for t in tips
                        if t.get("pos_sigma_mm", "") != ""] or [np.nan])
        if np.isfinite(sig).any():
            R.info(f"pose position sigma mm: median {np.nanmedian(sig):.3f}, "
                   f"max {np.nanmax(sig):.3f} (growth = keypoints dropping out)")
        dt = np.array([float(t["image_minus_pose_ms"]) for t in tips
                       if t.get("image_minus_pose_ms", "") != ""] or [np.nan])
        if np.isfinite(dt).any():
            R.info(f"image - pose stamp ms: median {np.nanmedian(dt):.1f}, "
                   f"range {np.nanmin(dt):.1f}..{np.nanmax(dt):.1f}")

    of = os.path.join(edir, "overlay_frames.csv")
    ov = read_csv(of) if os.path.isfile(of) else []
    mp4 = os.path.join(edir, "overlay.mp4")
    if ov and cv2 is not None and os.path.isfile(mp4):
        cap = cv2.VideoCapture(mp4)
        n = 0
        while cap.read()[0]:
            n += 1
        cap.release()
        if n != len(ov):
            R.err(f"overlay.mp4 has {n} frames, overlay_frames.csv {len(ov)} rows")
        else:
            R.ok(f"overlay.mp4 {n} frames == overlay_frames.csv rows")
    elif ov:
        R.warn("overlay.mp4 missing or cv2 unavailable")

    snap = os.path.join(edir, "snapshots")
    fm = os.path.join(edir, "final_mask.png")
    fa = os.path.join(snap, "final_accum.png")
    if os.path.isfile(fm):
        if not os.path.isfile(fa):
            R.err("snapshots/final_accum.png missing")
        elif cv2 is not None:
            a, b = cv2.imread(fm, 0), cv2.imread(fa, 0)
            if a is None or b is None or a.shape != b.shape or (a != b).any():
                R.err("snapshots/final_accum.png != final_mask.png")
            else:
                R.ok("snapshots/final_accum.png == final_mask.png")
        for f in ("final_cutline_x.npy", "final_cut_path.npy",
                  "final_tip_history.npy", "final_mask_tips.npy"):
            if not os.path.isfile(os.path.join(snap, f)):
                R.err(f"snapshots/{f} missing")
        n_snap = len(glob.glob(os.path.join(snap, "*_accum.png"))) - 1
        R.ok(f"snapshots: {n_snap} periodic + finals")

    ls = os.path.join(edir, "log_summary.json")
    if os.path.isfile(ls):
        s = json.load(open(ls))
        R.ok(f"storage {s.get('raw_mb_per_frame')} MB/frame, "
             f"{s.get('raw_gb_per_min')} GB/min as run, "
             f"{s.get('raw_gb_per_min_at_30fps')} GB/min at 30 fps; queue "
             f"high-water ordered {s.get('ordered_queue_high_water')}, "
             f"raw {s.get('raw_queue_high_water')}")
    else:
        R.err("log_summary.json missing")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    R = Report()
    for sdir in sys.argv[1:]:
        check_session(os.path.abspath(sdir), R)
    print(f"\n{len(R.errors)} error(s), {len(R.warnings)} warning(s)")
    sys.exit(1 if R.errors else 0)


if __name__ == "__main__":
    main()
