"""Experiment logging for ros_toolhead_pose_node.py (-p experiment:=true).

Imported only when the experiment flag is on, so a normal run never loads it.
Nothing here feeds back into tracking, smoothing, the mask, the cutline, the
EMA or the trigger: the node hands values over and this module only writes
them down.

The node's callbacks must stay as fast as they were, because the image
subscription keeps only the newest frame -- any time spent in the callback is
time in which the next frame can be overwritten. So the callbacks only
ENQUEUE. Two kinds of worker do the writing:

  ordered writer (1 thread)   every CSV row, the overlay video, snapshots,
                              episode open/close. One thread, so rows and
                              video frames stay in order.
  PNG writers   (N threads)   the raw frames, which are the expensive part.

Both queues are bounded. When one is full the LOGGING item is dropped (never a
processing frame), counted, and reported as a log_drop event, so a gap in the
logs is always visible as a gap rather than silently missing.

Session layout (one per node run):

  session_<stamp>/
    run_config.json        parameters, constants, checkpoints + sha256, git,
                           resize, calibration, ves_smoother params
    calibration/           copy of the YAML the projection used
    ves_smoother_params.yaml   `ros2 param dump /ves_smoother`, if reachable
    frames.csv             EVERY frame the node received, IDLE included
    events.csv             every event in the session
    metadata.csv           one row per episode (tumor_id, condition, ...)
    episode_NNN_<stamp>/
      final_mask.png ...   the node's usual outputs, unchanged
      meta.json            tumor_id, condition, operator, trial, overlay
      tips.csv             one row per processed tracking frame
      events.csv           this episode's events
      raw/NNNNNN.png       every received frame (540x540, as processed);
                           NNNNNN = session frame_idx
      overlay.mp4          what was displayed, with overlay_frames.csv
                           mapping each video frame to its frame_idx
      snapshots/           accumulated mask + tumor mask every N frames,
                           final cutline c[y], cut path and tip histories
      log_summary.json     frames saved/dropped, bytes, GB/min
"""
import csv
import hashlib
import json
import os
import queue
import shutil
import socket
import subprocess
import threading
import time
from datetime import datetime

import cv2
import numpy as np

FRAMES_COLS = [
    "frame_idx", "stamp_sec", "stamp_nsec", "stamp", "ros_time", "wall_time",
    "state", "state_after", "episode", "processed", "dropped", "drop_reason",
    "raw_saved", "proc_ms",
]

EVENTS_COLS = ["wall_time", "ros_time", "stamp", "frame_idx", "episode",
               "event", "detail"]

OVERLAY_COLS = ["video_frame", "frame_idx", "stamp_sec", "stamp_nsec", "kind"]

TIPS_COLS = [
    # identity and time
    "frame_idx", "episode_frame", "stamp_sec", "stamp_nsec", "stamp",
    "ros_time", "wall_time", "tip_source",
    # was a tip available this frame, and if not why
    "tip_valid", "reject_reason", "tracker_visible",
    # visual-servoing pose (pose_seed / pose_only)
    "pose_stamp", "image_minus_pose_ms", "pose_msgs_total", "pose_new",
    "pose_x", "pose_y", "pose_z", "pose_u", "pose_v",
    "cov_xx", "cov_yy", "cov_zz", "cov_xy", "cov_xz", "cov_yz",
    "pos_sigma_mm", "rot_sigma_deg", "cov_full",
    # tips, processing-frame pixels (proc_size space)
    "tip_raw_u", "tip_raw_v", "tip_smooth_u", "tip_smooth_v",
    # distance to the convex hull: the boundary the sweep's edge is cast to
    "hull_valid", "hull_dist_px", "hull_dist_mm", "hull_u", "hull_v", "tip_in_hull",
    # distance to the true (non-hull) tumor contour
    "contour_valid", "contour_dist_px", "contour_u", "contour_v",
    "tip_in_contour", "tip_on_tumor",
    # sweep
    "seg_end_u", "seg_end_v", "sweep_len_px", "angle_deg",
    "tumor_px", "accum_px",
    # display
    "overlay_on", "d_shown", "warn_px", "warn_mm", "warn_state",
    "proc_ms",
]

LABEL_NOTE = ("pixel columns are in the processing frame (proc_size, 540 by "
              "default); multiply by native/proc_size for camera pixels")

METADATA_COLS = [
    "session", "episode", "episode_dir", "tumor_id", "condition", "operator",
    "trial", "overlay_on", "show_distance_overlay", "warn_distance_px",
    "trigger_mode", "tip_source", "start_wall", "end_wall", "frames_processed",
    "final_mask_px", "end_reason",
]


def _stamp_parts(stamp):
    if stamp is None:
        return "", "", ""
    s, n = int(stamp.sec), int(stamp.nanosec)
    return s, n, f"{s + n * 1e-9:.9f}"


def _f(x, nd=3):
    """Format a float for CSV; blank for None/NaN."""
    if x is None:
        return ""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(x):
        return ""
    return f"{x:.{nd}f}"


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class ExperimentLog:

    def __init__(self, output_root, ros_now_ns, logger, contour_fn,
                 snapshot_every=15, png_level=1, png_workers=2,
                 raw_queue=240, ordered_queue=4000, overlay_fps=30.0):
        self.logger = logger
        self.ros_now_ns = ros_now_ns
        self.contour_fn = contour_fn
        self.snapshot_every = int(snapshot_every)
        self.png_params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_level)]
        self.overlay_fps = float(overlay_fps)

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_name = f"session_{stamp}"
        self.session_dir = os.path.join(output_root, self.session_name)
        os.makedirs(self.session_dir, exist_ok=True)

        # ---- main-thread state (touched only by the node's callbacks)
        self.frame_idx = -1
        self.episode = None          # name of the open episode, or None
        self.episode_dir = None
        self.episode_frame = 0
        self._tool_last = None
        self._pose_n_last = None
        self._snap_count = 0
        self._drops = {}             # kind -> count not yet reported
        self._drop_lock = threading.Lock()
        self.drops_total = {}

        # ---- config, shared with the background config thread
        self.config = {}
        self._cfg_lock = threading.Lock()

        # ---- writer-thread state
        self._files = {}
        self._writers = {}
        self._video = None
        self._video_n = 0
        self._ep_stats = None
        self._last_flush = time.time()

        self._oq = queue.Queue(maxsize=ordered_queue)
        self._pq = queue.Queue(maxsize=raw_queue)
        self._raw_bytes = 0
        self._raw_saved = 0
        self._raw_lock = threading.Lock()
        self._oq_high = 0
        self._pq_high = 0
        self._closed = False

        self._othread = threading.Thread(target=self._ordered_loop,
                                         name="xlog-ordered", daemon=True)
        self._othread.start()
        self._pthreads = []
        for i in range(max(int(png_workers), 1)):
            t = threading.Thread(target=self._png_loop, name=f"xlog-png{i}",
                                 daemon=True)
            t.start()
            self._pthreads.append(t)

        self._put(self._open_session_files, kind="session")

    # ================================================================ queues

    def _put(self, fn, *args, kind="row"):
        """Enqueue work for the ordered writer. Never blocks the caller."""
        try:
            self._oq.put_nowait((fn, args))
        except queue.Full:
            self._count_drop(kind)
            return False
        n = self._oq.qsize()
        if n > self._oq_high:
            self._oq_high = n
        self._report_drops()
        return True

    def _count_drop(self, kind):
        with self._drop_lock:
            self._drops[kind] = self._drops.get(kind, 0) + 1
            self.drops_total[kind] = self.drops_total.get(kind, 0) + 1

    def _report_drops(self):
        with self._drop_lock:
            if not self._drops:
                return
            pending, self._drops = self._drops, {}
        detail = ";".join(f"{k}={v}" for k, v in sorted(pending.items()))
        try:
            self._oq.put_nowait((self._write_event,
                                 (self._event_row("log_drop", detail),)))
        except queue.Full:
            with self._drop_lock:
                for k, v in pending.items():
                    self._drops[k] = self._drops.get(k, 0) + v

    def _ordered_loop(self):
        while True:
            try:
                item = self._oq.get(timeout=1.0)
            except queue.Empty:
                self._maybe_flush(force=True)
                continue
            if item is None:
                self._oq.task_done()
                break
            fn, args = item
            try:
                fn(*args)
            except Exception as e:  # a logging fault must never stop logging
                try:
                    self.logger.error(f"experiment log: {fn.__name__}: {e}")
                except Exception:
                    pass
            self._oq.task_done()
            self._maybe_flush()

    def _png_loop(self):
        while True:
            item = self._pq.get()
            if item is None:
                self._pq.task_done()
                break
            path, img, convert = item
            try:
                if convert is not None:
                    img = convert(img)
                ok, buf = cv2.imencode(".png", img, self.png_params)
                if ok:
                    with open(path, "wb") as f:
                        f.write(buf.tobytes())
                    with self._raw_lock:
                        self._raw_bytes += len(buf)
                        self._raw_saved += 1
            except Exception as e:
                try:
                    self.logger.error(f"experiment log: raw frame {path}: {e}")
                except Exception:
                    pass
            self._pq.task_done()

    def _maybe_flush(self, force=False):
        now = time.time()
        if not force and now - self._last_flush < 2.0:
            return
        self._last_flush = now
        for f in self._files.values():
            try:
                f.flush()
            except Exception:
                pass

    # ============================================================== helpers

    def _event_row(self, event, detail="", stamp=None):
        s = _stamp_parts(stamp)[2] if stamp is not None else ""
        return [f"{time.time():.6f}", self.ros_now_ns(), s,
                self.frame_idx if self.frame_idx >= 0 else "",
                self.episode or "", event, detail]

    def _open_csv(self, key, path, cols):
        f = open(path, "w", newline="")
        w = csv.writer(f)
        w.writerow(cols)
        self._files[key] = f
        self._writers[key] = w

    def _close_csv(self, key):
        f = self._files.pop(key, None)
        self._writers.pop(key, None)
        if f is not None:
            f.close()

    # ============================================================== session

    def _open_session_files(self):
        self._open_csv("frames", os.path.join(self.session_dir, "frames.csv"),
                       FRAMES_COLS)
        self._open_csv("events", os.path.join(self.session_dir, "events.csv"),
                       EVENTS_COLS)
        p = os.path.join(self.session_dir, "metadata.csv")
        self._open_csv("metadata", p, METADATA_COLS)

    def write_config(self, cfg):
        """Merge keys into run_config.json and rewrite it."""
        with self._cfg_lock:
            self.config.update(cfg)
            snap = json.loads(json.dumps(self.config, default=str))
        self._put(self._write_config_file, snap, kind="config")

    def _write_config_file(self, cfg):
        p = os.path.join(self.session_dir, "run_config.json")
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, p)

    def copy_calibration(self, path):
        if not path or not os.path.isfile(path):
            return None
        d = os.path.join(self.session_dir, "calibration")
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, os.path.basename(path))
        shutil.copy2(path, dst)
        return {
            "original_path": path,
            "copied_to": os.path.relpath(dst, self.session_dir),
            "mtime": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(),
            "sha256": sha256_of(path),
        }

    def gather_background(self, checkpoints, repo_dir, smoother_node="/ves_smoother"):
        """Hashes, git state and the smoother's parameters, off the main thread
        (hashing a checkpoint and asking another node both take seconds)."""
        def work():
            ck = {}
            for name, path in checkpoints.items():
                if path is None:
                    ck[name] = None
                    continue
                if isinstance(path, str) and not os.path.isfile(path):
                    ck[name] = {"path": path, "exists": False}
                    continue
                try:
                    ck[name] = {"path": path, "exists": True,
                                "bytes": os.path.getsize(path),
                                "sha256": sha256_of(path)}
                except Exception as e:
                    ck[name] = {"path": path, "error": str(e)}
            self.write_config({"checkpoints": ck})

            git = {}
            for key, cmd in (("commit", ["git", "rev-parse", "HEAD"]),
                             ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
                             ("dirty", ["git", "status", "--porcelain",
                                        "--untracked-files=no"])):
                try:
                    out = subprocess.run(cmd, cwd=repo_dir, capture_output=True,
                                         text=True, timeout=10)
                    git[key] = out.stdout.strip()
                except Exception as e:
                    git[key] = f"unavailable ({e})"
            if "dirty" in git:
                git["dirty"] = bool(git["dirty"]) if not str(
                    git["dirty"]).startswith("unavailable") else git["dirty"]
            self.write_config({"git": git})

            # ves_smoother parameters. Read-only: `ros2 param dump` only asks
            # the node for its values; it commands nothing.
            info = {"node": smoother_node}
            try:
                out = subprocess.run(["ros2", "param", "dump", smoother_node],
                                     capture_output=True, text=True, timeout=20)
                if out.returncode == 0 and out.stdout.strip():
                    p = os.path.join(self.session_dir, "ves_smoother_params.yaml")
                    with open(p, "w") as f:
                        f.write(out.stdout)
                    info["dump_file"] = "ves_smoother_params.yaml"
                    try:
                        import yaml
                        y = yaml.safe_load(out.stdout) or {}
                        params = {}
                        for v in y.values():
                            if isinstance(v, dict) and "ros__parameters" in v:
                                params = v["ros__parameters"]
                        info["params"] = params
                    except Exception as e:
                        info["parse_error"] = str(e)
                else:
                    info["unavailable"] = (out.stderr.strip() or
                                           f"exit {out.returncode}")[:500]
            except Exception as e:
                info["unavailable"] = str(e)
            self.write_config({"ves_smoother": info})
            self.event("config_complete", "hashes, git, ves_smoother params")

        threading.Thread(target=work, name="xlog-config", daemon=True).start()

    # =============================================================== events

    def event(self, name, detail="", stamp=None):
        self._put(self._write_event, self._event_row(name, detail, stamp),
                  kind="event")

    def _write_event(self, row):
        w = self._writers.get("events")
        if w is not None:
            w.writerow(row)
        w = self._writers.get("ep_events")
        if w is not None:
            w.writerow(row)

    def tool_signal(self, held, value, stamp, source):
        """Raw edges of the tool signal, before any debouncing."""
        if self._tool_last is None:
            self._tool_last = held
            self.event("tool_initial", f"{'on' if held else 'off'} value={value:g} "
                       f"source={source}", stamp)
            return
        if held != self._tool_last:
            self._tool_last = held
            self.event("tool_on" if held else "tool_off",
                       f"value={value:g} source={source}", stamp)

    # =============================================================== frames

    def frame_begin(self, header):
        self.frame_idx += 1
        return {"idx": self.frame_idx, "stamp": header.stamp,
                "ros": self.ros_now_ns(), "wall": time.time(),
                "t0": time.perf_counter(), "episode": self.episode,
                "raw_saved": 0}

    def raw_frame(self, rec, img, convert=None):
        """Queue one received frame for saving. `img` is the processing frame,
        or the raw message when `convert` turns it into one."""
        if self.episode_dir is None:
            return
        path = os.path.join(self.episode_dir, "raw", f"{rec['idx']:06d}.png")
        try:
            self._pq.put_nowait((path, img if convert is not None else img.copy(),
                                 convert))
            rec["raw_saved"] = 1
            n = self._pq.qsize()
            if n > self._pq_high:
                self._pq_high = n
        except queue.Full:
            rec["raw_saved"] = 0
            self._count_drop("raw_png")
            self.event("log_drop_raw", f"frame_idx={rec['idx']}")

    def frame_end(self, rec, state, state_after=None, processed=False,
                  dropped=""):
        s = _stamp_parts(rec["stamp"])
        ms = (time.perf_counter() - rec["t0"]) * 1e3
        row = [rec["idx"], s[0], s[1], s[2], rec["ros"], f"{rec['wall']:.6f}",
               state, state_after if state_after is not None else state,
               rec["episode"] or "", int(bool(processed)), int(bool(dropped)),
               dropped, rec["raw_saved"], f"{ms:.2f}"]
        self._put(self._write_row, "frames", row, kind="frames_row")
        rec["proc_ms"] = ms

    def _write_row(self, key, row):
        w = self._writers.get(key)
        if w is not None:
            w.writerow(row)

    # ============================================================== episode

    def episode_begin(self, episode_dir, meta):
        self.episode_dir = episode_dir
        self.episode = os.path.basename(episode_dir)
        self.episode_frame = 0
        self._snap_count = 0
        self._pose_n_last = None
        self._ep_meta = dict(meta)
        self._ep_meta.update({"session": self.session_name,
                              "episode": self.episode,
                              "start_wall": time.time()})
        # Directories are made here, not in the writer: the PNG workers may
        # reach the first raw frame before the writer reaches this episode.
        os.makedirs(os.path.join(episode_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(episode_dir, "snapshots"), exist_ok=True)
        with self._raw_lock:
            bytes0, saved0 = self._raw_bytes, self._raw_saved
        self._put(self._open_episode, episode_dir, dict(self._ep_meta),
                  bytes0, saved0, kind="episode")
        self.event("episode_start", json.dumps(meta, default=str))

    def _open_episode(self, d, meta, bytes0, saved0):
        self._open_csv("tips", os.path.join(d, "tips.csv"), TIPS_COLS)
        self._open_csv("ep_events", os.path.join(d, "events.csv"), EVENTS_COLS)
        self._open_csv("overlay", os.path.join(d, "overlay_frames.csv"),
                       OVERLAY_COLS)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(dict(meta, units_note=LABEL_NOTE), f, indent=2,
                      default=str)
        self._ep_dir_w = d
        self._video = None
        self._video_n = 0
        self._ep_stats = {"t0": time.time(), "bytes0": bytes0,
                          "saved0": saved0}

    def episode_end(self, reason, outputs=None):
        """Close the episode. `outputs` carries the final arrays to snapshot,
        or None when the episode produced no mask."""
        if self.episode_dir is None:
            return
        self.event("episode_end", reason)
        meta = dict(self._ep_meta)
        meta["end_wall"] = time.time()
        meta["end_reason"] = reason
        meta["frames_processed"] = self.episode_frame
        meta["final_mask_px"] = ("" if outputs is None or outputs.get("accum") is None
                                 else int(outputs["accum"].sum()))
        self._put(self._close_episode, self.episode_dir, meta, outputs,
                  dict(self.drops_total), kind="episode")
        self.episode_dir = None
        self.episode = None

    def _close_episode(self, d, meta, outputs, drops):
        snap = os.path.join(d, "snapshots")
        if outputs is not None:
            if outputs.get("cutline_x") is not None:
                np.save(os.path.join(snap, "final_cutline_x.npy"),
                        outputs["cutline_x"])
            for key in ("cut_path", "tip_history", "mask_tips"):
                if outputs.get(key) is not None:
                    np.save(os.path.join(snap, f"final_{key}.npy"),
                            np.asarray(outputs[key], dtype=np.float32).reshape(-1, 2))
            if outputs.get("accum") is not None:
                cv2.imwrite(os.path.join(snap, "final_accum.png"),
                            outputs["accum"].astype(np.uint8) * 255)
            if outputs.get("T") is not None:
                cv2.imwrite(os.path.join(snap, "final_tumor.png"),
                            outputs["T"].astype(np.uint8) * 255)

        if self._video is not None:
            self._video.release()
            self._video = None
        self._close_csv("tips")
        self._close_csv("ep_events")
        self._close_csv("overlay")

        # Raw frames are written by other threads; wait for this episode's to
        # land before measuring them.
        self._pq.join()
        st = self._ep_stats or {"t0": time.time(), "bytes0": 0, "saved0": 0}
        with self._raw_lock:
            nbytes = self._raw_bytes - st["bytes0"]
            nsaved = self._raw_saved - st["saved0"]
        dur = max(time.time() - st["t0"], 1e-6)
        summary = {
            "raw_frames_saved": nsaved,
            "raw_bytes": nbytes,
            "raw_mb_per_frame": round(nbytes / max(nsaved, 1) / 1e6, 3),
            # Measured over this episode's wall time, so it reflects the
            # frame rate the node actually achieved; the 30 fps figure is
            # for sizing disks.
            "raw_gb_per_min": round(nbytes / 1e9 / (dur / 60.0), 3),
            "raw_gb_per_min_at_30fps": round(
                nbytes / max(nsaved, 1) * 30 * 60 / 1e9, 3),
            "overlay_video_frames": self._video_n,
            "log_drops_session_total": drops,
            "ordered_queue_high_water": self._oq_high,
            "raw_queue_high_water": self._pq_high,
            "end_reason": meta.get("end_reason"),
        }
        with open(os.path.join(d, "log_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        self._write_event(self._event_row(
            "episode_log_summary", json.dumps(summary)))
        try:
            self.logger.info(
                f"Experiment log: {nsaved} raw frames, {nbytes / 1e9:.2f} GB, "
                f"~{summary['raw_gb_per_min']:.2f} GB/min -> {d}")
        except Exception:
            pass

        w = self._writers.get("metadata")
        if w is not None:
            w.writerow([meta.get(k, "") for k in METADATA_COLS])
        self._maybe_flush(force=True)

    # ================================================================= tips

    def tip(self, rec, info):
        """One tracking frame. `info` holds references and plain values from
        the node; everything derived (contour distance, projections) is
        computed in the writer thread."""
        if self.episode_dir is None:
            return
        self.episode_frame += 1
        info["episode_frame"] = self.episode_frame
        n = info.get("pose_msgs_total")
        info["pose_new"] = ("" if n is None or self._pose_n_last is None
                            else int(n != self._pose_n_last))
        self._pose_n_last = n
        info["proc_ms"] = (time.perf_counter() - rec["t0"]) * 1e3
        self._put(self._write_tip, rec["idx"], rec["stamp"], rec["ros"],
                  rec["wall"], info, kind="tips_row")

        self._snap_count += 1
        if (self.snapshot_every > 0 and info.get("accum") is not None
                and self._snap_count % self.snapshot_every == 1):
            self._put(self._write_snapshot, self.episode_dir, rec["idx"],
                      info["accum"].copy(), info.get("T"), kind="snapshot")

    def _write_snapshot(self, d, idx, accum, T):
        snap = os.path.join(d, "snapshots")
        cv2.imwrite(os.path.join(snap, f"{idx:06d}_accum.png"),
                    accum.astype(np.uint8) * 255)
        if T is not None:
            cv2.imwrite(os.path.join(snap, f"{idx:06d}_tumor.png"),
                        T.astype(np.uint8) * 255)

    def _write_tip(self, idx, stamp, ros, wall, info):
        s = _stamp_parts(stamp)
        r = {k: "" for k in TIPS_COLS}
        r.update({"frame_idx": idx, "episode_frame": info["episode_frame"],
                  "stamp_sec": s[0], "stamp_nsec": s[1], "stamp": s[2],
                  "ros_time": ros, "wall_time": f"{wall:.6f}",
                  "tip_source": info.get("tip_source", ""),
                  "proc_ms": _f(info.get("proc_ms"), 2)})

        tip = info.get("tip_raw")
        sm = info.get("tip_smooth")
        r["tip_valid"] = int(tip is not None)
        r["reject_reason"] = info.get("reject_reason", "")
        if info.get("tracker_visible") is not None:
            r["tracker_visible"] = int(bool(info["tracker_visible"]))
        if tip is not None:
            r["tip_raw_u"], r["tip_raw_v"] = _f(tip[0]), _f(tip[1])
        if sm is not None:
            r["tip_smooth_u"], r["tip_smooth_v"] = _f(sm[0]), _f(sm[1])

        # ---- pose
        pose = info.get("pose")
        if info.get("pose_subscribed"):
            r["pose_msgs_total"] = info.get("pose_msgs_total", "")
            r["pose_new"] = info.get("pose_new", "")
        ps = info.get("pose_stamp")
        if ps is not None:
            r["pose_stamp"] = _stamp_parts(ps)[2]
            if stamp is not None:
                dt = ((stamp.sec - ps.sec) + 1e-9 * (stamp.nanosec - ps.nanosec))
                r["image_minus_pose_ms"] = _f(dt * 1e3, 2)
        if pose is not None:
            x, y, z = pose
            r["pose_x"], r["pose_y"], r["pose_z"] = _f(x, 6), _f(y, 6), _f(z, 6)
            K, dist, cwh, WH = (info.get("K"), info.get("dist"),
                                info.get("calib_wh"), info.get("WH"))
            if K is not None and z > 1e-6 and cwh is not None and WH is not None:
                uv, _ = cv2.projectPoints(
                    np.array([[[x, y, z]]], dtype=np.float64),
                    np.zeros((3, 1)), np.zeros((3, 1)), K, dist)
                r["pose_u"] = _f(uv[0, 0, 0] * WH[0] / float(cwh[0]))
                r["pose_v"] = _f(uv[0, 0, 1] * WH[1] / float(cwh[1]))
        cov = info.get("cov")
        if cov is not None and len(cov) == 36:
            c = np.asarray(cov, dtype=np.float64).reshape(6, 6)
            # GTSAM order: rotation first (rows/cols 0-2), position (3-5).
            P = c[3:, 3:]
            R = c[:3, :3]
            r["cov_xx"], r["cov_yy"], r["cov_zz"] = (
                f"{P[0, 0]:.6e}", f"{P[1, 1]:.6e}", f"{P[2, 2]:.6e}")
            r["cov_xy"], r["cov_xz"], r["cov_yz"] = (
                f"{P[0, 1]:.6e}", f"{P[0, 2]:.6e}", f"{P[1, 2]:.6e}")
            r["pos_sigma_mm"] = _f(np.sqrt(max(np.trace(P), 0.0)) * 1e3, 4)
            r["rot_sigma_deg"] = _f(np.degrees(np.sqrt(max(np.trace(R), 0.0))), 4)
            r["cov_full"] = ";".join(f"{v:.6e}" for v in c.ravel())

        # ---- hull distance, exactly the value the sweep geometry used
        hd, hp, perim = info.get("hull_dist"), info.get("hull_pt"), info.get("perim")
        r["hull_valid"] = int(hd is not None)
        if hd is not None and sm is not None:
            r["hull_dist_px"] = _f(hd, 6)
            r["hull_dist_mm"] = _f(info.get("hull_dist_mm"), 6)
            r["hull_u"], r["hull_v"] = _f(hp[0]), _f(hp[1])
            if perim is not None and len(perim) >= 3:
                r["tip_in_hull"] = int(cv2.pointPolygonTest(
                    perim.reshape(-1, 1, 2).astype(np.float32),
                    (float(sm[0]), float(sm[1])), False) >= 0)

        # ---- true contour distance
        T = info.get("T")
        if T is not None and sm is not None:
            c = self.contour_fn(T)
            if c is not None and len(c):
                d = np.hypot(c[:, 0] - sm[0], c[:, 1] - sm[1])
                k = int(np.argmin(d))
                r["contour_valid"] = 1
                r["contour_dist_px"] = _f(d[k])
                r["contour_u"], r["contour_v"] = _f(c[k, 0]), _f(c[k, 1])
                r["tip_in_contour"] = int(cv2.pointPolygonTest(
                    c.reshape(-1, 1, 2), (float(sm[0]), float(sm[1])), False) >= 0)
            else:
                r["contour_valid"] = 0
            H, W = T.shape
            xi = int(round(min(max(sm[0], 0), W - 1)))
            yi = int(round(min(max(sm[1], 0), H - 1)))
            r["tip_on_tumor"] = int(bool(T[yi, xi]))
            r["tumor_px"] = int(T.sum())

        se = info.get("seg_end")
        if se is not None and sm is not None:
            r["seg_end_u"], r["seg_end_v"] = _f(se[0]), _f(se[1])
            r["sweep_len_px"] = _f(np.hypot(se[0] - sm[0], se[1] - sm[1]))
        if info.get("angle_deg") is not None and hd is not None:
            r["angle_deg"] = _f(info["angle_deg"], 2)
        if info.get("accum_px") is not None:
            r["accum_px"] = info["accum_px"]

        r["overlay_on"] = int(bool(info.get("overlay_on")))
        r["d_shown"] = info.get("d_shown", "")
        r["warn_px"] = _f(info.get("warn_px"), 1)
        r["warn_mm"] = _f(info.get("warn_mm"), 2)
        r["warn_state"] = info.get("warn_state", "")

        w = self._writers.get("tips")
        if w is not None:
            w.writerow([r[k] for k in TIPS_COLS])

    # ============================================================== overlay

    def overlay_frame(self, rec, img, kind):
        """Queue one displayed overlay (processing resolution) for the video.
        `img` must not be modified afterwards; the node never does."""
        if self.episode_dir is None or rec is None:
            return
        self._put(self._write_overlay, rec["idx"], rec["stamp"], img, kind,
                  kind="overlay_frame")

    def _write_overlay(self, idx, stamp, img, kind):
        if "overlay" not in self._writers:
            return
        if self._video is None:
            h, w = img.shape[:2]
            p = os.path.join(self._ep_dir_w, "overlay.mp4")
            self._video = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"),
                                          self.overlay_fps, (w, h))
        self._video.write(img)
        s = _stamp_parts(stamp)
        self._writers["overlay"].writerow([self._video_n, idx, s[0], s[1], kind])
        self._video_n += 1

    # ================================================================ close

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.episode_dir is not None:
            self.episode_end("node_shutdown")
        self.event("session_end", json.dumps(self.drops_total))
        # Ordered writer first: closing an episode waits on the PNG queue.
        self._oq.put(None)
        self._othread.join(timeout=120)
        for _ in self._pthreads:
            self._pq.put(None)
        for t in self._pthreads:
            t.join(timeout=60)
        for k in list(self._files):
            self._close_csv(k)
        with self._cfg_lock:
            snap = json.loads(json.dumps(self.config, default=str))
        snap["log_drops_total"] = dict(self.drops_total)
        self._write_config_file(snap)
