#!/usr/bin/env python3
"""Gather one live session into a folder per cut, with the toolhead and margin data side by side.

The two nodes run as separate processes and write separately:

    ~/Toolhead_Tracking/Output/session_<stamp>/episode_NNN_<stamp>/   toolhead (numbered per session)
    ~/Margin_Segmentation_Model/Output/episode_NNN_<stamp>/           margin   (numbered across sessions)

This pairs each toolhead episode with the margin episode of the same cut by the
camera timestamps both record (tips.csv / frames.csv), falling back to the folder
start times, and writes

    ~/resection_data/session_<stamp>/
        session_summary.csv        one row per cut
        session_config/            run_config.json, calibration/, frames.csv, events.csv, logs
        cut_01_<HHMMSS>/
            toolhead/              raw/, snapshots/, tips.csv, final_mask.png, episode_info.json, ...
            margin/                margin + tumor masks and overlays, frames.csv, episode.json
        cut_02_<HHMMSS>/ ...

    python3 tools/organize_session.py                      # newest toolhead session
    python3 tools/organize_session.py --after EPOCH        # sessions started after a time (run_all.sh)
    python3 tools/organize_session.py --session DIR --copy # one session; keep the originals

Files are MOVED by default (same disk: instant, no extra space); --copy keeps the originals.
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys
from datetime import datetime

import cv2

HOME = os.path.expanduser("~")
TOOLHEAD_OUT = os.path.join(HOME, "Toolhead_Tracking", "Output")
MARGIN_OUT = os.path.join(HOME, "Margin_Segmentation_Model", "Output")
DEST = os.path.join(HOME, "resection_data")
PAIR_SLACK_S = 3.0          # folder start times this close count as the same cut

SUMMARY_COLS = [
    "cut", "start", "duration_s", "toolhead_frames", "final_mask_px",
    "fraction_estimate", "fraction_final", "fraction_final_reason",
    "min_dist_px", "min_dist_mm", "time_within_warning_s",
    "margin_frames", "margin_final_px", "tumor_final_px",
    "toolhead_episode", "margin_episode", "paired",
]


def folder_time(name):
    """datetime from '..._YYYYmmdd-HHMMSS' at the end of a folder name, or None."""
    try:
        return datetime.strptime(name.rsplit("_", 1)[-1][:15], "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def stamp_range(rows, sec="stamp_sec", nsec="stamp_nsec"):
    t = []
    for r in rows:
        try:
            t.append(int(r[sec]) + int(r[nsec]) * 1e-9)
        except (KeyError, ValueError):
            pass
    return (min(t), max(t)) if t else None


def floats(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r[key]))
        except (KeyError, ValueError):
            pass
    return out


def mask_px(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if os.path.isfile(path) else None
    return "" if m is None else int((m > 127).sum())


def toolhead_stats(d):
    info = {}
    p = os.path.join(d, "episode_info.json")
    if os.path.isfile(p):
        with open(p) as f:
            info = json.load(f)
    tips = read_csv(os.path.join(d, "tips.csv"))
    px, mm = floats(tips, "hull_dist_px"), floats(tips, "hull_dist_mm")
    # Time within the warning: the stamp gap after each warned frame, capped so a
    # stall in the stream does not count as seconds of warning.
    warn = 0.0
    for a, b in zip(tips, tips[1:]):
        if a.get("warn_state") == "warn":
            try:
                dt = (int(b["stamp_sec"]) - int(a["stamp_sec"])) + (int(b["stamp_nsec"]) - int(a["stamp_nsec"])) * 1e-9
                warn += min(max(dt, 0.0), 1.0)
            except (KeyError, ValueError):
                pass
    return {
        "toolhead_frames": info.get("frames_processed", len(tips)),
        "duration_s": info.get("duration_sec", ""),
        "final_mask_px": info.get("final_mask_px", mask_px(os.path.join(d, "final_mask.png"))),
        "fraction_estimate": info.get("fraction_estimate", ""),
        "fraction_final": info.get("fraction_final", ""),
        "fraction_final_reason": info.get("fraction_final_reason", ""),
        "min_dist_px": round(min(px), 1) if px else "",
        "min_dist_mm": round(min(mm), 2) if mm else "",
        "time_within_warning_s": round(warn, 2),
    }, stamp_range(tips)


def margin_stats(d):
    meta = {}
    p = os.path.join(d, "episode.json")
    if os.path.isfile(p):
        with open(p) as f:
            meta = json.load(f)
    return {
        "margin_frames": meta.get("frames", ""),
        "margin_final_px": mask_px(os.path.join(d, "margin_final_mask.png")),
        "tumor_final_px": mask_px(os.path.join(d, "tumor_final_mask.png")),
    }, stamp_range(read_csv(os.path.join(d, "frames.csv")))


def pair(toolhead, margin):
    """[(toolhead_dir or None, margin_dir or None)] in time order. Overlap of camera
    timestamps decides; folder start times within PAIR_SLACK_S are the fallback."""
    used, pairs = set(), []
    for t in toolhead:
        best, best_score = None, 0.0
        for i, m in enumerate(margin):
            if i in used:
                continue
            if t["range"] and m["range"]:
                score = min(t["range"][1], m["range"][1]) - max(t["range"][0], m["range"][0])
                score = score + 1e3 if score > 0 else 0.0           # any overlap beats a name match
            else:
                score = 0.0
            if not score and t["time"] and m["time"]:
                gap = abs((t["time"] - m["time"]).total_seconds())
                score = PAIR_SLACK_S - gap if gap <= PAIR_SLACK_S else 0.0
            if score > best_score:
                best, best_score = i, score
        if best is not None:
            used.add(best)
        pairs.append((t, margin[best] if best is not None else None))
    pairs += [(None, m) for i, m in enumerate(margin) if i not in used]
    key = lambda p: (p[0] or p[1])["time"] or datetime.min
    return sorted(pairs, key=key)


def transfer(src, dst, copy):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if copy:
        (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)
    else:
        shutil.move(src, dst)


def organize(session_dir, margin_out, dest, copy, logs=()):
    session = os.path.basename(os.path.normpath(session_dir))
    t0 = folder_time(session)
    out = os.path.join(dest, session)
    if os.path.exists(out):
        sys.exit(f"Already organized: {out}")

    toolhead = []
    for d in sorted(glob.glob(os.path.join(session_dir, "episode_*"))):
        stats, rng = toolhead_stats(d)
        toolhead.append({"dir": d, "time": folder_time(os.path.basename(d)), "range": rng, "stats": stats})
    # Margin episodes of this session: started after the session began, and before
    # its last toolhead episode ended (plus a minute for a cut still being written).
    end = max([t["time"] for t in toolhead if t["time"]] or [t0 or datetime.max])
    margin = []
    for d in sorted(glob.glob(os.path.join(margin_out, "episode_*"))):
        mt = folder_time(os.path.basename(d))
        if mt is None or (t0 and (mt - t0).total_seconds() < -PAIR_SLACK_S) or (mt - end).total_seconds() > 60:
            continue
        stats, rng = margin_stats(d)
        margin.append({"dir": d, "time": mt, "range": rng, "stats": stats})

    rows = []
    for n, (t, m) in enumerate(pair(toolhead, margin), 1):
        when = (t or m)["time"]
        cut = f"cut_{n:02d}_{when.strftime('%H%M%S') if when else 'unknown'}"
        row = {k: "" for k in SUMMARY_COLS}
        row.update({"cut": cut, "start": when.isoformat(sep=" ") if when else "",
                    "paired": "yes" if t and m else ("toolhead only" if t else "margin only")})
        if t:
            row.update(t["stats"], toolhead_episode=os.path.basename(t["dir"]))
            transfer(t["dir"], os.path.join(out, cut, "toolhead"), copy)
        if m:
            row.update(m["stats"], margin_episode=os.path.basename(m["dir"]))
            transfer(m["dir"], os.path.join(out, cut, "margin"), copy)
        rows.append(row)

    # Everything left in the toolhead session folder is session-wide.
    cfg = os.path.join(out, "session_config")
    os.makedirs(cfg, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(session_dir, "*"))):
        if not os.path.basename(p).startswith("episode_"):
            transfer(p, os.path.join(cfg, os.path.basename(p)), copy)
    for p in logs:
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(cfg, os.path.basename(p)))
    if not copy and not os.listdir(session_dir):
        os.rmdir(session_dir)

    with open(os.path.join(out, "session_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} cuts -> {out}")
    for r in rows:
        print(f"  {r['cut']}  {r['paired']:<13}  final {r['fraction_final'] or '--'}  "
              f"min {r['min_dist_mm'] or '--'} mm  margin {r['margin_final_px'] or '--'} px")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="one toolhead session folder (default: the newest)")
    ap.add_argument("--after", type=float, help="organize every toolhead session started after this Unix time")
    ap.add_argument("--toolhead-out", default=TOOLHEAD_OUT)
    ap.add_argument("--margin-out", default=MARGIN_OUT)
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--copy", action="store_true", help="copy instead of move")
    ap.add_argument("--logs", nargs="*", default=[], help="run_all.sh log files to keep with the session")
    args = ap.parse_args()

    if args.session:
        sessions = [args.session]
    else:
        sessions = sorted(glob.glob(os.path.join(args.toolhead_out, "session_*")))
        if args.after is not None:
            start = datetime.fromtimestamp(args.after)
            sessions = [s for s in sessions
                        if (folder_time(os.path.basename(s)) or datetime.min) >= start.replace(microsecond=0)]
        else:
            sessions = sessions[-1:]
    if not sessions:
        print("No toolhead session to organize.")
        return
    for s in sessions:
        organize(s, args.margin_out, args.dest, args.copy, args.logs)


if __name__ == "__main__":
    main()
