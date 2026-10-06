#!/usr/bin/env python3
"""Hand-label tool tip and nearest tumor boundary on an episode's raw frames.

    python3 tools/label_tool.py Output/session_.../episode_001_... --stride 10

Shows the episode's raw/ frames (exactly what the node processed, no overlay)
at every `stride`-th frame. On each frame:

    left click #1   the tool tip
    left click #2   the corresponding point on the tumor boundary
    Enter / Space   accept and go to the next frame (needs both clicks)
    s               skip this frame (recorded as skipped)
    u               undo the last click; with no clicks, step back and
                    remove the previous frame's label
    b               back one frame without removing anything
    q / Esc         save and quit

labels.csv (in the episode folder unless --out) is rewritten after every
accept/skip, so quitting or crashing loses at most the current frame.
Re-running resumes: frames already in labels.csv are not shown again unless
--relabel is given.

Coordinates are in the raw frames' pixels -- the node's processing frame
(540x540 by default), the same space as tips.csv. frame_idx and the image
stamp come from the session's frames.csv, so labels join to tips.csv on
frame_idx.
"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np

LABEL_COLS = ["frame_idx", "stamp_sec", "stamp_nsec", "tip_u", "tip_v",
              "boundary_u", "boundary_v", "dist_px", "status", "labeler",
              "labeled_at"]


def load_stamps(episode_dir):
    """frame_idx -> (sec, nsec) from the session's frames.csv, if present."""
    p = os.path.join(os.path.dirname(os.path.abspath(episode_dir)), "frames.csv")
    out = {}
    if os.path.isfile(p):
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                out[int(r["frame_idx"])] = (r["stamp_sec"], r["stamp_nsec"])
    return out


class Labels:
    """State and file handling, kept apart from the GUI so it can be tested."""

    def __init__(self, episode_dir, stride=1, start=0, out=None, labeler="",
                 relabel=False):
        self.episode_dir = episode_dir
        files = sorted(glob.glob(os.path.join(episode_dir, "raw", "*.png")))
        if not files:
            raise SystemExit(f"No raw/*.png in {episode_dir}")
        self.files = files[start::max(int(stride), 1)]
        self.idx_of = [int(os.path.splitext(os.path.basename(p))[0])
                       for p in self.files]
        self.stamps = load_stamps(episode_dir)
        self.out = out or os.path.join(episode_dir, "labels.csv")
        self.labeler = labeler
        self.rows = {}                       # frame_idx -> row dict
        if os.path.isfile(self.out):
            with open(self.out, newline="") as f:
                for r in csv.DictReader(f):
                    self.rows[int(r["frame_idx"])] = r
        self.clicks = []
        self.pos = 0
        self._relabel_all = bool(relabel)
        if not relabel:
            while self.pos < len(self.files) and self.idx_of[self.pos] in self.rows:
                self.pos += 1
        self.history = []                    # positions labeled this sitting

    # ------------------------------------------------------------- state
    @property
    def done(self):
        return self.pos >= len(self.files)

    @property
    def frame_idx(self):
        return self.idx_of[self.pos]

    def click(self, u, v):
        if len(self.clicks) < 2:
            self.clicks.append((float(u), float(v)))

    def _row(self, status):
        s = self.stamps.get(self.frame_idx, ("", ""))
        r = {k: "" for k in LABEL_COLS}
        r.update({"frame_idx": self.frame_idx, "stamp_sec": s[0],
                  "stamp_nsec": s[1], "status": status,
                  "labeler": self.labeler,
                  "labeled_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        if status == "labeled":
            (tu, tv), (bu, bv) = self.clicks
            r.update({"tip_u": f"{tu:.1f}", "tip_v": f"{tv:.1f}",
                      "boundary_u": f"{bu:.1f}", "boundary_v": f"{bv:.1f}",
                      "dist_px": f"{np.hypot(bu - tu, bv - tv):.2f}"})
        return r

    def accept(self):
        if len(self.clicks) < 2:
            return False
        self._commit("labeled")
        return True

    def skip(self):
        self._commit("skipped")

    def _commit(self, status):
        self.rows[self.frame_idx] = self._row(status)
        self.save()
        self.history.append(self.pos)
        self.clicks = []
        self.pos += 1
        if not self._relabel_all:
            while self.pos < len(self.files) and self.idx_of[self.pos] in self.rows:
                self.pos += 1

    def undo(self):
        if self.clicks:
            self.clicks.pop()
            return
        if self.history:
            self.pos = self.history.pop()
            self.rows.pop(self.frame_idx, None)
            self.save()

    def back(self):
        self.clicks = []
        if self.pos > 0:
            self.pos -= 1
        self._relabel_all = True         # stepping back means: show labeled too

    def save(self):
        tmp = self.out + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LABEL_COLS)
            w.writeheader()
            for k in sorted(self.rows):
                w.writerow({c: self.rows[k].get(c, "") for c in LABEL_COLS})
        os.replace(tmp, self.out)


def gui(L, scale):
    import cv2
    win = "label_tool"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            L.click(x / scale, y / scale)

    cv2.setMouseCallback(win, on_mouse)
    cache = {}
    while not L.done:
        p = L.files[L.pos]
        if p not in cache:
            cache.clear()
            cache[p] = cv2.imread(p)
        img = cache[p]
        vis = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_NEAREST) if scale != 1 else img.copy()
        # Only the clicks are drawn; the frame itself is the raw frame.
        pts = [(int(round(u * scale)), int(round(v * scale))) for u, v in L.clicks]
        if len(pts) == 2:
            cv2.line(vis, pts[0], pts[1], (0, 255, 255), 1, cv2.LINE_AA)
        for k, pt in enumerate(pts):
            cv2.drawMarker(vis, pt, (0, 255, 0) if k == 0 else (0, 0, 255),
                           cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
        prev = L.rows.get(L.frame_idx)
        cv2.setWindowTitle(
            win, f"frame {L.frame_idx}  ({L.pos + 1}/{len(L.files)})  "
                 f"click {'tip' if not L.clicks else 'boundary' if len(L.clicks) == 1 else 'done'}"
                 + (f"  [already {prev['status']}]" if prev else "")
                 + "   Enter=accept s=skip u=undo b=back q=quit")
        cv2.imshow(win, vis)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10, 32):
            L.accept()
        elif k == ord("s"):
            L.skip()
        elif k == ord("u"):
            L.undo()
        elif k == ord("b"):
            L.back()
        elif k in (ord("q"), 27):
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode_dir")
    ap.add_argument("--stride", type=int, default=10,
                    help="label every Nth raw frame (default 10)")
    ap.add_argument("--start", type=int, default=0,
                    help="offset into the raw frame list")
    ap.add_argument("--scale", type=float, default=1.5,
                    help="display magnification; labels stay in raw pixels")
    ap.add_argument("--labeler", default=os.environ.get("USER", ""))
    ap.add_argument("--out", default=None)
    ap.add_argument("--relabel", action="store_true",
                    help="show frames that already have a label")
    a = ap.parse_args()
    L = Labels(a.episode_dir, a.stride, a.start, a.out, a.labeler, a.relabel)
    print(f"{len(L.files)} frames at stride {a.stride}; "
          f"{len(L.rows)} already in {L.out}")
    gui(L, a.scale)
    n = sum(r["status"] == "labeled" for r in L.rows.values())
    print(f"Saved {L.out}: {n} labeled, {len(L.rows) - n} skipped")


if __name__ == "__main__":
    main()
