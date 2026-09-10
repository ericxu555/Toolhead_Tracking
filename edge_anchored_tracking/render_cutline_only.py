"""Render the CUTLINE ALONE (no mask fill, no tumor boundary) on top of the
raw video, for demonstration purposes -- so a viewer can see how the tool
tip's trajectory builds up the left edge of the resection mask over time,
independent of the tumor-segmentation/perpendicular-bottom machinery.

Reuses the exact same cutline construction as cutline_sweep_mask_walk.py
(per-row cutline_x array, EMA-blended writes, monotonic frontier_y fill) so
the line drawn here matches what that pipeline actually uses as the mask's
left edge -- just visualized without the rest of the geometry.

Usage: render_cutline_only.py [video_dir] [tip_csv] [outdir] [tip_smooth=15]
"""
import sys, os, glob, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
from PIL import Image

video_dir = sys.argv[1]
tip_csv = sys.argv[2]
outdir = sys.argv[3] if len(sys.argv) > 3 else \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cutline_only_out")
TIP_SMOOTH = int(sys.argv[4]) if len(sys.argv) > 4 else 15
os.makedirs(outdir, exist_ok=True)


def smooth_tips(tips_by_frame, win):
    """Identical to cutline_sweep_mask_walk.py's smooth_tips: edge-replicated
    padding so the moving average doesn't drag early/late frames toward a
    phantom zero (the bug fixed earlier this session)."""
    if win <= 1:
        return dict(tips_by_frame)
    frames = sorted(tips_by_frame)
    xs = np.array([tips_by_frame[f][0] for f in frames])
    ys = np.array([tips_by_frame[f][1] for f in frames])
    k = min(win, len(xs))
    if k >= 2:
        pad = k // 2
        ker = np.ones(k) / k
        xs_p = np.pad(xs, pad, mode="edge")
        ys_p = np.pad(ys, pad, mode="edge")
        xs = np.convolve(xs_p, ker, mode="same")[pad:pad + len(xs)]
        ys = np.convolve(ys_p, ker, mode="same")[pad:pad + len(ys)]
    return {f: (float(xs[i]), float(ys[i])) for i, f in enumerate(frames)}


def load_tips(tip_csv):
    tips, trusted = {}, {}
    for r in csv.DictReader(open(tip_csv)):
        f = int(r["frame"])
        tips[f] = (float(r["x"]), float(r["y"]))
        trusted[f] = (r.get("mode", "D") != "M")
    return tips, trusted


def main():
    files = sorted(glob.glob(os.path.join(video_dir, "*.png")))
    first = cv2.imread(files[0]); H, W = first.shape[:2]

    tips_raw, trusted = load_tips(tip_csv)
    tips = smooth_tips(tips_raw, TIP_SMOOTH)
    print(f"{len(files)} frames; {len(tips)} tips; smooth={TIP_SMOOTH}")

    cutline_x = np.full(H, np.nan, dtype=np.float32)
    frontier_y = -1

    vw = cv2.VideoWriter(os.path.join(outdir, "cutline_only.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))

    for i, fp in enumerate(files):
        is_trusted = trusted.get(i, True)
        if i in tips and is_trusted:
            tx, ty = tips[i]
            tyi = int(round(np.clip(ty, 0, H - 1)))
            cutline_x[tyi] = tx if np.isnan(cutline_x[tyi]) else 0.5 * cutline_x[tyi] + 0.5 * tx
            if tyi > frontier_y:
                prev_y = frontier_y if frontier_y >= 0 else tyi
                prev_x = cutline_x[prev_y] if not np.isnan(cutline_x[prev_y]) else tx
                for yy in range(prev_y + 1, tyi + 1):
                    frac = (yy - prev_y) / max(tyi - prev_y, 1)
                    cutline_x[yy] = prev_x + frac * (tx - prev_x)
                frontier_y = tyi

        vis = cv2.imread(fp)

        # Draw the accumulated cutline as a polyline (yellow), top-down.
        visited_rows = np.where(~np.isnan(cutline_x))[0]
        if frontier_y >= 0:
            visited_rows = visited_rows[visited_rows <= frontier_y]
        if len(visited_rows) >= 2:
            pts = np.array([[cutline_x[r], r] for r in visited_rows], dtype=np.int32)
            cv2.polylines(vis, [pts], False, (0, 255, 255), 2)

        # Current tip: magenta if trusted, orange if the tracker flagged it lost.
        if i in tips:
            tx, ty = tips[i]
            tip_col = (255, 0, 255) if is_trusted else (0, 165, 255)
            cv2.circle(vis, (int(round(tx)), int(round(ty))), 5, tip_col, -1)

        cv2.putText(vis, f"t={i} cutline (no mask){'' if is_trusted else ' FROZEN'}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        vw.write(vis)

    vw.release()
    print(f"DONE -> {outdir}")


if __name__ == "__main__":
    main()
