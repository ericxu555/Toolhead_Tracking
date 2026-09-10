"""TAPNext++ tip tracking + shaft-geometric fallback (same idea as the CoTracker
hybrid, on a backbone that stalls-in-place under occlusion instead of drifting
wildly). Tracks tip + real shaft points (not just discardable local support
points) jointly. When the tip is occluded, derive it from the shaft's rigid
motion (PCA axis + calibrated arclength offset) instead of trusting the model's
frozen/stalled tip position.
Reads click_seed_multi.json (point 0 = tip, rest = shaft).
Usage: tapnextpp_hybrid.py [video_dir] [seed_json] [outdir]
"""
import sys, os, glob, json, pathlib, urllib.request
import numpy as np, cv2, torch
from tapnet.tapnextpp.votsp2026.model import TAPNextPP

CHECKPOINT_URL = "https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt"
EMA = 0.6

def _get_checkpoint():
    cache_dir = pathlib.Path.home() / ".cache" / "tapnextpp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "tapnextpp_512.ckpt"
    if not dest.exists():
        urllib.request.urlretrieve(CHECKPOINT_URL, dest)
    return str(dest)

def filter_collapsed(S, idx, pts0_shaft):
    """Drop shaft points that have collapsed onto a neighbor (a known point-
    tracker failure in low-texture/blurred regions) or gone out of their
    original relative order -- both are real degeneracy signals, not a
    threshold guessed on the tip's own behavior. Keeps at least 3 points."""
    orig_gaps = np.linalg.norm(np.diff(pts0_shaft, axis=0), axis=1)
    min_sep = max(3.0, 0.35 * np.median(orig_gaps))   # collapsed if <35% of its
                                                        # original spacing to neighbor
    keep = np.ones(len(idx), dtype=bool)
    for k in range(len(idx) - 1):
        if np.linalg.norm(S[k] - S[k + 1]) < min_sep:
            keep[k + 1] = False   # drop the later (more tip-distal) of the pair
    if keep.sum() < 3:
        return S, idx   # not enough left to filter safely -- use all
    return S[keep], idx[keep]

def shaft_axis(S, idx):
    c = S.mean(0); d = S - c
    w, v = np.linalg.eigh(d.T @ d)
    u = v[:, -1]
    proj = d @ u
    if len(idx) > 1 and np.corrcoef(idx, proj)[0, 1] > 0:
        u = -u; proj = -proj
    return c, u, float(proj.std() + 1e-6)

if len(sys.argv) < 3:
    raise SystemExit("Usage: tapnextpp_hybrid.py <video_dir> <seed_json> [outdir]")
video_dir = sys.argv[1]
seed_json = sys.argv[2]
outdir = sys.argv[3] if len(sys.argv) > 3 else \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "tapnextpp_hybrid_out")
os.makedirs(outdir, exist_ok=True)

seed = json.load(open(seed_json))
pts0 = np.array(seed["points"], dtype=np.float32)   # [0]=tip, rest=shaft
N = len(pts0)
files = sorted(glob.glob(os.path.join(video_dir, "*.png")))
first = cv2.imread(files[0]); H, W = first.shape[:2]
print(f"{len(files)} frames; tip={pts0[0].tolist()}; {N-1} shaft pts")

ckpt = _get_checkpoint()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TAPNextPP.from_checkpoint(ckpt, device=device, input_resolution=512)
print(f"TAPNext++ loaded on {device}")

vw = cv2.VideoWriter(os.path.join(outdir, "hybrid_tip.mp4"),
                     cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
traj, rows = [], []
SAMPLE = {0, 300, 470, 510, 540, 560, 570, 596, 600, 650, 713, 745, 750, len(files)-1}
rho = None; tip_s = pts0[0].copy(); state = None
n_direct = n_geo = n_reject = 0
# Consistency gate: max allowed disagreement (px) between the model's direct tip
# and the geometric shaft-axis estimate before we distrust the "visible" tip.
# Catches CONFIDENT drift (model says visible but is slowly wrong), which
# visibility alone can't catch. Tuned relative to typical shaft segment length.
DISAGREE_THRESH = 25.0
for i, fp in enumerate(files):
    frame = cv2.imread(fp)
    if i == 0:
        positions, visible, state = model.track_frame(frame, query_points_xy=pts0)
    else:
        positions, visible, state = model.track_frame(frame, state=state)

    sidx = np.arange(1, N)
    Sf, sidxf = filter_collapsed(positions[sidx], sidx, pts0[1:])
    have_axis = len(sidxf) >= 3
    if have_axis:
        c, u, sig = shaft_axis(Sf, sidxf)

    geo_tip = (c + rho * sig * u).astype(np.float32) if (have_axis and rho is not None) else None

    if visible[0]:
        direct_tip = positions[0].astype(np.float32)
        disagree = (np.linalg.norm(direct_tip - geo_tip)
                   if geo_tip is not None else 0.0)
        if geo_tip is not None and disagree > DISAGREE_THRESH:
            # Model claims visible but disagrees sharply with shaft geometry --
            # confident drift. Trust geometry, do NOT recalibrate rho from this.
            tip = geo_tip; mode = "G"; n_geo += 1; n_reject += 1
        else:
            tip = direct_tip; mode = "D"; n_direct += 1
            if have_axis:
                rho = float((tip - c) @ u / sig)   # only recalibrate on trusted reads
    elif geo_tip is not None:
        tip = geo_tip; mode = "G"; n_geo += 1
    else:
        tip = positions[0].astype(np.float32); mode = "M"   # model's own (stalled) guess

    tip_s = EMA * tip + (1 - EMA) * tip_s
    traj.append((int(tip_s[0]), int(tip_s[1])))
    rho_str = f"{rho:.3f}" if rho is not None else ""
    shaft_str = ";".join(f"{positions[n,0]:.0f},{positions[n,1]:.0f},{int(visible[n])}"
                         for n in range(1, N))
    rows.append(f"{i},{tip_s[0]:.1f},{tip_s[1]:.1f},{mode},{rho_str},{shaft_str}")

    img = frame.copy()
    for n in range(1, N):
        cv2.circle(img, (int(positions[n, 0]), int(positions[n, 1])), 4,
                  (255, 140, 0) if visible[n] else (255, 140, 0), -1 if visible[n] else 1)
    for k in range(1, len(traj)):
        cv2.line(img, traj[k-1], traj[k], (0, 0, 255), 2)
    col = {"D": (0, 255, 0), "G": (255, 0, 255), "M": (0, 165, 255)}[mode]
    cv2.circle(img, traj[-1], 10, col, 2)
    cv2.putText(img, f"t={i} {mode}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    vw.write(img)
    if i in SAMPLE:
        cv2.imwrite(os.path.join(outdir, f"h_frame{i:04d}.png"), img)
    if i % 100 == 0:
        print(f"  frame {i}/{len(files)}  mode={mode}")
vw.release()
open(os.path.join(outdir, "tip_track.csv"), "w").write("frame,x,y,mode,rho,shaft_pts\n" + "\n".join(rows) + "\n")
print(f"DONE direct={n_direct} geo={n_geo} (rejected_disagreement={n_reject}) -> {outdir}")
