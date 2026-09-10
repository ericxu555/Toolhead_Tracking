"""Solution 1b -- CUTLINE-SWEEP MASK with a ROTATING (perpendicular) bottom.

Companion to `cutline_sweep_mask.py` (horizontal-only bottom), which it does
not modify. Postdoc's request: instead of always cutting the mask off with an
image-horizontal line through the tool tip, the bottom edge should run
PERPENDICULAR to the tissue surface -- horizontal where the tumor boundary is
vertical (early in the cut), rotating toward vertical as the boundary curves
around the crescent.

Mask boundaries each frame:
  LEFT   -- the cutline: the accumulated tip trajectory (per-row cutline_x).
  RIGHT  -- the tumor mask itself (`& T`).
  BOTTOM -- a line through the tip at `angle_deg`, where angle_deg is
            perpendicular to the tangent of the tumor's OUTER perimeter at
            the point nearest the tip. Recomputed independently every frame:
            no walk, no accumulated position, nothing that can ratchet or
            drift. At angle 0 this reduces exactly to the horizontal model.
  TOP    -- depends on where the cut began (decided once, see below).

TOP EDGE, two cases:
  Case A -- the tip enters the tumor while cutting (it began inside, with
            un-resected tissue above it). The tumor's own upper boundary
            would wrongly include that tissue, so the top edge is the
            cutline's row at the moment of entry (`cut_start_y`).
  Case B -- the tip never enters (began above/outside, cut runs down past
            the tumor). The cutline and the tumor boundary already meet and
            enclose the region, so no artificial top edge is needed and the
            `& T` clip serves as the top.
  Decided on the frame the tip FIRST tests inside the notch-bridged hull.
  Keyed on that event rather than on mask area because early frames are
  unusable -- binary10swa has not found the tumor yet (nodaggernorec4 is 1650 px
  at f0, ~30000 by f50, component count flailing 1/3/2/1 meanwhile). The
  entry event needs no invented threshold and fires when the geometry becomes
  real. This replaces the earlier vertical-ray "closure segment", which
  addressed the same gap less directly.

WHY THE CONVEX HULL (outer_perimeter): binary10swa excludes the metal toolhead
from the tumor mask, and those cutouts are INDENTATIONS reaching in from the
outside edge -- not enclosed holes. (Measured Full_exp5 f300: hole-filling
changes area by 0 px; hull concavity is 5519 px.) So the raw contour dives
into the notch and passes within ~3 px of the tip: every "nearest boundary
point" query snapped to the notch rim and every tangent fitted there was
noise, with three smoothing windows disagreeing by up to 79 deg. Taking the
convex hull bridges the notches and recovers the smooth perimeter the tumor
actually presents -- the tip then sits a sane 5-27 px away, smoothing scales
agree within a few degrees, and the angle sweeps 88 -> 62 -> 36 -> 11 -> 1
(flattest, mid-cut) -> 24 -> 50 -> 84 deg around the crescent.
CAVEAT: the hull also bridges GENUINE concavities. Measured concavity is a
stable 10-26% across all five videos, consistent with it being dominated by
the two tool notches, so it is safe on this dataset; revisit for markedly
more irregular tumors.

TEMPORAL SMOOTHING: the tumor probability is EMA-blended across frames before
thresholding, so a single flickery binary10swa frame cannot jump the boundary.
Separate from the tip-position smoothing (TIP_SMOOTH).

Mask is CUMULATIVE/monotonic: once resected, stays resected. When the tip
exits the tumor (cut done) no new region is produced and the mask simply
stops growing -- no special-casing needed.

HISTORY -- do not retry these. Four "walk along the contour" schemes all
failed, and all for the SAME underlying reason (they measured against the
tool-notch rim, not the tumor): (1) angle forced monotonic via max() -- one
spurious 86 deg reading at f0 poisons the whole video; monotonicity belongs
on position, not angle. (2) row-lookup every frame -- pins the boundary point
to the tip's own row, dy=0, so the segment stays horizontal forever.
(3) wide-arc perpendicular search -- escapes around the CLOSED contour to the
far wall (f1410: 181 px away, spanning the tumor). (4) per-frame step cap --
bounds the RATE of drift but not the ACCUMULATED drift, so it still laps.
Also note a separate bug that made the angle look cosmetic for a while: the
rotated bottom test was ANDed with a leftover `rows <= frontier_y` cutoff,
which pinned the mask's lowest row and hid the rotation entirely (sweeping
0->89 deg changed area by <7% and lowest_row not at all).

Usage: cutline_sweep_mask_walk.py [video_dir] [tip_csv] [outdir]
                                  [resect_side=right] [tip_smooth=15]
                                  [mask_ema=0.5] [dilate=0]
"""
import sys, os, glob, csv
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import numpy as np, cv2, torch
from PIL import Image
from src.inference_engine import InferenceEngine

video_dir = sys.argv[1]
tip_csv = sys.argv[2]
outdir = sys.argv[3] if len(sys.argv) > 3 else \
    os.path.join(REPO_ROOT, "edge_anchored_tracking", "cutline_sweep_walk_out")
RESECT_SIDE = sys.argv[4] if len(sys.argv) > 4 else "right"
TIP_SMOOTH = int(sys.argv[5]) if len(sys.argv) > 5 else 15
MASK_EMA = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5   # 1.0 = no smoothing (use raw frame)
DILATE = int(sys.argv[7]) if len(sys.argv) > 7 else 0
os.makedirs(outdir, exist_ok=True)

SPATIAL_MODEL = os.environ.get(
    "SPATIAL_MODEL",
    os.path.join(REPO_ROOT, "checkpoints_tumor_binary10", "best_model_epoch_swa.pth"))
SPATIAL_META = os.environ.get(
    "SPATIAL_META",
    os.path.join(REPO_ROOT, "checkpoints_tumor_binary10", "best_model_epoch_swa_metadata.json"))

SEG_TANGENT_MAX_COS = 0.12  # a long sweep may not run along the tumor edge
SEG_TANGENT_MIN_LEN = 120.0 # ...only checked beyond this length, in px
SEG_CANDIDATES = 200     # boundary hits to try before giving up on a frame
CROSS_FRACTION = 0.25    # share of the segment allowed past the cut path
ARC_MIN_SAMPLES = 12     # below this, a wrong-side fraction is noise
ARC_SIDE_MARGIN = 0.25   # fraction gap needed to justify the longer arc
ARC_FRAC = 0.12          # fraction of contour length used for local tangent fit


def raw_tumor_mask(engine, img, H, W):
    """Un-smoothed per-frame tumor probability (soft), for EMA blending."""
    probs, _ = engine.predict_single(img)
    p = probs[1].cpu().numpy() if probs.shape[0] > 1 else probs[0].cpu().numpy()
    p = np.array(Image.fromarray(p).resize((W, H), Image.BILINEAR))
    return p.astype(np.float32)


def largest_component(mask_bool):
    nc, lbl, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8))
    if nc > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (lbl == big)
    return mask_bool


def contour_of(T):
    cnts, _ = cv2.findContours(T.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    # Ensure a consistent CLOCKWISE winding (image y-down coords): signed
    # area via the shoelace formula; OpenCV contours from RETR_EXTERNAL are
    # typically CCW in standard (y-up) math coords, which is CW in image
    # (y-down) coords already in most cases, but don't assume -- check.
    area2 = np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1])
    if area2 < 0:   # negative => counter-clockwise in image coords; flip
        c = c[::-1]
    return c



def outer_perimeter(T, resample_px=1.0):
    """The tumor's TRUE outer boundary, with the toolhead notches BRIDGED.

    binary10swa excludes the metal toolhead from the tumor mask, and those
    cutouts are indentations reaching in from the outside edge (measured:
    5519 px of concavity on Full_exp5 f300). The raw contour therefore dives
    into the notch and runs within ~3 px of the tool tip, so "nearest
    boundary point to the tip" snaps to the notch rim and any tangent fitted
    there is noise -- three smoothing windows disagreed by up to 79 deg.

    The convex hull bridges the notches and recovers the smooth perimeter the
    tumor actually presents (the tip then sits a sane 5-27 px from it, and
    tangents agree across smoothing scales to within a few degrees). The hull
    is resampled to ~`resample_px` spacing so tangent fits have evenly
    distributed support instead of clustering at hull vertices.

    Returns an (N,2) float32 array of perimeter points, or None.
    """
    c = contour_of(T)
    if c is None:
        return None
    hull = cv2.convexHull(c.astype(np.int32)).reshape(-1, 2).astype(np.float32)
    if len(hull) < 3:
        return None
    pts = []
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        n = max(int(np.linalg.norm(b - a) / resample_px), 1)
        for k in range(n):
            pts.append(a + (b - a) * (k / n))
    return np.array(pts, dtype=np.float32)


def perp_direction_at_tip(perim, tip_xy, arc_frac=0.12):
    """The REAL, correctly-SIGNED perpendicular vector from the tip toward
    the tumor's outer perimeter, plus the angle and nearest point.

    BUG THIS REPLACES: the perpendicular was previously computed from a PCA
    eigenvector, whose sign is ARBITRARY (numpy.linalg.eigh gives no sign
    guarantee). That signed vector was then immediately FOLDED into
    [0,90] deg via arctan2(abs(y),abs(x)) before being used anywhere, and a
    second function tried to RECONSTRUCT a direction from that folded angle
    with plain cos/sin -- which is only correct in one of four possible
    quadrants. Measured on Full_exp5: the raw eigenvector pointed AWAY from
    the true boundary at every single sampled frame (dot product with the
    tip->nearest-boundary-point vector was -1.000, -1.000, -0.999, -1.000 at
    f300/705/1057/1410) -- i.e. it was backwards 100% of the time, not
    intermittently. The reconstruction in the old segment_end_on_perimeter
    then guessed a direction close to straight down at steep angles instead
    of toward the actual edge (measured: ray landed 4px from the tip instead
    of ~25px at the real boundary).

    Fix: orient the tangent-derived perpendicular using the UNAMBIGUOUS
    tip->nearest-boundary vector (whose sign is correct by construction), and
    use that ONE oriented vector both for reporting the angle and for casting
    the ray -- no fold, no reconstruction, no round-trip.

    Returns (angle_deg in [0,90] for display, unit direction vector pointing
    from the tip toward the boundary, nearest perimeter point, distance).
    """
    tip = np.array(tip_xy, dtype=np.float32)
    d = np.linalg.norm(perim - tip[None, :], axis=1)
    idx = int(np.argmin(d))
    n = len(perim)
    w = max(int(n * arc_frac), 21)
    h = w // 2
    loc = perim[(np.arange(idx - h, idx + h + 1)) % n]
    dd = loc - loc.mean(axis=0)
    _, V = np.linalg.eigh(dd.T @ dd)
    tang = V[:, -1]
    perp = np.array([-tang[1], tang[0]])

    to_boundary = perim[idx] - tip
    dist = float(d[idx])
    if dist > 1e-3 and float(perp @ to_boundary) < 0:
        perp = -perp                      # orient toward the boundary
    perp_u = perp / (np.linalg.norm(perp) + 1e-9)

    ang = np.degrees(np.arctan2(abs(perp_u[1]), abs(perp_u[0])))
    return float(np.clip(ang, 0.0, 90.0)), perp_u, perim[idx], dist



def _ray_to_boundary(origin, direction, perim, fallback_len):
    """Where a ray from `origin` leaves the tumor, by actual intersection.

    This has to be a real ray/polygon crossing, not a nearest-point search.
    Ranking perimeter points by distance from the ray line picks whichever
    point lies closest to that line, and since the perimeter is a closed loop
    passing near the tip on both sides, that can be a point beside the tip
    rather than the one the ray travels to. Measured on failure_case1 f300:
    the endpoint came back at (378,273), 88 px away and strictly INSIDE the
    tumor. Both barrier ends then floated in the interior, the barrier never
    reached the outline, and the "two sides" stayed connected around it -- a
    58749 / 2 px split, so the guard rejected nothing at all.

    Taking the farthest crossing means the barrier leaves the tumor for good,
    which is what closes the partition.
    """
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if perim is None or len(perim) < 3:
        return origin + direction * fallback_len

    P = np.asarray(perim, dtype=np.float64)
    A = P
    B = np.roll(P, -1, axis=0)
    E = B - A                                   # edge vectors

    denom = direction[0] * E[:, 1] - direction[1] * E[:, 0]
    ok = np.abs(denom) > 1e-12
    if not ok.any():
        return origin + direction * fallback_len

    AO = A[ok] - origin
    t = (AO[:, 0] * E[ok][:, 1] - AO[:, 1] * E[ok][:, 0]) / denom[ok]
    u = (AO[:, 0] * direction[1] - AO[:, 1] * direction[0]) / denom[ok]
    hit = (t > 1e-6) & (u >= 0.0) & (u <= 1.0)
    if not hit.any():
        return origin + direction * fallback_len

    # Farthest crossing, then a little beyond, so the barrier ends outside.
    t_max = float(t[hit].max())
    return origin + direction * (t_max + 2.0)


def _side_partition(cut_path, tip_xy, H, W, perim=None, thickness=2,
                    start_stub=None):
    """Split the frame into the two sides of the cut curve.

    A continuous curve has exactly two sides however it bends, so the side a
    point lies on is a question about CONNECTIVITY, not about direction. That
    matters because every direction-based answer fails somewhere: comparing
    against `cutline_x[row]` is only defined while the cut is single-valued in
    row and breaks on the horizontal stretch of an inverted-C path, while a
    local tangent flips whenever the tool reverses and the cut doubles back
    into a Z.

    The cut is an open curve, so on its own it does not separate the frame.
    Its START already sits on the tumor boundary (the tool entered there); the
    leading end is extended from the tip along the cut's near-tip heading until
    it leaves the frame. Drawing that closed barrier and labelling connected
    components then yields the two sides directly, with no side convention, no
    tangent at evaluation time and no threshold.

    Returns an int32 label image (0 on the barrier itself, 1 and 2 for the two
    sides), or None if the path is too short to separate anything.
    """
    if cut_path is None or len(cut_path) < 2:
        return None
    P = np.asarray(cut_path, dtype=np.float64)

    # Heading over the last stretch of real travel, so a single jittery sample
    # cannot set the direction the barrier is extended in.
    tail = P[-12:] if len(P) >= 12 else P
    hv = tail[-1] - tail[0]
    n = float(np.hypot(hv[0], hv[1]))
    if n < 1e-6:
        hv = P[-1] - P[0]
        n = float(np.hypot(hv[0], hv[1]))
    if n < 1e-6:
        return None
    hv = hv / n

    diag = float(H + W)
    tipv = np.asarray(tip_xy, dtype=np.float64)
    # The leading end leaves the tumor straight DOWN from the keypoint.
    #
    # This end only has to close the partition, so it does not need to follow
    # the cut's heading -- and using that heading makes the barrier depend on
    # whichever way the tool happened to be moving, which swings as the tool
    # rounds the bottom of an inverted-C path. A vertical ray is stable frame
    # to frame and always exits, since the cut runs downward through the tumor.
    far = _ray_to_boundary(tipv, np.array([0.0, 1.0]), perim, diag)

    barrier = np.zeros((H, W), dtype=np.uint8)
    pts = np.vstack([P, far[None, :]]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(barrier, [pts], False, 1, thickness)

    # The START end is anchored ONCE, not redrawn each frame.
    #
    # The cut begins where the tool entered the tumor, so that end needs a
    # stub out to the boundary only while the cut is too short to reach it
    # itself. Recomputing the stub every frame keeps a ray pinned to the
    # cut's first point for the whole episode: on failure_case1 it points
    # (+1.00, 0.00), straight along the top of the tumor, and that stub
    # slices the upper-left region away from the resected side so the guard
    # admits it -- the 10656 px of untouched tissue there.
    #
    # Once the cut is long enough to bound the region on its own, the cut IS
    # the constraint and no stub is needed.
    if start_stub is not None:
        cv2.line(barrier, tuple(np.int32(P[0])),
                 tuple(np.int32(start_stub)), 1, thickness)

    # Close the barrier against the tumor OUTLINE, and label only what is
    # inside it. Terminating the two ends on the perimeter is not enough on
    # its own: the barrier is then an open curve strictly inside the frame,
    # which separates nothing -- you can walk around either endpoint. Measured
    # on failure_case1, that produced a 290428 / 2 px split, so the guard
    # never rejected anything and the mask matched the ablation with the guard
    # removed entirely.
    #
    # Drawing the outline turns the cut into a chord of the tumor, splitting
    # its interior into exactly the two regions the cut separates. Outside the
    # tumor is not a side of the cut and takes no label.
    inside = np.zeros((H, W), dtype=np.uint8)
    if perim is not None and len(perim) >= 3:
        cv2.fillPoly(inside, [np.asarray(perim, np.int32).reshape(-1, 1, 2)], 1)
    else:
        inside[:] = 1
    free = ((barrier == 0) & (inside > 0)).astype(np.uint8)

    n_lab, lab = cv2.connectedComponents(free, connectivity=4)
    if n_lab < 3:
        return None

    # A self-crossing cut encloses pockets, and each pocket comes back as its
    # own component -- on failure_case1, 302 of 484 frames produced three or
    # more, up to nine. A label then names a pocket rather than a side, and the
    # guard rejects sweeps into the rest of the resected side. But a pocket is
    # not a third side: the curve still has exactly two. Pockets are re-merged
    # into whichever side they touch, leaving the two genuine regions.
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    if len(counts) > 3:
        order = np.argsort(counts)[::-1]
        a, b = int(order[0]), int(order[1])
        remap = np.zeros(len(counts), dtype=np.int32)
        remap[a] = a
        remap[b] = b
        # Assign every pocket to the larger region it borders.
        for lb in range(1, len(counts)):
            if lb in (a, b) or counts[lb] == 0:
                continue
            m = (lab == lb).astype(np.uint8)
            grown = cv2.dilate(m, np.ones((3, 3), np.uint8))
            touch = lab[(grown > 0) & (m == 0)]
            touch = touch[touch > 0]
            if touch.size:
                nb = np.bincount(touch, minlength=len(counts))
                nb[lb] = 0
                remap[lb] = a if nb[a] >= nb[b] else b
            else:
                remap[lb] = a
        lab = remap[lab]
    return lab


def _side_of(lab, pt, H, W):
    """Which side label a point falls on (0 if it sits on the barrier)."""
    x = int(round(min(max(float(pt[0]), 0), W - 1)))
    y = int(round(min(max(float(pt[1]), 0), H - 1)))
    return int(lab[y, x])


def _crosses_cutline(tip, end_pt, cutline_x, resect_side, n_samples=40, tol=1.0,
                     side_lab=None, keep_side=None):
    """True if the straight tip->end_pt segment strays past the cut path.

    When a side partition is supplied the question is answered by CONNECTIVITY:
    the cut curve has two sides however it bends, and the sweep must stay on the
    resected one. That holds for any shape -- the inverted C these tools trace,
    and the Z it becomes when the tool reverses -- because it never asks which
    way the curve points.

    Without a partition it falls back to the original row-indexed comparison:
    the cutline is stored as one x per image row, which is exact while the cut
    descends. That fallback is what the opening frames use, before the path is
    long enough to separate the frame.
    """
    if side_lab is not None and keep_side:
        bad = seen = 0
        Hs, Ws = side_lab.shape
        for k in range(n_samples + 1):
            f = k / n_samples
            px = float(tip[0]) + (float(end_pt[0]) - float(tip[0])) * f
            py = float(tip[1]) + (float(end_pt[1]) - float(tip[1])) * f
            lb = _side_of(side_lab, (px, py), Hs, Ws)
            if lb == 0:
                continue                  # on the cut itself: no side
            seen += 1
            if lb != keep_side:
                bad += 1
        if seen == 0:
            return False
        return (bad / seen) > CROSS_FRACTION

    H = len(cutline_x)
    bad = seen = 0
    for k in range(n_samples + 1):
        f = k / n_samples
        px = float(tip[0]) + (float(end_pt[0]) - float(tip[0])) * f
        py = float(tip[1]) + (float(end_pt[1]) - float(tip[1])) * f
        r = int(round(min(max(py, 0), H - 1)))
        cx = cutline_x[r]
        if np.isnan(cx):
            continue                      # row the cut has not reached yet
        seen += 1
        if resect_side == "left":
            if px > cx + tol:
                bad += 1
        else:
            if px < cx - tol:
                bad += 1
    if seen == 0:
        return False                      # nothing to cross
    return (bad / seen) > CROSS_FRACTION


def segment_end_on_perimeter(perim, tip_xy, direction, cutline_x=None,
                             side_lab=None, keep_side=None, resect_side=None):
    """Where the bottom segment, leaving the tip along the (already correctly
    oriented) unit `direction`, meets the tumor's outer perimeter.

    The bottom edge must be a FINITE segment, not an infinite half-plane: a
    tilted line through the tip has no natural extent and sweeps in the whole
    upper-left of the tumor (measured f300: area 16009 vs 1701 for the
    horizontal baseline). Terminating it on the perimeter mirrors how the
    horizontal version stopped at the tumor's edge, and needs no invented
    radius.

    Takes `direction` directly (from perp_direction_at_tip) rather than
    re-deriving it from an angle -- that reconstruction was the bug. Returns
    the perimeter point closest to the ray ahead of the tip, or None.
    """
    tip = np.array(tip_xy, dtype=np.float32)
    dx, dy = float(direction[0]), float(direction[1])
    d = perim - tip[None, :]
    t = d[:, 0] * dx + d[:, 1] * dy      # projection along the ray
    ahead = t > 0
    if not ahead.any():
        return None
    perp_dist = np.abs(-d[:, 0] * dy + d[:, 1] * dx)   # distance off the ray
    cost = perp_dist + 0.01 * t          # nearest hit, mild preference for close
    cost[~ahead] = np.inf
    if not np.isfinite(cost).any():
        return None

    # THE BOTTOM EDGE MUST NOT CROSS THE CUTLINE.
    #
    # The bottom edge runs from the tip to the tumor boundary. The resected
    # region is on the resection side of the cut path, so that edge must stay
    # on that side too -- if it crosses back over the cut, the polygon covers
    # cutline that was already drawn and the mask appears to sweep backwards.
    #
    # Measured on the screen-recording case at the reported 0:12 window: the
    # segment sampled 61 points along its length and lay LEFT of the recorded
    # cutline on 53 of 61 at frame 380 and on 61 of 61 at frame 420, while
    # frames 340 and 360 (which look correct) crossed on 0 of 61.
    #
    # Note this is not a length test. At those frames the segment is short
    # (8 to 28 px); an earlier attempt to gate on segment length therefore
    # changed nothing, because the failing segments are not the long ones.
    order = np.argsort(cost)
    for cand in order[:SEG_CANDIDATES]:
        if not np.isfinite(cost[cand]):
            break
        # A LONG SWEEP MUST NOT RUN ALONG THE TUMOR'S EDGE.
        #
        # perp_direction_at_tip orients the sweep toward the perimeter point
        # NEAREST the tip. When the tip is close to the tumor's boundary, that
        # nearest point is the edge the tip is beside, not the direction the
        # resection should sweep -- so the segment runs roughly TANGENT to the
        # boundary and the polygon closes over a huge arc of untouched tumor.
        #
        # Measured on full_balanced_exp2 f160: |cos| between the segment and
        # the local boundary tangent was 0.438 over a 206 px segment, and the
        # mask went from 114 px to 9329 px in one frame with the tip having
        # moved 0.22 px. Because the region claimed touches the cut path, the
        # component filter can never remove it, so it persists to the end of
        # the episode -- the left-side mess. The same signature appears at
        # f295 (0.316 / 221 px) and on cond2_exp1 f14 and f35.
        #
        # Legitimate sweeps are nearly PERPENDICULAR to the boundary they end
        # on: the rebuild frames on DaggerMatched_NoRec_exp1 f223-228 measure
        # |cos| 0.019-0.022 while claiming just as much area, and cond2_exp1
        # f189 claims 23945 px over a 12.6 px segment. So the test pairs
        # tangency WITH length: a short tangent sweep is harmless, and a long
        # perpendicular one is exactly what resection looks like.
        #
        # The two populations are far apart -- every damaging frame measures
        # 0.216-0.464, every legitimate one 0.016-0.022 -- so the cutoff is
        # not tuned: 0.08, 0.12 and 0.18 all give byte-identical results on
        # all three episodes. 0.12 sits in the middle of that gap.
        #
        # This only skips the candidate; the search continues, so a better
        # endpoint is used instead of the frame being dropped.
        _cp = perim[cand]
        _sv0 = float(_cp[0]) - float(tip[0])
        _sv1 = float(_cp[1]) - float(tip[1])
        _sl = float(np.hypot(_sv0, _sv1))
        if _sl > SEG_TANGENT_MIN_LEN:
            _np_ = len(perim)
            _w = max(int(_np_ * 0.05), 5)
            _a = perim[(cand - _w) % _np_]
            _b = perim[(cand + _w) % _np_]
            _tv0 = float(_b[0]) - float(_a[0])
            _tv1 = float(_b[1]) - float(_a[1])
            _tl = float(np.hypot(_tv0, _tv1))
            if _tl > 1e-6:
                _cos = abs((_tv0 * _sv0 + _tv1 * _sv1) / (_tl * _sl))
                if _cos > SEG_TANGENT_MAX_COS:
                    continue
        if cutline_x is None or not _crosses_cutline(
                tip, _cp, cutline_x,
                resect_side if resect_side is not None else RESECT_SIDE,
                side_lab=side_lab, keep_side=keep_side):
            return _cp

    # Every candidate crosses the cut path. Contribute no bottom edge rather
    # than drawing one back over the cutline; the accumulated mask is
    # monotonic, so the region is claimed later once the geometry allows.
    return None





def shaft_direction_near_tip(T, perim, tip_xy, resect_side="right", min_px=300):
    """Direction along the TOOL SHAFT near the tip, fit from binary10swa's own
    exclusion of the toolhead -- used when the tumor perimeter has no real
    geometry near the tip to find a bottom-cut direction from.

    WHY THIS IS NEEDED: as the cut deepens and the local tumor boundary
    curves toward horizontal, the correct bottom-cut angle steepens toward
    vertical (by design -- see perp_direction_at_tip). But right where the
    angle is steepest, the tool shaft physically occludes the tissue below
    the tip, so binary10swa has no real tumor pixels there and the perimeter
    search lands on a noisy/short segment instead of a real edge. Measured
    on Balanced_exp1_590038: at f350 (angle 27 deg, shallow) the mask hugs
    the tip with no gap; by f450 (angle 79 deg) a gap has opened and persists
    through f560+ -- exactly where the perimeter search starts landing near
    the shaft occlusion instead of real tissue.

    binary10swa already segments the shaft CLEANLY as an exclusion region (a
    hole in the tumor mask, inside the hull, right where the metal prong
    is) -- more reliable than inferring shaft orientation from the single
    tracked tip point, which carries no direction information by itself.
    This fits the shaft's own principal axis via PCA/SVD on that excluded
    region and returns it as a unit direction pointing DOWN (toward
    increasing y, i.e. away from the tip along the shaft) -- oriented to
    have positive dot product with (0,1) so it's unambiguous regardless of
    the PCA sign. Only the exclusion component NEAREST the tip is used, so
    the LEFT tool's own exclusion (which can also split the mask) is
    ignored. Returns None if no sizable nearby exclusion is found.
    """
    if perim is None or len(perim) < 3:
        return None
    H, W = T.shape
    hull = cv2.convexHull(perim.astype(np.int32))
    solid = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(solid, [hull], 1)
    excluded = solid.astype(bool) & ~T
    nc, lbl, stats, cent = cv2.connectedComponentsWithStats(excluded.astype(np.uint8))
    tx, ty = tip_xy
    best, best_d = None, None
    for k in range(1, nc):
        if stats[k, cv2.CC_STAT_AREA] < min_px:
            continue
        cx, cy = cent[k]
        d = float(np.hypot(cx - tx, cy - ty))
        if best_d is None or d < best_d:
            best_d, best = d, k
    if best is None:
        return None
    ys, xs = np.where(lbl == best)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    pts -= pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts, full_matrices=False)
    axis = Vt[0]
    if axis[1] < 0:
        axis = -axis                      # orient downward (away from tip)
    return axis / (np.linalg.norm(axis) + 1e-9)


def tip_starts_inside(T, tip_xy, min_component_px=500, x_tol=20):
    """Decide, ONCE at the start, whether the cut begins INSIDE the tumor.

    Signal (user's): the tip and shaft are one body, and binary10swa excludes
    that body from the tumor mask. So while the tip is still OUTSIDE, the
    shaft cuts a channel from the tumor's edge inward and SEVERS the mask
    into two pieces. The moment the tip enters, the exclusion is surrounded
    by tumor and the mask is connected again. Hence: split => started
    outside; connected => started inside.

    The LEFT tool can also split the mask, which would look identical to a
    plain component count, so a split only counts when the gap is on the
    tracked (right) tip's side -- i.e. at least two sizable components
    straddle the tip's x. Components below `min_component_px` are ignored as
    segmentation speckle.
    """
    nc, lbl, stats, _ = cv2.connectedComponentsWithStats(T.astype(np.uint8))
    big = [k for k in range(1, nc) if stats[k, cv2.CC_STAT_AREA] >= min_component_px]
    if len(big) <= 1:
        return True                      # connected => tip began inside
    tx = float(tip_xy[0])
    near = 0
    for k in big:
        x0 = stats[k, cv2.CC_STAT_LEFT] - x_tol
        x1 = stats[k, cv2.CC_STAT_LEFT] + stats[k, cv2.CC_STAT_WIDTH] + x_tol
        if x0 <= tx <= x1:
            near += 1
    return near <= 1                     # split on the tip's side => outside


def local_tangent_vec(contour, idx, arc_frac=ARC_FRAC):
    """Unit tangent vector of the boundary at `idx` (wide-arc PCA)."""
    n_pts = len(contour)
    neighborhood = max(int(n_pts * arc_frac), 21)
    n = neighborhood // 2
    idxs = (np.arange(idx - n, idx + n + 1)) % n_pts
    local = contour[idxs]
    d = local - local.mean(axis=0)
    w, v = np.linalg.eigh(d.T @ d)
    t = v[:, -1]
    return t / (np.linalg.norm(t) + 1e-9)




def smooth_tips(tips_by_frame, win):
    """Moving-average the tip trajectory over `win` frames.

    BUG THIS FIXES: `np.convolve(..., mode="same")` zero-pads past the ends
    of the array. Near frame 0 that means the window averages in several
    PHANTOM ZEROS from before the trajectory began, dragging the smoothed x
    toward 0 (the left image edge) regardless of where the tip actually
    starts. Measured on Balanced_exp1_590038: the tip is genuinely constant
    at raw x=319 for its first ~7 frames, but smoothing pulled the reported
    frame-0 position to x=170 -- a 149px phantom jump left, with nothing in
    the real trajectory to justify it. That corrupted the cutline's start
    point and produced a mask segment pointing at the wrong part of the
    tumor for the video's first several frames. This reproduced on every
    video regardless of tumor position, confirming it's a smoothing-window
    edge artifact, not a per-video geometry issue.

    Fix: pad by EDGE REPLICATION (repeat the first/last real sample) instead
    of zeros, so frames near the start/end average against real values.
    """
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
    if not files:
        raise SystemExit(
            f"No PNG frames found in {video_dir}\n"
            f"  This stage expects the same folder of numbered PNG frames used\n"
            f"  for tracking. See 'Input data' in README.md.")
    first = cv2.imread(files[0]); H, W = first.shape[:2]

    tips_raw, trusted = load_tips(tip_csv)
    tips = smooth_tips(tips_raw, TIP_SMOOTH)
    n_untrusted = sum(1 for v in trusted.values() if not v)
    print(f"{len(files)} frames; {len(tips)} tips; side={RESECT_SIDE} smooth={TIP_SMOOTH} "
          f"mask_ema={MASK_EMA} untrusted(M)={n_untrusted}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = InferenceEngine(model_path=SPATIAL_MODEL, metadata_path=SPATIAL_META, device=device)
    print("binary10swa loaded")

    cutline_x = np.full(H, np.nan, dtype=np.float32)
    # Ordered trace of tip positions, kept ALONGSIDE cutline_x. The row
    # array cannot represent horizontal motion: `cutline_x[tyi] = tx`
    # stores one x per integer row, so when the tool sweeps sideways every
    # sample lands on the same row and overwrites the last (25 of 50 frames
    # at 0:08-0:09 of failure_case2). Worse, rows written on different
    # passes are never joined: row 169 held x=267.7 from f176 on the way
    # down while row 170 held x=382.7 from f278 on the way back up -- a
    # 115 px step across one row, drawn as the horizontal line in the mask.
    # The polyline records the path as travelled, so consecutive samples
    # connect regardless of direction. cutline_x stays exactly as it was;
    # every other consumer still reads it.
    cut_path = []
    frontier_y = -1
    prev_tip_y = None

    # No walk state. The bottom-cut angle is computed independently each
    # frame from the tumor's OUTER perimeter (tool notches bridged) at the
    # point nearest the tip -- see perp_direction_at_tip(). Nothing accumulates,
    # so nothing can ratchet or escape.
    last_angle_deg = 0.0    # carried only when a frame has no usable perimeter

    # Case decision, made ONCE, on the frame where the tip FIRST tests inside
    # the tumor's notch-bridged hull:
    #   Case A -- the tip enters the tumor while the cut is still in progress
    #       (it began inside, with un-resected tissue above it). The tumor's
    #       own upper boundary would wrongly include that tissue, so the top
    #       edge is the cutline's row at that moment -- `cut_start_y`.
    #   Case B -- the tip never enters (it began above/outside and the cut
    #       runs down past the tumor). The cutline and the tumor boundary
    #       already meet and enclose the region, so no artificial top edge is
    #       needed and the `& T` clip serves as the top.
    # Replaces the old vertical-ray "closure segment", which addressed the
    # same gap less directly.
    #
    # Keyed on the tip-enters-hull EVENT rather than on mask area: early
    # frames are unusable for either signal because binary10swa has not found
    # the tumor yet (nodaggernorec4 is 1650 px at f0, ~30000 by f50, and its
    # component count flails 1/3/2/1 meanwhile). The entry event needs no
    # invented threshold and fires exactly when the geometry becomes real --
    # measured on that video, the hull test flips -18 px -> +21 px at f50,
    # where the split-signal also reads INSIDE, so the two agree there.
    case_decided = False
    starts_inside = False        # stays False if the tip never enters (Case B)
    cut_start_y = None

    xs_grid = np.arange(W)[None, :].astype(np.float32)
    ys_grid = np.arange(H)[:, None].astype(np.float32)

    ema_mask_prob = None  # temporally-smoothed tumor probability

    vw = cv2.VideoWriter(os.path.join(outdir, "cutline_sweep_walk.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    accum_mask = np.zeros((H, W), dtype=bool)
    _START_STUB = [None]  # start-end anchor, fixed on first use
    _KEEP_SIDE = [None]   # which partition label is the resection
    _KEEP_SEED = [None]   # a point known to lie on that side
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(DILATE | 1, 3),) * 2) if DILATE > 0 else None

    for i, fp in enumerate(files):
        img_rgb = np.array(Image.open(fp).convert("RGB"))
        raw_prob = raw_tumor_mask(engine, img_rgb, H, W)
        ema_mask_prob = raw_prob if ema_mask_prob is None else \
            MASK_EMA * raw_prob + (1 - MASK_EMA) * ema_mask_prob
        T = largest_component(ema_mask_prob > 0.5)
        # The tumor's TRUE outer perimeter, with the toolhead notches bridged
        # (convex hull). The raw contour dives into the tool cutouts and runs
        # ~3 px from the tip, which makes any local tangent there meaningless;
        # bridging recovers the smooth boundary the tumor actually presents.
        # T itself (notches and all) is still what gets masked/displayed.
        perim = outer_perimeter(T)

        is_trusted = trusted.get(i, True)
        if i in tips and is_trusted:
            tx, ty = tips[i]
            tyi = int(round(np.clip(ty, 0, H - 1)))
            # LATEST PASS OWNS THE ROW.
            #
            # cutline_x holds one x per image row, so a row the tool crosses
            # twice has two competing values. Blending them 50/50 (the old
            # rule) leaves the row stranded halfway between the two passes:
            # measured on failure_case2, 17 rows are crossed by both an
            # outbound leftward move (f177-198, x~238) and the return sweep
            # (f264-315, x~373), up to 135 px apart. The polygon spans
            # cutline -> tumor boundary, so those stale leftward values kept
            # dragging the sweep ~150 px left of the tool.
            #
            # The tool's current position is the better description of where
            # the cut now runs, so a revisit overwrites rather than averages.
            # This only moves where FUTURE polygons are drawn -- accumulation
            # is `accum_mask | frame_mask`, so mask already earned on earlier
            # frames is not deleted by the rewrite.
            if not cut_path or (abs(cut_path[-1][0] - tx) > 1e-6 or
                                abs(cut_path[-1][1] - ty) > 1e-6):
                cut_path.append((float(tx), float(ty)))
            cutline_x[tyi] = tx
            # Fill the rows between the previous tip row and this one, so the
            # cut path stays continuous when the tip moves several rows in a
            # frame. This runs in BOTH directions: advancing down the image
            # extends the path, and moving back up rewrites the rows the tool
            # is re-crossing (see the latest-pass-wins note above). Only the
            # downward case advances the frontier.
            if prev_tip_y is not None and prev_tip_y != tyi:
                prev_y = prev_tip_y
                prev_x = cutline_x[prev_y] if not np.isnan(cutline_x[prev_y]) else tx
                step = 1 if tyi > prev_y else -1
                for yy in range(prev_y + step, tyi + step, step):
                    frac = (yy - prev_y) / float(tyi - prev_y)
                    cutline_x[yy] = prev_x + frac * (tx - prev_x)
            elif tyi > frontier_y:
                prev_y = frontier_y if frontier_y >= 0 else tyi
                prev_x = cutline_x[prev_y] if not np.isnan(cutline_x[prev_y]) else tx
                for yy in range(prev_y + 1, tyi + 1):
                    frac = (yy - prev_y) / max(tyi - prev_y, 1)
                    cutline_x[yy] = prev_x + frac * (tx - prev_x)
            if tyi > frontier_y:
                frontier_y = tyi
            prev_tip_y = tyi

            # Decide the case on the frame the tip FIRST enters the hull.
            if not case_decided and perim is not None and len(perim) >= 3:
                inside = cv2.pointPolygonTest(
                    perim.reshape(-1, 1, 2), (float(tx), float(ty)), False) >= 0
                if inside:
                    case_decided = True
                    starts_inside = True
                    cut_start_y = tyi
                    print(f"  case A at f{i}: tip entered tumor, "
                          f"top edge = cut start row {tyi}")

        # Bottom-cut angle: perpendicular to the tangent of the tumor's OUTER
        # perimeter at the point nearest the tip. Recomputed independently
        # every frame -- there is no walk, no accumulated position, so there
        # is nothing that can ratchet or escape around the contour. Verified
        # on Full_exp5: three smoothing windows agree within a few degrees,
        # and the angle sweeps 88 -> 36 -> 1 (flattest, mid-cut) -> 50 -> 84
        # as the tip travels around the crescent, which is the intended
        # horizontal-at-the-side / vertical-at-the-ends behavior.
        angle_deg = last_angle_deg
        perp_dir = None
        if is_trusted and i in tips and perim is not None and len(perim) >= 40:
            angle_deg, perp_dir, _, _ = perp_direction_at_tip(perim, tips[i])
            last_angle_deg = angle_deg

        # Mask = tumor, on the resected side of the cutline, ABOVE the bottom
        # cut -- a line through the tip at `angle_deg`. At angle 0 this is
        # exactly the original horizontal-only model's `row <= frontier_y`.
        # Build the region as ONE CLOSED POLYGON:
        #   LEFT   -- down the cutline, from where the cut began to the tip,
        #   BOTTOM -- the FINITE segment from the tip out to the perimeter,
        #   RIGHT/TOP -- back along the tumor perimeter to the start.
        # A filled polygon (not a stack of half-planes) keeps the region
        # bounded without any invented radius. An infinite tilted half-plane
        # swept in the whole upper-left of the tumor (f300: area 16009 vs
        # 1701 for the horizontal baseline).
        frame_mask = np.zeros((H, W), dtype=bool)
        if frontier_y >= 0 and perim is not None and i in tips and perp_dir is not None:
            # ROOT CAUSE (found 2026-08-03, pinpointed via frame-by-frame
            # trace on full_balanced_exp2_479979 f1380-1410): cutline_x[row]
            # is stamped once, from whichever frame's y happened to round to
            # that row, and never revisited. When the tip's y is NOT
            # monotonic -- e.g. it touches row 208 once, then hovers at rows
            # 206-207 for 50+ frames while x keeps drifting left -- row 208
            # is left permanently stamped with an OLDER, more-rightward x
            # (285.1) than the FRESHER rows 206-207 (247.6, 251.2) that the
            # tip is actually sitting on now. Including row 208 in the
            # polygon (via `<= frontier_y`) put a stale, temporally-earlier
            # point after two fresher ones, producing a 65px zigzag in the
            # cutline edge -- the "Z shape" / mechanical retraction the user
            # saw whenever the tip stalls instead of moving monotonically.
            #
            # FIX: clip to the CURRENT frame's tip row (tyi), not the
            # historical max (frontier_y). This changes NOTHING when motion
            # is monotonic (tyi == frontier_y every frame, which is the
            # normal/working case) -- verified this is a no-op there. When
            # the tip has drifted back above frontier_y, it only excludes
            # rows temporally AHEAD of where the tip is right now; nothing is
            # lost, since accum_mask is monotonic and already captured that
            # row's contribution in the frame where it WAS current. Verified:
            # this drops exactly the stale row 208 at f1410 and nothing else
            # (rows 203-207 unchanged).
            tyi_now = int(round(np.clip(tips[i][1], 0, H - 1)))
            visited_rows = np.where(~np.isnan(cutline_x))[0]
            visited_rows = visited_rows[visited_rows <= tyi_now]
            if len(visited_rows) >= 2:
                tx, ty = tips[i]
                # Which of the cut's two sides is the resection. Seeded on
                # the first frame that can be partitioned, from a probe placed
                # just right of the tip (the resection starts right of the
                # cutline), then carried across frames by continuity: the side
                # keeps its identity even as the curve bends, so a reversal
                # into a Z cannot swap it. This is also what stops an early
                # frame putting the mask on the wrong side and having it
                # removed later, as happened on cond2_exp1.
                if (_START_STUB[0] is None and len(cut_path) >= 2
                        and perim is not None and len(perim) >= 3):
                    # Leave the tumor PERPENDICULAR to its boundary.
                    #
                    # Using the cut's own reverse heading sends this stub
                    # wherever the tool happened to be moving when it entered,
                    # which on failure_case1 is up-RIGHT along the top edge:
                    # a 29 px stub running parallel to the boundary instead of
                    # out through it. The barrier then never crossed the
                    # outline at that end, the two sides stayed connected
                    # around the cut's start, and the partition came back
                    # 58743 / 2 px -- so the guard rejected nothing.
                    #
                    # The boundary normal always exits, wherever the cut
                    # begins: it reduces to straight up when the tool enters at
                    # the top, and still leaves cleanly if a cut starts on a
                    # side.
                    _p0 = np.asarray(cut_path[0], dtype=np.float64)
                    _pp = np.asarray(perim, dtype=np.float64)
                    _k = int(np.argmin(np.hypot(_pp[:, 0] - _p0[0],
                                                _pp[:, 1] - _p0[1])))
                    _m = len(_pp)
                    _w = max(int(_m * 0.05), 5)
                    _ta = _pp[(_k - _w) % _m]
                    _tb = _pp[(_k + _w) % _m]
                    _tv = _tb - _ta
                    _tn = float(np.hypot(_tv[0], _tv[1]))
                    if _tn > 1e-6:
                        _nv = np.array([-_tv[1], _tv[0]]) / _tn
                        # Point it OUT of the tumor.
                        _probe = _p0 + _nv * 6.0
                        if cv2.pointPolygonTest(
                                np.asarray(perim, np.int32),
                                (float(_probe[0]), float(_probe[1])),
                                False) >= 0:
                            _nv = -_nv
                        _START_STUB[0] = _ray_to_boundary(
                            _p0, _nv, perim, float(H + W))
                _lab = _side_partition(cut_path, (tx, ty), H, W, perim=perim,
                                       start_stub=_START_STUB[0])
                _keep = None
                if _lab is not None:
                    if _KEEP_SIDE[0] is None:
                        _off = 8.0 if RESECT_SIDE != "left" else -8.0
                        for _d in (_off, _off * 2, _off * 4):
                            _c = _side_of(_lab, (tx + _d, ty), H, W)
                            if _c:
                                _KEEP_SIDE[0] = _c
                                _KEEP_SEED[0] = np.array([tx + _d, ty])
                                break
                    elif _KEEP_SEED[0] is not None:
                        # Re-identify by position: labels are renumbered every
                        # frame, so the stored seed says which side is which.
                        _c = _side_of(_lab, _KEEP_SEED[0], H, W)
                        if _c:
                            _KEEP_SIDE[0] = _c
                    _keep = _KEEP_SIDE[0]
                seg_end = segment_end_on_perimeter(perim, (tx, ty), perp_dir,
                                                   cutline_x, side_lab=_lab,
                                                   keep_side=_keep)
                if seg_end is not None:
                    # The polygon's LEFT edge is the path as the tool actually
                    # travelled it, not a per-row resampling of it. Sampling by
                    # row flattened every sideways move onto one row and left a
                    # 115 px break where two passes met; the ordered trace has
                    # neither problem and needs no interpolation.
                    #
                    # Truncation still applies, but on the PATH: points ahead of
                    # the tip's current row are dropped, which is the same rule
                    # as before (it prevents a stale row temporally ahead of the
                    # tip from zigzagging the edge) expressed on the trace.
                    # Truncate to a CONTIGUOUS PREFIX, never a filter. Keeping
                    # every point with y <= tip_y scatters the selection: on
                    # failure_case2 f300 it dropped 92 interior points and
                    # spliced (263.0, 175.1) straight onto (371.3, 175.2), a
                    # 108 px horizontal jump -- the same artifact in a new
                    # place. Cutting at the first point that runs ahead of the
                    # tip keeps the trace connected, so consecutive vertices
                    # are always neighbours the tool actually travelled
                    # between.
                    _stop = len(cut_path)
                    for _k in range(len(cut_path) - 1, -1, -1):
                        if cut_path[_k][1] <= ty + 0.5:
                            _stop = _k + 1
                            break
                    cut_pts = cut_path[:_stop]
                    if len(cut_pts) < 2:
                        cut_pts = list(cut_path)
                    # Close the path on the tool tip itself.
                    if (abs(cut_pts[-1][0] - tx) > 1e-6 or
                            abs(cut_pts[-1][1] - ty) > 1e-6):
                        cut_pts.append((float(tx), float(ty)))
                    cut_pts_arr = np.array(cut_pts, dtype=np.float32)
                    start_pt = cut_pts[0]

                    # Perimeter arc from the segment end back round to the
                    # cut's start. Two candidate arcs exist (the two ways
                    # round the closed perimeter); pick whichever stays
                    # CLOSER to the accumulated cutline -- the cutline and
                    # the tumor's outer boundary are both roughly nested
                    # inverted-C shapes, and the resection always hugs close
                    # to the boundary, so the correct arc runs near-parallel
                    # to the cutline the whole way. Point-count ("take the
                    # shorter arc") is NOT reliable: with tip smoothing on,
                    # measured 315 vs 321 points (nearly tied) and it picked
                    # the WRONG (84px mean distance) arc over the right one
                    # (19px mean distance) -- a coin flip that depends on
                    # smoothing/EMA parameters, not on the actual geometry.
                    i_end = int(np.argmin(np.linalg.norm(
                        perim - np.array(seg_end, dtype=np.float32), axis=1)))
                    i_start = int(np.argmin(np.linalg.norm(
                        perim - np.array(start_pt, dtype=np.float32), axis=1)))
                    n = len(perim)
                    fwd = (i_start - i_end) % n
                    arc_fwd = [perim[(i_end + k) % n] for k in range(fwd + 1)]
                    arc_bwd = [perim[(i_end - k) % n] for k in range(n - fwd + 1)]

                    def _mean_dist_to_cutline(arc_pts):
                        arc_np = np.array(arc_pts, dtype=np.float32)
                        # distance from each arc point to its nearest cutline
                        # point, vectorized
                        diffs = arc_np[:, None, :] - cut_pts_arr[None, :, :]
                        return float(np.mean(np.min(np.linalg.norm(diffs, axis=2), axis=1)))

                    # BASELINE + ARC SIDE CONSTRAINT ONLY.
                    # Everything else in this file is the original algorithm:
                    # the direction still points at the nearest boundary, the
                    # endpoint search is unchanged. Only the arc's winding is
                    # constrained, so this isolates the effect of that one
                    # change from the direction/endpoint fixes in v12.
                    def _frac_wrong_side(arc_pts):
                        wrong = total = 0
                        for ax, ay in arc_pts:
                            r = int(round(min(max(ay, 0), H - 1)))
                            cx = cutline_x[r]
                            if np.isnan(cx):
                                continue
                            total += 1
                            if RESECT_SIDE == "left":
                                if ax > cx + 1.0:
                                    wrong += 1
                            else:
                                if ax < cx - 1.0:
                                    wrong += 1
                        return wrong / max(total, 1)

                    def _n_samples(arc_pts):
                        """How many arc points sit at rows the cutline has reached."""
                        t = 0
                        for _ax, _ay in arc_pts:
                            _r = int(round(min(max(_ay, 0), H - 1)))
                            if not np.isnan(cutline_x[_r]):
                                t += 1
                        return t

                    # Decide with the SAMPLE COUNT in hand, not just the
                    # fraction. Measured on 881426 f57: the bwd arc was 3
                    # points long, so its "0.67 wrong side" was 2 points out
                    # of 3 -- noise. On that single frame the rule preferred
                    # the 510-point fwd arc (0.60), and because accumulation
                    # is monotonic that one frame's wrap-around mask was
                    # unioned in and never cleared. It is the entire reason
                    # the final mask nearly doubled (10.5k -> 19.7k px).
                    #
                    # Two guards, both about not trusting a weak signal:
                    #   * an arc with too few samples cannot veto the other;
                    #   * a marginal fraction difference never buys a wildly
                    #     longer arc, since the local arc is the correct one
                    #     whenever the evidence is thin.
                    _wf, _wb = _frac_wrong_side(arc_fwd), _frac_wrong_side(arc_bwd)
                    _nf, _nb = _n_samples(arc_fwd), _n_samples(arc_bwd)
                    _short, _long = (arc_fwd, arc_bwd) if len(arc_fwd) <= len(arc_bwd)                                     else (arc_bwd, arc_fwd)
                    _ws = _wf if _short is arc_fwd else _wb
                    _wl = _wb if _short is arc_fwd else _wf
                    _ns = _nf if _short is arc_fwd else _nb
                    _nl = _nb if _short is arc_fwd else _nf

                    if _ns < ARC_MIN_SAMPLES or _nl < ARC_MIN_SAMPLES:
                        # Not enough of either arc lies at rows the cutline has
                        # reached; the fractions carry no information.
                        arc = _short
                    elif _ws - _wl > ARC_SIDE_MARGIN:
                        # The short arc is clearly the one crossing to the
                        # wrong side, so take the long way round.
                        arc = _long
                    else:
                        arc = _short

                    poly = cut_pts + [(float(seg_end[0]), float(seg_end[1]))] + \
                           [(float(p[0]), float(p[1])) for p in arc]
                    filled = np.zeros((H, W), dtype=np.uint8)
                    cv2.fillPoly(filled, [np.array(poly, np.int32).reshape(-1, 1, 2)], 1)
                    frame_mask = T & filled.astype(bool)
                    # Clip the filled region to the resected side of the cut.
                    #
                    # Vetting only the sweep ENDPOINT is not enough: the
                    # polygon is cut_pts + seg_end + perimeter ARC, and the arc
                    # can wrap the far side of the tumor and fill the
                    # unresected half however the endpoint was chosen. Measured
                    # on failure_case1 f159, one frame put 40247 px on the
                    # wrong side of the cut and 11069 px of it survived to the
                    # end. The partition already says which side is resected,
                    # so applying it to the region itself closes that path.
                    if _lab is not None and _keep:
                        frame_mask &= ~((_lab > 0) & (_lab != _keep))

        # REVERTED 2026-08-03: tried dropping the trailing `& T` here (see
        # git history / prior session) on the theory that it was retroactively
        # erasing accumulated pixels. WRONG FIX -- user correctly pointed out
        # this makes the mask static and lets false positives accumulate
        # forever with no way to clear. The `& T` re-clip is load-bearing:
        # it's what allows a wrongly-included pixel to drop back out once
        # binary10swa stops calling it tumor. The real bug causing the visible
        # gap/"Z-shape" at the tip is still unresolved -- see investigation
        # notes below this function.
        # `& T` retracts a pixel once binary10swa stops calling it tumor,
        # which is what lets a false positive drop back out. But T is also
        # false wherever a TOOL occludes tissue -- the model does not label
        # metal as tumor -- and that dropout is temporary while the deletion
        # is permanent. Measured on failure_case3 f388: 933 px were cut by
        # `& T` and 25 frames later 870 of them (93%) were tumor again, but
        # only 397 had returned to the mask, because accumulation re-adds only
        # through the current sweep polygon. Each pass of a shaft over
        # resected ground therefore ratcheted the mask down -- the loss at
        # 0:14-0:15 of failure_case3 as the tool sweeps back down.
        #
        # The tools are exactly the notches the convex hull bridges: the
        # toolhead cutouts are indentations reaching in from the outside edge,
        # so `hull & ~T` is the tool region without needing a tool detector.
        #
        # Only the part of the notch ABOVE the tip row is exempt. That is the
        # shaft running back from the tool, over ground the cut has already
        # passed; tissue there does not stop being resected because metal is
        # in front of it. Below the tip the tool sits on tissue the cut has
        # not reached yet, which is not resected and must still be free to
        # retract, so `& T` is left to act there unchanged.
        # Only the notch COMPONENT the tool tip belongs to, above the tip
        # row. The hull bridges genuine tumor concavity as well as the two
        # tool notches, so exempting the whole notch let the mask persist
        # outside the tumor as T's boundary fluctuated -- the accumulation
        # past the boundary at 0:14 of failure_case2 (1460 px). Selecting the
        # component that touches the tip keeps the shaft and drops unrelated
        # concavity, with no threshold.
        _occluded = np.zeros((H, W), dtype=bool)
        if perim is not None and len(perim) >= 3 and i in tips:
            _hf = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(_hf, [perim.astype(np.int32).reshape(-1, 1, 2)], 1)
            _notch = _hf.astype(bool) & ~T
            _nn, _nlab = cv2.connectedComponents(_notch.astype(np.uint8),
                                                 connectivity=8)
            if _nn > 1:
                _txi = int(round(np.clip(tips[i][0], 0, W - 1)))
                _tyi = int(round(np.clip(tips[i][1], 0, H - 1)))
                # The tip sits just inside the tissue, so search a small
                # neighbourhood for the notch it borders.
                _lb = 0
                for _rad in (0, 2, 4, 7, 11, 16):
                    _y0, _y1 = max(_tyi - _rad, 0), min(_tyi + _rad + 1, H)
                    _x0, _x1 = max(_txi - _rad, 0), min(_txi + _rad + 1, W)
                    _win = _nlab[_y0:_y1, _x0:_x1]
                    _nz = _win[_win > 0]
                    if _nz.size:
                        _vals, _cnt = np.unique(_nz, return_counts=True)
                        _lb = int(_vals[int(np.argmax(_cnt))])
                        break
                if _lb:
                    _occluded = (_nlab == _lb)
                    _occluded[_tyi:, :] = False
        accum_mask = (accum_mask | frame_mask) & (T | _occluded)

        # HARD BOUND: never hold mask outside the tumor's own outer boundary.
        # The occlusion exemption is what allows a pixel to sit outside T at
        # all, and once one does it keeps re-qualifying while the shaft covers
        # it, so as T's boundary fluctuates the exempt region can drift past
        # the tumor edge. The hull is that outer boundary with the tool
        # notches bridged, so clipping to it lets tissue under a shaft survive
        # while nothing survives beyond the tumor itself.
        if perim is not None and len(perim) >= 3:
            _bound = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(_bound, [perim.astype(np.int32).reshape(-1, 1, 2)], 1)
            accum_mask &= _bound.astype(bool)

        # KEEP ONLY THE REGION CONNECTED TO THE CUT.
        #
        # The resection grows outward from the cut path and stays attached to
        # it, so any component not touching the cutline came from a bad frame.
        # Accumulation is monotonic, so without this such a component survives
        # to the end: measured on the screen-recording case, one frame planted
        # a 5,461 px blob 90 px left of the mask and it persisted for the
        # remaining 370 frames.
        #
        # This fixes the outcome rather than guessing at the cause, so it has
        # no thresholds and does not depend on tip position or frame timing.
        n_lab, lab = cv2.connectedComponents(accum_mask.astype(np.uint8), connectivity=8)
        if n_lab > 2:
            keep = set()
            for r in np.where(~np.isnan(cutline_x))[0]:
                c = int(round(cutline_x[r]))
                for off in (0, -2, 2, -5, 5):
                    cc = c + off
                    if 0 <= cc < W and lab[r, cc]:
                        keep.add(int(lab[r, cc]))
            if keep:
                accum_mask = np.isin(lab, list(keep))
        disp = accum_mask
        if kernel is not None:
            disp = cv2.dilate(disp.astype(np.uint8), kernel).astype(bool) & T

        vis = cv2.imread(fp)
        vis[disp] = (0.5 * np.array([0, 0, 255]) + 0.5 * vis[disp]).astype(np.uint8)
        if i in tips:
            tx, ty = tips[i]; txi, tyi = int(round(tx)), int(round(ty))
            tip_col = (255, 0, 255) if is_trusted else (0, 165, 255)
            cv2.circle(vis, (txi, tyi), 5, tip_col, -1)
        cv2.putText(vis, f"t={i} area={int(disp.sum())} angle={angle_deg:.0f}"
                    f"{'' if is_trusted else ' FROZEN'}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        vw.write(vis)
        last_vis = vis

    vw.release()

    # The mask is cumulative, so the state after the final frame is the
    # complete resected region for the episode. Save it as a plain binary
    # image for downstream analysis (area, overlap with ground truth), plus
    # an overlay on the last frame for visual confirmation.
    final_mask = disp
    mask_path = os.path.join(outdir, "final_mask.png")
    cv2.imwrite(mask_path, (final_mask.astype(np.uint8) * 255))

    overlay_path = os.path.join(outdir, "final_overlay.png")
    cv2.imwrite(overlay_path, last_vis)

    area_px = int(final_mask.sum())
    print(f"final mask: {area_px} px ({100.0 * area_px / (H * W):.2f}% of frame) "
          f"-> {mask_path}")
    print(f"final overlay -> {overlay_path}")

    print(f"DONE side={RESECT_SIDE} smooth={TIP_SMOOTH} mask_ema={MASK_EMA} dilate={DILATE} "
          f"final_angle={last_angle_deg:.1f} "
          f"case={'A (top=row %s)' % cut_start_y if starts_inside else 'B (top=tumor boundary)'} "
          f"-> {outdir}")


if __name__ == "__main__":
    main()
