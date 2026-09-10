"""Two-pass SAM2 tip refinement: a single-click SAM2 prompt often segments
only the shaft (missing the tooltip) or, less often, only the tooltip
(missing the shaft) -- because one point isn't always enough for SAM2 to
commit to the whole elongated tool. This adds a second pass: skeletonize the
first-pass mask, walk WALK_PX pixels along the skeleton's own path (not a
straight line) from the point nearest the original click, in both
directions, to get two more points ON the tool body. Re-prompt SAM2 with
all 3 points (original + both walked points) and use THAT mask for the
final tip refinement (skeleton-endpoint-nearest-original, same as before).

Usage: refine_tip_with_sam2_v2.py <video_dir> <auto_seed.json> <out_dir> [walk_px=5] [refined_seed_out.json]

If refined_seed_out.json is given, writes the refined point back out in the
same {"points": [[x,y]], "W", "H", "frame"} schema auto_seed_tip.py uses, so
it can be fed straight into tapnextpp_hybrid.py in place of the raw seed.
"""
import sys, os, json, tempfile, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2, torch
from PIL import Image
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_seed_tool import _mask_shape_ok

video_dir = sys.argv[1]
seed_json_path = sys.argv[2]
out_dir = sys.argv[3]
WALK_PX = int(sys.argv[4]) if len(sys.argv) > 4 else 5
refined_seed_out = sys.argv[5] if len(sys.argv) > 5 else None
os.makedirs(out_dir, exist_ok=True)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CKPT = os.path.join(REPO_ROOT, "checkpoints_sam2", "sam2.1_hiera_small.pt")
# SAM2's video predictor wants a real directory of frames, so each pass stages
# its single frame into a scratch dir. Defaults to the OS temp dir; override
# SAM2_SCRATCH to put it on a faster local disk if temp is slow or small.
SCRATCH = os.environ.get("SAM2_SCRATCH", os.path.join(tempfile.gettempdir(),
                                                       "sam2_scratch"))

DEBUG_DUMP_DIR = out_dir
DEBUG_TAG = os.path.basename(os.path.dirname(os.path.normpath(video_dir)))


def skeleton_graph(mask):
    """Skeletonize and return (skel, ordered pixel-adjacency dict) so we can
    walk the skeleton's actual path (not straight-line distance)."""
    skel = skeletonize(mask).astype(np.uint8)
    ys, xs = np.where(skel)
    pts = set(zip(xs.tolist(), ys.tolist()))
    neighbors = {}
    for (x, y) in pts:
        nbrs = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if (x + dx, y + dy) in pts:
                    nbrs.append((x + dx, y + dy))
        neighbors[(x, y)] = nbrs
    return skel, pts, neighbors


def skeleton_endpoints_from_graph(pts, neighbors):
    return [p for p in pts if len(neighbors[p]) == 1]


def component_containing(mask, xy):
    """Restrict `mask` to the single connected component containing pixel
    `xy`. BUG THIS FIXES: SAM2's pass-2 mask can include a small,
    disconnected spurious blob elsewhere in the frame (noise/tissue
    misclassified as foreground). Skeletonizing the WHOLE mask gives that
    speck its own tiny skeleton and its own endpoint -- and since
    endpoint-nearest-to-orig_xy doesn't check connectivity, a close-but-fake
    endpoint on the spurious blob can beat the true (further-away-but-real)
    tip endpoint on the actual tool. Confirmed on 339542: pass-2 mask had a
    disconnected speck near (355,112), ~40px from the real tool, and its
    skeleton endpoint was selected as "refined" instead of the real tip."""
    x, y = int(round(xy[0])), int(round(xy[1]))
    mask_u8 = mask.astype(np.uint8)
    nc, lbl = cv2.connectedComponents(mask_u8)
    sizes = {i: int((lbl == i).sum()) for i in range(1, nc)}
    print(f"    component_containing: xy=({x},{y}) num_components={nc-1} sizes={sizes}")
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
        comp_id = lbl[y, x]
        print(f"      pixel at (x,y) belongs to component_id={comp_id} (0=background)")
        if comp_id != 0:
            return lbl == comp_id
    # orig_xy itself isn't foreground (e.g. sits just outside the mask) --
    # fall back to the largest component rather than nothing.
    if nc <= 1:
        print("      no components at all, returning mask unchanged")
        return mask
    largest_id = 1 + int(np.argmax(list(sizes.values())))
    print(f"      FALLBACK to largest component: id={largest_id} size={sizes[largest_id]}")
    return lbl == largest_id


def pca_tip_point(mask):
    """Find the tip as the extreme point of the mask along its own
    principal (long) axis, in the tip direction (smaller y).

    REPLACES skeleton-endpoint selection: skeleton endpoints require a
    pixel with exactly one skeleton neighbor, but a ROUNDED/BLUNT tip
    skeletonizes into a small loop or Y-junction with NO degree-1 pixel at
    all -- confirmed on 339542, where every skeleton pixel in the true tip
    region had degree >= 2, so the endpoint search silently found zero
    candidates there and picked a false endpoint ~365px away at the frame
    boundary instead. PCA doesn't depend on skeleton topology: it uses
    every foreground pixel, so it degrades gracefully for both pointed and
    blunt tips."""
    ys, xs = np.where(mask)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid)
    axis = vt[0]
    # Orient the axis so positive projection = toward the tip (smaller y).
    if axis[1] > 0:
        axis = -axis
    proj = (pts - centroid) @ axis
    tip_idx = int(np.argmax(proj))
    return pts[tip_idx]


def nearest_skeleton_point(pts, xy):
    pts_arr = np.array(list(pts))
    d = np.linalg.norm(pts_arr - np.array(xy), axis=1)
    return tuple(pts_arr[int(np.argmin(d))].tolist())


def _extrapolate_from_path(path, remaining_px):
    """The skeleton (i.e. the SAM2 mask) ran out before walk_px was reached
    on the tip side. Fit a straight line through the ENTIRE branch walked so
    far (least-squares direction via SVD over all path pixels, not just the
    last few) -- far less sensitive to local noise/wiggle near the end than
    a short-tail fit, since it uses the branch's overall trend. Continue in
    that direction for the remaining distance, past the mask's own
    boundary -- this is what lets the point land on the tooltip even when
    the mask itself stopped short at the shaft."""
    branch = np.array(path, dtype=np.float64)
    if len(branch) < 2:
        return np.array(path[-1], dtype=np.float64)
    centroid = branch.mean(axis=0)
    _, _, vt = np.linalg.svd(branch - centroid)
    direction = vt[0]
    # Orient the fitted direction to point the way the path was actually
    # travelling (from the first to the last point of the branch).
    travel = branch[-1] - branch[0]
    if np.dot(direction, travel) < 0:
        direction = -direction
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return np.array(path[-1], dtype=np.float64)
    direction = direction / norm
    return np.array(path[-1], dtype=np.float64) + direction * remaining_px


def walk_along_skeleton(neighbors, start, walk_px):
    """BFS outward from `start` along skeleton adjacency, in EACH available
    direction independently, accumulating Euclidean path length until
    walk_px is reached.

    Image y increases DOWNWARD, and the true tool tip consistently has a
    SMALLER y than the auto-detected inner-tube point (confirmed on the
    22-video manual-click validation earlier this session: dy =
    manual_y - auto_y was negative in 22/22 videos). So only the direction
    whose walked path ends at a smaller y than `start` is treated as
    "toward the tip" and gets extrapolated past a short skeleton branch (fit
    a line through the whole branch, continue past the mask boundary). The
    other direction (toward frame-entry / larger y) is left as whatever the
    skeleton actually reaches, no extrapolation -- extending there doesn't
    help find the tip and risks drifting off the real tool body.

    Returns a list of (point, path_len, extrapolated) tuples -- usually 1 or
    2 entries (more if the skeleton branches at `start`, e.g. a Y-junction).
    """
    results = []
    for first_step in neighbors[start]:
        path = [start, first_step]
        path_len = np.linalg.norm(np.array(first_step) - np.array(start))
        cur = first_step
        visited = {start, first_step}
        while path_len < walk_px:
            candidates = [n for n in neighbors[cur] if n not in visited]
            if not candidates:
                break
            nxt = min(candidates, key=lambda n: np.linalg.norm(np.array(n) - np.array(cur)))
            step_len = np.linalg.norm(np.array(nxt) - np.array(cur))
            path_len += step_len
            cur = nxt
            path.append(cur)
            visited.add(cur)

        is_tip_direction = cur[1] < start[1]  # smaller y = toward the tip
        print(f"    branch from {start}: {len(path)} pixels, path_len={path_len:.2f}, "
              f"end={cur}, tip_dir={is_tip_direction}, "
              f"needs_extrapolation={path_len < walk_px and is_tip_direction}")

        if path_len >= walk_px or not is_tip_direction:
            results.append((np.array(cur, dtype=np.float64), path_len, False))
        else:
            remaining = walk_px - path_len
            extrapolated_pt = _extrapolate_from_path(path, remaining)
            print(f"      extrapolating: branch={path}  remaining={remaining:.2f}  -> {extrapolated_pt}")
            results.append((extrapolated_pt, walk_px, True))
    return results


def sam2_pass(predictor, img_path, points_xy):
    """Run one SAM2 single-frame segmentation with the given positive
    points, return the boolean mask."""
    with tempfile.TemporaryDirectory(dir=SCRATCH) as td:
        Image.open(img_path).convert("RGB").save(
            os.path.join(td, "000000.jpg"), "JPEG", quality=95)
        state = predictor.init_state(video_path=td, offload_video_to_cpu=True, offload_state_to_cpu=True)
        predictor.add_new_points_or_box(
            state, frame_idx=0, obj_id=1,
            points=np.array(points_xy, dtype=np.float32),
            labels=np.ones(len(points_xy), dtype=np.int32))
        mask = None
        for oidx, oids, logits in predictor.propagate_in_video(state):
            mask = (logits[0, 0] > 0.0).cpu().numpy()
            break
        predictor.reset_state(state)
    return mask


def refine(video_dir, seed):
    files = sorted(glob.glob(os.path.join(video_dir, "*.png")))
    frame_idx = seed.get("frame", 0)
    img_path = files[frame_idx]
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    orig_xy = np.array(seed["points"][0], dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from sam2.build_sam import build_sam2_video_predictor
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_module('sam2', version_base='1.2'):
        predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CKPT, device=device)
    os.makedirs(SCRATCH, exist_ok=True)

    # --- Pass 1: single click ---
    mask1 = sam2_pass(predictor, img_path, [orig_xy])
    if device == "cuda":
        torch.cuda.empty_cache()
    if mask1 is None:
        return dict(status="no pass-1 mask", extra_points=[], mask1=None, mask2=None,
                    refined_xy=orig_xy)

    ok1, frac1, aspect1 = _mask_shape_ok(mask1, H, W)
    mask1_main = component_containing(mask1, orig_xy)
    skel1, pts1, nbrs1 = skeleton_graph(mask1_main)
    if len(pts1) == 0:
        return dict(status=f"pass-1 mask empty skeleton (frac={frac1:.4f} aspect={aspect1:.2f})",
                     extra_points=[], mask1=mask1, mask2=None, refined_xy=orig_xy)

    nearest = nearest_skeleton_point(pts1, orig_xy)
    walked = walk_along_skeleton(nbrs1, nearest, WALK_PX)
    extra_points = [np.array(w[0], dtype=np.float32) for w in walked]
    extra_extrapolated = [bool(w[2]) for w in walked]

    # --- Pass 2: original + walked points ---
    all_points = [orig_xy] + extra_points
    mask2 = sam2_pass(predictor, img_path, all_points)
    if device == "cuda":
        torch.cuda.empty_cache()

    result = dict(extra_points=extra_points, extra_extrapolated=extra_extrapolated,
                  mask1=mask1, mask2=mask2,
                  pass1_status=f"frac={frac1:.4f} aspect={aspect1:.2f} ok={ok1}")

    if mask2 is None:
        result.update(status="no pass-2 mask", refined_xy=orig_xy)
        return result

    ok2, frac2, aspect2 = _mask_shape_ok(mask2, H, W)
    if DEBUG_DUMP_DIR:
        cv2.imwrite(os.path.join(DEBUG_DUMP_DIR, f"{DEBUG_TAG}_mask2_raw.png"),
                    (mask2.astype(np.uint8) * 255))
    if not ok2:
        # mask2 itself is rejected -- clear it from the result so the
        # visualization falls back to mask1 (or nothing) instead of
        # drawing a mask we've already decided not to trust. Previously
        # mask2 stayed in `result` unconditionally (set at dict-creation
        # time above) and the viz always preferred it when present,
        # making a correctly-handled rejection look like a visible bug
        # (confirmed on 412793: an 18%-of-frame, aspect=0.00 mask got
        # rejected exactly as intended, but was still drawn).
        result["mask2"] = None
        result.update(status=f"pass-2 mask failed shape check (frac={frac2:.4f} aspect={aspect2:.2f})",
                       refined_xy=orig_xy)
        return result

    # Restrict to the component that actually contains the original click,
    # so a spurious disconnected blob elsewhere can't contribute a false
    # "nearer" endpoint (see component_containing's docstring).
    mask2_main = component_containing(mask2, orig_xy)
    if DEBUG_DUMP_DIR:
        cv2.imwrite(os.path.join(DEBUG_DUMP_DIR, f"{DEBUG_TAG}_mask2_main.png"),
                    (mask2_main.astype(np.uint8) * 255))
        print(f"    mask2 total px={int(mask2.sum())}  mask2_main px={int(mask2_main.sum())}")

    refined_xy = np.array(pca_tip_point(mask2_main), dtype=np.float32)
    print(f"    pca tip point (after component filter): {tuple(refined_xy.tolist())}")
    result.update(status=f"ok (pass2 frac={frac2:.4f} aspect={aspect2:.2f}, pca-tip)",
                   refined_xy=refined_xy)
    return result


def main():
    seed = json.load(open(seed_json_path))
    orig_xy = np.array(seed["points"][0], dtype=np.float32)
    frame_idx = seed.get("frame", 0)

    result = refine(video_dir, seed)
    ex_flags = result.get("extra_extrapolated", [False] * len(result["extra_points"]))
    ex_str = [(tuple(p.tolist()), "extrapolated" if f else "on-skeleton")
              for p, f in zip(result["extra_points"], ex_flags)]
    print(f"orig={tuple(orig_xy.tolist())}  "
          f"extra_points={ex_str}  "
          f"refined={tuple(result['refined_xy'].tolist())}  {result['status']}")

    files = sorted(glob.glob(os.path.join(video_dir, "*.png")))
    img = cv2.imread(files[frame_idx])
    vis = img.copy()
    mask_to_show = result.get("mask2") if result.get("mask2") is not None else result.get("mask1")
    if mask_to_show is not None:
        overlay = vis.copy()
        overlay[mask_to_show] = (0.35 * np.array([255, 140, 0]) + 0.65 * overlay[mask_to_show]).astype(np.uint8)
        vis = overlay

    op = (int(round(orig_xy[0])), int(round(orig_xy[1])))
    cv2.circle(vis, op, 8, (0, 165, 255), -1)   # orange = original click
    cv2.circle(vis, op, 8, (0, 0, 0), 2)
    cv2.putText(vis, "orig", (op[0] + 10, op[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

    extrapolated_flags = result.get("extra_extrapolated", [False] * len(result["extra_points"]))
    for k, ep in enumerate(result["extra_points"]):
        pt = (int(round(ep[0])), int(round(ep[1])))
        was_extrapolated = extrapolated_flags[k] if k < len(extrapolated_flags) else False
        color = (0, 200, 255) if was_extrapolated else (0, 255, 0)  # yellow-orange = extrapolated, green = on-skeleton
        cv2.circle(vis, pt, 8, color, -1)
        cv2.circle(vis, pt, 8, (0, 0, 0), 2)
        tag = f"walk{k}{'*' if was_extrapolated else ''}"
        cv2.putText(vis, tag, (pt[0] + 10, pt[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    rp = tuple(int(round(v)) for v in result["refined_xy"])
    cv2.circle(vis, rp, 8, (255, 0, 255), -1)   # magenta = final refined point
    cv2.circle(vis, rp, 8, (0, 0, 0), 2)
    cv2.putText(vis, "refined", (rp[0] + 10, rp[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)

    tag = os.path.basename(os.path.dirname(os.path.normpath(video_dir)))
    out_path = os.path.join(out_dir, f"{tag}_v2refine.png")
    cv2.imwrite(out_path, vis)
    print(f"saved -> {out_path}")

    if refined_seed_out:
        # refine() already falls back to refined_xy=orig_xy on every
        # non-"ok" status (rejected pass-1/pass-2 mask), so result["refined_xy"]
        # is always safe to use directly here.
        out_xy = result["refined_xy"]
        seed_H, seed_W = seed.get("H"), seed.get("W")
        if seed_H is None or seed_W is None:
            seed_H, seed_W = img.shape[:2]
        with open(refined_seed_out, "w") as f:
            json.dump({"points": [[float(out_xy[0]), float(out_xy[1])]],
                       "W": seed_W, "H": seed_H, "frame": frame_idx}, f)
        print(f"refined seed -> {refined_seed_out} ({'refined' if result['status'].startswith('ok') else 'fallback to orig'})")


if __name__ == "__main__":
    main()
