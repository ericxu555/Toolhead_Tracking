"""Automatically produce a tip-seed JSON (the file click_tooltip_multi.py
normally makes by hand) by running the collaborator's keypoint_rcnn.onnx
detector.

SELECTION RULE, v2 (revised again after the 110114 failure case -- see
Tool Tip Tracking Repos/test/sam2_refine_v2/20260613-014424-110114_v2refine.png):
the model's left/right CLASS label is unreliable, but it very often fires
TWICE on the same true tip -- once under each label, both landing within a
few pixels of each other. Plain "top-2 by confidence, pick rightmost" (the
v1 rule) breaks when both of the top-2 are two such same-tip guesses for
ONE tool, pushing the correct opposite-tool detection to 3rd+ and out of
consideration entirely -- confirmed on 110114: top-2 were left@0.970 and
left@0.929 (1.3px apart, same tip), while the correct right@0.837 (the
other tool's real tip) never got compared.

v2: pair up detections whose left/right labels differ AND whose positions
are within PAIR_DIST_PX of each other -- these are (almost certainly) two
guesses for the same physical tip. Take the two highest-combined-confidence
pairs (one per tool), represent each pair by its higher-confidence member's
own point, and pick whichever of those two is further right (x) -- same
rightmost-by-position logic as v1, just applied to pair representatives
instead of raw top-2 detections. Falls back to the v1 rule (top-2 by
confidence, pick rightmost, no pairing) when fewer than 2 valid pairs exist.

Preprocessing/postprocessing mirrors tool_tip_tracking-main's
tracker.py::run_inference() (crop to square, resize to 256x256, BGR->RGB,
normalize, rescale keypoints back to original resolution).

Scans forward from frame 0 until the chosen tip's own confidence clears
CONFIDENCE_THRESHOLD, since the tool is not always visible/confident on the
very first frame. Writes a seed JSON compatible with tapnextpp_hybrid.py's
{"points": [[x,y]], "W", "H", "frame"} schema, using its "frame" field to
seed at whichever frame actually succeeded.

Usage: auto_seed_tip.py <model.onnx> <video_dir> <out_seed.json> [max_frames=30]
"""
import sys, os, glob, json
import numpy as np
import cv2
import onnxruntime as ort

model_path = sys.argv[1]
video_dir = sys.argv[2]
out_json = sys.argv[3]
MAX_FRAMES = int(sys.argv[4]) if len(sys.argv) > 4 else 30

CONFIDENCE_THRESHOLD = 0.5
PAIR_DIST_PX = 5.0  # max left/right distance to count as "same physical tip"

sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def run_inference(image_bgr):
    """Same crop/resize/normalize recipe as tool_tip_tracking-main's
    tracker.py::run_inference() -- center-crop to square, resize to the
    model's fixed 256x256 input, rescale keypoints back to original coords."""
    H, W = image_bgr.shape[:2]
    side = min(H, W)
    crop_y = (H - side) // 2
    crop_x = (W - side) // 2
    cropped = image_bgr[crop_y:crop_y + side, crop_x:crop_x + side]

    resized = cv2.resize(cropped, (256, 256))
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    scaled = resized_rgb.astype(np.float32) / 255.0
    chw = np.rollaxis(scaled, 2, 0)
    inp = np.expand_dims(chw, 0)

    inputs = {sess.get_inputs()[0].name: inp}
    outputs = sess.run(None, inputs)
    labels, scores, keypoints = outputs[1], outputs[2], outputs[3]

    scale = side / 256.0
    keypoints_xy = keypoints[:, 0, 0:2] * scale
    keypoints_xy[:, 0] += crop_x
    keypoints_xy[:, 1] += crop_y

    return labels, scores, keypoints_xy


def _rightmost_of_top2_fallback(scores, keypoints_xy):
    """v1 rule: top-2 by confidence, pick rightmost by x. Used when fewer
    than 2 left/right pairs are found."""
    order = np.argsort(-scores)[:2]
    top_kp = keypoints_xy[order]
    top_scores = scores[order]
    rightmost_i = int(np.argmax(top_kp[:, 0]))
    return float(top_scores[rightmost_i]), top_kp[rightmost_i]


def _cluster_detections(scores, keypoints_xy, radius=PAIR_DIST_PX):
    """Group ALL detections (regardless of label) into clusters of mutually
    close points (union-find on the "within radius" graph), representing
    one physical tip location each. Two detections `radius` apart are
    merged transitively -- i.e. a chain of nearby points all end up in one
    cluster, not just direct pairs.

    Returns a list of index-lists, one per cluster."""
    n = len(scores)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(keypoints_xy[i] - keypoints_xy[j]) <= radius:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


STRONG_PARTNER_CONFIDENCE = 0.8  # partner confidence needed to override rightmost-x


def find_tip(labels, scores, keypoints_xy):
    """v9 rule: fixes a gap in v8 found on 828449 -- v8 picked its top-2
    CANDIDATES by raw per-detection confidence, which breaks when the
    model fires MULTIPLE overlapping guesses for the SAME physical tip
    (e.g. left@0.996 and right@0.617 both ~1px apart, on one tool): both
    of those crowd into the "top-2 by confidence" slots, and the correct
    OTHER tool's detection (right@0.421, rank 3) never enters the
    comparison at all. No downstream tiebreak (rightmost-x, strong-partner
    override) can recover from the candidate pool itself being wrong.

    Fix: first CLUSTER all detections into physical-tip groups (mutually
    within PAIR_DIST_PX, regardless of label -- see _cluster_detections),
    so left@0.996+right@0.617 collapse into one cluster and no longer
    consume two of the "top-2" slots. Rank clusters by their best member's
    confidence, take the top-2 CLUSTERS (guaranteeing two distinct physical
    locations are compared, not two guesses about one). Represent each
    cluster by its highest-confidence member.

    Everything downstream is identical to v8: default to rightmost-x
    between the two cluster representatives; override to the
    higher-confidence side only if one candidate has an opposite-label
    partner (within its own cluster) whose own confidence exceeds
    STRONG_PARTNER_CONFIDENCE. See v8's docstring (preserved in git
    history) for why comparing partner confidences directly, or penalizing
    internal pair disagreement, were both tried and reverted -- that
    reasoning is unchanged here, only the candidate pool changed.

    Verified against all 4 known conflict cases (110114, 144158, 230193,
    232468): clustering produces the IDENTICAL top-2 points for each of
    them as v8's raw top-2 did, since none of those cases had a same-tip
    duplicate crowding out the real second tool -- this is a strict
    superset fix, not a behavior change for previously-passing cases.

    Falls back to the v1 rule (top-2 by confidence, pick rightmost) if
    fewer than 2 clusters exist at all."""
    if len(scores) == 0:
        return None
    if len(scores) == 1:
        return float(scores[0]), keypoints_xy[0]

    def best_partner_score(i, members):
        this_label = labels[i]
        best = None
        for j in members:
            if j == i or labels[j] == this_label:
                continue
            if best is None or scores[j] > best:
                best = scores[j]
        return best

    clusters = _cluster_detections(scores, keypoints_xy)
    if len(clusters) < 2:
        return _rightmost_of_top2_fallback(scores, keypoints_xy)

    ranked = sorted(clusters, key=lambda members: -max(scores[i] for i in members))
    top2 = ranked[:2]

    candidates = []
    for members in top2:
        best_i = max(members, key=lambda i: scores[i])
        candidates.append((keypoints_xy[best_i], float(scores[best_i]),
                            best_partner_score(best_i, members)))

    strong = [c for c in candidates if c[2] is not None and c[2] > STRONG_PARTNER_CONFIDENCE]
    if len(strong) == 1:
        winner = strong[0]
    else:
        winner = max(candidates, key=lambda c: c[0][0])  # rightmost-x (default, or tie)
    return winner[1], winner[0]


VOTE_CLUSTER_RADIUS = 15.0  # across-frame clustering radius (looser than within-frame PAIR_DIST_PX,
                             # since the true tip can drift a few px frame to frame)
MIN_VOTES = 5                # M-of-N window size (matches classic radar/automotive track-confirmation
                             # defaults of N=4-5, see iteration log research notes)
PLURALITY_FRACTION = 0.5     # winning cluster must exceed this fraction of the accumulated window


def _write_seed(xy, frame_idx, W, H, out_json):
    seed = {"points": [[float(xy[0]), float(xy[1])]], "W": W, "H": H, "frame": frame_idx}
    with open(out_json, "w") as f:
        json.dump(seed, f)


def _plurality_vote(buffer):
    """buffer: list of (frame_idx, score, xy) sub-threshold observations.
    Cluster their xy positions (radius=VOTE_CLUSTER_RADIUS) and check
    whether the largest cluster holds a strict majority (> PLURALITY_FRACTION)
    of the buffer. Returns (frame_idx, score, xy) of the winning cluster's
    best-scoring member, or None if no cluster reaches plurality yet.

    v10: added to handle 828449, where the correct tool's detections never
    individually clear CONFIDENCE_THRESHOLD across many consecutive frames
    (range 0.09-0.42) while an unrelated wrong-tool detection spikes above
    threshold once (0.991) -- the old "first frame >= threshold" rule locked
    onto that single spike. A consistently-recurring LOW-confidence location
    is stronger evidence than one single HIGH-confidence outlier -- this is
    the same principle ByteTrack uses for low-score detection recovery, and
    matches classic M-of-N radar track-confirmation logic. See iteration log
    for full research citations and hand-verification against 828449."""
    xy_arr = np.array([b[2] for b in buffer])
    n = len(buffer)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a in range(n):
        for b in range(a + 1, n):
            if np.linalg.norm(xy_arr[a] - xy_arr[b]) <= VOTE_CLUSTER_RADIUS:
                union(a, b)

    groups = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)

    best_group = max(groups.values(), key=len)
    if len(best_group) / n <= PLURALITY_FRACTION:
        return None

    best_member = max(best_group, key=lambda idx: buffer[idx][1])  # highest score in winning cluster
    return buffer[best_member]


def main():
    files = sorted(glob.glob(os.path.join(video_dir, "*.png")))
    if not files:
        raise SystemExit(f"No frames found in {video_dir}")

    first = cv2.imread(files[0])
    H, W = first.shape[:2]

    sub_threshold_buffer = []  # (frame_idx, score, xy) for frames that didn't clear CONFIDENCE_THRESHOLD

    for i, fp in enumerate(files[:MAX_FRAMES]):
        img = cv2.imread(fp)
        if img is None:
            continue
        labels, scores, keypoints_xy = run_inference(img)
        result = find_tip(labels, scores, keypoints_xy)
        if result is None:
            print(f"  frame {i}: no detections")
            continue
        score, xy = result
        print(f"  frame {i}: score={score:.3f} xy=({xy[0]:.1f},{xy[1]:.1f})")
        if score >= CONFIDENCE_THRESHOLD:
            _write_seed(xy, i, W, H, out_json)
            print(f"AUTO-SEEDED at frame {i} (score={score:.3f}) -> {out_json}")
            return

        sub_threshold_buffer.append((i, score, xy))
        if len(sub_threshold_buffer) >= MIN_VOTES:
            vote = _plurality_vote(sub_threshold_buffer)
            if vote is not None:
                v_frame, v_score, v_xy = vote
                _write_seed(v_xy, v_frame, W, H, out_json)
                print(f"AUTO-SEEDED by plurality vote across frames "
                      f"{sub_threshold_buffer[0][0]}-{i} "
                      f"(best member: frame {v_frame}, score={v_score:.3f}) -> {out_json}")
                return

    raise SystemExit(
        f"No detection >= {CONFIDENCE_THRESHOLD} AND no plurality-vote cluster "
        f"(>{PLURALITY_FRACTION*100:.0f}% agreement across {MIN_VOTES}+ frames) "
        f"in the first {min(MAX_FRAMES, len(files))} frames -- fall back to manual click."
    )


if __name__ == "__main__":
    main()
