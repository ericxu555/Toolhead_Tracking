"""Auto-seed the tool shaft point for a new video, without a human click.

Classical color-based tool detection was tested earlier and FAILED (tools take
on warm endoscope lighting, blend with char) -- so this does NOT try to detect
the tool's pixels directly. Instead: the two human-validated seeds we have
(exp1, Full_exp4) both land in a narrow, consistent region (x~280-340,
y~140-145), consistent with a fixed camera/endoscope rig where the tool always
enters frame in roughly the same place. This tries a small grid of candidate
points around that region, and for EACH one runs a cheap 10-frame SAM2
propagation to VERIFY the result actually looks tool-shaped (elongated,
reasonable area, not empty, not a huge blob) before trusting it -- inference
with a check, not a blind guess.

Returns the best verified (x,y) seed, or None if nothing passes (video should
be skipped and flagged, not run on a bad guess).
"""
import os, glob, tempfile
import numpy as np, cv2, torch
from PIL import Image

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "checkpoints_sam2", "sam2.1_hiera_small.pt")
# Scratch dir for staging frames into SAM2's video predictor. Defaults to the
# OS temp dir; override SAM2_SCRATCH to use a faster local disk.
SCRATCH = os.environ.get("SAM2_SCRATCH", os.path.join(tempfile.gettempdir(),
                                                       "sam2_scratch"))

CANDIDATES = [
    (310, 142), (285, 142), (336, 141), (310, 120), (310, 165),
    (280, 120), (340, 120), (280, 170), (340, 170),
]
VERIFY_FRAMES = 10
MIN_AREA_FRAC = 0.003   # tool mask must cover at least this fraction of frame
MAX_AREA_FRAC = 0.15    # ...and no more than this (else it's grabbed tissue)
MIN_ASPECT = 1.8        # elongated: major/minor axis ratio via bounding box


def _mask_shape_ok(mask, H, W):
    area = int(mask.sum())
    frac = area / (H * W)
    if frac < MIN_AREA_FRAC or frac > MAX_AREA_FRAC:
        return False, frac, 0.0
    ys, xs = np.where(mask)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    aspect = max(h, w) / max(1, min(h, w))
    return aspect >= MIN_ASPECT, frac, aspect


def find_tool_seed(video_dir, predictor=None, device="cuda"):
    files = sorted(glob.glob(os.path.join(video_dir, "*.png")))[:VERIFY_FRAMES]
    if len(files) < 3:
        return None, "too few frames"
    first = cv2.imread(files[0]); H, W = first.shape[:2]

    own_predictor = predictor is None
    if own_predictor:
        from sam2.build_sam import build_sam2_video_predictor
        from hydra import initialize_config_module
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()
        with initialize_config_module('sam2', version_base='1.2'):
            predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CKPT, device=device)

    os.makedirs(SCRATCH, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(dir=SCRATCH) as td:
        for j, fp in enumerate(files):
            Image.open(fp).convert("RGB").save(os.path.join(td, f"{j:06d}.jpg"), "JPEG", quality=95)

        for cx, cy in CANDIDATES:
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            state = predictor.init_state(video_path=td, offload_video_to_cpu=True, offload_state_to_cpu=True)
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=1,
                points=np.array([[cx, cy]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32))
            last_mask = None
            for oidx, oids, logits in predictor.propagate_in_video(state):
                last_mask = (logits[0, 0] > 0.0).cpu().numpy()
            predictor.reset_state(state)
            if device == "cuda":
                torch.cuda.empty_cache()
            if last_mask is None:
                continue
            ok, frac, aspect = _mask_shape_ok(last_mask, H, W)
            results.append((ok, aspect, frac, (cx, cy)))
            print(f"    candidate ({cx},{cy}): area_frac={frac:.4f} aspect={aspect:.2f} "
                  f"{'PASS' if ok else 'fail'}")

    passing = [r for r in results if r[0]]
    if not passing:
        return None, f"no candidate passed shape check ({len(results)} tried)"
    # Prefer highest aspect ratio among passing (most elongated = most tool-like).
    passing.sort(key=lambda r: -r[1])
    best = passing[0]
    return best[3], f"aspect={best[1]:.2f} frac={best[2]:.4f}"


if __name__ == "__main__":
    import sys
    vd = sys.argv[1]
    seed, info = find_tool_seed(vd)
    print(f"RESULT: {vd} -> {seed}  ({info})")
