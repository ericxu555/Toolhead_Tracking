"""Fully autonomous toolhead-following pipeline: auto-detects the tool tip
(no manual click) via the collaborator's keypoint_rcnn.onnx (v10 -- see
edge_anchored_tracking/_debug/autoseed_iteration_log.md for that algorithm's
design/validation history), refines it with SAM2 two-pass centerline
refinement, then runs the existing tip-tracking + cutline-sweep pipeline
exactly as before.

Does NOT modify tapnextpp_hybrid.py or cutline_sweep_mask_walk.py -- both
are invoked as subprocesses, unchanged, so the manually-seeded workflow
those scripts support keeps working exactly as it does today. This script
only replaces the click_tooltip_multi.py step (now a 2-stage auto-seed +
refine, in place of the earlier 1-stage auto-seed).

Tip selection (auto_seed_tip.py, v10): within-frame, detections are
clustered by physical location (keypoint_rcnn.onnx's left/right CLASS label
is unreliable on our footage) and the two clusters representing the two
tools are compared, defaulting to rightmost-x with a strong-partner
override. Across-frame, the first frame whose winner clears
CONFIDENCE_THRESHOLD is used immediately; if no frame clears it, a
plurality vote across a small window of sub-threshold frames can still
commit to a seed once one location dominates -- prevents a single
high-confidence wrong-tool spike from overriding a consistently-recurring
correct (but individually low-confidence) detection. Exhaustively validated
against the full 204-video G:/Ablation corpus; see the iteration log.

Tip refinement (refine_tip_with_sam2_v2.py): corrects a KNOWN SYSTEMATIC
OFFSET in the raw detection -- keypoint_rcnn.onnx's "tip" is the end of the
tool's inner tube, not the cautery/spatula tip itself, per
tool_tip_tracking-main's README, producing a ~25-32px offset (mostly in y,
deeper into frame) versus a manual click on the true tip. The refiner seeds
SAM2 with the raw point, skeletonizes the resulting mask, walks along the
skeleton toward the tip direction (extrapolating past the mask boundary if
needed) to get 1-2 more on-tool points, re-seeds SAM2 with all points, and
finds the true tip via PCA projection onto the refined mask's principal
axis. Falls back to the raw (unrefined) point if either SAM2 pass's mask
fails a shape sanity check (see refine()'s status field) -- never worse
than the unrefined baseline, only better when the refinement is trustworthy.

Usage: run_autonomous_toolhead.py <video_dir> <out_root_dir> [tag]
                                  [--no-refine] [--manual]
  video_dir      folder of frameNNNNNN.png
  out_root_dir   created if absent
  tag            optional label used in intermediate file/folder names (default: derived
                 from out_root_dir's basename)
  --no-refine    skip the SAM2 stage and track straight from the raw
                 detection. SAM2 need not be installed. The seed keeps the
                 inner-tube offset described above.
  --manual       skip auto-detection and click the tip by hand instead.

Produces, inside out_root_dir:
  auto_seed.json               the auto-detected seed
  refined_seed.json            the SAM2-refined seed fed to tracking (omitted under --no-refine)
  sam2_refine_viz/*_v2refine.png   refinement debug visualization (omitted under --no-refine)
  tip_track/tip_track.csv      TAPNext++ output
  cutline_sweep_walk.mp4       full mask overlay (from cutline_sweep_mask_walk.py)
  cutline_only.mp4             trajectory-only visualization (from render_cutline_only.py)
"""
import sys, os, glob, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
PYTHON = sys.executable

# The tool-tip detector weights are not in this repo (236 MB, over GitHub's
# limit) -- download them into checkpoints_keypoint/, or point KEYPOINT_MODEL
# elsewhere. Same override convention as SPATIAL_MODEL/SPATIAL_META in
# cutline_sweep_mask_walk.py. See "Step 4. Download the model weights" in
# README.md for the download link.
KEYPOINT_MODEL = os.environ.get(
    "KEYPOINT_MODEL",
    os.path.join(REPO_ROOT, "checkpoints_keypoint", "keypoint_rcnn.onnx"))

FLAGS = ("--manual", "--no-refine")
args = [a for a in sys.argv[1:] if a not in FLAGS]
MANUAL = "--manual" in sys.argv
NO_REFINE = "--no-refine" in sys.argv

if len(args) < 2:
    raise SystemExit(
        "Usage: run_autonomous_toolhead.py <video_dir> <out_root_dir> [tag]\n"
        "                                  [--no-refine] [--manual]\n"
        "  --no-refine  detect the tip automatically but skip the SAM2\n"
        "               refinement stage. SAM2 does not need to be installed.\n"
        "               The seed keeps the detector's inner-tube offset.\n"
        "  --manual     skip auto-detection; open the click GUI to seed the tip\n"
        "               (use when auto-seeding picks the wrong tool)")

video_dir = args[0]
out_root = args[1]
tag = args[2] if len(args) > 2 else os.path.basename(os.path.normpath(out_root))

os.makedirs(out_root, exist_ok=True)


def run_step(name, cmd):
    """Run one pipeline stage, announcing it and reporting how long it took.

    Stages take minutes, so the banner and the elapsed time are what tell a
    user the run is progressing rather than hung.
    """
    print(f"\n{'=' * 62}")
    print(f"  {name}")
    print(f"{'=' * 62}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(
            f"\nFAILED: {name}\n"
            f"  exit code {result.returncode}. The stage's own error is above.\n"
            f"  Stopping; no further stages will run.")
    print(f"  [done in {time.time() - t0:.0f}s]", flush=True)


def main():
    # Confirm the inputs before committing to a run that takes minutes, so a
    # wrong folder or a missing weight file surfaces immediately.
    n_frames = len(glob.glob(os.path.join(video_dir, "*.png")))
    if MANUAL:
        mode = "manual click (SAM2 not used)"
    elif NO_REFINE:
        mode = "automatic, SAM2 refinement skipped (--no-refine)"
    else:
        mode = "automatic with SAM2 refinement"
    print(f"\n{'=' * 62}")
    print("  Cutline-sweep resection mask pipeline")
    print(f"{'=' * 62}")
    print(f"  input frames : {video_dir}  ({n_frames} PNG files)")
    print(f"  output       : {out_root}")
    print(f"  mode         : {mode}")
    print("  Each stage below takes minutes; elapsed time is printed as it goes.",
          flush=True)
    if n_frames == 0:
        raise SystemExit(
            f"\nNo PNG frames found in {video_dir}\n"
            f"  The pipeline expects a folder of numbered PNG frames from one\n"
            f"  episode. See 'Input data' in README.md.")

    if MANUAL:
        # Manual fallback: the click GUI gives a true-tip click directly, so
        # the SAM2 refinement step (which exists only to correct the
        # DETECTOR's inner-tube offset) is skipped -- running it on an
        # already-correct click would move a good point for no reason.
        seed_json = os.path.join(out_root, "manual_seed.json")
        frames = sorted(glob.glob(os.path.join(video_dir, "*.png")))
        if not frames:
            raise SystemExit(f"No PNG frames found in {video_dir}")
        run_step("1/3 Manual tip seed (click GUI)", [
            PYTHON, os.path.join(REPO_ROOT, "click_tooltip_multi.py"),
            frames[0], seed_json,
        ])
        seed_for_tracking = seed_json
        n_steps = 3
    else:
        if not os.path.isfile(KEYPOINT_MODEL):
            raise SystemExit(
                f"keypoint_rcnn.onnx not found at:\n  {KEYPOINT_MODEL}\n"
                f"Set KEYPOINT_MODEL to its location, e.g.\n"
                f"  export KEYPOINT_MODEL=/path/to/keypoint_rcnn.onnx\n"
                f"See 'Step 4. Download the model weights' in README.md.\n"
                f"Or re-run with --manual to click the tip by hand instead.")

        n_steps = 3 if NO_REFINE else 4
        seed_json = os.path.join(out_root, "auto_seed.json")
        run_step(f"1/{n_steps} Auto-seed tip (keypoint_rcnn.onnx, v10 cluster+plurality-vote)", [
            PYTHON, os.path.join(HERE, "auto_seed_tip.py"),
            KEYPOINT_MODEL, video_dir, seed_json,
        ])

        if NO_REFINE:
            # Baseline path: track straight from the raw detection. SAM2 is
            # never imported, so it does not need to be installed. The seed
            # keeps the detector's inner-tube offset described below.
            seed_for_tracking = seed_json
        else:
            refine_viz_dir = os.path.join(out_root, "sam2_refine_viz")
            refined_seed_json = os.path.join(out_root, "refined_seed.json")
            run_step("2/4 Refine seed with SAM2 two-pass centerline walk", [
                PYTHON, os.path.join(HERE, "refine_tip_with_sam2_v2.py"),
                video_dir, seed_json, refine_viz_dir, "5", refined_seed_json,
            ])
            seed_for_tracking = refined_seed_json

    track_dir = os.path.join(out_root, "tip_track")
    run_step(f"{n_steps - 1}/{n_steps} Track tip across video (TAPNext++, unmodified)", [
        PYTHON, os.path.join(REPO_ROOT, "tapnextpp_hybrid.py"),
        video_dir, seed_for_tracking, track_dir,
    ])
    tip_csv = os.path.join(track_dir, "tip_track.csv")

    run_step(f"{n_steps}/{n_steps}a Cutline-sweep mask pipeline (unmodified)", [
        PYTHON, os.path.join(HERE, "cutline_sweep_mask_walk.py"),
        video_dir, tip_csv, out_root,
        "right", "15", "0.5", "0",
    ])

    run_step(f"{n_steps}/{n_steps}b Cutline-only visualization (unmodified)", [
        PYTHON, os.path.join(HERE, "render_cutline_only.py"),
        video_dir, tip_csv, out_root, "15",
    ])

    if MANUAL:
        mode = "manual click"
    elif NO_REFINE:
        mode = "automatic, no SAM2 refinement"
    else:
        mode = "fully autonomous, no manual click"
    print(f"\n{'=' * 62}")
    print(f"  FINISHED ({mode})")
    print(f"{'=' * 62}")
    print(f"  Results in: {out_root}\n")
    print("  Look at these first:")
    print(f"    cutline_only.mp4        tracked tip and its path, no mask fill")
    print(f"                            (check the tip is on the correct tool)")
    print(f"    cutline_sweep_walk.mp4  resected region shaded over the video")
    print("\n  Intermediate files:")
    if MANUAL:
        print(f"    manual_seed.json        the point you clicked")
    elif NO_REFINE:
        print(f"    auto_seed.json          detected tip")
    else:
        print(f"    auto_seed.json          raw detected tip")
        print(f"    refined_seed.json       SAM2-refined tip used for tracking")
    print(f"    tip_track/tip_track.csv per-frame tip positions")
    print("\n  If the tip was seeded on the wrong tool, re-run with --manual.",
          flush=True)


if __name__ == "__main__":
    main()
