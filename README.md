# Cutline-Sweep Resection Mask Pipeline

Surgical tumor resection tracking from 2D endoscope video. The pipeline
combines a per-frame binary tumor segmentation model with a tracked tool-tip
position to build a cumulative resected-region mask, using a geometric
"cutline-sweep" construction. Neither model carries temporal state; the
accumulation is handled entirely by the geometry.

## Pipeline overview

1. **Tumor boundary segmentation.** A U-Net with a ResNet-34 encoder segments
   the entire tumor in every frame. Trained with Focal, Tversky, and boundary
   loss.
2. **Tool-tip seeding.** A Keypoint R-CNN detector locates the tip
   automatically, and SAM2 then refines that point to the true distal cautery
   tip. Both steps are automatic; a manual click remains available as a
   fallback.
3. **Tool-tip tracking.** TAPNext++, a transformer point tracker, follows the
   seeded tip across the video.
4. **Cutline-sweep mask construction.** The segmentation and tracking outputs
   are combined into a closed polygon each frame, bounded by the cutline, the
   tumor boundary, and a perpendicular bottom edge. Polygons accumulate into a
   monotonic, self-correcting resection mask.

## Setup

Setup comes in two parts. **Basic setup** gets the pipeline running with
automatic tip detection. **SAM2 tip refinement** then adds one install and one
checkpoint, and is recommended: it corrects a systematic offset in the
detected tip and measurably improves the resulting mask. Both parts are
fully automatic.

If you want to get a first result quickly, or cannot install SAM2 yet, the
basic setup alone works with the `--no-refine` flag.

Run every command from the repository root with the virtual environment
activated. A CUDA-capable GPU is recommended; the pipeline falls back to CPU
automatically but runs considerably slower.

## Basic setup

### Step 1. Clone the repository and create an environment

```bash
git clone https://github.com/ericxu555/Toolhead_Tracking.git
cd Toolhead_Tracking

python -m venv venv
# Windows:       venv\Scripts\activate
# macOS/Linux:   source venv/bin/activate
```

### Step 2. Install PyTorch

Install PyTorch first and separately, because the CUDA builds are not
distributed on PyPI and will not resolve from a plain `pip install`. Use the
selector at https://pytorch.org/get-started/locally/ to get the command for
your CUDA version. For CUDA 12.8, which this pipeline was developed against:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

For CPU only:

```bash
pip install torch torchvision
```

Verify before continuing:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Step 3. Install the remaining Python dependencies

```bash
pip install -r requirements.txt
```

Then verify the two pieces that are easy to get silently wrong:

```bash
python -c "from tapnet.tapnextpp.votsp2026.model import TAPNextPP; print('tapnext ok')"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

The first line must print `tapnext ok`. If it raises
`ModuleNotFoundError: No module named 'einops'`, tapnet imports einops without
declaring it as a dependency, so pip did not install it. This is a hard
blocker, because tracking cannot start:

```bash
pip install einops
```

The second line lists the ONNX Runtime execution providers. On a CUDA machine
`CUDAExecutionProvider` should appear. If you only see
`['AzureExecutionProvider', 'CPUExecutionProvider']`, the CPU build is
installed and the tool-tip detector will run on CPU.

That is usually fine. The detector runs only during seeding, on at most a few
dozen frames at the start of each episode, not per frame. Nothing else in the
pipeline uses ONNX Runtime.

If you do want it on the GPU, note that `onnxruntime-gpu` builds are tied to a
specific CUDA major version, and a mismatch with your PyTorch build fails with
`libcublasLt.so.NN: cannot open shared object file`. Check what PyTorch is
using and install a matching build rather than the newest one:

```bash
python -c "import torch; print(torch.version.cuda)"
# CUDA 12.x, e.g. torch 2.11.0+cu128:
pip uninstall -y onnxruntime && pip install "onnxruntime-gpu==1.22.0"
```

Do not install a new CUDA toolkit to satisfy ONNX Runtime. PyTorch does the
heavy lifting here, and changing the system CUDA version would mean rebuilding
that whole stack to speed up one step that runs once per cut.

### Step 4. Download the model weights

Two sets of weights are required for the basic setup. Neither is included in
this repository because both exceed GitHub's file size limit.

**4a. Tumor segmentation checkpoint.** Download both files from the link below
and place them in `checkpoints_tumor_binary10/`:

> https://drive.google.com/drive/folders/18w_LTyrIgShmxdWPbZUZF8BunnrfqyQo?usp=sharing

```
checkpoints_tumor_binary10/
  best_model_epoch_swa.pth
  best_model_epoch_swa_metadata.json
```

**4b. Keypoint R-CNN tool-tip detector.** Download `keypoint_rcnn.onnx`
(approximately 236 MB) and place it in `checkpoints_keypoint/`:

> https://drive.google.com/drive/folders/18rmqv1qCeWYj1ps9pURw5bngLHZDsngY

TAPNext++'s checkpoint (`tapnextpp_512.ckpt`) is not listed here because it
downloads itself to `~/.cache/tapnextpp/` on first run.

After this step:

```
checkpoints_tumor_binary10/  best_model_epoch_swa.pth, best_model_epoch_swa_metadata.json
checkpoints_keypoint/        keypoint_rcnn.onnx
```

### Step 5. Run the pipeline

```bash
python edge_anchored_tracking/run_autonomous_toolhead.py \
    <video_frame_dir> <out_root_dir> --no-refine
```

`--no-refine` skips the SAM2 stage. The tip is still detected automatically;
only the refinement step is omitted. See Usage below for the arguments and
outputs.

This is a working pipeline. For the recommended configuration, continue to
the next section and add SAM2 refinement.

## SAM2 tip refinement (recommended)

The tip detector reports the end of the tool's inner tube rather than the
distal cautery tip, a systematic offset of roughly 25 to 32 pixels,
predominantly in y. The refinement stage corrects this by segmenting the tool
with SAM2 and locating the true tip along its centerline. If the segmentation
fails a shape check, it keeps the original detection, so the result is never
worse than the basic setup.

Complete the basic setup first, then follow the three steps below.

### Step 6. Install SAM2

SAM2 is not published on PyPI and must be installed from source. It is a
single command and requires no further configuration; the model configs ship
inside the package.

```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

Verify:

```bash
python -c "import sam2; print('sam2 OK')"
```

On some systems SAM2 prints a warning at runtime about being unable to import
`_C`, its optional compiled extension. This is expected and harmless.

### Step 7. Download the SAM2 checkpoint

Download `sam2.1_hiera_small.pt` (approximately 184 MB) into
`checkpoints_sam2/`:

```bash
mkdir -p checkpoints_sam2
curl -L -o checkpoints_sam2/sam2.1_hiera_small.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

### Step 8. Run with refinement

Drop the `--no-refine` flag:

```bash
python edge_anchored_tracking/run_autonomous_toolhead.py \
    <video_frame_dir> <out_root_dir>
```

The run then reports four stages instead of three, and writes both
`auto_seed.json` (the raw detection) and `refined_seed.json` (the corrected
point actually used for tracking).

## Configuration

If you keep the weights outside the repository, point at them with environment
variables instead of editing any script. All four are optional and have
working defaults.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SPATIAL_MODEL` | `checkpoints_tumor_binary10/best_model_epoch_swa.pth` | Segmentation weights |
| `SPATIAL_META` | `checkpoints_tumor_binary10/best_model_epoch_swa_metadata.json` | Segmentation metadata |
| `KEYPOINT_MODEL` | `checkpoints_keypoint/keypoint_rcnn.onnx` | Tool-tip detector |
| `SAM2_SCRATCH` | System temp directory | Scratch space for SAM2 frame staging (refinement only) |

```bash
export SPATIAL_MODEL=/path/to/best_model_epoch_swa.pth
export SPATIAL_META=/path/to/best_model_epoch_swa_metadata.json
export KEYPOINT_MODEL=/path/to/keypoint_rcnn.onnx
```

Segmentation model summary, from the metadata file: U-Net with ResNet-34
encoder, 512x512 input, trained with Focal (gamma=2.0), Tversky (alpha=0.3,
beta=0.7), and Boundary (sigma=3.0) loss, SWA from epoch 50, validation
Dice 0.976.

## Troubleshooting

**`No matching distribution found for torch==...+cu128`.** The CUDA builds of
PyTorch are not on PyPI. Install PyTorch separately first, as described in
Step 2, then run `pip install -r requirements.txt`.

**`ModuleNotFoundError: No module named 'einops'`.** Raised from inside tapnet
when TAPNext++ is imported. tapnet uses einops but does not declare it, so pip
does not always install it. Tracking cannot run without it:

```bash
pip install einops
```

**`libcublasLt.so.13: cannot open shared object file`, or `Failed to create
CUDAExecutionProvider`.** The installed `onnxruntime-gpu` was built for a
different CUDA major version than PyTorch. The node falls back to CPU and
still runs. To fix it, install a build matching `torch.version.cuda` (for
CUDA 12.x, `pip install "onnxruntime-gpu==1.22.0"`), or simply use the CPU
build with `pip install onnxruntime`. Do not upgrade the system CUDA toolkit
for this; the detector runs only during seeding.

**`ModuleNotFoundError: No module named 'sam2'`.** SAM2 is not on PyPI and is
not installed by `requirements.txt`. See "SAM2 tip refinement
(recommended)", or run with `--no-refine` to skip it.

**`cannot import name '_C' from 'sam2'`.** A warning, not an error. SAM2's
optional compiled extension is unavailable, and the pipeline runs correctly
without it. No action needed.

**`keypoint_rcnn.onnx not found`.** The detector weights are missing. Complete
Step 4b, or set `KEYPOINT_MODEL` to the file's location. To proceed without
them, run with `--manual` and click the tip by hand.

**Automatic seeding selects the wrong tool.** Re-run that video with
`--manual`. The rest of the pipeline is unchanged.

**`torch.cuda.is_available()` returns `False`.** The pipeline will run on CPU
but considerably slower. Reinstall PyTorch using the index URL matching your
CUDA version, as described in Step 2.

## Usage

### Input data

The pipeline takes **a folder containing every frame of one resection
episode**, as individual PNG files. It does not accept a video file, and it
does not work from a single image: the mask is built up frame by frame as the
tool moves, so the whole sequence is required.

```
my_case/
  frame000001.png
  frame000002.png
  frame000003.png
  ...
```

Point the pipeline at that folder. One folder is one episode; process
episodes separately rather than combining them.

Three requirements:

1. **PNG only.** Files with other extensions in the folder are ignored.
2. **Filenames must sort into playback order.** Frames are read with a plain
   alphabetical sort, so zero-padded names such as `frame000001.png` are
   required. Unpadded names break the ordering, because `frame10.png` sorts
   before `frame2.png`.
3. **One episode per folder**, with no unrelated images mixed in.

Any frame size works; the models resize internally. The example videos are
540x540.

To convert an existing video file into this layout:

```bash
ffmpeg -i my_case.mp4 my_case/frame%06d.png
```

### Pipeline output

Two videos, two still images, and the intermediate seed and tracking
files.

**`cutline_only.mp4`** shows the tracked tip and its accumulated path with no
mask fill. Check this first: it is the quickest way to confirm the tip was
seeded on the correct instrument.

**`cutline_sweep_walk.mp4`** is the main result, with the resected region
shaded red over the original video and the tracked tip marked in magenta. The
mask is cumulative and only grows; if it stops growing while the tool is still
cutting, tracking has likely drifted.

**`final_mask.png`** is that same cumulative mask after the last frame, saved
as a plain binary image (255 = resected, 0 = background) at the input frame
size. This is the file to use for quantitative analysis. **`final_overlay.png`**
is the corresponding last video frame, for checking the mask by eye.

### Automatic workflow

Detects the tool tip, refines it with SAM2, tracks it, and builds the
resection mask. Add `--no-refine` to skip the SAM2 stage if it is not
installed.

```bash
python edge_anchored_tracking/run_autonomous_toolhead.py \
    <video_frame_dir> <out_root_dir> [tag] [--no-refine] [--manual]
```

For example:

```bash
python edge_anchored_tracking/run_autonomous_toolhead.py \
    /data/case_20260612/endoscope \
    /results/case_20260612 --no-refine
```

Requires `keypoint_rcnn.onnx` in `checkpoints_keypoint/`, or `KEYPOINT_MODEL`
pointing at it.

Outputs, written to `<out_root_dir>`:

```
final_mask.png                  final resection mask, binary (for analysis)
final_overlay.png               final mask over the last frame
cutline_sweep_walk.mp4          full mask overlay, all frames
cutline_only.mp4                trajectory-only visualization
auto_seed.json                  detector output
tip_track/tip_track.csv         TAPNext++ output
```

With refinement enabled, two more are written: `refined_seed.json`, the
corrected point actually used for tracking, and `sam2_refine_viz/`, showing
what the refinement did.

### Manual seeding fallback

The `--manual` flag skips automatic detection and opens the click GUI to seed
the tip by hand. The remainder of the pipeline is unchanged.

```bash
python edge_anchored_tracking/run_autonomous_toolhead.py \
    <video_frame_dir> <out_root_dir> --manual
```

It is also the way to process episodes before the detector weights are
available. Refinement is skipped in this mode, because it exists solely to
correct the detector's inner-tube offset and a hand click is already placed
on the true tip. See "When seeding goes wrong" below for when to reach for
this.

With `--manual`, the first three outputs listed above are replaced by a single
`manual_seed.json`.

### When seeding goes wrong

Two failure modes are worth knowing about. Both are uncommon, and both are
handled by re-running that episode with `--manual`.

**The tip detector picks the wrong tool, or a point that is not on a tool.**
Two instruments are usually visible, and the detector occasionally selects the
wrong one or returns a point off the tool entirely. Tracking then follows
whatever it was given, so the resulting mask is wrong from the start rather
than degrading gradually.

**The refined seed is still far from the tip.** Refinement corrects the
detector's inner-tube offset, but it does not guarantee the point lands
exactly on the distal tip. If the refined seed is still visibly off, a manual
click is the more reliable option for that episode.

Both are visible in `cutline_only.mp4`, which draws the tracked point without
the mask fill and is the quickest way to check that seeding was sensible
before trusting the mask.

If the tip detector cannot find a confident detection at all, the run stops
with a message saying so rather than guessing. That is an expected outcome on
difficult footage, not a crash, and `--manual` is the intended response.

### Running individual stages

The seeding stages can also be run standalone:

```bash
# Detect the tip. Writes {"points": [[x,y]], "W", "H", "frame"}.
python edge_anchored_tracking/auto_seed_tip.py \
    <keypoint_rcnn.onnx> <video_frame_dir> <out_seed.json> [max_frames=30]

# Refine to the true distal tip. The final argument writes the refined seed.
python edge_anchored_tracking/refine_tip_with_sam2_v2.py \
    <video_frame_dir> <auto_seed.json> <viz_out_dir> [walk_px=5] [refined_seed.json]
```

The manual workflow's individual stages:

```bash
# 1. Click the tool tip. First click is the tip; 2-3 further clicks down the
#    shaft improve tracking robustness under occlusion. Close the window to save.
python click_tooltip_multi.py <path/to/frame000001.png> <out_seed.json>

# 2. Track the tip. Produces tip_track.csv with columns frame, x, y, mode.
#    mode="M" indicates the tracker lost the tip and used a shaft-derived fallback.
python tapnextpp_hybrid.py <video_frame_dir> <seed.json> <track_outdir>

# 3. Build the mask. Produces cutline_sweep_walk.mp4.
python edge_anchored_tracking/cutline_sweep_mask_walk.py \
    <video_frame_dir> <track_outdir>/tip_track.csv <output_dir>

# 4. Optional. Renders the tracked tip and accumulated trajectory without mask
#    fill, which isolates tracking behavior from the mask geometry.
python edge_anchored_tracking/render_cutline_only.py \
    <video_frame_dir> <track_outdir>/tip_track.csv <output_dir> [tip_smooth=15]
```

### Mask parameters

`cutline_sweep_mask_walk.py` accepts three optional parameters after the
output directory. The defaults are what the automatic workflow uses.

| Parameter | Default | Effect |
| --- | --- | --- |
| `tip_smooth` | `15` | Moving-average window, in frames, applied to the tip trajectory. Larger values smooth the cutline but respond more slowly to real direction changes. |
| `mask_ema` | `0.5` | Blends the tumor probability across frames so one noisy segmentation cannot jump the boundary. `1.0` disables it. |
| `dilate` | `0` | Radius in pixels of an optional dilation of the displayed mask. `0` disables it. |

### Runtime

On a CUDA GPU, a 469 frame episode takes about three minutes end to end,
dominated by tracking and mask construction. Each stage prints its own
elapsed time as it runs. CPU-only runs are substantially slower. SAM2
refinement runs once per episode, not per frame, so it adds a few seconds
regardless of episode length.

## Real-time use with ROS 2

`edge_anchored_tracking/ros_toolhead_node.py` runs the same pipeline against a
live camera topic instead of a folder of frames, publishing the resection mask
overlaid on the endoscopic image as it is built.

No algorithmic change was needed. The pipeline was already causal: TAPNext++
carries its state from frame to frame, and the mask accumulates per frame with
no lookahead. The node imports the geometry from
`cutline_sweep_mask_walk.py` and calls it directly, so the real-time and
offline paths run the same code.

### Interface

| | |
| --- | --- |
| Subscribes | `/ves_camera/image` (`sensor_msgs/Image`) |
| Publishes | `/toolhead/overlay` (`sensor_msgs/Image`), the mask drawn on the live frame |
| | `/toolhead/mask` (`sensor_msgs/Image`, mono8), 255 = resected |
| | `/toolhead/tip` (`geometry_msgs/PointStamped`), tip in camera pixels |
| Services | `/toolhead/start`, `/toolhead/stop` (`std_srvs/srv/Trigger`) |

### Running one cut

Start the node, then bracket each cut with the two services. One start/stop
cycle is one episode, so the node can be left running across many cuts.

```bash
python3 edge_anchored_tracking/ros_toolhead_node.py

# before the cut
ros2 service call /toolhead/start std_srvs/srv/Trigger
# ... perform the cut ...
ros2 service call /toolhead/stop std_srvs/srv/Trigger
```

View the result live:

```bash
ros2 run rqt_image_view rqt_image_view /toolhead/overlay
```

On `start`, the node buffers frames until the tip detector commits to a
location, refines that point once with SAM2, and begins tracking. This
normally takes well under a second. On `stop`, it writes the episode's final
images and resets, ready for the next cut.

### Episode output

Each cut is written to its own timestamped directory under `Output/`:

```
Output/episode_001_20260903-142317/
  final_mask.png        cumulative resection mask, binary (for analysis)
  final_overlay.png     that mask over the last frame
  cutline_overlay.png   the tracked tip trajectory, no mask fill
  episode_info.json     frame counts, duration, final mask area
```

These match the offline pipeline's outputs, so downstream analysis is
unchanged.

### Parameters

```bash
python3 edge_anchored_tracking/ros_toolhead_node.py --ros-args \
    -p image_topic:=/ves_camera/image_rect \
    -p output_root:=/data/toolhead_output
```

| Parameter | Default | Purpose |
| --- | --- | --- |
| `image_topic` | `/ves_camera/image` | Camera topic to subscribe to |
| `output_root` | `Output/` | Where episode directories are written |
| `proc_size` | `540` | Frames are resized to this before processing, to match the resolution the models were trained at |
| `resect_side` | `right` | Side of the cutline the resection lies on |
| `refine` | `true` | Run SAM2 tip refinement once per episode |

### Resolution

The models were trained on 540x540 frames. If the camera publishes at a
different resolution (the reference system publishes 1080x1080), every frame
is downscaled to `proc_size` before processing, so the tip-smoothing window,
the SAM2 skeleton walk and the mask-shape gates all see the scale they were
tuned at. Overlays and masks are published back at the camera's native
resolution, and the published tip coordinate is scaled to match.

### Keeping up with the camera

Frames that arrive while a frame is still being processed are dropped rather
than queued, so the overlay stays aligned with what the camera is showing
instead of falling progressively behind. The node reports how many frames were
processed and how many dropped when the episode ends. Segmentation and mask
construction run at roughly 8 frames per second on a CUDA GPU, so on a 30 fps
camera expect most frames to be skipped; the mask is unaffected, since it
accumulates from whichever frames are processed.

### Requirements

Beyond the setup above, this needs a ROS 2 installation (developed against
Jazzy) with `rclpy`, `sensor_msgs`, `geometry_msgs` and `std_srvs` importable
from the same Python environment as PyTorch:

```bash
python3 -c "import rclpy, sensor_msgs, geometry_msgs, std_srvs; print('ros ok')"
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Both must succeed from the same interpreter. The GPU build of ONNX Runtime is
recommended here, as noted in Step 3.

## Repository layout

```
src/                          segmentation model and inference engine
  model.py                    U-Net/ResNet-34 architecture definition
  inference_engine.py         loads checkpoint, runs per-frame inference
  preprocessor.py / postprocessor.py
edge_anchored_tracking/
  run_autonomous_toolhead.py  end-to-end driver for the automatic workflow
  ros_toolhead_node.py        ROS 2 node: same pipeline on a live camera topic
  auto_seed_tip.py            automatic tip detection (keypoint_rcnn.onnx)
  refine_tip_with_sam2_v2.py  SAM2 two-pass refinement to the true distal tip
  auto_seed_tool.py           shared mask-quality helpers
  cutline_sweep_mask_walk.py  main pipeline: cutline, polygon, accumulation
  render_cutline_only.py      cutline-only visualization, no mask fill
tapnextpp_hybrid.py           tool-tip tracking (TAPNext++ with shaft fallback)
click_tooltip_multi.py        GUI tool to seed the tip tracker manually
checkpoints_tumor_binary10/   segmentation weights
checkpoints_keypoint/         keypoint_rcnn.onnx
checkpoints_sam2/             SAM2 checkpoint
requirements.txt
```
