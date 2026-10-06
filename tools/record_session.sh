#!/usr/bin/env bash
# Record the raw inputs of a study session to an MCAP bag, alongside (not
# instead of) the node's own -p experiment:=true logs.
#
#   tools/record_session.sh [output_root]      # Ctrl-C to stop
#
# The bag holds what the node's logs cannot: every camera frame at native
# resolution (the node's depth-1 subscription never sees frames that arrive
# while it is busy), the smoother's tip pose before any of our smoothing,
# both arms' joint states, and the tool signal. Replay it offline with
# tools/export_bag_frames.py.
#
# Read-only with respect to the robot: it subscribes and asks
# /ves_smoother for its parameter values; it publishes and commands nothing.
set -euo pipefail

ROOT="${1:-$HOME/toolhead_bags}"
OUT="$ROOT/session_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"

TOPICS=(
  /ves_camera/image_rect
  /ves_camera/camera_info
  /ves_smoother/right/tip_pose
  /ves_smoother/right/tool_tip_pixels
  /ves/left/joint/measured_jp
  /ves/right/joint/measured_jp      # carries the 'tool' joint (the trigger)
  /toolhead/tip
  /toolhead/mask
)
# Deliberately NOT recorded: /ves_camera/image (unrectified duplicate),
# /ves_smoother/overlay_image and /toolhead/overlay (renderings, and large).

# Smoother parameters at session start: camera_lag_seconds, tip offsets,
# ignore_keypoints_mode. A failure here is reported, not fatal.
if ros2 param dump /ves_smoother > "$OUT/ves_smoother_params.yaml" 2> "$OUT/param_dump.err"; then
  echo "ves_smoother params -> $OUT/ves_smoother_params.yaml"
else
  echo "WARNING: could not dump /ves_smoother params (see $OUT/param_dump.err)" >&2
fi
ros2 topic list -t > "$OUT/topics_at_start.txt" || true
for t in "${TOPICS[@]}"; do
  grep -q "^$t " "$OUT/topics_at_start.txt" || echo "WARNING: $t not currently published" >&2
done

{
  echo "date: $(date -Iseconds)"
  echo "host: $(hostname)"
  echo "topics: ${TOPICS[*]}"
  echo "storage: mcap, preset zstd_fast"
} > "$OUT/record_info.txt"

echo "Recording to $OUT/bag  (Ctrl-C to stop)"
exec ros2 bag record -s mcap --storage-preset-profile zstd_fast \
  -o "$OUT/bag" "${TOPICS[@]}"
