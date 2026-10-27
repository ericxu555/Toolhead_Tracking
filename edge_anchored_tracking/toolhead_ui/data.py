"""What a source hands the window: one frame's streams and readouts.

Every source (live ROS, offline replay, demo) produces FrameData, so the window
never knows where the numbers came from.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Resected-fraction states, as the toolhead node publishes them on /toolhead/status.
#   baseline  capturing the tumor mask before the tool enters; no number yet
#   blocked   the tool is in view, so no baseline can be captured; no number
#   ready     baseline captured, waiting for the cut
#   estimate  live value, swept area / baseline tumor area (deformation makes it approximate)
#   final     measured after the cut button is released, with the tool out of the way
#   none      no baseline existed at the press, so this cut has no estimate
BASELINE, BLOCKED, READY, ESTIMATE, FINAL, NONE = (
    "baseline", "blocked", "ready", "estimate", "final", "none")


@dataclass
class FrameData:
    streams: List[Tuple[str, Optional[np.ndarray]]] = field(default_factory=list)  # (caption, BGR or None)
    d_mm: Optional[float] = None          # tip to tumor-hull distance; None without a pose or a hull
    d_px: Optional[float] = None          # the same in pixels, shown only when there is no mm value
    warn_mm: float = 2.0
    warn: bool = False                    # decided by the source (the node, for live)
    fraction: Optional[float] = None      # 0..1
    fraction_state: str = BASELINE
    index: int = 0
    total: int = 0                        # 0 for a live source, which has no timeline
