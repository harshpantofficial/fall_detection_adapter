# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Skeleton → transformer features, and the per-stream sequence window.

Ported from the reference fall-detection Space (``app.py``:
``extract_features_from_yolo_result`` and ``normalize_skeleton_frame``)
with the arithmetic unchanged, because the transformer was trained on
exactly these numbers. Two details of that code look like accidents and
are kept on purpose:

* **Feature order is the joint NAMES sorted alphabetically**, not COCO
  order: ``"Left Ankle"`` is slot 0, ``"Right Wrist"`` slot 16. Feeding
  COCO order would hand the model a scrambled body that still has the
  right statistics, which is the kind of bug that degrades accuracy
  without failing anything.
* **When no shoulder or hip is visible the frame is NOT normalized** and
  goes in as plain frame fractions; joints at or below the confidence
  floor are zeroed only when a reference point exists. A frame with no
  person is 51 zeros.

Pure numpy — no model, no I/O — so the tests pin every branch.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np

#: The transformer's window, in frames. Fixed by the trained model's
#: input shape ``[1, 30, 51]``.
TIMESTEPS: int = 30

#: Joints at or below this confidence are ignored when picking the
#: reference point and zeroed after normalization (``>`` in the
#: reference). The comparison is against a float32 feature, and
#: float32(0.3) is 0.30000001 — so a joint at exactly 0.3 counts as
#: VISIBLE, in the reference as here.
MIN_KEYPOINT_CONFIDENCE: float = 0.3

#: COCO-17 in the order YOLO emits them, in the reference's spelling.
COCO_NAMES: tuple[str, ...] = (
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
)
#: Training order: the same names, sorted.
TRAINING_NAMES: tuple[str, ...] = tuple(sorted(COCO_NAMES))
_SLOT: dict[str, int] = {name: i for i, name in enumerate(TRAINING_NAMES)}
#: COCO index → training slot.
COCO_TO_SLOT: tuple[int, ...] = tuple(_SLOT[name] for name in COCO_NAMES)
NUM_FEATURES: int = len(TRAINING_NAMES) * 3


def _xyc(slot: int) -> tuple[int, int, int]:
    return slot * 3, slot * 3 + 1, slot * 3 + 2


def _midpoint(frame: np.ndarray, left: str, right: str,
              min_conf: float) -> tuple[float, float]:
    """Mean of the two joints that are visible; one if only one is; NaN if neither."""
    lx, ly, lc = _xyc(_SLOT[left])
    rx, ry, rc = _xyc(_SLOT[right])
    left_ok, right_ok = frame[lc] > min_conf, frame[rc] > min_conf
    if left_ok and right_ok:
        return (frame[lx] + frame[rx]) / 2, (frame[ly] + frame[ry]) / 2
    if left_ok:
        return frame[lx], frame[ly]
    if right_ok:
        return frame[rx], frame[ry]
    return np.nan, np.nan


def normalize_skeleton(frame: np.ndarray,
                       min_conf: float = MIN_KEYPOINT_CONFIDENCE) -> np.ndarray:
    """Centre the skeleton on the torso and scale by torso length.

    The reference point is the middle of mid-shoulder and mid-hip (or
    whichever of the two exists); the scale is the shoulder-to-hip
    distance when both exist, else 1. No reference point → the frame is
    returned as it came."""
    out = np.copy(frame)
    sx, sy = _midpoint(frame, "Left Shoulder", "Right Shoulder", min_conf)
    hx, hy = _midpoint(frame, "Left Hip", "Right Hip", min_conf)

    if not np.isnan(sx) and not np.isnan(hx):
        ref_x, ref_y = (sx + hx) / 2, (sy + hy) / 2
    elif not np.isnan(sx):
        ref_x, ref_y = sx, sy
    elif not np.isnan(hx):
        ref_x, ref_y = hx, hy
    else:
        return frame

    scale = 1.0
    if not np.isnan(sx) and not np.isnan(hx):
        torso = np.sqrt((sx - hx) ** 2 + (sy - hy) ** 2)
        if torso > 1e-6:
            scale = torso

    for slot in range(len(TRAINING_NAMES)):
        x, y, c = _xyc(slot)
        if frame[c] > min_conf:
            out[x] = (frame[x] - ref_x) / scale
            out[y] = (frame[y] - ref_y) / scale
        else:
            out[x] = 0.0
            out[y] = 0.0
    return out


def subject_index(persons: Sequence[dict[str, Any]]) -> int | None:
    """The person the model watches: the LARGEST box, as in the
    reference. One subject per camera is the trained model's assumption,
    not ours — it sees one skeleton per frame."""
    best, best_area = None, -1.0
    for i, person in enumerate(persons):
        x1, y1, x2, y2 = (float(v) for v in person["bbox"])
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best, best_area = i, area
    return best


def features_for(keypoints: Sequence[Sequence[float]] | None,
                 width: int, height: int) -> np.ndarray:
    """One frame's 51 features from a person's 17 pixel-space
    ``[x, y, conf]`` keypoints (COCO order). ``None`` → no person → zeros."""
    frame = np.zeros(NUM_FEATURES, dtype=np.float32)
    if keypoints is None:
        return frame
    for coco_index, (kx, ky, kc) in enumerate(keypoints[:len(COCO_NAMES)]):
        x, y, c = _xyc(COCO_TO_SLOT[coco_index])
        frame[x] = kx / width
        frame[y] = ky / height
        frame[c] = float(kc)
    return normalize_skeleton(frame)


class SequenceWindow:
    """The last ``TIMESTEPS`` frames of one stream.

    One per camera stream, never shared: the model reads motion across
    the window, so frames from two cameras interleaved in one window
    would read as a body teleporting. Bounded by construction."""

    def __init__(self, timesteps: int = TIMESTEPS) -> None:
        self._frames: deque[np.ndarray] = deque(maxlen=timesteps)
        self.timesteps = timesteps

    def push(self, features: np.ndarray) -> None:
        self._frames.append(features)

    def clear(self) -> None:
        self._frames.clear()

    @property
    def filled(self) -> int:
        return len(self._frames)

    @property
    def full(self) -> bool:
        return len(self._frames) == self.timesteps

    def batch(self) -> np.ndarray:
        """``[1, TIMESTEPS, 51]`` float32, the transformer's input."""
        return np.asarray(self._frames, dtype=np.float32)[np.newaxis, ...]
