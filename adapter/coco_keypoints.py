"""
COCO-17 keypoint names and skeleton indices for YOLO pose models.

Standard COCO keypoints:
 0: Nose           1: Left Eye       2: Right Eye
 3: Left Ear       4: Right Ear      5: Left Shoulder
 6: Right Shoulder 7: Left Elbow     8: Right Elbow
 9: Left Wrist    10: Right Wrist   11: Left Hip
12: Right Hip     13: Left Knee     14: Right Knee
15: Left Ankle    16: Right Ankle
"""
from __future__ import annotations

COCO_KEYPOINTS: tuple[str, ...] = (
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
)

assert len(COCO_KEYPOINTS) == 17, "COCO-17 must have exactly 17 keypoints"

KEYPOINT_STRIDE: int = 3

COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms + shoulders
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
)

# Semantic index constants for fast lookup
NOSE_IDX = 0
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12
LEFT_KNEE_IDX = 13
RIGHT_KNEE_IDX = 14
LEFT_ANKLE_IDX = 15
RIGHT_ANKLE_IDX = 16


def keypoint_index_to_name(index: int) -> str:
    """Return the COCO keypoint name for a slot index."""
    if 0 <= index < len(COCO_KEYPOINTS):
        return COCO_KEYPOINTS[index]
    return f"keypoint_{index}"