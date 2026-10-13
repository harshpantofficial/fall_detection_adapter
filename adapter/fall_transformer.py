"""
FallTransformer — Sequence-based TFLite Transformer for Fall Classification.

Ported from fall_detection/fall_detection_core.py and fall_detection/app.py.
Extracts 17 COCO keypoints from YOLO26-pose, normalises them relative to the torso
centre and torso height scale, buffers a 30-timestep sliding sequence, and executes
the TFLite Transformer model to detect falls.
"""
from __future__ import annotations

import logging
import math
import os
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Keypoint definitions matching fall_detection_core.py
YOLO_KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
]

YOUR_KEYPOINT_NAMES_TRAINING = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
]

SORTED_YOUR_KEYPOINT_NAMES = sorted(YOUR_KEYPOINT_NAMES_TRAINING)
KEYPOINT_DICT_TRAINING = {name: i for i, name in enumerate(SORTED_YOUR_KEYPOINT_NAMES)}
NUM_KEYPOINTS_TRAINING = len(KEYPOINT_DICT_TRAINING)
NUM_FEATURES = NUM_KEYPOINTS_TRAINING * 3  # 51 features

YOLO_IDX_TO_NAME = {
    i: name for i, name in enumerate(YOLO_KEYPOINT_NAMES)
    if name in KEYPOINT_DICT_TRAINING
}

INPUT_TIMESTEPS = 30
FALL_CONFIDENCE_THRESHOLD = 0.90
MIN_KEYPOINT_CONFIDENCE_FOR_NORMALIZATION = 0.3
FALL_EVENT_COOLDOWN_FRAMES = 300


def get_kpt_indices_training_order(keypoint_name: str) -> tuple[int, int, int]:
    """Return the (x_idx, y_idx, conf_idx) positions in the flat feature vector."""
    kp_idx = KEYPOINT_DICT_TRAINING[keypoint_name]
    return kp_idx * 3, kp_idx * 3 + 1, kp_idx * 3 + 2


def normalize_skeleton_frame(
    frame_features_sorted: np.ndarray,
    min_confidence: float = MIN_KEYPOINT_CONFIDENCE_FOR_NORMALIZATION,
) -> np.ndarray:
    """Normalise a flat keypoint feature vector relative to the torso centre.

    Coordinates are translated to a mid-torso origin and scaled by torso
    height so that the classifier is invariant to position and person size.
    Low-confidence keypoints are zeroed out rather than used as anchors.
    """
    normalized_frame = np.copy(frame_features_sorted)

    try:
        ls_x_idx, ls_y_idx, ls_c_idx = get_kpt_indices_training_order("Left Shoulder")
        rs_x_idx, rs_y_idx, rs_c_idx = get_kpt_indices_training_order("Right Shoulder")
        lh_x_idx, lh_y_idx, lh_c_idx = get_kpt_indices_training_order("Left Hip")
        rh_x_idx, rh_y_idx, rh_c_idx = get_kpt_indices_training_order("Right Hip")
    except Exception:
        return frame_features_sorted

    ls_c = frame_features_sorted[ls_c_idx]
    rs_c = frame_features_sorted[rs_c_idx]
    lh_c = frame_features_sorted[lh_c_idx]
    rh_c = frame_features_sorted[rh_c_idx]
    ls_x, ls_y = frame_features_sorted[ls_x_idx], frame_features_sorted[ls_y_idx]
    rs_x, rs_y = frame_features_sorted[rs_x_idx], frame_features_sorted[rs_y_idx]
    lh_x, lh_y = frame_features_sorted[lh_x_idx], frame_features_sorted[lh_y_idx]
    rh_x, rh_y = frame_features_sorted[rh_x_idx], frame_features_sorted[rh_y_idx]

    # Mid-shoulder anchor
    mid_shoulder_x, mid_shoulder_y = np.nan, np.nan
    valid_ls = ls_c > min_confidence
    valid_rs = rs_c > min_confidence
    if valid_ls and valid_rs:
        mid_shoulder_x, mid_shoulder_y = (ls_x + rs_x) / 2.0, (ls_y + rs_y) / 2.0
    elif valid_ls:
        mid_shoulder_x, mid_shoulder_y = ls_x, ls_y
    elif valid_rs:
        mid_shoulder_x, mid_shoulder_y = rs_x, rs_y

    # Mid-hip anchor
    mid_hip_x, mid_hip_y = np.nan, np.nan
    valid_lh = lh_c > min_confidence
    valid_rh = rh_c > min_confidence
    if valid_lh and valid_rh:
        mid_hip_x, mid_hip_y = (lh_x + rh_x) / 2.0, (lh_y + rh_y) / 2.0
    elif valid_lh:
        mid_hip_x, mid_hip_y = lh_x, lh_y
    elif valid_rh:
        mid_hip_x, mid_hip_y = rh_x, rh_y

    # Reference point = centre of torso
    ref_x, ref_y = np.nan, np.nan
    if not np.isnan(mid_shoulder_x) and not np.isnan(mid_hip_x):
        ref_x = (mid_shoulder_x + mid_hip_x) / 2.0
        ref_y = (mid_shoulder_y + mid_hip_y) / 2.0
    elif not np.isnan(mid_shoulder_x):
        ref_x, ref_y = mid_shoulder_x, mid_shoulder_y
    elif not np.isnan(mid_hip_x):
        ref_x, ref_y = mid_hip_x, mid_hip_y

    # Scale = torso height (shoulder-to-hip distance)
    scale = 1.0
    if not np.isnan(mid_shoulder_x) and not np.isnan(mid_hip_x):
        torso_height = np.sqrt(
            (mid_shoulder_x - mid_hip_x) ** 2 + (mid_shoulder_y - mid_hip_y) ** 2
        )
        if torso_height > 1e-6:
            scale = torso_height

    if np.isnan(ref_x):
        return frame_features_sorted

    for kp_name in SORTED_YOUR_KEYPOINT_NAMES:
        x_idx, y_idx, c_idx = get_kpt_indices_training_order(kp_name)
        if frame_features_sorted[c_idx] > min_confidence:
            normalized_frame[x_idx] = (frame_features_sorted[x_idx] - ref_x) / scale
            normalized_frame[y_idx] = (frame_features_sorted[y_idx] - ref_y) / scale
        else:
            normalized_frame[x_idx] = 0.0
            normalized_frame[y_idx] = 0.0

    return normalized_frame


def extract_features_from_keypoints(
    kps_data: np.ndarray | list[list[float]],
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Convert raw 17-keypoint [x, y, conf] data into a flat normalised feature vector (51,)."""
    frame_features = np.zeros(NUM_FEATURES, dtype=np.float32)
    if kps_data is None or len(kps_data) == 0:
        return frame_features

    kps_arr = np.asarray(kps_data, dtype=np.float32)
    if kps_arr.ndim == 3:
        kps_arr = kps_arr[0]

    for yolo_idx, kp_name in YOLO_IDX_TO_NAME.items():
        if yolo_idx >= len(kps_arr):
            continue
        x_norm = float(kps_arr[yolo_idx, 0]) / max(1, frame_width)
        y_norm = float(kps_arr[yolo_idx, 1]) / max(1, frame_height)
        conf = float(kps_arr[yolo_idx, 2])
        x_idx, y_idx, c_idx = get_kpt_indices_training_order(kp_name)
        frame_features[x_idx] = x_norm
        frame_features[y_idx] = y_norm
        frame_features[c_idx] = conf

    return normalize_skeleton_frame(frame_features)


def compute_torso_angle(kpts: np.ndarray | list[list[float]]) -> float:
    """Calculate torso angle from vertical in degrees (0 = upright, 90 = horizontal)."""
    if kpts is None or len(kpts) < 13:
        return 0.0
    kpts_arr = np.asarray(kpts)
    if kpts_arr.ndim == 3:
        kpts_arr = kpts_arr[0]
    try:
        ls, rs = kpts_arr[5], kpts_arr[6]
        lh, rh = kpts_arr[11], kpts_arr[12]

        if ls[2] > 0.3 and rs[2] > 0.3:
            ms_x, ms_y = (ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0
        elif ls[2] > 0.3:
            ms_x, ms_y = float(ls[0]), float(ls[1])
        elif rs[2] > 0.3:
            ms_x, ms_y = float(rs[0]), float(rs[1])
        else:
            return 0.0

        if lh[2] > 0.3 and rh[2] > 0.3:
            mh_x, mh_y = (lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0
        elif lh[2] > 0.3:
            mh_x, mh_y = float(lh[0]), float(lh[1])
        elif rh[2] > 0.3:
            mh_x, mh_y = float(rh[0]), float(rh[1])
        else:
            return 0.0

        dx = ms_x - mh_x
        dy = mh_y - ms_y  # positive when shoulders are above hips
        angle_rad = math.atan2(abs(dx), max(1e-5, abs(dy)))
        deg = round(math.degrees(angle_rad), 1)
        if dy < -15.0:  # inverted or lying flat
            deg = max(deg, 75.0)
        return deg
    except Exception:
        return 0.0


class StreamState:
    """Holds the sliding frame buffer and cooldown counter for one camera stream."""

    def __init__(self, timesteps: int = INPUT_TIMESTEPS, cooldown: int = FALL_EVENT_COOLDOWN_FRAMES) -> None:
        self.frame_buffer: deque = deque(maxlen=timesteps)
        self.last_fall_frame: int = -cooldown
        self.frame_count: int = 0


class FallTransformer:
    """Inference engine for the sequence-based TFLite Transformer."""

    def __init__(
        self,
        tflite_model_path: str | None = None,
        timesteps: int = INPUT_TIMESTEPS,
        threshold: float = FALL_CONFIDENCE_THRESHOLD,
        cooldown_frames: int = FALL_EVENT_COOLDOWN_FRAMES,
    ) -> None:
        self.timesteps = timesteps
        self.threshold = threshold
        self.cooldown_frames = cooldown_frames
        self._tflite_path = tflite_model_path
        self._interpreter: Any = None
        self._input_details: Any = None
        self._output_details: Any = None
        self._streams: dict[str, StreamState] = {}
        self._init_tflite()

    def _init_tflite(self) -> None:
        if not self._tflite_path or not os.path.isfile(self._tflite_path):
            logger.warning("FallTransformer: TFLite weights not found at %s", self._tflite_path)
            return

        try:
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                try:
                    import tflite_runtime.interpreter as tflite
                    Interpreter = tflite.Interpreter
                except ImportError:
                    import tensorflow as tf
                    Interpreter = tf.lite.Interpreter

            self._interpreter = Interpreter(model_path=self._tflite_path)
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            logger.info("FallTransformer: TFLite Transformer loaded successfully from %s", self._tflite_path)
        except Exception as exc:
            logger.exception("FallTransformer: failed to initialize TFLite interpreter from %s: %s", self._tflite_path, exc)
            self._interpreter = None

    def get_stream_state(self, stream_id: str) -> StreamState:
        if stream_id not in self._streams:
            self._streams[stream_id] = StreamState(self.timesteps, self.cooldown_frames)
        return self._streams[stream_id]

    def reset_stream(self, stream_id: str) -> None:
        self._streams.pop(stream_id, None)

    def analyze_frame(
        self,
        keypoints: np.ndarray | list[list[float]] | None,
        frame_width: int,
        frame_height: int,
        stream_id: str = "default",
    ) -> dict[str, Any]:
        """Process one frame's keypoints for a given stream and return fall classification.

        Returns
        -------
        dict with keys:
            - fall_detected: bool
            - fall_confidence: float (0.0 to 1.0)
            - fall_state: "standing" | "falling" | "fallen" | "unknown"
            - torso_angle_deg: float
            - frame_count: int
            - warmup_done: bool
        """
        state = self.get_stream_state(stream_id)
        state.frame_count += 1

        if keypoints is not None and len(keypoints) > 0:
            features = extract_features_from_keypoints(keypoints, frame_width, frame_height)
            torso_angle = compute_torso_angle(keypoints)
        else:
            features = np.zeros(NUM_FEATURES, dtype=np.float32)
            torso_angle = 0.0

        state.frame_buffer.append(features)

        fall_detected = False
        confidence = 0.0
        warmup_done = len(state.frame_buffer) == self.timesteps

        if warmup_done and self._interpreter is not None:
            input_seq = np.array(state.frame_buffer, dtype=np.float32)[np.newaxis, ...]
            self._interpreter.set_tensor(self._input_details[0]["index"], input_seq)
            self._interpreter.invoke()
            output_data = self._interpreter.get_tensor(self._output_details[0]["index"])
            confidence = float(output_data[0][0])

            cooldown_ok = (state.frame_count - state.last_fall_frame) >= self.cooldown_frames
            if confidence >= self.threshold and cooldown_ok:
                fall_detected = True
                state.last_fall_frame = state.frame_count

        if fall_detected:
            fall_state = "fallen"
        elif torso_angle >= 35.0 and confidence >= 0.4:
            fall_state = "falling"
        elif warmup_done:
            fall_state = "standing"
        else:
            fall_state = "standing"

        return {
            "fall_detected": fall_detected,
            "fall_confidence": round(confidence, 4),
            "fall_state": fall_state,
            "torso_angle_deg": torso_angle,
            "frame_count": state.frame_count,
            "warmup_done": warmup_done,
        }