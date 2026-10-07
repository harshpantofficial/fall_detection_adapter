# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Shared fall-detection inference logic.
Used by both the Gradio demo (app.py) and the OpenNVR adapter (adapter.py).
Import this module instead of duplicating the keypoint constants and model
loading in multiple places.
"""

import numpy as np
from collections import deque
from ai_edge_litert.interpreter import Interpreter as tflite_Interpreter
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "fall_detection_transformer.tflite"
YOLO_POSE_MODEL = "yolo26n-pose.pt"
INPUT_TIMESTEPS = 30
FALL_CONFIDENCE_THRESHOLD = 0.90
MIN_KEYPOINT_CONFIDENCE_FOR_NORMALIZATION = 0.3
FALL_EVENT_COOLDOWN_FRAMES = 300


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
NUM_FEATURES = NUM_KEYPOINTS_TRAINING * 3

# Map YOLO output index → training dict index (only names that appear in both)
YOLO_IDX_TO_NAME = {
    i: name for i, name in enumerate(YOLO_KEYPOINT_NAMES)
    if name in KEYPOINT_DICT_TRAINING
}


class StreamState:
    """Holds the sliding frame buffer and cooldown counter for one camera stream.

    The OpenNVR adapter maintains one :class:`StreamState` per ``stream_id``
    so concurrent camera feeds never bleed into each other's buffers.
    """

    def __init__(self) -> None:
        self.frame_buffer: deque = deque(maxlen=INPUT_TIMESTEPS)
        self.last_fall_frame: int = -FALL_EVENT_COOLDOWN_FRAMES
        self.frame_count: int = 0


def load_models():
    """Load YOLO26-pose and the TFLite Transformer once at startup.
    Returns
    -------
    tuple
        ``(yolo_pose, interpreter, input_details, output_details)``
    """
    print(f"Loading YOLO26 pose model: {YOLO_POSE_MODEL}")
    yolo_pose = YOLO(YOLO_POSE_MODEL)
    print("YOLO26 pose model loaded.")

    interpreter = tflite_Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"TFLite Transformer loaded: {MODEL_PATH}")
    print(f"  Expected input shape: {tuple(input_details[0]['shape'])}")
    print(f"  NUM_FEATURES: {NUM_FEATURES}")

    return yolo_pose, interpreter, input_details, output_details


def get_kpt_indices_training_order(keypoint_name: str):
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

    Parameters
    ----------
    frame_features_sorted:
        Flat ``float32`` array of length ``NUM_FEATURES`` arranged as
        ``[x0, y0, conf0, x1, y1, conf1, …]`` in *sorted* keypoint order.
    min_confidence:
        Keypoints below this confidence are treated as invisible.

    Returns
    -------
    np.ndarray
        Normalised copy of the input array.
    """
    normalized_frame = np.copy(frame_features_sorted)

    try:
        ls_x_idx, ls_y_idx, ls_c_idx = get_kpt_indices_training_order("Left Shoulder")
        rs_x_idx, rs_y_idx, rs_c_idx = get_kpt_indices_training_order("Right Shoulder")
        lh_x_idx, lh_y_idx, lh_c_idx = get_kpt_indices_training_order("Left Hip")
        rh_x_idx, rh_y_idx, rh_c_idx = get_kpt_indices_training_order("Right Hip")
    except Exception:  # noqa: BLE001
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
        mid_shoulder_x, mid_shoulder_y = (ls_x + rs_x) / 2, (ls_y + rs_y) / 2
    elif valid_ls:
        mid_shoulder_x, mid_shoulder_y = ls_x, ls_y
    elif valid_rs:
        mid_shoulder_x, mid_shoulder_y = rs_x, rs_y

    # Mid-hip anchor
    mid_hip_x, mid_hip_y = np.nan, np.nan
    valid_lh = lh_c > min_confidence
    valid_rh = rh_c > min_confidence
    if valid_lh and valid_rh:
        mid_hip_x, mid_hip_y = (lh_x + rh_x) / 2, (lh_y + rh_y) / 2
    elif valid_lh:
        mid_hip_x, mid_hip_y = lh_x, lh_y
    elif valid_rh:
        mid_hip_x, mid_hip_y = rh_x, rh_y

    # Reference point = centre of torso
    ref_x, ref_y = np.nan, np.nan
    if not np.isnan(mid_shoulder_x) and not np.isnan(mid_hip_x):
        ref_x = (mid_shoulder_x + mid_hip_x) / 2
        ref_y = (mid_shoulder_y + mid_hip_y) / 2
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
        # Cannot compute a reference point — return unchanged
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


def extract_features_from_yolo_result(result, frame_width: int, frame_height: int) -> np.ndarray:
    """Convert a YOLO pose result into a flat normalised keypoint feature vector.

    Only the largest detected person (by bounding-box area) is used.

    Parameters
    ----------
    result:
        Single ``ultralytics`` result object (``results[0]``).
    frame_width, frame_height:
        Dimensions of the source frame used to normalise raw pixel coordinates.

    Returns
    -------
    np.ndarray
        ``float32`` array of shape ``(NUM_FEATURES,)`` in sorted keypoint order.
    """
    frame_features = np.zeros(NUM_FEATURES, dtype=np.float32)

    if result.keypoints is None or len(result.keypoints) == 0:
        return frame_features

    if result.boxes is not None and len(result.boxes) > 0:
        areas = result.boxes.xywh[:, 2] * result.boxes.xywh[:, 3]
        best_idx = int(areas.argmax())
    else:
        best_idx = 0

    kps = result.keypoints[best_idx]
    kps_data = kps.data[0].cpu().numpy()  # (17, 3): x, y, conf

    for yolo_idx, kp_name in YOLO_IDX_TO_NAME.items():
        x_norm = kps_data[yolo_idx, 0] / frame_width
        y_norm = kps_data[yolo_idx, 1] / frame_height
        conf = float(kps_data[yolo_idx, 2])
        x_idx, y_idx, c_idx = get_kpt_indices_training_order(kp_name)
        frame_features[x_idx] = x_norm
        frame_features[y_idx] = y_norm
        frame_features[c_idx] = conf

    return normalize_skeleton_frame(frame_features)


def run_inference(
    frame_bgr: np.ndarray,
    stream_state: StreamState,
    yolo_pose,
    interpreter,
    input_details: list,
    output_details: list,
    device: str = "cuda",
):
    """Process one BGR frame for a given stream and return the fall verdict.

    Updates ``stream_state`` in-place (advances frame counter, updates the
    sliding buffer, records the last-fall frame for cooldown).

    Parameters
    ----------
    frame_bgr:
        OpenCV BGR frame (``np.ndarray``).
    stream_state:
        The :class:`StreamState` instance for this camera stream.
    yolo_pose:
        Loaded ``ultralytics.YOLO`` model.
    interpreter:
        Loaded ``tflite_Interpreter`` instance.
    input_details, output_details:
        Tensor detail dicts returned by ``interpreter.get_*_details()``.
    device:
        Torch device string passed to YOLO (``"cuda"`` or ``"cpu"``).

    Returns
    -------
    tuple[bool, float, object]
        ``(fall_detected, confidence, yolo_result)``
        where ``yolo_result`` is the raw ultralytics result (useful for bbox
        extraction and debug annotation).
    """
    h, w = frame_bgr.shape[:2]
    stream_state.frame_count += 1

    # --- YOLO26 pose inference ---
    results = yolo_pose(frame_bgr, verbose=False, device=device)
    result = results[0]

    # --- Build feature vector and push onto the sliding window ---
    features = extract_features_from_yolo_result(result, w, h)
    stream_state.frame_buffer.append(features)

    fall_detected = False
    confidence = 0.0

    # --- TFLite Transformer inference (only when buffer is full) ---
    if len(stream_state.frame_buffer) == INPUT_TIMESTEPS:
        input_seq = np.array(stream_state.frame_buffer, dtype=np.float32)[np.newaxis, ...]
        interpreter.set_tensor(input_details[0]["index"], input_seq)
        interpreter.invoke()
        confidence = float(interpreter.get_tensor(output_details[0]["index"])[0][0])

        cooldown_ok = (
            stream_state.frame_count - stream_state.last_fall_frame
        ) >= FALL_EVENT_COOLDOWN_FRAMES

        if confidence >= FALL_CONFIDENCE_THRESHOLD and cooldown_ok:
            fall_detected = True
            stream_state.last_fall_frame = stream_state.frame_count

    return fall_detected, confidence, result


def annotate_frame(
    frame_bgr: np.ndarray,
    yolo_result,
    fall_detected: bool,
    confidence: float,
    frame_count: int,
    input_timesteps: int = INPUT_TIMESTEPS,
) -> bytes:
    """Render pose skeleton, bounding box, and fall overlay onto a frame.

    The returned PNG bytes are attached to the OpenNVR ``ResultMessage.frame``
    field so KAI-C stores them as the event snapshot image (shown in the VMS
    alert timeline).  They are also written locally to ``fall_snapshots/``.

    Visual elements
    ---------------
    * YOLO26 pose skeleton — keypoints + limb lines (via ``result.plot()``)
    * Black status bar (top-left) with:
        - Warm-up progress  ``Warming up... (n/30)``
        - Normal status     ``Normal``          (green text)
        - Fall status       ``FALL DETECTED!``  (red text)
        - Confidence score
    * On fall: thick **red** bounding box + ``"FALL"`` label above the person

    Parameters
    ----------
    frame_bgr:
        Raw BGR frame from the camera (``np.ndarray``).
    yolo_result:
        Ultralytics result object returned by ``run_inference()``.
    fall_detected:
        Whether the transformer classified this window as a fall.
    confidence:
        Transformer output probability (0–1).
    frame_count:
        Number of frames processed so far in this stream — used to display
        warm-up progress before the buffer is full.
    input_timesteps:
        Sliding-window length; defaults to the global ``INPUT_TIMESTEPS``.

    Returns
    -------
    bytes
        PNG-encoded annotated frame.
    """
    import cv2  # local import — not needed if running in environments without display

    # Draw YOLO pose keypoints and limb skeleton
    annotated = yolo_result.plot(kpt_radius=4, line_width=2)
    H, W = annotated.shape[:2]

    # ── Status text ──────────────────────────────────────────────────
    if frame_count < input_timesteps:
        status_text   = f"Warming up... ({frame_count}/{input_timesteps})"
        overlay_color = (30, 200, 200)     # yellow
    elif fall_detected:
        status_text   = "FALL DETECTED!"
        overlay_color = (0, 0, 255)        # red
    else:
        status_text   = "Normal"
        overlay_color = (0, 200, 0)        # green

    conf_text = f"Conf: {confidence:.2f}" if frame_count >= input_timesteps else ""

    # Black background bar (top-left) for readability
    cv2.rectangle(annotated, (0, 0), (380, 70), (0, 0, 0), -1)
    cv2.putText(annotated, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, overlay_color, 2)
    cv2.putText(annotated, conf_text, (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    # ── Fall bounding box ────────────────────────────────────────────
    if fall_detected and yolo_result.boxes is not None and len(yolo_result.boxes) > 0:
        areas    = yolo_result.boxes.xywh[:, 2] * yolo_result.boxes.xywh[:, 3]
        best_idx = int(areas.argmax())
        cx, cy, bw, bh = yolo_result.boxes.xywh[best_idx].cpu().numpy()

        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)

        # Thick red rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # "FALL" label above the box
        label_y = max(y1 - 10, 20)
        cv2.putText(annotated, "FALL", (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # ── Encode to PNG ────────────────────────────────────────────────
    success, buf = cv2.imencode(".png", annotated)
    if not success:
        raise RuntimeError("cv2.imencode failed — cannot produce PNG bytes")
    return buf.tobytes()