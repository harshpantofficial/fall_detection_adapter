"""
FallTransformer — keypoint-based fall classification engine.

Analyzes human pose keypoints (COCO-17 format) to detect falls and posture states:
- "standing": Upright posture, torso angle near vertical.
- "falling": Transitionary tilt / acute torso angle.
- "fallen": Horizontal posture on floor / wide aspect ratio / low center of mass.
- "unknown": Insufficient visible keypoints to assess posture reliably.

Supports both:
1. Pure geometric keypoint analysis (torso vector, aspect ratio, vertical elevation).
2. Optional sequence-based TFLite transformer model (30 timesteps) when weights are provided.
"""
from __future__ import annotations

import logging
import math
import os
from collections import deque
from typing import Any

from adapters.fall_detection.coco_keypoints import (
    LEFT_ANKLE_IDX,
    LEFT_HIP_IDX,
    LEFT_KNEE_IDX,
    LEFT_SHOULDER_IDX,
    NOSE_IDX,
    RIGHT_ANKLE_IDX,
    RIGHT_HIP_IDX,
    RIGHT_KNEE_IDX,
    RIGHT_SHOULDER_IDX,
)

logger = logging.getLogger(__name__)

# Fall classification thresholds
TORSO_ANGLE_FALLEN_DEG = 60.0
TORSO_ANGLE_FALLING_DEG = 35.0
MIN_CONFIDENCE_THRESHOLD = 0.3
ASPECT_RATIO_FALLEN_THRESHOLD = 1.15


class FallTransformer:
    """Classifies human posture and fall conditions from COCO keypoints."""

    def __init__(self, tflite_model_path: str | None = None) -> None:
        self._tflite_path = tflite_model_path
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._track_buffers: dict[str, deque[list[float]]] = {}
        self._init_tflite()

    def _init_tflite(self) -> None:
        if not self._tflite_path or not os.path.isfile(self._tflite_path):
            return

        try:
            # Check for available TFLite runtime
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
            logger.info("FallTransformer: TFLite transformer loaded from %s", self._tflite_path)
        except Exception as exc:
            logger.warning(
                "FallTransformer: could not load TFLite model (%s); falling back to geometric engine: %s",
                self._tflite_path,
                exc,
            )
            self._interpreter = None

    def analyze_person(
        self,
        person: dict[str, Any],
        frame_width: int,
        frame_height: int,
        track_id: str | None = None,
    ) -> dict[str, Any]:
        """Classify posture for a single detected person.

        Parameters
        ----------
        person:
            Dict containing:
            - "bbox": [x1, y1, x2, y2] in pixels
            - "score": detection score float
            - "keypoints": list of 17 [x, y, conf] keypoint lists
        frame_width, frame_height:
            Source image dimensions for normalization
        track_id:
            Optional tracking ID for maintaining temporal sequence buffers

        Returns
        -------
        dict with added fields:
            - fall_state: 'standing' | 'falling' | 'fallen' | 'unknown'
            - torso_angle_deg: angle of torso vector relative to vertical (0-90)
            - aspect_ratio: bbox width / height
            - fall_confidence: confidence score [0.0, 1.0]
        """
        kpts = person.get("keypoints", [])
        bbox = person.get("bbox", [0, 0, 0, 0])

        if len(kpts) < 17:
            return {
                "fall_state": "unknown",
                "torso_angle_deg": 0.0,
                "aspect_ratio": 1.0,
                "fall_confidence": 0.0,
            }

        # Calculate bounding box dimensions
        x1, y1, x2, y2 = bbox
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        aspect_ratio = round(bw / bh, 3)

        # Extract shoulders and hips
        ls = kpts[LEFT_SHOULDER_IDX]
        rs = kpts[RIGHT_SHOULDER_IDX]
        lh = kpts[LEFT_HIP_IDX]
        rh = kpts[RIGHT_HIP_IDX]

        valid_shoulders = ls[2] >= MIN_CONFIDENCE_THRESHOLD or rs[2] >= MIN_CONFIDENCE_THRESHOLD
        valid_hips = lh[2] >= MIN_CONFIDENCE_THRESHOLD or rh[2] >= MIN_CONFIDENCE_THRESHOLD

        if not (valid_shoulders and valid_hips):
            # Not enough keypoints to judge torso geometry
            # Fall back to aspect ratio if person is heavily flattened
            if aspect_ratio >= 1.5 and person.get("score", 0.0) >= 0.5:
                return {
                    "fall_state": "fallen",
                    "torso_angle_deg": 80.0,
                    "aspect_ratio": aspect_ratio,
                    "fall_confidence": round(float(person.get("score", 0.5)), 3),
                }
            return {
                "fall_state": "unknown",
                "torso_angle_deg": 0.0,
                "aspect_ratio": aspect_ratio,
                "fall_confidence": 0.0,
            }

        # Mid-shoulder coordinates
        if ls[2] >= MIN_CONFIDENCE_THRESHOLD and rs[2] >= MIN_CONFIDENCE_THRESHOLD:
            mid_shoulder_x = (ls[0] + rs[0]) / 2.0
            mid_shoulder_y = (ls[1] + rs[1]) / 2.0
            shoulder_conf = (ls[2] + rs[2]) / 2.0
        elif ls[2] >= MIN_CONFIDENCE_THRESHOLD:
            mid_shoulder_x, mid_shoulder_y, shoulder_conf = ls[0], ls[1], ls[2]
        else:
            mid_shoulder_x, mid_shoulder_y, shoulder_conf = rs[0], rs[1], rs[2]

        # Mid-hip coordinates
        if lh[2] >= MIN_CONFIDENCE_THRESHOLD and rh[2] >= MIN_CONFIDENCE_THRESHOLD:
            mid_hip_x = (lh[0] + rh[0]) / 2.0
            mid_hip_y = (lh[1] + rh[1]) / 2.0
            hip_conf = (lh[2] + rh[2]) / 2.0
        elif lh[2] >= MIN_CONFIDENCE_THRESHOLD:
            mid_hip_x, mid_hip_y, hip_conf = lh[0], lh[1], lh[2]
        else:
            mid_hip_x, mid_hip_y, hip_conf = rh[0], rh[1], rh[2]

        # Vector from hip to shoulder:
        # In image coordinates, y increases downward.
        # An upright person has shoulders ABOVE hips (shoulder_y < hip_y, dy < 0).
        dx = mid_shoulder_x - mid_hip_x
        dy = mid_hip_y - mid_shoulder_y  # positive when shoulders are above hips

        # Angle from vertical (0° is perfectly upright, 90° is horizontal)
        angle_rad = math.atan2(abs(dx), max(1e-5, abs(dy)))
        torso_angle_deg = round(math.degrees(angle_rad), 1)

        # Inversion check: if dy < -15 pixels, shoulders are below hips (upside down or laying down)
        if dy < -15.0:
            torso_angle_deg = max(torso_angle_deg, 75.0)

        # Base confidence from keypoint visibility
        geom_conf = min(shoulder_conf, hip_conf)

        # Posture classification rule
        is_flat_bbox = aspect_ratio >= ASPECT_RATIO_FALLEN_THRESHOLD
        is_flat_angle = torso_angle_deg >= TORSO_ANGLE_FALLEN_DEG

        if is_flat_angle or (is_flat_bbox and torso_angle_deg >= 45.0):
            fall_state = "fallen"
            fall_conf = round(float(geom_conf * (0.85 if is_flat_angle and is_flat_bbox else 0.7)), 3)
        elif torso_angle_deg >= TORSO_ANGLE_FALLING_DEG:
            fall_state = "falling"
            fall_conf = round(float(geom_conf * 0.6), 3)
        else:
            fall_state = "standing"
            fall_conf = 0.0

        # Optional TFLite sequence transformer refinement
        if self._interpreter is not None and track_id:
            tflite_conf = self._evaluate_sequence_tflite(kpts, frame_width, frame_height, track_id)
            if tflite_conf is not None:
                if tflite_conf >= 0.8:
                    fall_state = "fallen"
                    fall_conf = max(fall_conf, round(tflite_conf, 3))
                elif tflite_conf >= 0.5 and fall_state == "standing":
                    fall_state = "falling"
                    fall_conf = max(fall_conf, round(tflite_conf, 3))

        return {
            "fall_state": fall_state,
            "torso_angle_deg": torso_angle_deg,
            "aspect_ratio": aspect_ratio,
            "fall_confidence": fall_conf,
        }

    def _evaluate_sequence_tflite(
        self,
        keypoints: list[list[float]],
        frame_width: int,
        frame_height: int,
        track_id: str,
    ) -> float | None:
        """Run the 30-timestep TFLite transformer if buffer is full."""
        try:
            import numpy as np

            # Normalize keypoints to flat vector
            flat_feat = []
            for kp in keypoints:
                flat_feat.extend([kp[0] / max(1, frame_width), kp[1] / max(1, frame_height), kp[2]])

            buf = self._track_buffers.setdefault(track_id, deque(maxlen=30))
            buf.append(flat_feat)

            if len(buf) == 30:
                input_data = np.array(buf, dtype=np.float32)[np.newaxis, ...]
                self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
                self._interpreter.invoke()
                output_data = self._interpreter.get_tensor(self._output_details[0]["index"])
                return float(output_data[0][0])
        except Exception as exc:
            logger.debug("TFLite sequence inference failed: %s", exc)
        return None