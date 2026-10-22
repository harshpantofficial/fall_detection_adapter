# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
FallDetectionService — YOLO26-pose + a temporal transformer, one adapter.

Two models, one answer per frame:

1. **Pose** — ``yolo26n-pose.onnx`` on onnxruntime. This is the
   ``adapters/yolo_pose/`` service, inherited rather than copied: the
   letterbox, the NMS, the parameter validation, the §6 WebSocket loop
   and its error envelopes are all that adapter's, so the two cannot
   drift. A YOLO26 export with ``dynamic=True`` comes out in the same
   ``(1, 56, anchors)`` layout as YOLO11 (ultralytics turns ``end2end``
   off for dynamic exports), which is what makes the reuse possible.
2. **Fall classifier** — ``fall_detection_transformer.tflite`` on LiteRT.
   It reads the last 30 frames of ONE person's skeleton (``[1, 30, 51]``)
   and returns the probability that they fell.

The second model is temporal, so this adapter is stateful where the pose
adapter is not:

* **Over the §6 stream** (the path the fall-detection app uses) each
  session owns its window. A session is one camera — the app opens one
  per camera, KAI-C relays it 1:1 — so the window cannot mix cameras,
  and it dies with the socket. It lives in a ContextVar set around the
  inherited ``handle_stream``: each WebSocket runs in its own task and
  ``asyncio.to_thread`` copies the context, so the frame handler finds
  its own session's window without the loop being duplicated here.
* **Over HTTP /infer** a caller that sends ``stream_id`` gets a window
  keyed on it, bounded (``MAX_HTTP_STREAMS``, LRU) and dropped after
  ``HTTP_STREAM_IDLE_S`` of silence. Without ``stream_id`` a call is one
  frame with no history: pose comes back, ``fall.score`` is null.

Faithful to the reference Space (``app.py``) by construction, and
measured: same features (``features.py``), same subject (largest box),
same person floor (0.25, ultralytics' default), same 640 px input, same
0.90 threshold. On the Space's four example clips the first frame at or
above 0.90 is the same frame as the reference on all three falls, and
the non-fall clip peaks at 0.09 in both (see README → Fidelity).

What it does NOT do is decide that an alarm is due: no cooldown, no
debounce. ``fall.detected`` is "this frame's score is over the
threshold"; turning a run of those into one alarm is the app's job.
"""
from __future__ import annotations

import contextvars
import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any

from fastapi import WebSocket

from adapters.fall_detection.features import (
    NUM_FEATURES,
    TIMESTEPS,
    SequenceWindow,
    features_for,
    subject_index,
)
from adapters.yolo_pose.service import (
    MAX_IMAGE_BYTES,
    YoloPoseService,
    _float_param,
)
from opennvr_adapter_sdk import ServiceError
from opennvr_adapter_sdk.contract import (
    ErrorCategory,
    HardwareEvaluationResponse,
    HealthStatus,
    InferResponse,
    ModelInfo,
)

logger = logging.getLogger(__name__)

MODEL_NAME: str = "yolo26n-pose+fall-transformer"
MODEL_FRAMEWORK: str = "onnxruntime+litert"

#: The image sets FALL_WEIGHTS_DIR=/weights and mounts a volume there.
DEFAULT_WEIGHTS_DIR: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "model_weights")
)
POSE_WEIGHTS_FILENAME: str = "yolo26n-pose.onnx"
CLASSIFIER_WEIGHTS_FILENAME: str = "fall_detection_transformer.tflite"

#: The reference runs ultralytics' predict, whose person floor is 0.25 —
#: not the pose adapter's 0.4. Raising it changes which skeleton is the
#: subject on a borderline frame, and the classifier never saw that.
DEFAULT_POSE_CONF: float = 0.25
#: 640, not the pose adapter's 448: measured on the reference clips, 448
#: moves fall onset by up to seven frames; 640 matches it exactly.
DEFAULT_POSE_IMGSZ: int = 640
#: The reference's FALL_CONFIDENCE_THRESHOLD.
DEFAULT_FALL_THRESHOLD: float = 0.90

#: HTTP windows: how many callers may keep one, and how long one survives
#: silence. A stream that goes quiet for this long has a gap in its
#: window anyway, so starting over is the honest thing to do.
MAX_HTTP_STREAMS: int = 64
HTTP_STREAM_IDLE_S: float = 30.0

#: The current §6 session's window, set by ``handle_stream``.
_SESSION_WINDOW: contextvars.ContextVar[SequenceWindow | None] = contextvars.ContextVar(
    "fall_detection_session_window", default=None)


class FallDetectionService(YoloPoseService):
    """Pose (inherited) plus a per-stream window and the fall classifier."""

    def __init__(self, weights_dir: str | None = None) -> None:
        weights_dir = weights_dir or os.getenv("FALL_WEIGHTS_DIR", DEFAULT_WEIGHTS_DIR)
        super().__init__(weights_path=os.path.join(
            weights_dir, os.getenv("FALL_POSE_WEIGHTS", POSE_WEIGHTS_FILENAME)))
        # The pose service's first-boot fetch reads YOLO_POSE_MODEL_URL;
        # this adapter never fetches — both files are mounted.
        self._model_url = ""
        self._classifier_path = os.path.join(
            weights_dir, os.getenv("FALL_CLASSIFIER_WEIGHTS", CLASSIFIER_WEIGHTS_FILENAME))
        self._interpreter: Any | None = None
        self._input_index: int = 0
        self._output_index: int = 0
        # LiteRT interpreters are not thread-safe, and streams run
        # their frames on worker threads. The classifier is ~1 ms; a
        # lock costs nothing next to the pose model.
        self._interpreter_lock = threading.Lock()
        self._classifier_state: HealthStatus = HealthStatus.LOADING
        self._classifier_error: str | None = None
        self._pose_conf = _env_float("FALL_POSE_CONF", DEFAULT_POSE_CONF)
        self._pose_imgsz = int(_env_float("FALL_POSE_IMGSZ", DEFAULT_POSE_IMGSZ))
        self._threshold = _env_float("FALL_THRESHOLD", DEFAULT_FALL_THRESHOLD)
        self._http_windows: OrderedDict[str, tuple[float, SequenceWindow]] = OrderedDict()
        self._http_lock = threading.Lock()

    # ── lifecycle ──────────────────────────────────────────────────

    def load(self) -> None:
        """Both models, or not ready. Idempotent."""
        super().load()
        if self._classifier_state == HealthStatus.OK:
            return
        self.metrics.register_counter(
            "adapter_fall_scores_total",
            "Frames the fall classifier scored, by outcome.",
            label_key="outcome", allowed_values=["fall", "no_fall"])
        try:
            if not os.path.isfile(self._classifier_path):
                raise FileNotFoundError(
                    f"fall classifier not found at {self._classifier_path}")
            interpreter = _make_interpreter(self._classifier_path)
            interpreter.allocate_tensors()
            inp = interpreter.get_input_details()[0]
            out = interpreter.get_output_details()[0]
            shape = tuple(int(v) for v in inp["shape"])
            if shape[1:] != (TIMESTEPS, NUM_FEATURES):
                raise ValueError(
                    f"classifier expects input {shape}, this adapter feeds "
                    f"(1, {TIMESTEPS}, {NUM_FEATURES}) — is "
                    f"{self._classifier_path} the fall transformer?")
            self._interpreter = interpreter
            self._input_index = inp["index"]
            self._output_index = out["index"]
            # Warm-up: the first invoke pays for delegate setup.
            self._classify(_zeros_batch())
            self._classifier_state = HealthStatus.OK
            self._classifier_error = None
            logger.info("fall classifier ready: %s", self._classifier_path)
        except Exception as exc:
            self._classifier_state = HealthStatus.ERROR
            self._classifier_error = str(exc)
            logger.exception("fall classifier failed to load %s", self._classifier_path)

    def is_ready(self) -> bool:
        return super().is_ready() and self._classifier_state == HealthStatus.OK

    def fingerprint(self) -> str | None:
        """One fingerprint over BOTH files: swapping either model is a
        different adapter, and §11.3 drift detection has to see it."""
        try:
            return self._compute_fingerprint()
        except OSError:
            return self._fingerprint_cache

    def _compute_fingerprint(self) -> str:
        # The pose loader calls this too, before the classifier is
        # checked: a missing classifier must surface as the classifier's
        # load error, not as a pose failure. Not ready either way.
        digest = hashlib.sha256()
        for path in (self._weights_path, self._classifier_path):
            if path == self._classifier_path and not os.path.isfile(path):
                continue
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    def _weights_size_mb(self) -> float | None:
        try:
            size = os.path.getsize(self._weights_path) + os.path.getsize(self._classifier_path)
        except OSError:
            return None
        return round(size / (1024 * 1024), 2)

    def _adapter_model_version(self) -> str:
        return f"{MODEL_FRAMEWORK}/{MODEL_NAME}"

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=MODEL_NAME,
            version=self._adapter_model_version(),
            framework=MODEL_FRAMEWORK,
            size_mb=self._weights_size_mb(),
            modalities_in=["image"],
            modalities_out=["keypoints", "classification"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        report = super().hardware_evaluation()
        details = dict(report.details or {})
        details.update({
            "classifier_path": self._classifier_path,
            "classifier_loaded": self._classifier_state == HealthStatus.OK,
            "default_imgsz": self._pose_imgsz,
            "fall_threshold": self._threshold,
            "window_frames": TIMESTEPS,
        })
        update: dict[str, Any] = {"details": details}
        if self._classifier_state == HealthStatus.ERROR:
            from opennvr_adapter_sdk.contract import HardwareVerdict
            update["verdict"] = HardwareVerdict.BLOCKED
            update["reasoning"] = f"Fall classifier failed to load: {self._classifier_error}"
        return report.model_copy(update=update)

    # ── §6 stream: one window per session ──────────────────────────

    async def handle_stream(self, websocket: WebSocket) -> None:
        token = _SESSION_WINDOW.set(SequenceWindow())
        try:
            await super().handle_stream(websocket)
        finally:
            _SESSION_WINDOW.reset(token)

    # ── inference ──────────────────────────────────────────────────

    def _infer_image_bytes(self, image_bytes: bytes, params: dict[str, Any]) -> InferResponse:
        """Pose (the inherited path, every check included), then the
        subject's features into this stream's window, then a score once
        the window is full."""
        if self._classifier_state != HealthStatus.OK and super().is_ready():
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code=("weights_missing" if self._classifier_state == HealthStatus.ERROR
                      else "fall_detection.model_loading"),
                message=self._classifier_error or "Fall classifier still loading.",
                transient=self._classifier_state == HealthStatus.LOADING,
                http_status=503,
            )
        params = dict(params)
        stream_id = params.pop("stream_id", None)
        reset = bool(params.pop("reset", False))
        threshold = _float_param(params, "threshold", self._threshold, 0.0, 1.0)
        params.pop("threshold", None)
        # The pose adapter's own defaults are tuned for wrist geometry;
        # these are the ones the classifier was fed.
        params.setdefault("conf", self._pose_conf)
        params.setdefault("imgsz", self._pose_imgsz)

        pose = super()._infer_image_bytes(image_bytes, params)
        start = time.monotonic()
        result = dict(pose.result)
        persons = result.get("persons") or []
        dims = result.get("frame_dimensions") or {}
        width, height = int(dims.get("w") or 0), int(dims.get("h") or 0)

        window = self._window_for(stream_id, reset)
        subject = subject_index(persons)
        score: float | None = None
        if window is not None and width > 0 and height > 0:
            keypoints = persons[subject]["keypoints"] if subject is not None else None
            window.push(features_for(keypoints, width, height))
            if window.full:
                score = self._classify(window.batch())
        detected = score is not None and score >= threshold
        if score is not None:
            try:
                self.metrics.inc_counter("adapter_fall_scores_total",
                                         label_value="fall" if detected else "no_fall")
            except Exception:  # pragma: no cover - metrics must never break infer
                logger.debug("fall metrics recording failed", exc_info=True)

        result["fall"] = {
            "score": None if score is None else round(score, 4),
            "detected": detected,
            "threshold": threshold,
            # Which person the score is about: the largest box, as the
            # model was trained. null when nobody is in the frame.
            "subject_index": subject,
            # Frames in this stream's window. The score is null until it
            # reaches `size` — a new stream is blind for that long.
            "window": {"filled": window.filled if window else 0, "size": TIMESTEPS},
            "stateful": window is not None,
        }
        return InferResponse(
            model_name=MODEL_NAME,
            model_version=self._adapter_model_version(),
            inference_ms=pose.inference_ms + int((time.monotonic() - start) * 1000),
            result=result,
        )

    def _window_for(self, stream_id: Any, reset: bool) -> SequenceWindow | None:
        session = _SESSION_WINDOW.get()
        if session is not None:
            if reset:
                session.clear()
            return session
        if stream_id is None or str(stream_id).strip() == "":
            return None
        key = str(stream_id)[:128]
        now = time.monotonic()
        with self._http_lock:
            for stale in [k for k, (at, _w) in self._http_windows.items()
                          if now - at > HTTP_STREAM_IDLE_S]:
                del self._http_windows[stale]
            entry = self._http_windows.pop(key, None)
            window = entry[1] if entry is not None and not reset else SequenceWindow()
            self._http_windows[key] = (now, window)
            while len(self._http_windows) > MAX_HTTP_STREAMS:
                self._http_windows.popitem(last=False)
            return window

    def _classify(self, batch: Any) -> float:
        with self._interpreter_lock:
            self._interpreter.set_tensor(self._input_index, batch)
            self._interpreter.invoke()
            out = self._interpreter.get_tensor(self._output_index)
        return float(out.reshape(-1)[0])


def _zeros_batch():
    import numpy as np

    return np.zeros((1, TIMESTEPS, NUM_FEATURES), dtype=np.float32)


def _make_interpreter(path: str):
    """LiteRT (``ai-edge-litert``), which is what the reference uses;
    ``tflite_runtime`` as a fallback for hosts that already have it."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tflite_runtime.interpreter import Interpreter  # type: ignore[no-redef]
    return Interpreter(model_path=path)


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


__all__ = ["MAX_IMAGE_BYTES", "MODEL_NAME", "FallDetectionService"]
