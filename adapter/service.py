"""
FallDetectionService — OpenNVR AI Adapter for pose estimation and fall detection.

Combines YOLO26 pose estimation (Ultralytics, CUDA/CPU) with the sequence-based
FallTransformer TFLite engine. Implements the OpenNVR Adapter Contract v1
with full HTTP (/infer) and WebSocket (/infer/stream) streaming capabilities.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import platform
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from adapters.fall_detection.coco_keypoints import COCO_KEYPOINTS, COCO_SKELETON
from adapters.fall_detection.fall_transformer import (
    FALL_CONFIDENCE_THRESHOLD,
    INPUT_TIMESTEPS,
    FallTransformer,
)
from opennvr_adapter_sdk import AdapterService, BODY_BYTES_KEY, ServiceError
from opennvr_adapter_sdk.contract import (
    ErrorCategory,
    FrameMessage,
    FrameTransport,
    HandshakeAckMessage,
    HandshakeMessage,
    HardwareEvaluationResponse,
    HardwareVerdict,
    HealthStatus,
    InferResponse,
    ModelInfo,
    ResultMessage,
    StreamCloseCode,
)
from opennvr_adapter_sdk.model_fetch import ensure_model_file

logger = logging.getLogger(__name__)

MODEL_FRAMEWORK: str = "ultralytics+tflite"
MODEL_NAME: str = "yolo26n-pose-fall"
MAX_IMAGE_BYTES: int = 8 * 1024 * 1024  # 8 MiB max request size

DEFAULT_WEIGHTS_DIR: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "fall_detection")
)
WEIGHTS_FILENAME: str = "yolo26n-pose.pt"
TFLITE_FILENAME: str = "fall_detection_transformer.tflite"
MODEL_URL_ENV: str = "YOLO_POSE_MODEL_URL"
TFLITE_MODEL_ENV: str = "FALL_TRANSFORMER_TFLITE_PATH"

DEFAULT_CONF: float = 0.35
DEFAULT_IOU: float = 0.45
DEFAULT_IMGSZ: int = 640
IMGSZ_STRIDE: int = 32
MIN_IMGSZ: int = 160
MAX_IMGSZ: int = 1280
DEFAULT_MAX_PERSONS: int = 20
ABSOLUTE_MAX_PERSONS: int = 100
KEYPOINT_VISIBLE_CONF: float = 0.4

FALL_STATES = ("standing", "falling", "fallen", "unknown")


def _find_weights_file(env_var: str, default_filename: str) -> str:
    explicit = os.getenv(env_var)
    if explicit and os.path.exists(explicit):
        return explicit

    candidates = [
        explicit,
        os.path.join(os.getenv("YOLO_POSE_WEIGHTS_DIR", DEFAULT_WEIGHTS_DIR), default_filename),
        os.path.join(os.path.dirname(__file__), default_filename),
        os.path.join(os.path.dirname(__file__), "..", "..", "fall_detection", default_filename),
        os.path.join(os.path.dirname(__file__), "..", "fall_detection", default_filename),
        os.path.join("/weights", default_filename),
        os.path.join(os.getcwd(), default_filename),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return explicit or os.path.join(DEFAULT_WEIGHTS_DIR, default_filename)


class FallDetectionService(AdapterService):
    """Façade around YOLO26-pose (Ultralytics) and FallTransformer (TFLite)."""

    def __init__(self, weights_path: str | None = None) -> None:
        self._weights_path = weights_path or _find_weights_file("YOLO_POSE_WEIGHTS_PATH", WEIGHTS_FILENAME)
        self._model_url: str = os.getenv(MODEL_URL_ENV, "")
        self._tflite_path: str = _find_weights_file(TFLITE_MODEL_ENV, TFLITE_FILENAME)

        self._yolo: Any | None = None
        self._device: str = "cpu"
        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint_cache: str | None = None
        self._gpu_in_use: bool = False
        self._fall_transformer: FallTransformer | None = None

    def load(self) -> None:
        """Eagerly load YOLO26 weights and initialize the FallTransformer."""
        if self._load_state == HealthStatus.OK:
            return

        self.metrics.register_counter(
            "adapter_fall_state_total",
            "Person fall states detected.",
            label_key="state",
            allowed_values=FALL_STATES,
        )
        self.metrics.register_counter(
            "adapter_pose_persons_total",
            "Persons returned with a pose.",
        )
        self.metrics.register_counter(
            "adapter_pose_keypoints_visible_total",
            "Keypoints returned at or above visibility threshold.",
            label_key="keypoint",
            allowed_values=COCO_KEYPOINTS,
        )

        try:
            import torch
            from ultralytics import YOLO

            ensure_model_file(
                self._weights_path,
                self._model_url,
                label=f"{MODEL_NAME} weights",
                logger=logger,
            )

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._gpu_in_use = (self._device == "cuda")

            logger.info("Loading YOLO26 pose model from %s on %s...", self._weights_path, self._device)
            self._yolo = YOLO(self._weights_path)

            logger.info("Loading FallTransformer from %s...", self._tflite_path)
            self._fall_transformer = FallTransformer(
                tflite_model_path=self._tflite_path,
                timesteps=INPUT_TIMESTEPS,
                threshold=FALL_CONFIDENCE_THRESHOLD,
            )

            self._fingerprint_cache = self._compute_fingerprint()
            self._warm_up()

            self._load_state = HealthStatus.OK
            self._load_error = None
            logger.info(
                "FallDetectionService ready: weights=%s gpu=%s device=%s",
                self._weights_path,
                self._gpu_in_use,
                self._device,
            )
        except Exception as exc:
            self._load_state = HealthStatus.ERROR
            self._load_error = str(exc)
            logger.exception("FallDetectionService failed to load weights %s: %s", self._weights_path, exc)

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    def fingerprint(self) -> str | None:
        try:
            return self._compute_fingerprint()
        except OSError:
            return self._fingerprint_cache

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=MODEL_NAME,
            version=self._adapter_model_version(),
            framework=MODEL_FRAMEWORK,
            size_mb=self._weights_size_mb(),
            modalities_in=["image"],
            modalities_out=["keypoints", "persons", "fall_state"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        cpu_count = os.cpu_count() or 0
        device_name = "CPU"
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

        if self._load_state == HealthStatus.OK:
            if self._gpu_in_use:
                verdict = HardwareVerdict.OK
                reasoning = f"NVIDIA CUDA GPU detected and active ({device_name}). YOLO26-pose and Transformer loaded."
            elif cpu_count >= 4:
                verdict = HardwareVerdict.OK
                reasoning = f"Running on CPU with {cpu_count} cores."
            else:
                verdict = HardwareVerdict.WARN
                reasoning = f"Only {cpu_count} CPU cores and no CUDA device."
        elif self._load_state == HealthStatus.LOADING:
            verdict = HardwareVerdict.WARN
            reasoning = "Model is still loading."
        else:
            verdict = HardwareVerdict.BLOCKED
            reasoning = f"Weights failed to load: {self._load_error}"

        return HardwareEvaluationResponse(
            verdict=verdict,
            reasoning=reasoning,
            checked_at=datetime.now(timezone.utc),
            details={
                "gpu_required": True,
                "gpu_in_use": self._gpu_in_use,
                "device": self._device,
                "device_name": device_name,
                "cpu_count": cpu_count,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "weights_path": self._weights_path,
                "tflite_path": self._tflite_path,
            },
        )

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        """SDK /infer entry point."""
        image_bytes = payload.get(BODY_BYTES_KEY)
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message="Frame bytes are required.",
                transient=False,
                http_status=400,
            )
        params = {k: v for k, v in payload.items() if k != BODY_BYTES_KEY}
        return self._infer_image_bytes(bytes(image_bytes), params)

    async def handle_stream(self, websocket: WebSocket) -> None:
        """OpenNVR §6 WebSocket streaming inference protocol."""
        await websocket.accept()
        session_id = uuid.uuid4().hex

        # Handshake
        try:
            first_raw = await websocket.receive_text()
            handshake = HandshakeMessage.model_validate(json.loads(first_raw))
        except (WebSocketDisconnect, json.JSONDecodeError, Exception) as exc:
            logger.info("Stream handshake rejected: %s", exc)
            await websocket.close(
                code=StreamCloseCode.POLICY_REFUSED.value,
                reason="bad handshake",
            )
            return

        ack = HandshakeAckMessage(
            frame_transport=FrameTransport.WEBSOCKET,
            result_sink="websocket",
            max_inflight=1,
            session_id=session_id,
        )
        await websocket.send_text(json.dumps(ack.model_dump(mode="json")))

        if not self.is_ready():
            await websocket.close(
                code=StreamCloseCode.MODEL_ERROR.value,
                reason="model not loaded",
            )
            return

        stream_id = handshake.client_id or "default"
        logger.info("Stream session opened: session_id=%s stream_id=%s", session_id, stream_id)

        paused = False
        frames_done = 0
        session_start = time.monotonic()

        while True:
            try:
                msg = await websocket.receive()
            except WebSocketDisconnect:
                logger.info("Stream client disconnected: session_id=%s", session_id)
                return

            if msg.get("type") == "websocket.disconnect":
                return

            text = msg.get("text")
            if text is not None:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    await websocket.close(
                        code=StreamCloseCode.POLICY_REFUSED.value,
                        reason="non-JSON control message",
                    )
                    return

                msg_type = payload.get("type")
                if msg_type == "close":
                    return
                if msg_type == "pause":
                    paused = True
                    continue
                if msg_type == "resume":
                    paused = False
                    continue
                if msg_type == "stats":
                    gauges = self.metrics.gauges()
                    elapsed = max(time.monotonic() - session_start, 1e-6)
                    await websocket.send_text(json.dumps({
                        "type": "stats",
                        "inflight": gauges["inflight"],
                        "queue_depth": gauges["queue_depth"],
                        "fps": round(frames_done / elapsed, 2),
                    }))
                    continue
                if msg_type == "frame":
                    try:
                        frame_meta = FrameMessage.model_validate(payload)
                    except Exception:
                        await websocket.close(
                            code=StreamCloseCode.POLICY_REFUSED.value,
                            reason="bad frame metadata",
                        )
                        return

                    try:
                        binary_msg = await websocket.receive()
                    except WebSocketDisconnect:
                        return
                    frame_bytes = binary_msg.get("bytes")
                    if not isinstance(frame_bytes, (bytes, bytearray)) or not frame_bytes:
                        await websocket.close(
                            code=StreamCloseCode.POLICY_REFUSED.value,
                            reason="frame metadata must be followed by binary frame",
                        )
                        return

                    if paused:
                        continue

                    metrics = self.metrics
                    metrics.inc_inflight()
                    try:
                        result_dict = await asyncio.to_thread(
                            self._infer_frame_for_stream,
                            bytes(frame_bytes),
                            seq=frame_meta.seq,
                            ts_ms=frame_meta.ts_ms,
                            stream_id=stream_id,
                        )
                        await websocket.send_text(json.dumps(result_dict))
                        latency_seconds = result_dict.get("inference_ms", 0) / 1000.0
                        metrics.record_infer("ok", latency_seconds)
                        frames_done += 1
                    finally:
                        metrics.dec_inflight()
                    continue

                await websocket.close(
                    code=StreamCloseCode.POLICY_REFUSED.value,
                    reason=f"unknown message type: {msg_type}",
                )
                return

            if msg.get("bytes") is not None:
                await websocket.close(
                    code=StreamCloseCode.POLICY_REFUSED.value,
                    reason="binary frame received without preceding frame metadata",
                )
                return

    def _infer_image_bytes(
        self,
        image_bytes: bytes,
        params: dict[str, Any],
    ) -> InferResponse:
        """Run YOLO26-pose estimation and FallTransformer classification."""
        if self._load_state != HealthStatus.OK:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="fall_detection.model_loading",
                message=self._load_error or "Model still loading.",
                transient=(self._load_state == HealthStatus.LOADING),
                http_status=503,
            )

        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=f"Frame exceeds {MAX_IMAGE_BYTES} bytes.",
                transient=False,
                http_status=413,
            )

        conf = _float_param(params, "conf", DEFAULT_CONF, 0.0, 1.0, aliases=("confidence_threshold",))
        iou = _float_param(params, "iou", DEFAULT_IOU, 0.0, 1.0, aliases=("iou_threshold",))
        imgsz = _imgsz_param(params)
        max_persons = _int_param(params, "max_persons", DEFAULT_MAX_PERSONS, 1, ABSOLUTE_MAX_PERSONS)
        stream_id = str(params.get("stream_id") or params.get("camera_id") or "default")

        start = time.monotonic()
        try:
            img, width, height = _decode_image(image_bytes)
            results = self._yolo(
                img,
                verbose=False,
                device=self._device,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
            )
            yolo_res = results[0]

            persons, best_kpts, _ = self._extract_persons(yolo_res, max_persons)

            # Sequence fall transformer classification
            fall_result = {"fall_detected": False, "fall_confidence": 0.0, "fall_state": "standing", "torso_angle_deg": 0.0}
            if self._fall_transformer:
                fall_result = self._fall_transformer.analyze_frame(
                    best_kpts,
                    frame_width=width,
                    frame_height=height,
                    stream_id=stream_id,
                )

            fall_detected = bool(fall_result["fall_detected"])
            fall_conf = float(fall_result["fall_confidence"])
            fall_state = str(fall_result["fall_state"])
            torso_angle = float(fall_result["torso_angle_deg"])

            for p in persons:
                p["fall_state"] = fall_state
                p["fall_confidence"] = fall_conf
                p["confidence"] = fall_conf if fall_detected else p["score"]
                p["torso_angle_deg"] = torso_angle

        except DecodeError as exc:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=str(exc),
                transient=False,
                http_status=400,
            ) from exc
        except ServiceError:
            raise
        except Exception as exc:
            logger.exception("Inference failed")
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="inference_runtime_crash",
                message=f"Inference failed: {exc}",
                transient=False,
                http_status=500,
            ) from exc

        inference_ms = int((time.monotonic() - start) * 1000)

        # Record domain metrics
        try:
            self.metrics.inc_counter("adapter_pose_persons_total", len(persons))
            for p in persons:
                state = p.get("fall_state", "unknown")
                if state in FALL_STATES:
                    self.metrics.inc_counter("adapter_fall_state_total", label_value=state)
                for k_idx, kpt in enumerate(p.get("keypoints", [])):
                    if k_idx < len(COCO_KEYPOINTS) and kpt[2] >= KEYPOINT_VISIBLE_CONF:
                        self.metrics.inc_counter(
                            "adapter_pose_keypoints_visible_total",
                            label_value=COCO_KEYPOINTS[k_idx],
                        )
        except Exception:
            pass

        return InferResponse(
            model_name=MODEL_NAME,
            model_version=self._adapter_model_version(),
            inference_ms=inference_ms,
            result={
                "persons": persons,
                "fall_detected": fall_detected,
                "fall_confidence": fall_conf,
                "keypoint_names": list(COCO_KEYPOINTS),
                "frame_dimensions": {"w": width, "h": height},
            },
        )

    def _extract_persons(
        self,
        yolo_result: Any,
        max_persons: int,
    ) -> tuple[list[dict[str, Any]], np.ndarray | None, np.ndarray | None]:
        """Extract detected persons, finding the largest person for fall analysis."""
        persons: list[dict[str, Any]] = []
        best_kpts: np.ndarray | None = None
        best_box: np.ndarray | None = None
        max_area: float = -1.0

        if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
            return persons, best_kpts, best_box

        boxes_xyxy = yolo_result.boxes.xyxy.cpu().numpy()
        scores = yolo_result.boxes.conf.cpu().numpy()
        kpts_data = None
        if yolo_result.keypoints is not None:
            kpts_data = yolo_result.keypoints.data.cpu().numpy()

        for idx in range(min(len(boxes_xyxy), max_persons)):
            box = boxes_xyxy[idx]
            area = float((box[2] - box[0]) * (box[3] - box[1]))

            kpts_for_person = kpts_data[idx] if kpts_data is not None and idx < len(kpts_data) else None

            if area > max_area and kpts_for_person is not None:
                max_area = area
                best_kpts = kpts_for_person
                best_box = box

            formatted_kpts: list[list[float]] = []
            if kpts_for_person is not None:
                for k in kpts_for_person:
                    formatted_kpts.append([
                        round(float(k[0]), 1),
                        round(float(k[1]), 1),
                        round(float(k[2]), 4),
                    ])

            persons.append({
                "track_id": str(idx),
                "bbox": [
                    round(float(box[0]), 1),
                    round(float(box[1]), 1),
                    round(float(box[2]), 1),
                    round(float(box[3]), 1),
                ],
                "score": round(float(scores[idx]), 4),
                "keypoints": formatted_kpts,
            })

        return persons, best_kpts, best_box

    def _infer_frame_for_stream(
        self,
        image_bytes: bytes,
        seq: int,
        ts_ms: int,
        stream_id: str = "default",
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params = params or {}
        params["stream_id"] = stream_id
        try:
            infer = self._infer_image_bytes(image_bytes, params)
            return ResultMessage(
                seq=seq,
                ts_ms=ts_ms,
                inference_ms=infer.inference_ms,
                result=infer.result,
            ).model_dump(mode="json")
        except ServiceError as exc:
            envelope = exc.envelope().model_dump(mode="json")
            return ResultMessage(
                seq=seq,
                ts_ms=ts_ms,
                inference_ms=0,
                result=envelope,
            ).model_dump(mode="json")
        except Exception as exc:
            envelope = ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="inference_crash",
                message=str(exc),
                transient=False,
                http_status=500,
            ).envelope().model_dump(mode="json")
            return ResultMessage(
                seq=seq,
                ts_ms=ts_ms,
                inference_ms=0,
                result=envelope,
            ).model_dump(mode="json")

    def _warm_up(self) -> None:
        try:
            blank = np.zeros((DEFAULT_IMGSZ, DEFAULT_IMGSZ, 3), dtype=np.uint8)
            self._yolo(blank, verbose=False, device=self._device, imgsz=DEFAULT_IMGSZ)
        except Exception:
            pass

    def _adapter_model_version(self) -> str:
        return f"{MODEL_FRAMEWORK}/{MODEL_NAME}"

    def _weights_size_mb(self) -> float | None:
        try:
            total_size = os.path.getsize(self._weights_path)
            if self._tflite_path and os.path.exists(self._tflite_path):
                total_size += os.path.getsize(self._tflite_path)
            return round(total_size / (1024 * 1024), 2)
        except OSError:
            return None

    def _compute_fingerprint(self) -> str:
        if not os.path.exists(self._weights_path):
            return "sha256:unavailable"
        digest = hashlib.sha256()
        with open(self._weights_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
        if self._tflite_path and os.path.exists(self._tflite_path):
            with open(self._tflite_path, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"


# ── Parameter parsing & preprocessing helpers ────────────────────────


def _float_param(
    params: dict[str, Any],
    name: str,
    default: float,
    low: float,
    high: float,
    *,
    aliases: tuple[str, ...] = (),
) -> float:
    raw = params.get(name)
    for alias in aliases:
        if raw is None:
            raw = params.get(alias)
    if raw is None:
        return default
    if isinstance(raw, bool):
        raw = repr(raw)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ServiceError(
            ErrorCategory.TRANSPORT_ERROR,
            code="malformed_input",
            message=f"{name} must be a number, got {raw!r}.",
            transient=False,
            http_status=400,
        ) from exc
    if not low <= value <= high:
        raise ServiceError(
            ErrorCategory.TRANSPORT_ERROR,
            code="malformed_input",
            message=f"{name} must be between {low} and {high}.",
            transient=False,
            http_status=400,
        )
    return value


def _int_param(
    params: dict[str, Any],
    name: str,
    default: int,
    low: int,
    high: int,
) -> int:
    raw = params.get(name)
    if raw is None:
        return default
    if isinstance(raw, bool):
        raw = repr(raw)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ServiceError(
            ErrorCategory.TRANSPORT_ERROR,
            code="malformed_input",
            message=f"{name} must be an integer, got {raw!r}.",
            transient=False,
            http_status=400,
        ) from exc
    if not low <= value <= high:
        raise ServiceError(
            ErrorCategory.TRANSPORT_ERROR,
            code="malformed_input",
            message=f"{name} must be between {low} and {high}.",
            transient=False,
            http_status=400,
        )
    return value


def _imgsz_param(params: dict[str, Any]) -> int:
    value = _int_param(params, "imgsz", DEFAULT_IMGSZ, MIN_IMGSZ, MAX_IMGSZ)
    if value % IMGSZ_STRIDE != 0:
        raise ServiceError(
            ErrorCategory.TRANSPORT_ERROR,
            code="malformed_input",
            message=f"imgsz must be a multiple of {IMGSZ_STRIDE}.",
            transient=False,
            http_status=400,
        )
    return value


class DecodeError(Exception):
    pass


def _decode_image(image_bytes: bytes) -> tuple[Any, int, int]:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise DecodeError("Could not decode frame as JPEG/PNG.")
    height, width = img.shape[:2]
    return img, width, height