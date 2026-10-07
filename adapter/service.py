"""
FallDetectionService — OpenNVR AI Adapter for pose estimation and fall detection.

Combines YOLO pose estimation (ONNX Runtime, CUDA-accelerated) with the
FallTransformer keypoint analysis engine. Implements the OpenNVR Adapter Contract v1
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

from fastapi import WebSocket, WebSocketDisconnect

from adapters.fall_detection.coco_keypoints import COCO_KEYPOINTS, KEYPOINT_STRIDE
from adapters.fall_detection.fall_transformer import FallTransformer
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

MODEL_FRAMEWORK: str = "onnxruntime"
MODEL_NAME: str = "yolo11n-pose-fall"
MAX_IMAGE_BYTES: int = 8 * 1024 * 1024  # 8 MiB max request size

DEFAULT_WEIGHTS_DIR: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "model_weights")
)
WEIGHTS_FILENAME: str = "yolo11n-pose.onnx"
MODEL_URL_ENV: str = "YOLO_POSE_MODEL_URL"
TFLITE_MODEL_ENV: str = "FALL_TRANSFORMER_TFLITE_PATH"

DEFAULT_CONF: float = 0.4
DEFAULT_IOU: float = 0.45
DEFAULT_IMGSZ: int = 448
IMGSZ_STRIDE: int = 32
MIN_IMGSZ: int = 160
MAX_IMGSZ: int = 1280
DEFAULT_MAX_PERSONS: int = 20
ABSOLUTE_MAX_PERSONS: int = 100
KEYPOINT_VISIBLE_CONF: float = 0.5
EXPECTED_FEATURES: int = 5 + len(COCO_KEYPOINTS) * KEYPOINT_STRIDE

FALL_STATES = ("standing", "falling", "fallen", "unknown")


class FallDetectionService(AdapterService):
    """Façade around YOLO-pose ONNX and FallTransformer."""

    def __init__(self, weights_path: str | None = None) -> None:
        self._weights_path = weights_path or os.path.join(
            os.getenv("YOLO_POSE_WEIGHTS_DIR", DEFAULT_WEIGHTS_DIR),
            WEIGHTS_FILENAME,
        )
        self._model_url: str = os.getenv(MODEL_URL_ENV, "")
        self._tflite_path: str = os.getenv(
            TFLITE_MODEL_ENV,
            os.path.join(os.path.dirname(__file__), "fall_detection_transformer.tflite"),
        )
        self._session: Any | None = None
        self._input_name: str = "images"
        self._static_imgsz: int | None = None
        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint_cache: str | None = None
        self._gpu_in_use: bool = False
        self._fall_transformer: FallTransformer | None = None

    def load(self) -> None:
        """Eagerly load ONNX weights and initialize the fall transformer."""
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
            import onnxruntime as ort

            ensure_model_file(
                self._weights_path,
                self._model_url,
                label=f"{MODEL_NAME} weights",
                logger=logger,
            )

            # Prioritize CUDAExecutionProvider on systems with NVIDIA GPUs
            self._session = ort.InferenceSession(
                self._weights_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            input_meta = self._session.get_inputs()[0]
            self._input_name = input_meta.name
            self._static_imgsz = _static_input_size(getattr(input_meta, "shape", None))
            self._fingerprint_cache = self._compute_fingerprint()
            self._gpu_in_use = self._detect_gpu_in_use()

            # Initialize FallTransformer
            self._fall_transformer = FallTransformer(self._tflite_path)

            self._warm_up()
            self._load_state = HealthStatus.OK
            self._load_error = None
            logger.info(
                "FallDetectionService ready: weights=%s gpu=%s imgsz=%s",
                self._weights_path,
                self._gpu_in_use,
                self._static_imgsz or f"dynamic (default {DEFAULT_IMGSZ})",
            )
        except Exception as exc:
            self._load_state = HealthStatus.ERROR
            self._load_error = str(exc)
            logger.exception("FallDetectionService failed to load weights %s", self._weights_path)

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
            modalities_out=["keypoints", "fall_state"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        cpu_count = os.cpu_count() or 0
        if self._load_state == HealthStatus.OK:
            if self._gpu_in_use:
                verdict = HardwareVerdict.OK
                reasoning = "NVIDIA CUDA GPU detected and active; ONNX weights loaded."
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

        providers: list[str] = []
        try:
            import onnxruntime as ort
            providers = list(ort.get_available_providers())
        except Exception:
            pass

        return HardwareEvaluationResponse(
            verdict=verdict,
            reasoning=reasoning,
            checked_at=datetime.now(timezone.utc),
            details={
                "gpu_required": True,
                "gpu_in_use": self._gpu_in_use,
                "onnxruntime_providers": providers,
                "cpu_count": cpu_count,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "weights_path": self._weights_path,
                "default_imgsz": DEFAULT_IMGSZ,
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

        logger.info("Stream session opened: session_id=%s client_id=%s", session_id, handshake.client_id)

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
        """Run pose estimation and fall classification."""
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
        imgsz = self._resolve_imgsz(params)
        max_persons = _int_param(params, "max_persons", DEFAULT_MAX_PERSONS, 1, ABSOLUTE_MAX_PERSONS)

        start = time.monotonic()
        try:
            img, width, height = _decode_image(image_bytes)
            raw, scale, pad_x, pad_y = self._run_inference(img, imgsz)
            persons = self._shape_persons(
                raw,
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y,
                width=width,
                height=height,
                conf=conf,
                iou=iou,
                max_persons=max_persons,
            )

            # Classify fall state for every person
            if self._fall_transformer:
                for idx, person in enumerate(persons):
                    fall_info = self._fall_transformer.analyze_person(
                        person,
                        frame_width=width,
                        frame_height=height,
                        track_id=str(idx),
                    )
                    person.update(fall_info)
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
                    if kpt[2] >= KEYPOINT_VISIBLE_CONF:
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
                "keypoint_names": list(COCO_KEYPOINTS),
                "frame_dimensions": {"w": width, "h": height},
            },
        )

    def _infer_frame_for_stream(
        self,
        image_bytes: bytes,
        seq: int,
        ts_ms: int,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params = params or {}
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

    def _run_inference(self, img: Any, imgsz: int) -> tuple[Any, float, float, float]:
        import numpy as np

        blob, scale, pad_x, pad_y = _letterbox_blob(img, imgsz)
        outputs = self._session.run(None, {self._input_name: blob})
        raw = np.transpose(outputs[0], (0, 2, 1)).squeeze()
        if raw.ndim == 1:
            raw = np.expand_dims(raw, axis=0)

        if raw.shape[-1] != EXPECTED_FEATURES:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="yolo_pose.unexpected_model_output",
                message=(
                    f"Model emitted {raw.shape[-1]} features per prediction, "
                    f"expected {EXPECTED_FEATURES}."
                ),
                transient=False,
                http_status=500,
            )
        return raw, scale, pad_x, pad_y

    def _shape_persons(
        self,
        raw: Any,
        *,
        scale: float,
        pad_x: float,
        pad_y: float,
        width: int,
        height: int,
        conf: float,
        iou: float,
        max_persons: int,
    ) -> list[dict[str, Any]]:
        import numpy as np

        scores = raw[:, 4].astype(float)
        keep_mask = scores >= conf
        if not bool(keep_mask.any()):
            return []
        rows = raw[keep_mask]
        scores = scores[keep_mask]

        cx, cy, bw, bh = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        x1 = (cx - bw / 2.0 - pad_x) / scale
        y1 = (cy - bh / 2.0 - pad_y) / scale
        x2 = (cx + bw / 2.0 - pad_x) / scale
        y2 = (cy + bh / 2.0 - pad_y) / scale
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(float)
        np.clip(boxes[:, 0::2], 0.0, float(width), out=boxes[:, 0::2])
        np.clip(boxes[:, 1::2], 0.0, float(height), out=boxes[:, 1::2])

        order = _nms(boxes, scores, iou)[:max_persons]

        persons: list[dict[str, Any]] = []
        for index in order:
            row = rows[index]
            keypoints: list[list[float]] = []
            for slot in range(len(COCO_KEYPOINTS)):
                base = 5 + slot * KEYPOINT_STRIDE
                kx = (float(row[base]) - pad_x) / scale
                ky = (float(row[base + 1]) - pad_y) / scale
                kconf = float(row[base + 2])
                kx = min(max(kx, 0.0), float(width))
                ky = min(max(ky, 0.0), float(height))
                keypoints.append([
                    round(kx, 1),
                    round(ky, 1),
                    round(min(max(kconf, 0.0), 1.0), 4),
                ])
            box = boxes[index]
            persons.append({
                "bbox": [
                    round(float(box[0]), 1),
                    round(float(box[1]), 1),
                    round(float(box[2]), 1),
                    round(float(box[3]), 1),
                ],
                "score": round(float(scores[index]), 4),
                "keypoints": keypoints,
            })
        return persons

    def _resolve_imgsz(self, params: dict[str, Any]) -> int:
        requested = _imgsz_param(params)
        if self._static_imgsz is None:
            return requested
        if "imgsz" in params and requested != self._static_imgsz:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=f"Model exported with fixed {self._static_imgsz}px input.",
                transient=False,
                http_status=400,
            )
        return self._static_imgsz

    def _warm_up(self) -> None:
        try:
            import numpy as np

            imgsz = self._static_imgsz or DEFAULT_IMGSZ
            blank = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            blob, _, _, _ = _letterbox_blob(blank, imgsz)
            self._session.run(None, {self._input_name: blob})
        except Exception:
            pass

    def _adapter_model_version(self) -> str:
        return f"{MODEL_FRAMEWORK}/{MODEL_NAME}"

    def _weights_size_mb(self) -> float | None:
        try:
            return round(os.path.getsize(self._weights_path) / (1024 * 1024), 2)
        except OSError:
            return None

    def _compute_fingerprint(self) -> str:
        if not os.path.exists(self._weights_path):
            return "sha256:unavailable"
        digest = hashlib.sha256()
        with open(self._weights_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    def _detect_gpu_in_use(self) -> bool:
        try:
            providers = self._session.get_providers()
            return "CUDAExecutionProvider" in providers
        except Exception:
            return False


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
    import cv2
    import numpy as np

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise DecodeError("Could not decode frame as JPEG/PNG.")
    height, width = img.shape[:2]
    return img, width, height


def _static_input_size(shape: Any) -> int | None:
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        return None
    height, width = shape[-2], shape[-1]
    if isinstance(height, int) and isinstance(width, int) and height == width > 0:
        return int(height)
    return None


def _letterbox_blob(img: Any, imgsz: int) -> tuple[Any, float, float, float]:
    import cv2
    import numpy as np

    height, width = img.shape[:2]
    scale = min(imgsz / float(width), imgsz / float(height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    left = int(round((imgsz - new_w) / 2.0))
    top = int(round((imgsz - new_h) / 2.0))

    interpolation = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized

    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None, ...])
    return blob, scale, float(left), float(top)


def _nms(boxes: Any, scores: Any, iou_threshold: float) -> list[int]:
    import numpy as np

    areas = (boxes[:, 2] - boxes[:, 0]).clip(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clip(min=0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 1.0)
        order = rest[iou <= iou_threshold]
    return keep

