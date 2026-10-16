# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Fall Detection — OpenNVR AI Adapter.

Wraps YOLO26-pose keypoint extraction + TFLite Transformer binary classifier
as both a single-frame HTTP service and a full §6 WebSocket streaming adapter.

Endpoints exposed
-----------------
GET  /health              — liveness probe (polled by KAI-C every 30 s)
GET  /capabilities        — task list, GPU flag, model fingerprint (polled every 60 s)
GET  /hardware/evaluation — operator UI: can this host run the model well?
GET  /metrics             — Prometheus scrape (latency, queue depth, fall count)
POST /infer               — single-frame fallback (one HTTP round-trip)
WS   /infer/stream        — §6 streaming: one warm session per camera feed

Run (dev / in-process driver):
    cd fall_detection
    opennvr-adapter dev

Run (production):
    uvicorn adapter:app --host 0.0.0.0 --port 9000

Result conventions
------------------
* §5.1 object_detection — fall emitted as DetectionItem(label="fall", ...)
* Normal frame          — empty detection list []
* ResultMessage.frame   — PNG bytes (fall events only); null on normal frames
* Annotated PNGs also archived locally to ./fall_snapshots/
"""

import asyncio
import datetime
import json
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path

from opennvr_adapter_sdk import Adapter, Overloaded
from opennvr_adapter_sdk.contract import (
    CloseMessage,
    FrameMessage,
    HandshakeAckMessage,
    HandshakeMessage,
    PauseMessage,
    ResultMessage,
    ResumeMessage,
    StreamCloseCode,
    StreamMessageType,
    DetectionItem,
    NormalizedBBox,
)

from fall_detection_core import (
    FALL_CONFIDENCE_THRESHOLD,
    INPUT_TIMESTEPS,
    StreamState,
    annotate_frame,
    load_models,
    run_inference,
)

SNAPSHOT_DIR = Path("fall_snapshots")
SNAPSHOT_DIR.mkdir(exist_ok=True)
# Maximum frames buffered before the adapter signals PauseMessage to KAI-C.
# When the queue drains below MAX_QUEUE // 2, ResumeMessage is sent.
MAX_QUEUE = 8

adapter = Adapter(
    "fall-detection",
    version="1.0.0",
    vendor="Maruti POC",
    license="Apache-2.0",
    # §5.1 object_detection — makes this a drop-in wherever that task is expected.
    # Bounding boxes are normalised (0–1).
    tasks=["object_detection"],
    framework="tflite+ultralytics",
    # SDK hashes this file to produce the fingerprint KAI-C uses for drift detection.
    # Update the path if you switch the TFLite weights file.
    weights="fall_detection_transformer.tflite",
    gpu=True,       # request GPU from OpenNVR scheduler
    max_inflight=4, # concurrent camera streams
)

# defaultdict creates a fresh StreamState for any new stream_id automatically.
_streams: dict[str, StreamState] = defaultdict(StreamState)


@adapter.load()
def load():
    """Load YOLO26-pose and TFLite Transformer once at startup.

    Heavy ML imports belong here so a missing dependency surfaces as a red
    GET /health (status=error + message) rather than crashing the container
    before it can answer any requests.

    The returned dict becomes ``call.model`` inside every inference handler.
    """
    yolo_pose, interpreter, input_details, output_details = load_models()
    return {
        "yolo_pose"     : yolo_pose,
        "interpreter"   : interpreter,
        "input_details" : input_details,
        "output_details": output_details,
    }


def _decode_frame(data: bytes) -> np.ndarray:
    """Decode JPEG/PNG bytes → BGR ndarray.  Raises ``Overloaded`` on failure."""
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise Overloaded("Could not decode frame bytes — expected JPEG or PNG")
    return frame


def _build_detection_item(
    frame_bgr: np.ndarray,
    yolo_result,
    confidence: float,
    stream_id: str,
) -> DetectionItem:
    """Build a §5.1 DetectionItem from a fall event.

    Computes the normalised bounding box from the largest person detected by
    YOLO26.  Falls back to a full-frame box if no detection is available.
    """
    H, W = frame_bgr.shape[:2]
    bbox_x, bbox_y, bbox_w, bbox_h = 0.0, 0.0, 1.0, 1.0

    if yolo_result.boxes is not None and len(yolo_result.boxes) > 0:
        areas    = yolo_result.boxes.xywh[:, 2] * yolo_result.boxes.xywh[:, 3]
        best_idx = int(areas.argmax())
        cx, cy, bw, bh = yolo_result.boxes.xywh[best_idx].cpu().numpy()
        bbox_x = float((cx - bw / 2) / W)
        bbox_y = float((cy - bh / 2) / H)
        bbox_w = float(bw / W)
        bbox_h = float(bh / H)

    return DetectionItem(
        label="fall",
        confidence=confidence,
        bbox=NormalizedBBox(x=bbox_x, y=bbox_y, w=bbox_w, h=bbox_h),
        attributes={
            "stream_id"           : stream_id,
            "fall_threshold"      : FALL_CONFIDENCE_THRESHOLD,
            "model_warmup_frames" : INPUT_TIMESTEPS,
        },
    )


def _save_snapshot(png_bytes: bytes, stream_id: str) -> Path:
    """Write a fall snapshot PNG to the local archive directory."""
    ts  = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    out = SNAPSHOT_DIR / f"{stream_id}_{ts}.png"
    out.write_bytes(png_bytes)
    return out


@adapter.on_image()
def detect(call):
    """Process one video frame and return fall detections.

    Called by KAI-C for single-frame requests, and as a fallback when the
    caller does not open a WebSocket stream session.

    call.image     — raw image bytes (JPEG / PNG)
    call.model     — dict returned by load()
    call.stream_id — unique string identifying the originating camera stream

    Returns
    -------
    list
        []           → no fall this frame  (§5.1 convention)
        [detection]  → fall; carries label, confidence, and normalised bbox
    """
    frame = _decode_frame(call.image)

    stream_id = getattr(call, "stream_id", "default")
    state     = _streams[stream_id]
    m         = call.model

    fall_detected, confidence, yolo_result = run_inference(
        frame, state,
        m["yolo_pose"], m["interpreter"], m["input_details"], m["output_details"],
        device="cuda",
    )

    if not fall_detected:
        return []

    # Save annotated snapshot locally
    png_bytes = annotate_frame(
        frame, yolo_result, fall_detected, confidence,
        state.frame_count, INPUT_TIMESTEPS,
    )
    snap = _save_snapshot(png_bytes, stream_id)
    print(f"[POST /infer] Fall detected (conf={confidence:.3f}) — snapshot: {snap}")

    return [_build_detection_item(frame, yolo_result, confidence, stream_id)]


@adapter.on_stream()
async def stream_handler(websocket) -> None:
    """§6 WebSocket streaming handler — one session per live camera feed.

    Keeps a single warm model session open for the duration of the stream,
    avoiding per-frame HTTP overhead and maintaining the shared correlation_id
    that KAI-C uses to trace a sequence of frames as one event.

    Protocol
    --------
    1. Receive   HandshakeMessage   → send HandshakeAckMessage
    2. Loop:
       a. Receive  FrameMessage (binary JPEG/PNG bytes)
       b. Run inference in thread pool (YOLO + TFLite are blocking)
       c. Send     ResultMessage (seq echoed, optional fall PNG in .frame)
       d. If queue fills → send PauseMessage; drain → send ResumeMessage
    3. On CloseMessage or disconnect → close WebSocket normally

    Backpressure
    ------------
    When ``queue_depth >= MAX_QUEUE`` the adapter sends ``PauseMessage`` and
    KAI-C stops sending frames.  Once the depth drops below ``MAX_QUEUE // 2``
    the adapter sends ``ResumeMessage`` and KAI-C resumes.  This prevents
    unbounded memory growth under sustained GPU saturation.
    """
    loop  = asyncio.get_event_loop()
    model = websocket.model   # dict from load()

    queue_depth: int  = 0
    paused:      bool = False

    # ── 1. Handshake ─────────────────────────────────────────────────────
    msg = await websocket.receive()
    raw = msg.get("text") if isinstance(msg, dict) else msg
    if not raw:
        print("[stream] Empty handshake message, closing")
        return

    try:
        handshake = HandshakeMessage.model_validate_json(raw)
    except Exception as exc:
        print(f"[stream] Handshake parse failed: {exc}")
        return

    session_id = (
        f"fall-{handshake.camera_id}-"
        f"{datetime.datetime.utcnow().strftime('%H%M%S%f')}"
    )
    correlation_id = session_id

    ack_json = HandshakeAckMessage(
        session_id=session_id,
        correlation_id=correlation_id,
    ).model_dump_json()

    if hasattr(websocket, "send_text"):
        await websocket.send_text(ack_json)
    else:
        await websocket.send(ack_json)

    stream_id = handshake.camera_id or "default"
    state     = _streams[stream_id]

    print(f"[stream] Session opened: {session_id}  stream_id={stream_id}")

    # ── 2. Frame loop ─────────────────────────────────────────────────────
    try:
        while True:
            msg = await websocket.receive()
            if isinstance(msg, dict):
                if msg.get("type") == "websocket.disconnect":
                    break
                raw = msg.get("text")
            else:
                raw = msg

            if not raw:
                continue

            # Detect close before full parse (avoid parse errors on close frames)
            try:
                envelope = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                break

            msg_type = envelope.get("type")
            if msg_type in (StreamMessageType.CLOSE, "close"):
                break
            if msg_type in (StreamMessageType.PAUSE, "pause"):
                paused = True
                continue
            if msg_type in (StreamMessageType.RESUME, "resume"):
                paused = False
                continue

            try:
                frame_msg = FrameMessage.model_validate(envelope)
            except Exception as exc:
                print(f"[stream] Bad frame metadata: {exc}")
                break

            queue_depth += 1

            # In §6 streaming protocol: binary frame payload follows metadata frame
            bin_msg = await websocket.receive()
            if isinstance(bin_msg, dict):
                frame_data = bin_msg.get("bytes")
            else:
                frame_data = bin_msg

            if not frame_data:
                queue_depth -= 1
                continue

            # ── Backpressure: pause if queue is full ──────────────────
            if queue_depth >= MAX_QUEUE and not paused:
                pause_json = PauseMessage().model_dump_json()
                if hasattr(websocket, "send_text"):
                    await websocket.send_text(pause_json)
                else:
                    await websocket.send(pause_json)
                paused = True
                print(f"[stream] {session_id} → PauseMessage (queue={queue_depth})")

            # ── Run blocking inference in thread pool ─────────────────
            # YOLO and TFLite are synchronous; run_in_executor keeps the
            # async event loop free to process incoming messages concurrently.
            fall_detected, confidence, yolo_result = await loop.run_in_executor(
                None,
                lambda fd=frame_data: run_inference(
                    _decode_frame(fd),
                    state,
                    model["yolo_pose"],
                    model["interpreter"],
                    model["input_details"],
                    model["output_details"],
                    device="cuda" if model.get("device") == "cuda" else "cpu",
                ),
            )
            queue_depth -= 1

            # ── Resume when queue drains ──────────────────────────────
            if paused and queue_depth <= MAX_QUEUE // 2:
                resume_json = ResumeMessage().model_dump_json()
                if hasattr(websocket, "send_text"):
                    await websocket.send_text(resume_json)
                else:
                    await websocket.send(resume_json)
                paused = False
                print(f"[stream] {session_id} → ResumeMessage (queue={queue_depth})")

            # ── Build result payload ──────────────────────────────────
            detections: list[DetectionItem] = []
            png_bytes:  bytes | None        = None

            if fall_detected:
                decoded_frame = _decode_frame(frame_data)

                detections = [
                    _build_detection_item(decoded_frame, yolo_result, confidence, stream_id)
                ]

                # Annotate and encode as PNG for OpenNVR event snapshot
                png_bytes = annotate_frame(
                    decoded_frame, yolo_result, fall_detected, confidence,
                    state.frame_count, INPUT_TIMESTEPS,
                )

                # Archive snapshot locally
                snap = _save_snapshot(png_bytes, stream_id)
                print(
                    f"[stream] {session_id} FALL (conf={confidence:.3f}, "
                    f"frame={state.frame_count}) → {snap}"
                )

            # ── Send ResultMessage (seq echo required by §6) ──────────
            result_json = ResultMessage(
                seq=frame_msg.seq,
                result={
                    "detections": [d.model_dump() for d in detections],
                },
                frame=png_bytes,  # None on normal frames; PNG on fall
            ).model_dump_json()

            if hasattr(websocket, "send_text"):
                await websocket.send_text(result_json)
            else:
                await websocket.send(result_json)

    except Exception as exc:
        print(f"[stream] {session_id} error: {exc}")

    finally:
        try:
            await websocket.close(StreamCloseCode.NORMAL)
        except Exception:
            pass
        print(f"[stream] Session closed: {session_id}")


app = adapter.app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("adapter:app", host="0.0.0.0", port=9000, reload=False)