"""
Fall Detection Stream Worker & Detector — OpenNVR Native App.

Bridges RTSP camera substreams from MediaMTX to the Fall Detection AI Adapter
using the §6 WebSocket streaming protocol (InferStream).

Supports:
1. Pattern 2 (Continuous Streaming): Dedicated 24/7 frame loop at 5–15 FPS.
2. Pattern 1 (Tier-0 Gated Streaming): Sleep/idle until Tier-0 detects a person,
   then stream frames down InferStream while the person is in view.
3. Hybrid connectivity: Works inside full OpenNVR stack (KAI-C + OpenNVR Core)
   OR standalone direct mode (direct RTSP / video file + direct adapter WS).
4. Alerts: Evaluates fall persistence (fallen_seconds) and dispatches
   'person-fallen' critical alerts to NATS and the OpenNVR UI.
"""

from __future__ import annotations

import asyncio
import cv2
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Generator

# Auto-detect local opennvr_app_sdk if present in same directory
_LOCAL_SDK = Path(__file__).resolve().parent / "opennvr_app_sdk"
if _LOCAL_SDK.exists() and str(_LOCAL_SDK) not in sys.path:
    sys.path.insert(0, str(_LOCAL_SDK))

from opennvr_app_sdk import Alert, AlertType, InferStream, KaiCError, OpenNVR
from opennvr_app_sdk.alerts import build_dispatcher, set_default_source
from opennvr_app_sdk.credentials import bus_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fall-stream-detector")

# Identify app for alerts
set_default_source(kind="app", name="fall-detection", version="1.0.0")

# Configuration via environment variables with safe defaults
OPENNVR_URL = os.getenv("OPENNVR_URL", "http://localhost:8000")
KAIC_URL = os.getenv("KAIC_URL", os.getenv("OPENNVR_KAIC_URL", "http://localhost:8100"))
ADAPTER_NAME = os.getenv("FALL_ADAPTER_NAME", "fall-detection")
# Direct WS fallback (e.g. "ws://localhost:9010/infer/stream" for direct testing without KAI-C)
# This module is the direct-streaming path (no KAI-C proxy).
# fall_detection.py is the NATS-subscriber path (with KAI-C). Both are valid alternatives.
ADAPTER_WS_URL = os.getenv("ADAPTER_WS_URL", "")
# Direct RTSP / video file fallback (e.g. "rtsp://localhost:8554/cam1" or "fall_example_1.mp4")
RTSP_URL = os.getenv("RTSP_URL", "")

TARGET_FPS = float(os.getenv("STREAM_FPS", "10.0"))
FRAME_WIDTH = int(os.getenv("STREAM_WIDTH", "640"))
FALLEN_SECONDS = float(os.getenv("FALLEN_SECONDS", "3.0"))
ALERT_COOLDOWN_SECONDS = float(os.getenv("ALERT_COOLDOWN_SECONDS", "30.0"))
TIER0_GATING = os.getenv("ENABLE_TIER0_GATING", "false").lower() == "true"
PERSON_HOLD_TIME_S = float(os.getenv("PERSON_HOLD_TIME_S", "5.0"))
NATS_URL = os.getenv("NATS_URL", "nats://nats:4222")
INTERNAL_KEY = os.getenv("INTERNAL_API_KEY", os.getenv("OPENNVR_INTERNAL_API_KEY", ""))


class DirectInferStream:
    """Connects directly to an adapter's WS /infer/stream (bypassing KAI-C proxy)."""

    def __init__(self, ws_url: str, camera_id: str = "cam1", timeout: float = 5.0) -> None:
        self.ws_url = ws_url
        self.camera_id = camera_id
        self.timeout = timeout
        self.conn: Any = None
        self._seq = 0
        self.correlation_id: str | None = None

    def __enter__(self) -> "DirectInferStream":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def open(self) -> "DirectInferStream":
        from websockets.sync.client import connect

        self.conn = connect(self.ws_url, open_timeout=self.timeout)
        # §6.1 Handshake
        self.conn.send(
            json.dumps({
                "type": "handshake",
                "client_id": "fall-stream-worker",
                "camera_id": self.camera_id,
                "frame_transport": "websocket",
            })
        )
        ack_raw = self.conn.recv(timeout=self.timeout)
        ack = json.loads(ack_raw) if isinstance(ack_raw, str) else json.loads(ack_raw.decode())
        self.correlation_id = ack.get("correlation_id", f"direct-{self.camera_id}")
        self._seq = 0
        return self

    def infer(self, jpeg: bytes) -> dict[str, Any]:
        if not self.conn:
            self.open()
        self._seq += 1
        # 1. Metadata frame (JSON)
        self.conn.send(
            json.dumps({
                "type": "frame",
                "seq": self._seq,
                "ts_ms": int(time.monotonic() * 1000),
                "content_type": "image/jpeg",
            })
        )
        # 2. Binary frame (JPEG bytes)
        self.conn.send(jpeg)
        res_raw = self.conn.recv(timeout=self.timeout)
        payload = json.loads(res_raw) if isinstance(res_raw, str) else json.loads(res_raw.decode())
        return {
            "status": "ok",
            "result": payload.get("result") or {},
            "correlation_id": self.correlation_id,
        }

    def close(self) -> None:
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None


class CameraStreamSession:
    """Manages continuous or Tier-0 gated streaming for a single camera."""

    def __init__(
        self,
        nvr: OpenNVR | None,
        dispatcher: Any,
        camera_id: str,
        adapter_name: str = ADAPTER_NAME,
        fps: float = TARGET_FPS,
        width: int = FRAME_WIDTH,
        gated: bool = TIER0_GATING,
        fallen_seconds: float = FALLEN_SECONDS,
        cooldown_seconds: float = ALERT_COOLDOWN_SECONDS,
    ) -> None:
        self.nvr = nvr
        self.dispatcher = dispatcher
        self.camera_id = camera_id
        self.adapter_name = adapter_name
        self.fps = fps
        self.width = width
        self.gated = gated
        self.fallen_seconds = fallen_seconds
        self.cooldown_seconds = cooldown_seconds

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_person_seen: float = 0.0

        # Persistence tracking for falls
        self._first_fall_time: float | None = None
        self._alert_fired: bool = False
        self._last_alert_time: float = 0.0

    def note_tier0_person(self) -> None:
        """Called when Tier-0 detects an active person track on this camera."""
        self._last_person_seen = time.monotonic()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"stream-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def _should_infer(self) -> bool:
        if not self.gated:
            return True
        # Gated mode: active if a person was seen within PERSON_HOLD_TIME_S
        return (time.monotonic() - self._last_person_seen) <= PERSON_HOLD_TIME_S

    def _get_infer_stream_context(self):
        """Returns either a DirectInferStream or SDK InferStream context manager."""
        if ADAPTER_WS_URL:
            logger.info("[%s] Using direct WebSocket: %s", self.camera_id, ADAPTER_WS_URL)
            return DirectInferStream(ADAPTER_WS_URL, camera_id=self.camera_id)
        if self.nvr is not None:
            return self.nvr.ai.stream(self.adapter_name, camera_id=self.camera_id)
        # Fallback to direct adapter on port 9010
        fallback_url = "ws://localhost:9010/infer/stream"
        logger.info("[%s] Falling back to default direct adapter WS: %s", self.camera_id, fallback_url)
        return DirectInferStream(fallback_url, camera_id=self.camera_id)

    def _iter_frames(self) -> Generator[bytes, None, None]:
        """Yields JPEG frame bytes from RTSP URL, OpenCV, or OpenNVR SDK."""
        rtsp_source = RTSP_URL or os.getenv("CAMERA_RTSP_URL", "")

        # If direct RTSP/video source is specified, use OpenCV VideoCapture
        if rtsp_source:
            source = rtsp_source.format(camera_id=self.camera_id) if "{camera_id}" in rtsp_source else rtsp_source
            logger.info("[%s] Reading direct video source: %s", self.camera_id, source)
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video source: {source}")

            delay = 1.0 / self.fps
            try:
                while not self._stop.is_set():
                    t0 = time.monotonic()
                    ret, bgr = cap.read()
                    if not ret:
                        # If video file ended, loop back for testing
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, bgr = cap.read()
                        if not ret:
                            time.sleep(1.0)
                            continue

                    if self.width and bgr.shape[1] != self.width:
                        h = int(bgr.shape[0] * (self.width / bgr.shape[1]))
                        bgr = cv2.resize(bgr, (self.width, h))

                    success, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if success:
                        yield jpeg.tobytes()

                    elapsed = time.monotonic() - t0
                    sleep_time = delay - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            finally:
                cap.release()
            return

        # Otherwise use OpenNVR SDK frame stream (MediaMTX via OpenNVR Core)
        if self.nvr is not None:
            with self.nvr.stream(self.camera_id, width=self.width, fps=self.fps) as video:
                for frame in video.frames(timeout=10.0):
                    if self._stop.is_set():
                        return
                    bgr = frame.to_ndarray()
                    success, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if success:
                        yield jpeg.tobytes()
            return

        # Direct MediaMTX fallback if neither RTSP_URL nor Core is present
        mediamtx_url = f"rtsp://localhost:8554/{self.camera_id}"
        logger.info("[%s] Trying default MediaMTX RTSP: %s", self.camera_id, mediamtx_url)
        cap = cv2.VideoCapture(mediamtx_url)
        delay = 1.0 / self.fps
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                ret, bgr = cap.read()
                if not ret:
                    time.sleep(1.0)
                    continue
                success, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if success:
                    yield jpeg.tobytes()
                elapsed = time.monotonic() - t0
                if delay > elapsed:
                    time.sleep(delay - elapsed)
        finally:
            cap.release()

    def _run_loop(self) -> None:
        logger.info(
            "Starting camera worker [%s] (fps=%.1f, gated=%s)",
            self.camera_id, self.fps, self.gated,
        )

        while not self._stop.is_set():
            try:
                with self._get_infer_stream_context() as infer_stream:
                    logger.info(
                        "[%s] Connected to /infer/stream (session correlation_id=%s)",
                        self.camera_id, getattr(infer_stream, "correlation_id", "active"),
                    )

                    for raw_jpeg in self._iter_frames():
                        if self._stop.is_set():
                            return

                        # In gated mode: skip inference if no person is around
                        if not self._should_infer():
                            self._reset_fall_tracking()
                            time.sleep(0.1)
                            continue

                        try:
                            # Send JPEG down WebSocket & receive ResultMessage
                            response = infer_stream.infer(raw_jpeg)
                            self._evaluate_detections(response, raw_jpeg)
                        except (KaiCError, Exception) as err:
                            logger.warning(
                                "[%s] InferStream error: %s (reconnecting)",
                                self.camera_id, err,
                            )
                            break  # Re-open stream

            except Exception as exc:
                logger.error(
                    "[%s] Stream loop error: %s (retrying in 5s)",
                    self.camera_id, exc,
                )
                self._stop.wait(5.0)

    def _reset_fall_tracking(self) -> None:
        if self._alert_fired:
            logger.info("[%s] Person stood up or left view. Fall state cleared.", self.camera_id)
            self._dispatch_resolved_alert()
        self._first_fall_time = None
        self._alert_fired = False

    def _evaluate_detections(self, response: dict[str, Any], jpeg_bytes: bytes) -> None:
        now = time.monotonic()
        result = response.get("result") or {}
        detections = result.get("detections") or []

        fall_item = next((d for d in detections if d.get("label") == "fall"), None)

        if fall_item is not None:
            conf = float(fall_item.get("confidence", 0.0))
            if self._first_fall_time is None:
                self._first_fall_time = now

            duration = now - self._first_fall_time

            logger.info(
                "[%s] Fall posture detected (conf=%.2f, duration=%.1fs/%.1fs)",
                self.camera_id, conf, duration, self.fallen_seconds,
            )

            # Check persistence threshold and cooldown
            if duration >= self.fallen_seconds and not self._alert_fired:
                if (now - self._last_alert_time) >= self.cooldown_seconds:
                    self._fire_fall_alert(fall_item, duration, conf, jpeg_bytes)
                    self._alert_fired = True
                    self._last_alert_time = now
        else:
            if self._first_fall_time is not None:
                # Require 1.5s grace period before clearing to avoid frame-drop jitter
                if (now - self._first_fall_time) > 1.5:
                    self._reset_fall_tracking()

    def _fire_fall_alert(
        self,
        fall_item: dict[str, Any],
        duration: float,
        conf: float,
        jpeg_bytes: bytes,
    ) -> None:
        bbox = fall_item.get("bbox") or {}
        bbox_norm = [
            bbox.get("x", 0.0),
            bbox.get("y", 0.0),
            bbox.get("w", 1.0),
            bbox.get("h", 1.0),
        ]

        logger.critical(
            "🚨 CRITICAL ALERT: Confirmed Person Fall on [%s]! Down for %.1fs (conf=%.2f)",
            self.camera_id, duration, conf,
        )

        alert = Alert(
            type="person-fallen",
            severity="critical",
            camera_id=self.camera_id,
            description=f"Person fall detected on {self.camera_id} (down for {duration:.1f}s)",
            bbox=bbox_norm,
            attributes={
                "confidence": conf,
                "duration_seconds": round(duration, 1),
                "camera_id": self.camera_id,
            },
        )

        try:
            self.dispatcher.fire(alert)
        except Exception as exc:
            logger.error("[%s] Failed to fire alert: %s", self.camera_id, exc)

    def _dispatch_resolved_alert(self) -> None:
        alert = Alert(
            type="person-fallen-resolved",
            severity="info",
            camera_id=self.camera_id,
            description=f"Person stood back up on {self.camera_id}",
            attributes={"camera_id": self.camera_id},
        )
        try:
            self.dispatcher.fire(alert)
        except Exception:
            pass


class StreamOrchestrator:
    """Coordinates camera streams, Tier-0 triggers, and alert dispatchers."""

    def __init__(self, cameras: list[str]) -> None:
        self.camera_ids = cameras

        # Initialize OpenNVR client with safe defaults (handles standalone vs integrated)
        try:
            self.nvr: OpenNVR | None = OpenNVR(
                url=OPENNVR_URL,
                kaic_url=KAIC_URL,
                token=INTERNAL_KEY or None,
            )
        except Exception as exc:
            logger.warning("OpenNVR platform init: %s (direct mode available)", exc)
            self.nvr = None

        self.dispatcher = build_dispatcher(
            webhook_url=os.getenv("WEBHOOK_URL"),
            nats_url=NATS_URL,
            nats_token=INTERNAL_KEY or None,
            stdout=True,
        )
        self.sessions: dict[str, CameraStreamSession] = {}
        self._stop = False

    async def run(self) -> None:
        # Start a stream session for each camera
        for cid in self.camera_ids:
            session = CameraStreamSession(
                nvr=self.nvr,
                dispatcher=self.dispatcher,
                camera_id=cid,
            )
            self.sessions[cid] = session
            session.start()

        # If Tier-0 gating is enabled, listen to NATS events
        if TIER0_GATING:
            await self._run_nats_tier0_listener()
        else:
            logger.info("Continuous mode (Pattern 2) running for cameras: %s", self.camera_ids)
            while not self._stop:
                await asyncio.sleep(1.0)

    async def _run_nats_tier0_listener(self) -> None:
        logger.info(
            "Tier-0 Gating (Pattern 1) enabled. Subscribing to 'opennvr.inference.tier0.>' on %s",
            NATS_URL,
        )
        try:
            import nats
        except ImportError:
            logger.error("The 'nats-py' package is not installed. Run 'pip install nats-py' to enable Tier-0 gating.")
            return

        try:
            connect_kwargs = bus_connection(None, NATS_URL, INTERNAL_KEY or None)
            connect_kwargs.setdefault("connect_timeout", 5.0)
            connect_kwargs.setdefault("reconnect_time_wait", 1.0)
            connect_kwargs.setdefault("max_reconnect_attempts", -1)
            nc = await nats.connect(**connect_kwargs)
        except Exception as exc:
            logger.error("Could not connect to NATS for Tier-0 events: %s", exc)
            return

        async def on_tier0_msg(msg):
            try:
                data = json.loads(msg.data.decode())
                tracks = data.get("tracks") or []
                cam_id = data.get("camera_id")

                # Check if any track has 'person' label
                has_person = any(t.get("label") == "person" for t in tracks)
                if has_person and cam_id in self.sessions:
                    self.sessions[cam_id].note_tier0_person()
            except Exception as e:
                logger.debug("Error parsing Tier-0 event: %s", e)

        try:
            sub = await nc.subscribe("opennvr.inference.tier0.>", cb=on_tier0_msg)
            while not self._stop:
                await asyncio.sleep(1.0)
            await sub.unsubscribe()
        finally:
            await nc.close()

    def stop(self) -> None:
        self._stop = True
        for session in self.sessions.values():
            session.stop()
        if self.nvr:
            try:
                self.nvr.close()
            except Exception:
                pass


def main():
    cameras_env = os.getenv("CAMERAS", "cam1")
    cameras = [c.strip() for c in cameras_env.split(",") if c.strip()]

    orchestrator = StreamOrchestrator(cameras)

    def handle_signal(*_):
        logger.info("Stopping fall stream workers...")
        orchestrator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        asyncio.run(orchestrator.run())
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()