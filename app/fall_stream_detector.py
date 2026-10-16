"""
Fall Detection Stream Worker & Detector — OpenNVR Native App.

Bridges RTSP camera substreams from MediaMTX to the Fall Detection AI Adapter
using the §6 WebSocket streaming protocol (InferStream).

Supports:
1. Pattern 2 (Continuous Streaming): Dedicated 24/7 frame loop at 5–15 FPS.
2. Pattern 1 (Tier-0 Gated Streaming): Sleep/idle until Tier-0 detects a person,
   then stream frames down InferStream while the person is in view.
3. Alerts: Evaluates fall persistence (fallen_seconds) and dispatches
   'person-fallen' critical alerts to NATS and the OpenNVR UI.
"""

from __future__ import annotations

import asyncio
import base64
import cv2
import json
import logging
import os
import signal
import sys
import threading
import time
from typing import Any

from opennvr_app_sdk import Alert, AlertType, InferStream, KaiCError, OpenNVR
from opennvr_app_sdk.alerts import build_dispatcher, set_default_source
from opennvr_app_sdk.nats_loop import connect_nats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fall-stream-detector")

# Identify app for alerts
set_default_source(kind="app", name="fall-detection", version="1.0.0")

# Configuration via environment variables or defaults
ADAPTER_NAME = os.getenv("FALL_ADAPTER_NAME", "fall-detection")
TARGET_FPS = float(os.getenv("STREAM_FPS", "10.0"))
FRAME_WIDTH = int(os.getenv("STREAM_WIDTH", "640"))
FALLEN_SECONDS = float(os.getenv("FALLEN_SECONDS", "3.0"))
ALERT_COOLDOWN_SECONDS = float(os.getenv("ALERT_COOLDOWN_SECONDS", "30.0"))
TIER0_GATING = os.getenv("ENABLE_TIER0_GATING", "false").lower() == "true"
PERSON_HOLD_TIME_S = float(os.getenv("PERSON_HOLD_TIME_S", "5.0"))
NATS_URL = os.getenv("NATS_URL", "nats://nats:4222")
INTERNAL_KEY = os.getenv("INTERNAL_API_KEY", os.getenv("OPENNVR_INTERNAL_API_KEY", ""))


class CameraStreamSession:
    """Manages continuous or Tier-0 gated streaming for a single camera."""

    def __init__(
        self,
        nvr: OpenNVR,
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

    def _run_loop(self) -> None:
        logger.info(
            "Starting camera worker [%s] (fps=%.1f, gated=%s)",
            self.camera_id, self.fps, self.gated,
        )

        while not self._stop.is_set():
            try:
                # 1. Tap the RTSP stream from MediaMTX via OpenNVR SDK
                with self.nvr.stream(self.camera_id, width=self.width, fps=self.fps) as video:
                    # 2. Open warm WebSocket session to KAI-C / Adapter
                    with self.nvr.ai.stream(self.adapter_name, camera_id=self.camera_id) as infer_stream:
                        logger.info(
                            "[%s] Connected to /infer/stream (session correlation_id=%s)",
                            self.camera_id, infer_stream.correlation_id,
                        )

                        for frame in video.frames(timeout=10.0):
                            if self._stop.is_set():
                                return

                            # In gated mode: skip inference if no person is around
                            if not self._should_infer():
                                self._reset_fall_tracking()
                                time.sleep(0.1)
                                continue

                            # Frame comes as BGR raw bytes; encode to JPEG for §6 wire protocol
                            bgr = frame.to_ndarray()
                            success, jpeg_bytes = cv2.imencode(
                                ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80]
                            )
                            if not success:
                                continue

                            raw_jpeg = jpeg_bytes.tobytes()

                            try:
                                # 3. Send JPEG down WebSocket & receive ResultMessage
                                response = infer_stream.infer(raw_jpeg)
                                self._evaluate_detections(response, raw_jpeg)
                            except KaiCError as err:
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
            # Emit resolution alert if desired
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
            # No fall in current frame
            if self._first_fall_time is not None:
                # Require grace period before clearing to avoid jitter
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
        self.nvr = OpenNVR()
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
            nc = await connect_nats(NATS_URL, INTERNAL_KEY or None)
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

        sub = await nc.subscribe("opennvr.inference.tier0.>", cb=on_tier0_msg)
        while not self._stop:
            await asyncio.sleep(1.0)
        await sub.unsubscribe()
        await nc.close()

    def stop(self) -> None:
        self._stop = True
        for session in self.sessions.values():
            session.stop()
        self.nvr.close()


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