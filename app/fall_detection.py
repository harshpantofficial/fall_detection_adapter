# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fall detection — raise an alarm when someone falls, and show the
skeleton it is watching on the live view.

How it is put together
----------------------
One worker thread per picked camera, the same shape as guard-scan:

* **frames** come from core's scoped stream grant (``nvr.stream``), at
  ``fps`` — 10 by default. The fall model reads 30 frames at a time and
  was trained on 24-30 fps video; 10-15 fps is measured to still catch
  every fall in the reference clips, below ~5 fps it stops working.
* **inference** is the ``fall-detection`` adapter over KAI-C's §6 stream
  (``InferStream``), one session per camera. The adapter holds BOTH
  models — YOLO26-pose and the fall transformer — and the 30-frame
  window, per session. So a session is a window: when the feed breaks,
  this app closes the session and opens a new one rather than let the
  window stitch the gap into one motion.
* **alarms** are decided here (``fall_rules.FallMonitor``): the adapter
  scores frames, a run of hot frames becomes one alarm, with the frame
  that decided it, annotated, as evidence.
* **the live view** gets this app's own boxes with keypoints on them
  (``overlay.boxes.v1``): the person being watched, labelled with the
  score, and FALLEN for a few seconds after an alarm. Drawn only when
  an administrator lets this app draw (App Catalog), like every app.

Run:
    python fall_detection.py --config config.yml
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from opennvr_app_sdk import (
    Alert, AlertType, AppManifest, BaseAppConfig, FrameApp, InferStream,
    OpenNVR, Param, StateView, load_app_config,
)
from opennvr_app_sdk.alerts import AlertSource, build_dispatcher
from opennvr_app_sdk.domain_events import DomainEventPublisher

sys.path.insert(0, str(Path(__file__).parent))
from fall_rules import FallMonitor, FallPolicy  # noqa: E402

log = logging.getLogger("fall-detection")

APP_ID = "fall-detection"
FALL_ADAPTER = "fall-detection"
FALL_TASK = "fall_detection"

#: Consecutive failed frames before a blip is called an outage — on
#: /state, in the log and in the catalog's status dot.
INFER_FAILURES_BEFORE_UNHEALTHY = 3

#: Below this achieved frame rate the model's 30-frame window covers so
#: much time that a fall is a few frames of it; say so on /state.
MIN_USEFUL_FPS = 6.0

#: COCO-17 bones, for the evidence picture. Same list as the adapter's
#: coco_keypoints.COCO_SKELETON and the live overlay.
SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
)


class FallInferenceUnavailable(RuntimeError):
    """The adapter could not be reached, or answered with an error."""


MANIFEST = AppManifest(
    id=APP_ID,
    name="Fall Detection",
    version="0.1.0",
    category="safety",
    summary=("Raises an alarm when someone falls, from their pose over the "
             "last few seconds, and shows the skeleton it is watching on "
             "the live view."),
    requires_tasks=[FALL_TASK],
    # The adapter is not in the standard stack: the installer brings up
    # fall-detection-adapter (adapter_service_name("fall_detection")).
    requires_adapters=["fall_detection"],
    subscribes=None,          # drives its own frames, at video rate
    params=[
        Param("threshold", float, default=0.90, group="Alarm",
              label="Fall score that raises an alarm",
              description="0-1. The model's score for 'this person fell' over "
                          "the last 30 frames. 0.90 is what the model was "
                          "validated at; lower catches more and alarms more."),
        Param("cooldown_s", float, default=10.0, group="Alarm",
              label="Seconds between alarms on one camera",
              description="One fall stays above the threshold for a second or "
                          "more; this keeps it one alarm."),
        Param("alert_severity", str, default="critical", group="Alarm",
              choices=["low", "medium", "high", "critical"]),
        Param("min_hits", int, default=1, group="Alarm", advanced=True,
              label="Hot frames in a row before alarming",
              description="1 alarms on the first frame over the threshold. Raise "
                          "it if a camera alarms on single noisy frames."),
        Param("fps", float, default=10.0, group="Video", advanced=True,
              label="Frames analysed per second",
              description="The model was trained on 24-30 fps video; 10-15 is "
                          "measured to work, below ~6 it stops being reliable. "
                          "Higher costs adapter CPU."),
        Param("frame_width", int, default=640, group="Video", advanced=True,
              description="Decode width. The pose model runs at 640 px, so "
                          "wider only costs bandwidth."),
        Param("hold_s", float, default=5.0, group="Video", advanced=True,
              label="Seconds a fallen person stays marked on the live view"),
        Param("attach_snapshot", bool, default=True, group="Alarm", advanced=True,
              description="Attach the frame that raised the alarm, with the "
                          "skeleton drawn on it."),
    ],
    emits=[
        AlertType("fall", severity="critical",
                  description="A person fell."),
    ],
    state_schema=[
        StateView("falls_today", "Falls today", path="falls_today"),
        StateView("cameras", "Cameras", path="cameras", kind="table",
                  columns=["camera", "fps", "score", "window", "last_fall", "error"]),
        StateView("recent", "Recent falls", path="recent", kind="log"),
    ],
    # The first-class page: live video with the skeletons, the scores,
    # and the falls. Without it the falls live only in the alert inbox.
    provides=["falls"],
    overlay=True,
)


@dataclass
class FallConfig(BaseAppConfig):
    kaic_url: str = ""
    kaic_api_key: str | None = None
    fps: float = 10.0
    frame_width: int = 640
    threshold: float = 0.90
    min_hits: int = 1
    cooldown_s: float = 10.0
    hold_s: float = 5.0
    alert_severity: str = "critical"
    attach_snapshot: bool = True


def _stream_settings(cfg: dict) -> tuple:
    return (float(cfg.get("fps", 10.0) or 10.0),
            int(cfg.get("frame_width", 640) or 640))


# ── adapter result → what this app reasons about ──


def parse_result(message) -> dict:
    """The adapter's §6 result message → ``{"persons", "fall", "w", "h"}``.

    A §7 failure envelope travels in the same ``result`` slot as a real
    result; read as a result it is "nobody in frame", which would look
    healthy while nothing is being watched. Raise instead."""
    body = (message or {}).get("result") or message or {}
    if not isinstance(body, dict):
        raise FallInferenceUnavailable("adapter returned no result")
    error = body.get("error")
    if isinstance(error, dict) or body.get("status") == "error":
        detail = ""
        if isinstance(error, dict):
            detail = str(error.get("code") or error.get("message") or "")
        raise FallInferenceUnavailable(detail or "adapter returned an error envelope")
    fall = body.get("fall")
    if not isinstance(fall, dict):
        # A pose-only adapter registered under this name answers without
        # one: every frame would score None forever, silently.
        raise FallInferenceUnavailable(
            "adapter answered without a 'fall' block — is the "
            f"'{FALL_ADAPTER}' name registered to the fall-detection adapter?")
    dims = body.get("frame_dimensions") or {}
    return {
        "persons": [p for p in (body.get("persons") or []) if isinstance(p, dict)],
        "fall": fall,
        "w": int(dims.get("w") or 0),
        "h": int(dims.get("h") or 0),
    }


def overlay_boxes(parsed: dict, monitor: FallMonitor, now: float) -> list[dict]:
    """Boxes for the live view, normalized, with keypoints.

    The person the model is watching carries the verdict and its FALL
    score; anyone else in frame — and the watched person while the
    window fills — is a plain ``person`` with YOLO's detection
    confidence. Every box carries a number: OpenNVR core drops an app
    box whose score is null, and reads a missing one as 100%."""
    w, h = parsed["w"], parsed["h"]
    if w <= 0 or h <= 0:
        return []
    fall = parsed["fall"]
    subject = fall.get("subject_index")
    score = fall.get("score")
    window = fall.get("window") or {}
    boxes = []
    for i, person in enumerate(parsed["persons"]):
        bbox = person.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in bbox)
        person_score = float(person.get("score") or 0.0)
        if i == subject and monitor.fallen(now):
            label, box_score = "FALLEN", score if score is not None else person_score
        elif i == subject and score is not None:
            label, box_score = ("Fall risk" if score >= 0.5 else "Upright"), score
        elif i == subject:
            label = f"person (warming {window.get('filled', 0)}/{window.get('size', 30)})"
            box_score = person_score
        else:
            label, box_score = "person", person_score
        entry = {
            "id": i,
            "label": label,
            "score": round(float(box_score), 3),
            "box": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
            # 17 x [x, y, conf], normalized like the box. A core with the
            # keypoint patch draws the skeleton; a stock core ignores them.
            "keypoints": [
                [round(float(k[0]) / w, 4), round(float(k[1]) / h, 4),
                 round(float(k[2]), 3)]
                for k in (person.get("keypoints") or [])[:17] if len(k) >= 3
            ],
        }
        boxes.append(entry)
    return boxes


def annotate(image, parsed: dict):
    """The alarm frame with the watched person's box and skeleton drawn
    — what the evidence viewer shows. Returns a copy."""
    import cv2

    out = image.copy()
    subject = parsed["fall"].get("subject_index")
    for i, person in enumerate(parsed["persons"]):
        colour = (0, 0, 255) if i == subject else (200, 200, 200)
        x1, y1, x2, y2 = (int(v) for v in person.get("bbox") or (0, 0, 0, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        kps = person.get("keypoints") or []
        for a, b in SKELETON:
            if a < len(kps) and b < len(kps) and kps[a][2] > 0.3 and kps[b][2] > 0.3:
                cv2.line(out, (int(kps[a][0]), int(kps[a][1])),
                         (int(kps[b][0]), int(kps[b][1])), colour, 2)
    score = parsed["fall"].get("score")
    cv2.rectangle(out, (0, 0), (260, 34), (0, 0, 0), -1)
    cv2.putText(out, f"FALL {score:.2f}" if score is not None else "FALL",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out


class CameraWorker:
    """One camera: frames in, scores and alarms out."""

    def __init__(self, app: "FallDetectionApp", camera) -> None:
        self.app = app
        self.camera = camera
        self.handle = getattr(camera, "handle", str(camera))
        self.monitor = FallMonitor(app.policy())
        self.stream = None
        self.infer: InferStream | None = None
        self._stop = threading.Event()
        self._reopen = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream_settings: tuple | None = None
        self.frames = 0
        self.fps = 0.0
        self.window = "0/30"
        self.last_fall: str = ""
        self.last_error: str | None = None
        self._infer_error: str | None = None
        self._infer_failures = 0

    # ── lifecycle ──

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"fall-{self.handle}")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._reopen.set()
        if self.stream is not None:
            self.stream.close()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10.0)
        self._close_infer()

    def reopen(self, why: str) -> None:
        log.info("%s: reopening the stream (%s)", self.handle, why)
        self._reopen.set()

    def stream_settings_changed(self, cfg: dict) -> bool:
        return (self._stream_settings is not None
                and self._stream_settings != _stream_settings(cfg))

    @property
    def inference_down(self) -> bool:
        return self._infer_error is not None

    def _close_infer(self) -> None:
        infer, self.infer = self.infer, None
        if infer is not None:
            try:
                infer.close()
            except Exception:  # noqa: BLE001 - closing a dead socket
                log.debug("%s: closing inference session", self.handle, exc_info=True)

    def _new_infer(self) -> InferStream:
        """A fresh adapter session — and with it a fresh 30-frame window."""
        self._close_infer()
        self.infer = InferStream(self.app.kaic_url, self.app.kaic_key,
                                 adapter=FALL_ADAPTER, camera_id=self.handle,
                                 client_id=APP_ID)
        self.monitor.reset()
        return self.infer

    # ── the loop ──

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._session()
            except BaseException as exc:  # noqa: BLE001 - see guard-scan's _run
                self.last_error = str(exc) or exc.__class__.__name__
                log.warning("%s: worker failed (%s), retrying", self.handle, exc,
                            exc_info=True)
                if isinstance(exc, (KeyboardInterrupt, SystemExit)) and self._stop.is_set():
                    return
                self._stop.wait(10.0)

    def _session(self) -> None:
        cfg = self.app.camera_config(self.handle)
        self._reopen.clear()
        self._stream_settings = _stream_settings(cfg)
        fps, width = self._stream_settings
        self.monitor.policy = self.app.policy()
        self.stream = self.app.nvr.stream(self.camera, width=width, fps=fps)
        self._new_infer()
        last = time.monotonic()
        for frame in self.stream.frames(timeout=15.0):
            if self._stop.is_set() or self._reopen.is_set():
                self.stream.close()
                return
            if frame.restarted:
                # The feed broke: what the adapter's window holds ends
                # before the gap. A new session starts a new window.
                log.info("%s: feed restarted — new inference window", self.handle)
                self._new_infer()
            self._on_frame(frame)
            self.frames += 1
            now = time.monotonic()
            dt, last = now - last, now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

    def _on_frame(self, frame) -> None:
        import cv2

        image = frame.to_ndarray()
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return
        try:
            parsed = parse_result(self.infer.infer(buf.tobytes()))
        except Exception as exc:  # noqa: BLE001
            self._note_infer_failure(str(exc) or exc.__class__.__name__)
            # The session is suspect after a failure (InferStream tears
            # itself down); the next one is a new window.
            self.monitor.reset()
            return
        if self.inference_down:
            log.info("%s: fall inference recovered", self.handle)
        self._infer_failures = 0
        if self.last_error == self._infer_error:
            self.last_error = None
        self._infer_error = None

        fall = parsed["fall"]
        window = fall.get("window") or {}
        self.window = f"{window.get('filled', 0)}/{window.get('size', 30)}"
        alarm = self.monitor.observe(fall.get("score"), frame.wall_ts)
        if alarm is not None:
            self.last_fall = time.strftime("%H:%M:%S", time.localtime(alarm.at))
            self.app.on_fall(self.handle, alarm, image, parsed)
        self.app.publish_overlay(self.handle, overlay_boxes(parsed, self.monitor, frame.wall_ts),
                                 frame)

    def _note_infer_failure(self, reason: str) -> None:
        self._infer_failures += 1
        log.warning("%s: fall inference failed (%s) — is the '%s' adapter "
                    "registered with KAI-C?", self.handle, reason, FALL_ADAPTER)
        if self._infer_failures >= INFER_FAILURES_BEFORE_UNHEALTHY:
            self._infer_error = f"fall inference failing: {reason}"
            self.last_error = self._infer_error
            self._stop.wait(min(2.0 * self._infer_failures, 30.0))


class _NoPollSource:
    """The base class polls; this app streams (see guard-scan)."""

    def get_frame(self, camera_id: str) -> bytes | None:
        return None


class FallDetectionApp(FrameApp):
    """Roster, config, alarms and the overlay. Frames are the workers'."""

    def __init__(self, config) -> None:
        self.config = config
        self._nvr: OpenNVR | None = None
        self.kaic_url = getattr(config, "kaic_url", "") or ""
        self.kaic_key = getattr(config, "kaic_api_key", "") or ""
        self.events = DomainEventPublisher(
            getattr(config, "nats_alerts_url", "") or "",
            token=getattr(config, "nats_alerts_token", None),
            producer=f"app:{APP_ID}")
        self.workers: dict[str, CameraWorker] = {}
        self._workers_lock = threading.Lock()
        self.falls_today = 0
        self._day = time.strftime("%Y-%m-%d")
        self.recent: list[str] = []
        #: Overlay publishes, counted: the overlay is fire-and-forget, so
        #: without these a bus refusing every box looks like a healthy app
        #: whose boxes simply never appear.
        self._overlay_lock = threading.Lock()
        self.overlay = {"published": 0, "failed": 0, "last_ok_at": None,
                        "last_error": None, "last_boxes": 0}
        dispatcher = build_dispatcher(
            webhook_url=getattr(config, "webhook_url", None),
            nats_alerts_url=getattr(config, "nats_alerts_url", None),
            nats_alerts_token=getattr(config, "nats_alerts_token", None))
        self.dispatcher = dispatcher
        super().__init__(config, dispatcher, frame_source=_NoPollSource(),
                         cameras=[], poll_interval_seconds=30.0)

    # Built on first use, after registration minted this app's key — see
    # guard-scan's `nvr` property for why that ordering matters.
    @property
    def nvr(self) -> OpenNVR:
        if self._nvr is None:
            self._nvr = OpenNVR()
        return self._nvr

    @nvr.setter
    def nvr(self, client) -> None:
        self._nvr = client

    # ── config ──

    def camera_config(self, handle: str) -> dict:
        keys = ("fps", "frame_width", "threshold", "min_hits", "cooldown_s",
                "hold_s", "alert_severity", "attach_snapshot")
        return {k: getattr(self.config, k) for k in keys if hasattr(self.config, k)}

    def policy(self) -> FallPolicy:
        return FallPolicy.from_config(self.camera_config(""))

    def on_config_update(self, config: dict) -> None:
        """Live edits from the catalog form. Alarm knobs apply on the next
        frame; a new frame rate or width needs a new stream."""
        for key, value in (config or {}).items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        policy = self.policy()
        for handle, worker in self._workers():
            worker.monitor.policy = policy
            if worker.stream_settings_changed(self.camera_config(handle)):
                worker.reopen("frame rate or width changed")

    # ── results ──

    def on_fall(self, handle: str, alarm, image, parsed: dict) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._day:
            self._day, self.falls_today = today, 0
        self.falls_today += 1
        stamp = time.strftime("%H:%M:%S", time.localtime(alarm.at))
        self.recent.append(f"{stamp} {handle} fall {alarm.score:.2f}")
        del self.recent[:-20]
        log.warning("%s: FALL (score %.3f, peak %.3f)", handle, alarm.score, alarm.peak)

        images = {}
        if getattr(self.config, "attach_snapshot", True):
            images = self._evidence(image, parsed)
        severity = str(getattr(self.config, "alert_severity", "critical") or "critical")
        self.dispatcher.fire(Alert(
            # Named explicitly: the bus lets an app publish only under its
            # own id, and these fire on worker threads (guard-scan's note).
            source=AlertSource(kind="app", name=APP_ID, version=MANIFEST.version),
            title="Person fallen",
            description=(f"A fall was detected on {handle} "
                         f"(score {alarm.score:.2f}, threshold "
                         f"{self.policy().threshold:.2f})."),
            camera_id=handle,
            severity=severity,
            alert_type="fall",
            images=images,
            evidence={
                "score": round(alarm.score, 4),
                "peak": round(alarm.peak, 4),
                "hits": alarm.hits,
                "threshold": self.policy().threshold,
                "observed_at": alarm.at,
            },
            tags=["fall"],
        ))

    def _evidence(self, image, parsed: dict) -> dict:
        """The alarm frame, annotated, in the evidence store. Never in the
        alert itself (1 MB bus ceiling). An upload failure still raises
        the alarm, without the picture."""
        import cv2

        try:
            ok, buf = cv2.imencode(".jpg", annotate(image, parsed),
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            path = self.nvr.save_evidence(buf.tobytes()) if ok else None
        except Exception:  # noqa: BLE001
            log.warning("could not store the fall picture", exc_info=True)
            return {}
        return {"scene": path} if path else {}

    def publish_overlay(self, handle: str, boxes: list[dict], frame) -> None:
        """Hand this frame's boxes to the bus, and count what happened.

        ``publish_overlay`` never raises on bus trouble; it returns False.
        That return value used to be dropped, so a refused or dead bus
        connection meant no boxes and no trace of why."""
        error = None
        try:
            ok = self.events.publish_overlay(camera_id=handle, boxes=boxes, seq=frame.seq)
            if ok is False:
                error = "the bus did not accept the overlay (see the NATS lines in this log)"
        except Exception as exc:  # noqa: BLE001
            error = f"{exc.__class__.__name__}: {exc}"
        with self._overlay_lock:
            o = self.overlay
            o["last_boxes"] = len(boxes)
            if error is None:
                o["published"] += 1
                o["last_ok_at"] = time.strftime("%H:%M:%S")
                return
            o["failed"] += 1
            o["last_error"] = error
            failed = o["failed"]
        if failed in (1, 10, 100) or failed % 1000 == 0:
            log.warning("%s: overlay publish failed (#%d): %s", handle, failed, error)

    # ── state the catalog and the Falls page show ──

    def state_snapshot(self) -> dict:
        """``GET /state`` — what the catalog and the Falls page show. The
        SDK's contract server calls THIS name; a method called ``state``
        is never reached, and /state answers ``{}``."""
        cameras = []
        for handle, w in self._workers():
            error = w.last_error or ""
            if not error and w.frames > 30 and w.fps < MIN_USEFUL_FPS:
                error = (f"only {w.fps:.1f} fps reaching the model — below "
                         f"{MIN_USEFUL_FPS:.0f} a fall is a few frames of its window")
            score = w.monitor.last_score
            cameras.append({
                "camera": handle,
                "fps": round(w.fps, 1),
                "score": None if score is None else round(score, 3),
                "window": w.window,
                "fallen": w.monitor.fallen(time.time()),
                "last_fall": w.last_fall,
                "error": error,
            })
        with self._overlay_lock:
            overlay = dict(self.overlay)
        return {"falls_today": self.falls_today, "cameras": cameras,
                "recent": list(reversed(self.recent)), "overlay": overlay}

    def not_ready_reason(self) -> str | None:
        workers = self._workers()
        if not workers:
            return ("No cameras selected — select them in this app's configuration "
                    "(App Catalog → Configure → Cameras).")
        down = sorted(h for h, w in workers if w.inference_down)
        if len(down) == len(workers):
            return (f"Fall inference is failing on every camera — check the "
                    f"'{FALL_ADAPTER}' adapter is running and registered.")
        if down:
            return f"Fall inference is failing on {', '.join(down)}."
        return None

    # ── the FrameApp surface ──

    def setup(self) -> None:
        if not self._reconcile_roster():
            log.warning("no cameras selected for this app yet")

    def on_cameras_update(self, camera_ids) -> None:
        self._reconcile_roster()

    def _workers(self) -> list[tuple[str, CameraWorker]]:
        with self._workers_lock:
            return list(self.workers.items())

    def _reconcile_roster(self) -> int:
        """Workers = picked cameras. ``None`` from core keeps what runs;
        ``[]`` stops everything (guard-scan explains why both matter)."""
        try:
            cameras = self.nvr.roster()
        except Exception as exc:  # noqa: BLE001
            cameras = None
            log.warning("could not read the camera roster (%s)", exc)
        if cameras is None:
            return len(self.workers)
        picked = {c.handle: c for c in cameras}
        for handle, cam in picked.items():
            if handle not in self.workers:
                worker = CameraWorker(self, cam)
                with self._workers_lock:
                    self.workers[handle] = worker
                worker.start()
                log.info("watching %s (%s)", handle, getattr(cam, "name", handle))
        with self._workers_lock:
            departed = [(h, self.workers.pop(h)) for h in list(self.workers)
                        if h not in picked]
        for handle, worker in departed:
            log.info("%s is no longer selected — stopping", handle)
            worker.stop()
        return len(picked)

    def on_frame(self, camera_id: str, frame_bytes: bytes):
        """Unused: the workers read frames themselves, at video rate."""
        return None

    def handle_tick(self):
        self._reconcile_roster()
        for handle, w in self._workers():
            log.info("%s: %.1f fps, %d frames, window %s, score %s%s", handle,
                     w.fps, w.frames, w.window, w.monitor.last_score,
                     f", last error: {w.last_error}" if w.last_error else "")
        return []

    def stop(self) -> None:
        for _, worker in self._workers():
            worker.stop()
        super().stop()


def _fill_from_env(cfg: FallConfig) -> None:
    """Connection settings and the key come from the container's
    environment (the compose file sets them from .env), so the config
    file in the image holds no secret. A value set in the file wins."""
    import os

    key = os.environ.get("OPENNVR_INTERNAL_API_KEY") or None
    if not cfg.kaic_url:
        cfg.kaic_url = os.environ.get("KAIC_URL", "")
    if not cfg.kaic_api_key:
        cfg.kaic_api_key = key
    if not cfg.nats_alerts_url:
        cfg.nats_alerts_url = os.environ.get("NATS_ALERTS_URL") or None
    if not cfg.nats_alerts_token:
        cfg.nats_alerts_token = key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_app_config(args.config, FallConfig)
    _fill_from_env(cfg)
    app = FallDetectionApp(cfg)
    app.manifest = MANIFEST

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: app.stop())
        except ValueError:
            pass
    loop.run_until_complete(app.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
