"""
Fall Detection App — OpenNVR-native monitoring app on opennvr-app-sdk.

Monitors entire camera feeds for person falls (no zone restriction required).
Tracks individuals across frames, verifies persistence (to avoid false alarms from
brief crouching or floor exercises), fires high-priority alerts when a person remains down,
and clears itself when the person stands back up.

Lifecycle of one person:
    standing ──torso angle tilt──▶ falling ──settles on floor──▶ fallen
        ▲                                                          │
        │                                                    fallen_seconds
        │                                                          ▼
        └──────────── stood up ◀──────── alert resolved ◀───── ALERT FIRED
"""
from __future__ import annotations

import datetime as _dt
import html as _html
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from opennvr_app_sdk import (
    Action,
    Alert,
    AlertType,
    AppManifest,
    Detector,
    Entity,
    Param,
    StateView,
    app,
)
from opennvr_app_sdk.config import load_yaml
from opennvr_app_sdk.state import keyed_state

logger = logging.getLogger("fall-detection")

SEVERITIES: tuple[str, ...] = ("low", "medium", "high", "critical")

# States for person tracking
STANDING, FALLING, FALLEN = "standing", "falling", "fallen"

MANIFEST = AppManifest(
    id="fall-detection",
    name="Fall Detection",
    version="1.0.0",
    category="safety",
    summary=(
        "Monitors entire camera feeds for person falls, alerting immediately when an "
        "individual falls and remains down past the persistence threshold."
    ),
    requires_tasks=["pose_estimation", "fall_detection"],
    provides=["safety_monitoring"],
    subscribes="opennvr.inference.>",
    params=[
        Param(
            "fallen_seconds",
            float,
            default=3.0,
            description="How many seconds a person must remain fallen before firing an alert.",
        ),
        Param(
            "alert_cooldown_seconds",
            float,
            default=30.0,
            description="Minimum time between repeat alerts for the same camera.",
        ),
        Param(
            "alert_severity",
            str,
            default="critical",
            choices=list(SEVERITIES),
            description="Severity level of the fall alert.",
        ),
        Param(
            "track_ttl_seconds",
            float,
            default=10.0,
            description="How long after a person leaves view before their track is cleared.",
        ),
        Param(
            "attach_snapshot",
            bool,
            default=True,
            description="Capture still snapshot from camera when alert fires.",
        ),
        Param(
            "active_hours",
            "time_range",
            description="Operating window for alerts. Empty = active 24/7.",
        ),
    ],
    emits=[
        AlertType(
            "person-fallen",
            severity="critical",
            description="A person has fallen and remained down.",
        ),
        AlertType(
            "person-fallen-resolved",
            severity="info",
            description="A fallen person has stood up or the event was resolved.",
        ),
    ],
    state_schema=[
        StateView("fallen_now", "Fallen now", kind="metric", path="fallen_now"),
        StateView("standing_now", "Standing now", kind="metric", path="standing_now"),
        StateView("alerts_today", "Alerts today", kind="metric", path="today.alerts"),
        StateView(
            "persons",
            "Persons",
            kind="table",
            path="persons",
            columns=["camera", "track", "state", "fallen_s", "angle_deg", "alerted"],
        ),
        StateView(
            "per_camera",
            "Per camera",
            kind="table",
            path="per_camera",
            columns=["camera", "fallen", "standing", "alerts_today", "last"],
        ),
        StateView("recent", "Recent events", kind="log", path="recent", limit=12),
    ],
    actions=[
        Action(
            "acknowledge",
            "Acknowledge",
            params=[Param("camera", str, default=""), Param("track", str, default="")],
            description="Operator confirms awareness; stops escalation.",
        ),
        Action(
            "resolve",
            "Resolve",
            params=[Param("camera", str, required=True), Param("track", str, required=True)],
            description="Clear fallen state for person.",
        ),
    ],
    entities=[
        Entity(
            "fallen_now",
            "sensor",
            "Fallen individuals",
            state_path="fallen_now",
            state_class="measurement",
            icon="mdi:human-cane",
        ),
        Entity(
            "alerts_today",
            "sensor",
            "Fall alerts today",
            state_path="today.alerts",
            state_class="total_increasing",
        ),
    ],
    has_ui=True,
)


@dataclass
class ActiveHours:
    start: _dt.time
    end: _dt.time

    def contains(self, when: _dt.datetime) -> bool:
        t = when.time()
        if self.start <= self.end:
            return self.start <= t < self.end
        return t >= self.start or t < self.end

    @classmethod
    def parse(cls, raw: Any) -> ActiveHours | None:
        if not isinstance(raw, dict):
            return None
        s, e = str(raw.get("start") or "").strip(), str(raw.get("end") or "").strip()
        if not s or not e:
            return None
        try:
            return cls(_dt.time.fromisoformat(s), _dt.time.fromisoformat(e))
        except ValueError:
            return None


@dataclass
class CameraWatch:
    camera_id: str
    frame_width: int = 1920
    frame_height: int = 1080


@dataclass
class AppConfig:
    nats_url: str
    nats_token: str | None
    subject_pattern: str
    cameras: dict[str, CameraWatch]
    webhook_url: str | None
    nats_alerts_url: str | None = None
    nats_alerts_token: str | None = None
    nats_alerts_subject_prefix: str = "opennvr.alerts"
    contract_port: int | None = None
    contract_bind_host: str | None = None
    contract_host: str | None = None
    opennvr_url: str | None = None
    opennvr_token: str | None = None
    fallen_seconds: float = 3.0
    alert_severity: str = "critical"
    alert_cooldown_seconds: float = 30.0
    track_ttl_seconds: float = 10.0
    attach_snapshot: bool = True
    active_hours: ActiveHours | None = None
    consume_tier0: bool = False
    auto_cameras: bool = False


def load_config(path: str) -> AppConfig:
    raw = load_yaml(path)
    nats_url = str(raw.get("nats_url") or "").strip()
    if not nats_url:
        raise ValueError("config: 'nats_url' is required")

    subject = str(raw.get("subject_pattern") or "opennvr.inference.>").strip()
    cameras_raw = raw.get("cameras") or []
    auto_cameras = not cameras_raw

    cameras: dict[str, CameraWatch] = {}
    for c in cameras_raw:
        cid = str(c["camera_id"])
        w = int(c.get("frame_width", 1920))
        h = int(c.get("frame_height", 1080))
        cameras[cid] = CameraWatch(camera_id=cid, frame_width=w, frame_height=h)

    active_hours = ActiveHours.parse(raw.get("active_hours"))
    def _token(key: str) -> str | None:
        val = str(raw.get(key) or "").strip()
        if not val:
            return None
        if val.startswith("${"):
            # Docker Compose expands ${VAR} before writing env vars but the YAML
            # file is read verbatim at runtime — fall back to env vars.
            return (
                os.getenv("OPENNVR_INTERNAL_API_KEY")
                or os.getenv("INTERNAL_API_KEY")
            ) or None
        return val

    return AppConfig(
        nats_url=nats_url,
        nats_token=_token("nats_token"),
        subject_pattern=subject,
        cameras=cameras,
        webhook_url=str(raw["webhook_url"]) if raw.get("webhook_url") else None,
        nats_alerts_url=str(raw["nats_alerts_url"]) if raw.get("nats_alerts_url") else None,
        nats_alerts_token=_token("nats_alerts_token"),
        contract_port=int(raw["contract_port"]) if raw.get("contract_port") is not None else None,
        contract_bind_host=str(raw.get("contract_bind_host")) if raw.get("contract_bind_host") else None,
        contract_host=str(raw.get("contract_host")) if raw.get("contract_host") else None,
        opennvr_url=str(raw["opennvr_url"]) if raw.get("opennvr_url") else None,
        opennvr_token=_token("opennvr_token"),
        fallen_seconds=float(raw.get("fallen_seconds", 3.0)),
        alert_severity=str(raw.get("alert_severity", "critical")).lower(),
        alert_cooldown_seconds=float(raw.get("alert_cooldown_seconds", 30.0)),
        track_ttl_seconds=float(raw.get("track_ttl_seconds", 10.0)),
        attach_snapshot=bool(raw.get("attach_snapshot", True)),
        active_hours=active_hours,
        consume_tier0=bool(raw.get("consume_tier0", False)),
        auto_cameras=auto_cameras,
    )


class FallDetectionDetector(Detector):
    """Tracks persons across video streams, alerting on persistent falls anywhere in view."""

    manifest = MANIFEST

    def setup(self) -> None:
        self._persons = keyed_state(ttl=self.cfg.track_ttl_seconds, auto_gc=False)
        self._today_alerts: dict[str, int] = {}
        self._last_alert: dict[str, float] = {}
        self._last_seen: dict[str, float] = {}
        self._recent: deque[dict[str, Any]] = deque(maxlen=50)
        self._started_at = time.time()
        self._nvr = None
        self._nvr_tried = False
        self._stream_path_cache: dict[str, str | None] = {}
        self._stream_path_last_fetch: dict[str, float] = {}

    def handle_event(self, event: Any) -> list[Alert]:
        """Support both 'persons' (from pose adapter) and 'detections' (from generic pipeline)."""
        self._contract_note_event()
        if not isinstance(event, dict):
            return []

        camera_id = event.get("camera_id")
        if not camera_id:
            return []
        if not self.camera_picked(camera_id):
            return []

        result = event.get("result") or {}
        persons = result.get("persons")
        if persons is None:
            persons = result.get("detections")

        if not isinstance(persons, list):
            return []

        produced = self.on_detections(camera_id, persons, event)
        fired: list[Alert] = []
        if produced is not None:
            for alert in produced:
                self._dispatcher.fire(alert)
                fired.append(alert)
        self._contract_note_alerts(len(fired))
        return fired

    def on_cameras_update(self, camera_ids) -> None:
        super().on_cameras_update(camera_ids)
        for cid in camera_ids:
            cid_str = f"cam{cid}" if str(cid).isdigit() else str(cid)
            if cid_str not in self.cfg.cameras:
                self.cfg.cameras[cid_str] = CameraWatch(camera_id=cid_str)
            if str(cid) not in self.cfg.cameras:
                self.cfg.cameras[str(cid)] = CameraWatch(camera_id=str(cid))
        logger.info("Cameras updated: %s (now watching: %s)", camera_ids, list(self.cfg.cameras.keys()))

    def on_config_update(self, config: dict[str, Any]) -> None:
        """Apply OpenNVR catalog configuration updates live."""
        if "fallen_seconds" in config:
            try:
                self.cfg.fallen_seconds = float(config["fallen_seconds"])
            except (ValueError, TypeError):
                pass
        if "alert_severity" in config:
            self.cfg.alert_severity = str(config["alert_severity"]).lower()
        if "alert_cooldown_seconds" in config:
            try:
                self.cfg.alert_cooldown_seconds = float(config["alert_cooldown_seconds"])
            except (ValueError, TypeError):
                pass
        if "track_ttl_seconds" in config:
            try:
                self.cfg.track_ttl_seconds = float(config["track_ttl_seconds"])
            except (ValueError, TypeError):
                pass
        logger.info("FallDetection configuration updated live: %s", config)

    def on_detections(
        self,
        camera_id: str,
        detections: list[dict[str, Any]],
        event: dict[str, Any],
    ) -> list[Alert]:
        cam = self.cfg.cameras.get(camera_id)
        if cam is None:
            cam = CameraWatch(camera_id=str(camera_id))
            self.cfg.cameras[str(camera_id)] = cam

        now = time.time()
        event_ts = self.parse_event_ts(event.get("completed_at"))
        self._last_seen[camera_id] = now
        fired: list[Alert] = []

        # Process all detected individuals across the entire camera frame
        for idx, item in enumerate(detections):
            if not isinstance(item, dict):
                continue

            bbox = item.get("bbox")
            if not bbox:
                continue

            # Centre coordinates of bounding box
            if isinstance(bbox, list) and len(bbox) >= 4:
                centre_x = (bbox[0] + bbox[2]) / 2.0
                centre_y = (bbox[1] + bbox[3]) / 2.0
            elif isinstance(bbox, dict):
                centre_x = float(bbox.get("x", 0.0)) + float(bbox.get("w", 0.0)) / 2.0
                centre_y = float(bbox.get("y", 0.0)) + float(bbox.get("h", 0.0)) / 2.0
            else:
                centre_x, centre_y = 0.0, 0.0

            # Track ID or fallback to spatial index
            track_id = str(item.get("track_id") or f"person-{idx}")
            key = (camera_id, track_id)

            fall_state = item.get("fall_state", STANDING)
            torso_angle = float(item.get("torso_angle_deg", 0.0))
            fall_conf = float(item.get("fall_confidence", item.get("confidence", 0.0)))

            # Normalised bbox dict for the UI overlay
            if isinstance(bbox, dict):
                norm_bbox = {
                    "x": float(bbox.get("x", 0.0)),
                    "y": float(bbox.get("y", 0.0)),
                    "w": float(bbox.get("w", 0.0)),
                    "h": float(bbox.get("h", 0.0)),
                }
            elif isinstance(bbox, list) and len(bbox) >= 4:
                norm_bbox = {
                    "x": float(bbox[0]),
                    "y": float(bbox[1]),
                    "w": float(bbox[2]) - float(bbox[0]),
                    "h": float(bbox[3]) - float(bbox[1]),
                }
            else:
                norm_bbox = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}

            rec = self._persons.touch(key, at=event_ts)
            d = rec.data
            if not d:
                d.update({
                    "state": fall_state,
                    "fallen_since": event_ts if fall_state == FALLEN else 0.0,
                    "last_angle": torso_angle,
                    "last_conf": fall_conf,
                    "alerted": False,
                    "wall_offset": now - event_ts,
                    "anchor": (centre_x, centre_y),
                    "bbox": norm_bbox,
                })
            else:
                d["last_angle"] = torso_angle
                d["last_conf"] = fall_conf
                d["anchor"] = (centre_x, centre_y)
                d["bbox"] = norm_bbox

                # State transitions
                if fall_state == FALLEN:
                    if d["state"] != FALLEN:
                        d["state"] = FALLEN
                        d["fallen_since"] = event_ts
                    else:
                        # Check elapsed time down
                        elapsed_down = now - (d["fallen_since"] + d.get("wall_offset", 0.0))
                        if elapsed_down >= self.cfg.fallen_seconds and not d["alerted"]:
                            alert = self._raise_fall_alert(cam, track_id, d, now, elapsed_down)
                            if alert:
                                fired.append(alert)
                elif fall_state in (STANDING, "upright"):
                    if d["state"] == FALLEN and d["alerted"]:
                        # Person stood back up
                        resolve_alert = self._raise_resolve_alert(cam, track_id, d, now)
                        if resolve_alert:
                            fired.append(resolve_alert)
                    d["state"] = STANDING
                    d["fallen_since"] = 0.0
                    d["alerted"] = False

        # Run tick to sweep aged-out states
        fired.extend(self.tick(now))
        return fired

    def tick(self, now: float | None = None) -> list[Alert]:
        now = time.time() if now is None else now
        cutoff = now - self.cfg.track_ttl_seconds
        fired: list[Alert] = []

        for key, rec in list(self._persons.items()):
            cam = self.cfg.cameras.get(key[0])
            if cam is None:
                cam = CameraWatch(camera_id=str(key[0]))
                self.cfg.cameras[str(key[0])] = cam

            d = rec.data
            # Person left view
            if rec.last_seen + d.get("wall_offset", 0.0) < cutoff:
                if d.get("alerted"):
                    self._note(
                        key[0],
                        f"Track {key[1]} left view while in fallen status",
                        "info",
                        now,
                    )
                self._persons.pop(key)
                continue

            # Fall persistence check during quiet periods
            if d.get("state") == FALLEN and not d.get("alerted") and d.get("fallen_since"):
                elapsed = now - (d["fallen_since"] + d.get("wall_offset", 0.0))
                if elapsed >= self.cfg.fallen_seconds:
                    alert = self._raise_fall_alert(cam, key[1], d, now, elapsed)
                    if alert:
                        fired.append(alert)

        return fired

    def _raise_fall_alert(
        self,
        cam: CameraWatch,
        track_id: str,
        data: dict[str, Any],
        now: float,
        elapsed_down: float,
    ) -> Alert | None:
        cam_id = cam.camera_id
        if self.cfg.active_hours and not self.cfg.active_hours.contains(_dt.datetime.now()):
            return None

        # Cooldown check
        last = self._last_alert.get(cam_id, 0.0)
        if self.cfg.alert_cooldown_seconds > 0 and (now - last) < self.cfg.alert_cooldown_seconds:
            return None

        self._last_alert[cam_id] = now
        self._today_alerts[cam_id] = self._today_alerts.get(cam_id, 0) + 1
        data["alerted"] = True

        title = f"Fall Detected on {cam_id}"
        desc = (
            f"Person {track_id} has fallen on {cam_id} "
            f"and remained down for {int(elapsed_down)}s (torso angle: {data.get('last_angle', 0)}°)."
        )

        self._note(cam_id, title, self.cfg.alert_severity, now, track=track_id)

        evidence = {
            "track_id": track_id,
            "elapsed_down_s": round(elapsed_down, 1),
            "torso_angle_deg": data.get("last_angle", 0.0),
            "confidence": data.get("last_conf", 0.0),
            "anchor": list(data.get("anchor", (0, 0))),
        }

        images = self._snapshot(cam_id)
        return Alert(
            title=title,
            description=desc,
            camera_id=cam_id,
            severity=self.cfg.alert_severity,
            alert_type="person-fallen",
            evidence=evidence,
            images=images,
            tags=["fall-detection", "safety", cam_id],
        )

    def _raise_resolve_alert(
        self,
        cam: CameraWatch,
        track_id: str,
        data: dict[str, Any],
        now: float,
    ) -> Alert | None:
        cam_id = cam.camera_id
        title = f"Fall Cleared on {cam_id}"
        desc = f"Person {track_id} has stood back up on {cam_id}."
        self._note(cam_id, title, "info", now, track=track_id)

        return Alert(
            title=title,
            description=desc,
            camera_id=cam_id,
            severity="info",
            alert_type="person-fallen-resolved",
            evidence={"track_id": track_id},
            tags=["fall-detection", "resolved", cam_id],
        )

    def _snapshot(self, camera_id: str) -> dict[str, str]:
        if not self.cfg.attach_snapshot:
            return {}
        if self._nvr is None and not self._nvr_tried:
            self._nvr_tried = True
            try:
                from opennvr_app_sdk.client import OpenNVR
                self._nvr = OpenNVR(self.cfg.opennvr_url or None, timeout=3.0)
            except Exception:
                pass
        if self._nvr:
            try:
                jpeg = self._nvr.snapshot(camera_id)
                path = self._nvr.save_evidence(jpeg) if jpeg else None
                if path:
                    return {"snapshot": path}
            except Exception:
                pass
        return {}

    def _note(self, camera_id: str, message: str, level: str, now: float, **extra: Any) -> None:
        self._recent.append({
            "camera": camera_id,
            "message": message,
            "level": level,
            "time": now,
            **extra,
        })

    def on_action(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        if name == "acknowledge":
            cam_id = str(params.get("camera") or "").strip()
            track = str(params.get("track") or "").strip()
            if cam_id and track:
                rec = self._persons.get((cam_id, track))
                if rec:
                    rec.data["acknowledged"] = True
                    self._note(cam_id, f"Fall for {track} acknowledged", "info", now)
                    return {"ok": True, "camera": cam_id, "track": track}
            return {"ok": True}
        if name == "resolve":
            cam_id = str(params.get("camera") or "").strip()
            track = str(params.get("track") or "").strip()
            self._persons.pop((cam_id, track), None)
            self._note(cam_id, f"Fall for {track} resolved by operator", "info", now)
            return {"ok": True, "camera": cam_id, "track": track}
        raise KeyError(name)

    def state_snapshot(self) -> dict[str, Any]:
        now = time.time()
        persons_list = []
        fallen_count = 0
        standing_count = 0

        for (cam_id, track), rec in self._persons.items():
            d = rec.data
            st = d.get("state", STANDING)
            if st == FALLEN:
                fallen_count += 1
                elapsed = now - (d.get("fallen_since", now) + d.get("wall_offset", 0.0))
            else:
                standing_count += 1
                elapsed = 0.0

            persons_list.append({
                "camera": cam_id,
                "track": track,
                "state": st,
                "fallen_s": round(elapsed, 1) if st == FALLEN else 0,
                "angle_deg": d.get("last_angle", 0.0),
                "alerted": d.get("alerted", False),
                "confidence": d.get("last_conf", 0.0),
                "bbox": d.get("bbox", {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}),
            })

        per_camera = []
        for cam_id in self.cfg.cameras:
            c_fallen = sum(1 for p in persons_list if p["camera"] == cam_id and p["state"] == FALLEN)
            c_standing = sum(1 for p in persons_list if p["camera"] == cam_id and p["state"] != FALLEN)
            per_camera.append({
                "camera": cam_id,
                "fallen": c_fallen,
                "standing": c_standing,
                "alerts_today": self._today_alerts.get(cam_id, 0),
                "last": self._last_seen.get(cam_id),
            })

        recent_alerts = [
            r for r in self._recent if r.get("level") in ("high", "critical")
        ][-10:]

        # Provide proxied HLS URLs for all known cameras so the UI can attach video.
        # We proxy via /proxy/hls/{cam_id}/... on our own contract server so the
        # browser never needs to authenticate against MediaMTX directly.
        stream_urls: dict[str, str | None] = {}
        for cam_id in set(list(self.cfg.cameras.keys()) + [p["camera"] for p in persons_list]):
            last = self._stream_path_last_fetch.get(cam_id, 0.0)
            if now - last > 300:  # re-fetch every 5 min
                path = self._fetch_stream_path(cam_id)
                self._stream_path_cache[cam_id] = path
                self._stream_path_last_fetch[cam_id] = now
            path = self._stream_path_cache.get(cam_id)
            stream_urls[cam_id] = f"/proxy/hls/{cam_id}/index.m3u8" if path else None

        return {
            "fallen_now": fallen_count,
            "standing_now": standing_count,
            "today": {"alerts": sum(self._today_alerts.values())},
            "persons": persons_list,
            "per_camera": per_camera,
            "recent": list(self._recent),
            "recent_alerts": recent_alerts,
            "stream_urls": stream_urls,
            "since": self._started_at,
        }

    def start_contract_server(self):
        # super() sets self._contract_server and returns it
        contract = super().start_contract_server()
        if contract is not None:
            # /api/state — JSON state snapshot for the UI to poll (includes stream_urls)
            contract._routes["/api/state"] = self.state_snapshot  # type: ignore[attr-defined]
            # Patch the request handler to serve /proxy/hls/* as raw bytes
            self._patch_hls_proxy(contract)
        return contract

    def _patch_hls_proxy(self, contract: Any) -> None:
        """Monkey-patch the ContractServer's HTTP handler to serve HLS proxy routes.

        The ContractServer only supports JSON routes. We wrap do_GET to intercept
        /proxy/hls/{camera_id}/{tail...} and proxy the bytes from internal MediaMTX.
        """
        detector_ref = self
        http_server = getattr(contract, "_server", None) or getattr(contract, "_httpd", None)
        if http_server is None:
            # Find the underlying HTTPServer object on the contract
            for attr in vars(contract):
                obj = getattr(contract, attr, None)
                if hasattr(obj, "RequestHandlerClass"):
                    http_server = obj
                    break
        if http_server is None:
            logger.warning("HLS proxy: could not locate HTTPServer on ContractServer")
            return

        OriginalHandler = http_server.RequestHandlerClass

        class PatchedHandler(OriginalHandler):  # type: ignore[valid-type]
            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path.startswith("/proxy/hls/"):
                    # /proxy/hls/{camera_id}/{tail}
                    rest = path[len("/proxy/hls/"):]
                    parts = rest.split("/", 1)
                    camera_id = parts[0]
                    tail = parts[1] if len(parts) > 1 else "index.m3u8"
                    stream_path = detector_ref._stream_path_cache.get(camera_id)
                    if not stream_path:
                        stream_path = detector_ref._fetch_stream_path(camera_id)
                        if stream_path:
                            detector_ref._stream_path_cache[camera_id] = stream_path
                    if not stream_path:
                        self.send_response(503)
                        self.end_headers()
                        return
                    result = detector_ref._proxy_hls(stream_path, tail)
                    if result is None:
                        self.send_response(502)
                        self.end_headers()
                        return
                    body, ct = result
                    # Rewrite absolute segment URLs in playlists to go via our proxy
                    if "mpegurl" in ct.lower() or tail.endswith(".m3u8"):
                        import re as _re
                        text = body.decode("utf-8", errors="replace")
                        mediamtx_base = os.getenv("MEDIAMTX_INTERNAL_URL", "http://mediamtx:8888")
                        prefix = f"{mediamtx_base.rstrip('/')}/{stream_path}/"
                        text = text.replace(prefix, f"/proxy/hls/{camera_id}/")
                        body = text.encode("utf-8")
                        ct = "application/vnd.apple.mpegurl"
                    self.send_response(200)
                    self.send_header("Content-Type", ct)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                super().do_GET()

        http_server.RequestHandlerClass = PatchedHandler
        logger.info("HLS proxy patched into contract server")

    def _fetch_stream_path(self, camera_id: str) -> str | None:
        """Return the MediaMTX stream path for this camera via the app platform endpoint.

        Uses GET /api/v1/internal/app/cameras/{id}/stream with X-Internal-Api-Key.
        Returns the 'path' field (e.g. 'cam-1') which we use to build a proxy URL.
        """
        opennvr_url = getattr(self.cfg, "opennvr_url", None)
        if not opennvr_url:
            return None
        token = (
            getattr(self.cfg, "opennvr_token", None)
            or os.getenv("OPENNVR_INTERNAL_API_KEY")
            or os.getenv("INTERNAL_API_KEY")
        )
        try:
            cam_int = int(camera_id)
        except (ValueError, TypeError):
            logger.debug("stream-path: camera_id %r is not an integer, skipping", camera_id)
            return None
        try:
            import json as _json
            import urllib.request
            req = urllib.request.Request(
                f"{opennvr_url.rstrip('/')}/api/v1/internal/app/cameras/{cam_int}/stream",
                headers={"X-Internal-Api-Key": token} if token else {},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
            return data.get("path") or None
        except Exception as exc:
            logger.debug("stream-path fetch failed for camera %s: %s", camera_id, exc)
            return None

    def _proxy_hls(self, stream_path: str, tail: str) -> tuple[bytes, str] | None:
        """Fetch an HLS playlist or segment from internal MediaMTX and return (body, content_type)."""
        mediamtx_url = os.getenv("MEDIAMTX_INTERNAL_URL", "http://mediamtx:8189")
        url = f"{mediamtx_url.rstrip('/')}/{stream_path}/{tail.lstrip('/')}"
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=10) as resp:
                body = resp.read()
                ct = resp.headers.get("Content-Type", "application/octet-stream")
            return body, ct
        except Exception as exc:
            logger.debug("HLS proxy fetch failed %s: %s", url, exc)
            return None

    def ui_html(self) -> str:
        return """<!doctype html>
<title>Fall Detection Monitor</title>
<style>
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.5 system-ui, sans-serif; background: #f5f5f5; color: #1a1a1a; }
@media (prefers-color-scheme: dark) {
  body { background: #111; color: #e8e8e8; }
  .card { background: #1e1e1e; border-color: #333; }
  th { color: #aaa; }
  td { border-color: #333; }
  .badge { border-color: #444; }
}
header { background: #111; color: #fff; padding: .6rem 1rem; display: flex; align-items: center; gap: 1rem; }
header h1 { font-size: 1rem; margin: 0; font-weight: 600; }
#metrics { display: flex; gap: 1.5rem; font-size: .85rem; }
#metrics span { opacity: .7; }
#metrics b { font-size: 1.3rem; display: block; }
main { padding: 1rem; }
.section-title { font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em;
  opacity: .5; margin: 1.2rem 0 .5rem; }
.camera-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 1rem; }
.camera-tile { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; }
@media (prefers-color-scheme: dark) { .camera-tile { background: #1e1e1e; border-color: #333; } }
.tile-header { padding: .4rem .7rem; font-size: .8rem; font-weight: 600;
  background: #222; color: #fff; display: flex; justify-content: space-between; align-items: center; }
.tile-body { position: relative; background: #000; aspect-ratio: 16/9; }
.tile-body video { width: 100%; height: 100%; object-fit: contain; display: block; }
.tile-body canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
.tile-footer { padding: .3rem .7rem; font-size: .75rem; opacity: .6; }
.badge { display: inline-block; border-radius: 10px; padding: 1px 8px; font-size: .72rem;
  font-weight: 600; border: 1px solid transparent; }
.badge-fallen { background: #c62a2f; color: #fff; }
.badge-standing { background: #46a758; color: #fff; }
.badge-loading { background: #888; color: #fff; }
table { border-collapse: collapse; width: 100%; font-size: .82rem; }
th { text-align: left; padding: .35rem .5rem; color: #6b6f76; font-weight: 500; border-bottom: 2px solid #e0e0e0; }
td { padding: .35rem .5rem; border-bottom: 1px solid #eee; }
#alerts-list { list-style: none; margin: 0; padding: 0; }
#alerts-list li { padding: .3rem 0; font-size: .82rem; border-bottom: 1px solid #eee; }
#alerts-list li:last-child { border-bottom: none; }
.no-data { opacity: .4; font-size: .85rem; padding: .5rem 0; }
</style>
<header>
  <h1>🛡 Fall Detection Monitor</h1>
  <div id="metrics">
    <div><b id="m-fallen">—</b><span>fallen now</span></div>
    <div><b id="m-standing">—</b><span>standing</span></div>
    <div><b id="m-alerts">—</b><span>alerts today</span></div>
  </div>
</header>
<main>
  <div class="section-title">Camera Streams</div>
  <div id="camera-grid" class="camera-grid"><p class="no-data">Waiting for detections…</p></div>
  <div class="section-title">Active Persons</div>
  <table id="persons-table">
    <thead><tr><th>Camera</th><th>Track</th><th>State</th><th>Down</th><th>Angle</th><th>Alerted</th></tr></thead>
    <tbody id="persons-body"><tr><td colspan="6" class="no-data">No active persons.</td></tr></tbody>
  </table>
  <div class="section-title">Recent Alerts</div>
  <ul id="alerts-list"><li class="no-data">No recent alerts.</li></ul>
</main>
<script src="https://cdn.jsdelivr.net/npm/hls.js@1.5.15/dist/hls.min.js"></script>
<script>
const hlsInstances = {};

async function fetchState() {
  try {
    const r = await fetch('/api/state');
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

function ensureCameraTile(cameraId) {
  const gridId = 'tile-' + CSS.escape(cameraId);
  if (document.getElementById(gridId)) return;

  const grid = document.getElementById('camera-grid');
  // Clear placeholder text
  const placeholder = grid.querySelector('.no-data');
  if (placeholder) placeholder.remove();

  const tile = document.createElement('div');
  tile.id = gridId;
  tile.className = 'camera-tile';
  tile.innerHTML = `
    <div class="tile-header">
      <span>${cameraId}</span>
      <span id="badge-${CSS.escape(cameraId)}" class="badge badge-loading">loading</span>
    </div>
    <div class="tile-body">
      <video id="video-${CSS.escape(cameraId)}" autoplay muted playsinline></video>
      <canvas id="canvas-${CSS.escape(cameraId)}"></canvas>
    </div>
    <div id="footer-${CSS.escape(cameraId)}" class="tile-footer">Initialising stream…</div>
  `;
  grid.appendChild(tile);

  // HLS URL arrives via /api/state stream_urls — attached by attachStream()
}

function attachStream(cameraId, hlsUrl) {
  if (hlsInstances[cameraId]) return;
  const video = document.getElementById('video-' + CSS.escape(cameraId));
  const footer = document.getElementById('footer-' + CSS.escape(cameraId));
  if (!video) return;
  if (Hls.isSupported()) {
    const hls = new Hls({ lowLatencyMode: true });
    hls.loadSource(hlsUrl);
    hls.attachMedia(video);
    hls.on(Hls.Events.ERROR, (_, d) => {
      if (d.fatal && footer) footer.textContent = 'Stream error: ' + d.type;
    });
    hls.on(Hls.Events.MANIFEST_PARSED, () => {
      if (footer) footer.textContent = 'Live';
    });
    hlsInstances[cameraId] = hls;
  } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
    video.src = hlsUrl;
    hlsInstances[cameraId] = true;
    if (footer) footer.textContent = 'Live (native HLS)';
  } else {
    if (footer) footer.textContent = 'HLS not supported in this browser.';
  }
}

function drawOverlay(cameraId, persons) {
  const canvas = document.getElementById('canvas-' + CSS.escape(cameraId));
  const video = document.getElementById('video-' + CSS.escape(cameraId));
  if (!canvas || !video) return;

  const W = video.videoWidth || video.offsetWidth || 640;
  const H = video.videoHeight || video.offsetHeight || 360;
  canvas.width = W;
  canvas.height = H;

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  for (const p of persons) {
    const bbox = p.bbox || {};
    const x = (bbox.x || 0) * W;
    const y = (bbox.y || 0) * H;
    const w = (bbox.w || 0) * W;
    const h = (bbox.h || 0) * H;
    if (w < 2 || h < 2) continue;  // skip zero-area boxes

    const fallen = p.state === 'fallen';
    ctx.strokeStyle = fallen ? '#ff3333' : '#33ff66';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    const label = fallen
      ? `FALLEN ${(p.confidence * 100).toFixed(0)}%`
      : `person ${(p.confidence * 100).toFixed(0)}%`;
    ctx.font = 'bold 12px system-ui';
    ctx.fillStyle = fallen ? '#ff3333' : '#33ff66';
    ctx.fillText(label, x + 2, y > 14 ? y - 3 : y + 13);
  }
}

function timeSince(ts) {
  if (!ts) return '—';
  const s = Math.floor(Date.now() / 1000 - ts);
  return s < 60 ? s + 's ago' : Math.floor(s / 60) + 'm ago';
}

async function refresh() {
  const state = await fetchState();
  if (!state) return;

  document.getElementById('m-fallen').textContent = state.fallen_now ?? 0;
  document.getElementById('m-standing').textContent = state.standing_now ?? 0;
  document.getElementById('m-alerts').textContent = (state.today || {}).alerts ?? 0;

  // Group persons by camera
  const byCamera = {};
  for (const p of (state.persons || [])) {
    if (!byCamera[p.camera]) byCamera[p.camera] = [];
    byCamera[p.camera].push(p);
  }

  // Ensure a tile exists for every camera with detections; attach stream + draw overlays
  const streamUrls = state.stream_urls || {};
  for (const [camId, persons] of Object.entries(byCamera)) {
    ensureCameraTile(camId);
    if (streamUrls[camId] && !hlsInstances[camId]) {
      attachStream(camId, streamUrls[camId]);
    }
    drawOverlay(camId, persons);

    // Update badge
    const badge = document.getElementById('badge-' + CSS.escape(camId));
    if (badge) {
      const nFallen = persons.filter(p => p.state === 'fallen').length;
      if (nFallen > 0) {
        badge.className = 'badge badge-fallen';
        badge.textContent = nFallen + ' fallen';
      } else {
        badge.className = 'badge badge-standing';
        badge.textContent = persons.length + ' standing';
      }
    }
  }

  // Persons table
  const tbody = document.getElementById('persons-body');
  if (tbody) {
    if (!state.persons || state.persons.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" class="no-data">No active persons.</td></tr>';
    } else {
      tbody.innerHTML = state.persons.map(p => {
        const color = p.state === 'fallen' ? '#c62a2f' : '#46a758';
        return `<tr>
          <td>${p.camera}</td><td>${p.track}</td>
          <td><span class="badge" style="background:${color};color:#fff">${p.state}</span></td>
          <td>${p.fallen_s}s</td><td>${p.angle_deg.toFixed(1)}°</td>
          <td>${p.alerted ? '⚠️ YES' : 'no'}</td>
        </tr>`;
      }).join('');
    }
  }

  // Recent alerts
  const alertsList = document.getElementById('alerts-list');
  if (alertsList) {
    const alerts = state.recent_alerts || [];
    if (alerts.length === 0) {
      alertsList.innerHTML = '<li class="no-data">No recent alerts.</li>';
    } else {
      alertsList.innerHTML = alerts.slice().reverse().map(a =>
        `<li><b>${a.camera}</b> — ${a.message} <span style="opacity:.5">${timeSince(a.time)}</span></li>`
      ).join('');
    }
  }
}

setInterval(refresh, 1000);
refresh();
</script>"""


def main(argv: list[str] | None = None) -> int:
    return app(FallDetectionDetector, load_config=load_config).run(argv)


if __name__ == "__main__":
    raise SystemExit(main())