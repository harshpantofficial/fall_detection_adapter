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
    return AppConfig(
        nats_url=nats_url,
        nats_token=str(raw["nats_token"]) if raw.get("nats_token") else None,
        subject_pattern=subject,
        cameras=cameras,
        webhook_url=str(raw["webhook_url"]) if raw.get("webhook_url") else None,
        nats_alerts_url=str(raw["nats_alerts_url"]) if raw.get("nats_alerts_url") else None,
        nats_alerts_token=str(raw["nats_alerts_token"]) if raw.get("nats_alerts_token") else None,
        contract_port=int(raw["contract_port"]) if raw.get("contract_port") is not None else None,
        contract_bind_host=str(raw.get("contract_bind_host")) if raw.get("contract_bind_host") else None,
        contract_host=str(raw.get("contract_host")) if raw.get("contract_host") else None,
        opennvr_url=str(raw["opennvr_url"]) if raw.get("opennvr_url") else None,
        opennvr_token=str(raw["opennvr_token"]) if raw.get("opennvr_token") else None,
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

    def on_detections(
        self,
        camera_id: str,
        detections: list[dict[str, Any]],
        event: dict[str, Any],
    ) -> list[Alert]:
        cam = self.cfg.cameras.get(camera_id)
        if cam is None:
            return []

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
                })
            else:
                d["last_angle"] = torso_angle
                d["last_conf"] = fall_conf
                d["anchor"] = (centre_x, centre_y)

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
                self._persons.pop(key)
                continue

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

        return {
            "fallen_now": fallen_count,
            "standing_now": standing_count,
            "today": {"alerts": sum(self._today_alerts.values())},
            "persons": persons_list,
            "per_camera": per_camera,
            "recent": list(self._recent),
            "since": self._started_at,
        }

    def ui_html(self) -> str:
        snap = self.state_snapshot()
        esc = _html.escape
        now = time.time()

        def format_time(ts):
            if not ts:
                return "—"
            delta = int(now - ts)
            return f"{delta}s ago" if delta < 60 else f"{delta // 60}m ago"

        rows = "".join(
            f"<tr><td>{esc(p['camera'])}</td><td>{esc(p['track'])}</td>"
            f"<td><span class='pill' style='background:{'#c62a2f' if p['state'] == FALLEN else '#46a758'}'>"
            f"{esc(p['state'])}</span></td><td>{p['fallen_s']}s</td><td>{p['angle_deg']}°</td>"
            f"<td>{'YES' if p['alerted'] else 'NO'}</td></tr>"
            for p in snap["persons"]
        )

        table = ("<table><tr><th>Camera</th><th>Track</th><th>State</th><th>Down</th><th>Angle</th><th>Alerted</th></tr>"
                 + rows + "</table>") if rows else "<p class='dim'>No active persons tracked.</p>"

        cards = "".join(
            f"<section class='card'><h2>{esc(r['camera'])}</h2>"
            f"<div class='stats'><div><b>{r['fallen']}</b><span class='dim'>fallen</span></div>"
            f"<div><b>{r['standing']}</b><span class='dim'>standing</span></div>"
            f"<div><b>{r['alerts_today']}</b><span class='dim'>alerts</span></div></div>"
            f"<div class='dim small'>last seen {format_time(r['last'])}</div></section>"
            for r in snap["per_camera"]
        )

        return f"""<!doctype html>
<title>Fall Detection</title>
<style>
 body {{ font: 14px system-ui, sans-serif; margin: 1.2rem; color: #1a1a1a; background: #fafafa; }}
 h1 {{ font-size: 1.2rem; margin: 0 0 .2rem }} h2 {{ font-size: .95rem; margin: .8rem 0 .4rem }}
 .dim {{ color: #6b6f76; font-weight: 400 }} .small {{ font-size: .8rem }}
 .pill {{ color: #fff; border-radius: 10px; padding: 2px 8px; font-size: .75rem; font-weight: 600; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: .8rem; margin: .8rem 0 }}
 .card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: .7rem .9rem }}
 .stats {{ display: flex; gap: 1rem; margin: .4rem 0; }}
 .stats b {{ font-size: 1.2rem; display: block }}
 table {{ border-collapse: collapse; width: 100%; margin-top: .4rem; background: #fff; }}
 th, td {{ text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #e0e0e0; }}
 th {{ color: #6b6f76; font-weight: 500 }}
</style>
<h1>Fall Detection Monitor</h1>
<div class="dim">Currently <b>{snap['fallen_now']}</b> fallen · <b>{snap['standing_now']}</b> standing · <b>{snap['today']['alerts']}</b> alerts today</div>
<h2>Active Persons</h2>
{table}
<div class="grid">{cards or "<p class='dim'>No cameras configured.</p>"}</div>
"""


def main(argv: list[str] | None = None) -> int:
    return app(FallDetectionDetector, load_config=load_config).run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
    raise SystemExit(main())