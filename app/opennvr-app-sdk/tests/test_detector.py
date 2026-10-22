# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""Detector base tests — the event-parsing + on_detections dispatch
path, exercised through ``_handle_raw`` so no NATS broker is needed."""
from __future__ import annotations

import datetime as _dt
import json
from types import SimpleNamespace
from typing import Any

import pytest

from opennvr_app_sdk import (
    Alert,
    AlertDispatcher,
    AlertType,
    AppManifest,
    Detector,
    app,
)
from opennvr_app_sdk.alerts import get_default_source


class _RecorderChannel:
    name = "recorder"

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> bool:
        self.alerts.append(alert)
        return True


MANIFEST = AppManifest(
    id="echo-detector",
    name="Echo Detector",
    version="0.9.9",
    category="test",
    subscribes="opennvr.inference.>",
    emits=[AlertType("echo", severity="low")],
)


class EchoDetector(Detector):
    """Fires one alert per watched-label detection — just enough rule
    to prove the base's parse → on_detections → dispatch path."""

    manifest = MANIFEST

    def setup(self) -> None:
        self.setup_ran = True

    def on_detections(
        self, camera_id: str, detections: list[dict[str, Any]], event: dict[str, Any],
    ):
        for det in detections:
            if not isinstance(det, dict):
                continue
            if det.get("label") == "person":
                yield Alert(
                    title="person seen",
                    description="echo",
                    camera_id=camera_id,
                    correlation_id=str(event.get("correlation_id") or ""),
                    evidence={"ts": self.parse_event_ts(event.get("completed_at"))},
                )


def _build() -> tuple[EchoDetector, _RecorderChannel]:
    recorder = _RecorderChannel()
    dispatcher = AlertDispatcher([recorder])
    cfg = SimpleNamespace(
        nats_url="nats://test:4222",
        nats_token=None,
        subject_pattern="opennvr.inference.>",
    )
    fixed_now = _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)
    detector = EchoDetector(cfg, dispatcher, clock=lambda: fixed_now)
    return detector, recorder


def _event(**overrides: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "correlation_id": "corr-42",
        "camera_id": "cam-1",
        "completed_at": "2026-01-02T03:04:05Z",
        "result": {
            "detections": [
                {"label": "person", "confidence": 0.9,
                 "bbox": {"x": 0.4, "y": 0.4, "w": 0.1, "h": 0.1}},
                {"label": "bike", "confidence": 0.8,
                 "bbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}},
            ],
        },
    }
    event.update(overrides)
    return event


# ── _handle_raw: decode + dispatch without a broker ────────────────


def test_handle_raw_dispatches_alerts_to_channel():
    detector, recorder = _build()
    fired = detector._handle_raw(json.dumps(_event()).encode())
    assert len(fired) == 1
    assert recorder.alerts == fired
    alert = fired[0]
    assert alert.camera_id == "cam-1"
    assert alert.correlation_id == "corr-42"


def test_handle_raw_skips_non_json_without_raising():
    detector, recorder = _build()
    assert detector._handle_raw(b"\x00not json{{") == []
    assert recorder.alerts == []


def test_handle_raw_isolates_on_detections_exceptions():
    detector, recorder = _build()

    class Boom(EchoDetector):
        def on_detections(self, camera_id, detections, event):
            raise RuntimeError("rule kaboom")

    boom = Boom(detector.cfg, AlertDispatcher([recorder]))
    # MUST NOT raise — one bad event can't kill a long-lived subscriber.
    assert boom._handle_raw(json.dumps(_event()).encode()) == []
    assert recorder.alerts == []


# ── handle_event: defensive payload parsing ────────────────────────


def test_handle_event_rejects_malformed_shapes():
    detector, recorder = _build()
    assert detector.handle_event(None) == []
    assert detector.handle_event("not a dict") == []
    assert detector.handle_event({"result": {"detections": []}}) == []  # no camera_id
    assert detector.handle_event({"camera_id": "cam-1"}) == []  # no result
    assert detector.handle_event(
        {"camera_id": "cam-1", "result": {"detections": "not a list"}}
    ) == []
    assert detector.handle_event(
        {"camera_id": "cam-1", "result": "not a dict"}
    ) == []
    assert recorder.alerts == []


def test_handle_event_accepts_list_return():
    detector, recorder = _build()

    class ListDetector(EchoDetector):
        def on_detections(self, camera_id, detections, event):
            return [Alert(title="t", description="d", camera_id=camera_id)]

    d = ListDetector(detector.cfg, AlertDispatcher([recorder]))
    assert len(d.handle_event(_event())) == 1
    assert len(recorder.alerts) == 1


def test_handle_event_accepts_none_return():
    detector, recorder = _build()

    class QuietDetector(EchoDetector):
        def on_detections(self, camera_id, detections, event):
            return None

    d = QuietDetector(detector.cfg, AlertDispatcher([recorder]))
    assert d.handle_event(_event()) == []
    assert recorder.alerts == []


def test_on_detections_is_abstract():
    detector, _ = _build()
    base = Detector(detector.cfg, AlertDispatcher([_RecorderChannel()]))
    with pytest.raises(NotImplementedError):
        base.handle_event(_event())


# ── Timestamp parsing ──────────────────────────────────────────────


def test_parse_event_ts_iso_with_z_suffix():
    detector, _ = _build()
    ts = detector.parse_event_ts("2026-01-02T03:04:05Z")
    expected = _dt.datetime(
        2026, 1, 2, 3, 4, 5, tzinfo=_dt.timezone.utc,
    ).timestamp()
    assert ts == expected


def test_parse_event_ts_naive_assumed_utc():
    detector, _ = _build()
    ts = detector.parse_event_ts("2026-01-02T03:04:05")
    expected = _dt.datetime(
        2026, 1, 2, 3, 4, 5, tzinfo=_dt.timezone.utc,
    ).timestamp()
    assert ts == expected


def test_parse_event_ts_falls_back_to_clock_on_garbage():
    detector, _ = _build()
    fixed = _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc).timestamp()
    assert detector.parse_event_ts("not-a-timestamp") == fixed
    assert detector.parse_event_ts(None) == fixed
    assert detector.parse_event_ts(12345) == fixed  # non-str falls back too


# ── Lifecycle glue ─────────────────────────────────────────────────


def test_alert_source_scoped_to_handler_not_process_global():
    # Constructing a detector must NOT mutate the process default —
    # several detectors can share one process (the camera agent's
    # create_monitor case) without clobbering each other.
    before = get_default_source()
    detector, recorder = _build()
    assert get_default_source() == before
    # But alerts created inside the rule inherit the app identity.
    detector.handle_event(_event())
    assert recorder.alerts[0].source.name == "echo-detector"
    assert recorder.alerts[0].source.version == "0.9.9"
    # And the default is restored once the handler returns.
    assert get_default_source() == before


def test_setup_hook_runs_at_construction():
    detector, _ = _build()
    assert detector.setup_ran is True


def test_keyed_state_convenience():
    detector, _ = _build()
    states = detector.keyed_state(ttl=3.0)
    rec = states.touch("k", at=1.0)
    assert rec.age == 0.0


def test_stop_sets_stop_event():
    detector, _ = _build()
    assert not detector._stop_event.is_set()
    detector.stop()
    assert detector._stop_event.is_set()


# ── app() runner wiring ────────────────────────────────────────────


def test_app_requires_a_config_loader():
    with pytest.raises(TypeError, match="load_config"):
        app(EchoDetector)


def test_app_returns_2_on_config_error(capsys):
    def bad_loader(path: str):
        raise ValueError("nope, bad config")

    runner = app(EchoDetector, load_config=bad_loader)
    rc = runner.run(["--config", "whatever.yml"])
    assert rc == 2
    assert "config error: nope, bad config" in capsys.readouterr().err


def test_app_uses_class_load_config_when_present():
    class WithLoader(EchoDetector):
        @classmethod
        def load_config(cls, path: str):
            raise ValueError("loader was called")

    runner = app(WithLoader)
    assert runner.run(["--config", "x.yml"]) == 2  # proves the loader ran
