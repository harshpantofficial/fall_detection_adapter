# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""AlertSubscriber base tests — the decode + on_alert dispatch path,
exercised through ``_handle_raw`` so no NATS broker is needed."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from opennvr_app_sdk import (
    AlertSubscriber,
    AlertType,
    AppManifest,
    alert_app,
)

MANIFEST = AppManifest(
    id="echo-bridge",
    name="Echo Bridge",
    version="0.9.9",
    category="test",
    subscribes="opennvr.alerts.>",
    emits=[AlertType("never", severity="low")],
)


class EchoBridge(AlertSubscriber):
    """Records every (alert, subject) pair — just enough sink to prove
    the base's decode → on_alert path."""

    manifest = MANIFEST

    def setup(self) -> None:
        self.setup_ran = True
        self.seen: list[tuple[dict[str, Any], str]] = []

    def on_alert(self, alert: dict[str, Any], subject: str) -> None:
        self.seen.append((alert, subject))


def _build() -> EchoBridge:
    cfg = SimpleNamespace(
        nats_url="nats://test:4222",
        nats_token=None,
        subject_pattern="opennvr.alerts.>",
    )
    return EchoBridge(cfg)


def _envelope(**overrides: Any) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "alert_id": "alrt_abc123",
        "fired_at": "2026-05-21T14:33:24+00:00",
        "title": "Person loitering",
        "description": "Detected loitering.",
        "severity": "medium",
        "source": {"kind": "app", "name": "loitering-detection", "version": "1.0.0"},
        "camera_id": "cam-back-shed",
        "correlation_id": "corr-xyz",
        "evidence": {},
        "tags": ["loitering"],
    }
    envelope.update(overrides)
    return envelope


# ── _handle_raw: decode + dispatch without a broker ────────────────


def test_handle_raw_delivers_alert_dict_and_subject():
    bridge = _build()
    subject = "opennvr.alerts.app.loitering-detection.cam-back-shed"
    ok = bridge._handle_raw(json.dumps(_envelope()).encode(), subject=subject)
    assert ok is True
    assert bridge.seen == [(_envelope(), subject)]


def test_handle_raw_skips_non_json_without_raising():
    bridge = _build()
    assert bridge._handle_raw(b"\x00not json{{") is False
    assert bridge.seen == []


def test_handle_raw_isolates_on_alert_exceptions():
    bridge = _build()

    class Boom(EchoBridge):
        def on_alert(self, alert, subject):
            raise RuntimeError("sink kaboom")

    boom = Boom(bridge.cfg)
    # MUST NOT raise — one bad envelope can't kill a long-lived bridge.
    assert boom._handle_raw(json.dumps(_envelope()).encode()) is False


def test_handle_raw_accepts_non_dict_json():
    """The sink receives whatever JSON arrived — a non-dict payload is
    the sink's problem to reject, not the loop's problem to crash on."""
    bridge = _build()
    assert bridge._handle_raw(b"[1, 2, 3]", subject="s") is True
    assert bridge.seen == [([1, 2, 3], "s")]


def test_on_alert_is_abstract():
    bridge = _build()
    base = AlertSubscriber(bridge.cfg)
    with pytest.raises(NotImplementedError):
        base.on_alert(_envelope(), "subject")
    # Through the loop path it is isolated like any handler failure.
    assert base._handle_raw(json.dumps(_envelope()).encode()) is False


# ── Contract wiring (spec §03) ─────────────────────────────────────


def test_contract_counts_decoded_alerts_as_events():
    bridge = _build()
    assert bridge.health_snapshot()["events_seen"] == 0
    bridge._handle_raw(json.dumps(_envelope()).encode(), subject="s")
    bridge._handle_raw(b"not json")  # skipped — not counted
    health = bridge.health_snapshot()
    assert health["events_seen"] == 1
    assert health["alerts_fired"] == 0  # subscribers consume, not emit
    assert health["last_event_age_s"] is not None


def test_manifest_snapshot_defaults_to_empty_dict():
    cfg = SimpleNamespace(
        nats_url="nats://test:4222",
        nats_token=None,
        subject_pattern="opennvr.alerts.>",
    )

    class Bare(AlertSubscriber):
        def on_alert(self, alert, subject):
            pass

    assert Bare(cfg).manifest_snapshot() == {}
    assert EchoBridge(cfg).manifest_snapshot()["id"] == "echo-bridge"


# ── Lifecycle glue ─────────────────────────────────────────────────


def test_setup_hook_runs_at_construction():
    bridge = _build()
    assert bridge.setup_ran is True


def test_config_compat_alias():
    bridge = _build()
    assert bridge._config is bridge.cfg


def test_stop_sets_stop_event():
    bridge = _build()
    assert not bridge._stop_event.is_set()
    bridge.stop()
    assert bridge._stop_event.is_set()


# ── alert_app() runner wiring ──────────────────────────────────────


def test_alert_app_requires_a_config_loader():
    with pytest.raises(TypeError, match="load_config"):
        alert_app(EchoBridge)


def test_alert_app_returns_2_on_config_error(capsys):
    def bad_loader(path: str):
        raise ValueError("nope, bad config")

    runner = alert_app(EchoBridge, load_config=bad_loader)
    rc = runner.run(["--config", "whatever.yml"])
    assert rc == 2
    assert "config error: nope, bad config" in capsys.readouterr().err


def test_alert_app_uses_class_load_config_when_present():
    class WithLoader(EchoBridge):
        @classmethod
        def load_config(cls, path: str):
            raise ValueError("loader was called")

    runner = alert_app(WithLoader)
    assert runner.run(["--config", "x.yml"]) == 2  # proves the loader ran
