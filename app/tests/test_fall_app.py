# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The app around the rule: reading the adapter, the overlay, the alarm.

No core, no camera, no adapter: the SDK pieces the worker talks to are
faked at their edges, so what runs is the app's own code."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fall_detection as fd  # noqa: E402
from fall_rules import FallMonitor, FallPolicy  # noqa: E402

W, H = 640, 360


def person(x1=64, y1=36, x2=192, y2=324, conf=0.9):
    return {"bbox": [x1, y1, x2, y2], "score": 0.9,
            "keypoints": [[x1 + i, y1 + 10 * i, conf] for i in range(17)]}


def message(score=None, persons=None, subject=0, filled=30):
    persons = [person()] if persons is None else persons
    return {"status": "ok", "result": {
        "persons": persons, "frame_dimensions": {"w": W, "h": H},
        "fall": {"score": score, "detected": score is not None and score >= 0.9,
                 "threshold": 0.9, "subject_index": subject if persons else None,
                 "window": {"filled": filled, "size": 30}, "stateful": True}}}


# ── parse_result ──


def test_parse_reads_persons_fall_and_size():
    p = fd.parse_result(message(0.5))
    assert (p["w"], p["h"]) == (W, H) and p["fall"]["score"] == 0.5 and len(p["persons"]) == 1


@pytest.mark.parametrize("bad", [
    {"result": {"status": "error", "error": {"code": "weights_missing"}}},
    {"result": {"persons": []}},              # a pose-only adapter: no fall block
    {"result": "nope"},
])
def test_an_error_is_an_outage_not_an_empty_scene(bad):
    with pytest.raises(fd.FallInferenceUnavailable):
        fd.parse_result(bad)


# ── overlay ──


def test_overlay_is_normalized_with_keypoints_and_a_verdict_on_the_subject():
    other = person(400, 40, 480, 300)
    p = fd.parse_result(message(0.97, persons=[person(), other], subject=0))
    boxes = fd.overlay_boxes(p, FallMonitor(), now=0.0)
    subject, bystander = boxes
    assert subject["label"] == "Fall risk" and subject["score"] == 0.97
    assert subject["box"] == pytest.approx([0.1, 0.1, 0.2, 0.8])
    assert len(subject["keypoints"]) == 17
    assert subject["keypoints"][0] == pytest.approx([0.1, 0.1, 0.9])
    assert bystander["label"] == "person" and bystander["score"] == 0.9   # YOLO confidence


def test_overlay_says_fallen_after_an_alarm_and_watching_while_warming():
    m = FallMonitor(FallPolicy(hold_s=5.0))
    m.observe(0.95, 10.0)
    assert fd.overlay_boxes(fd.parse_result(message(0.2)), m, 12.0)[0]["label"] == "FALLEN"
    warming = fd.overlay_boxes(fd.parse_result(message(None, filled=12)), FallMonitor(), 0.0)
    assert warming[0]["label"] == "person (warming 12/30)" and warming[0]["score"] == 0.9


def test_nobody_in_frame_draws_nothing():
    assert fd.overlay_boxes(fd.parse_result(message(None, persons=[])), FallMonitor(), 0.0) == []


def test_annotate_draws_on_a_copy():
    image = np.zeros((H, W, 3), dtype=np.uint8)
    out = fd.annotate(image, fd.parse_result(message(0.95)))
    assert out.any() and not image.any()


# ── the worker, with the SDK faked at its edges ──


class FakeInfer:
    def __init__(self, replies):
        self.replies = list(replies)
        self.closed = False

    def infer(self, jpeg):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def close(self):
        self.closed = True


class FakeApp:
    def __init__(self):
        self.falls, self.overlays = [], []
        self.config = SimpleNamespace(attach_snapshot=True)

    def policy(self):
        return FallPolicy()

    def on_fall(self, handle, alarm, image, parsed):
        self.falls.append((handle, alarm))

    def publish_overlay(self, handle, boxes, frame):
        self.overlays.append((handle, boxes))


def frame(t, restarted=False):
    data = np.zeros((H, W, 3), dtype=np.uint8).tobytes()
    return SimpleNamespace(to_ndarray=lambda: np.frombuffer(data, np.uint8).reshape(H, W, 3),
                           wall_ts=t, seq=int(t * 10), width=W, height=H,
                           restarted=restarted)


def worker_with(replies):
    app = FakeApp()
    w = fd.CameraWorker(app, SimpleNamespace(handle="cam3", name="Lobby"))
    w.infer = FakeInfer(replies)
    return app, w


def test_a_fall_run_raises_one_alarm_and_draws_every_frame():
    scores = [None] * 3 + [0.2, 0.95, 0.97, 0.96, 0.3]
    app, w = worker_with([message(s) for s in scores])
    for i in range(len(scores)):
        w._on_frame(frame(100 + i * 0.1))
    assert [h for h, _ in app.falls] == ["cam3"]
    assert app.falls[0][1].score == 0.95
    assert len(app.overlays) == len(scores)
    assert app.overlays[-1][1][0]["label"] == "FALLEN"
    assert w.window == "30/30"


def test_three_failures_mark_the_camera_down_and_a_reply_recovers_it(monkeypatch):
    app, w = worker_with([RuntimeError("404")] * 3 + [message(0.1)])
    monkeypatch.setattr(w._stop, "wait", lambda s: None)
    for i in range(3):
        w._on_frame(frame(i))
    assert w.inference_down and "fall inference failing" in w.last_error
    assert app.overlays == []
    w._on_frame(frame(4))
    assert not w.inference_down and w.last_error is None


def test_cameras_keep_their_own_runs():
    a_app, a = worker_with([message(0.95)])
    b_app, b = worker_with([message(0.1)])
    a._on_frame(frame(1)); b._on_frame(frame(1))
    assert len(a_app.falls) == 1 and b_app.falls == []
    assert a.monitor is not b.monitor


# ── the alarm the app fires ──


def test_on_fall_fires_one_alert_with_evidence(monkeypatch):
    fired = []
    app = fd.FallDetectionApp.__new__(fd.FallDetectionApp)
    app.config = fd.FallConfig(alert_severity="high")
    app.falls_today, app._day, app.recent = 0, fd.time.strftime("%Y-%m-%d"), []
    app.dispatcher = SimpleNamespace(fire=fired.append)
    app._nvr = SimpleNamespace(save_evidence=lambda jpeg: "evidence/abc.jpg" if jpeg else None)
    from fall_rules import FallAlarm

    image = np.zeros((H, W, 3), dtype=np.uint8)
    app.on_fall("cam3", FallAlarm(at=1e9, score=0.95, peak=0.97, hits=1), image,
                fd.parse_result(message(0.95)))
    (alert,) = fired
    assert alert.alert_type == "fall" and alert.camera_id == "cam3"
    assert alert.severity == "high" and alert.images == {"scene": "evidence/abc.jpg"}
    assert alert.source.name == "fall-detection"
    assert alert.evidence["peak"] == 0.97 and app.falls_today == 1


def test_every_box_carries_a_number_for_stock_core():
    """Stock OpenNVR core drops an app box whose score is null."""
    p = fd.parse_result(message(None, persons=[person(), person(400, 40, 480, 300)], filled=3))
    assert all(isinstance(b["score"], float) for b in fd.overlay_boxes(p, FallMonitor(), 0.0))


def test_connection_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("KAIC_URL", "http://opennvr-core:8100")
    monkeypatch.setenv("OPENNVR_INTERNAL_API_KEY", "k")
    monkeypatch.setenv("NATS_ALERTS_URL", "nats://nats-apps:4222")
    cfg = fd.FallConfig()
    fd._fill_from_env(cfg)
    assert (cfg.kaic_url, cfg.kaic_api_key, cfg.nats_alerts_url, cfg.nats_alerts_token) == (
        "http://opennvr-core:8100", "k", "nats://nats-apps:4222", "k")
    explicit = fd.FallConfig(kaic_url="http://elsewhere:8100")
    fd._fill_from_env(explicit)
    assert explicit.kaic_url == "http://elsewhere:8100"


def test_manifest_requires_its_own_task_not_pose_estimation():
    """`pose_estimation` in requires_tasks would light up the guard-scan
    page (appVerticals' legacy rule) for this app."""
    assert fd.MANIFEST.requires_tasks == ["fall_detection"]
    assert fd.MANIFEST.provides == ["falls"] and fd.MANIFEST.overlay


# ── what /state and the overlay report ──


def _bare_app():
    app = fd.FallDetectionApp.__new__(fd.FallDetectionApp)
    app.config = fd.FallConfig()
    app.falls_today, app._day, app.recent = 0, fd.time.strftime("%Y-%m-%d"), []
    app._overlay_lock = fd.threading.Lock()
    app.overlay = {"published": 0, "failed": 0, "last_ok_at": None,
                   "last_error": None, "last_boxes": 0}
    app.workers, app._workers_lock = {}, fd.threading.Lock()
    return app


def test_state_is_served_through_the_name_the_sdk_calls():
    """The contract server serves ``state_snapshot``; an app method named
    anything else leaves GET /state answering {} (the field report)."""
    from opennvr_app_sdk.contract import ContractMixin

    assert fd.FallDetectionApp.state_snapshot is not ContractMixin.state_snapshot
    snap = _bare_app().state_snapshot()
    assert set(snap) >= {"falls_today", "cameras", "recent", "overlay"}


def test_overlay_publish_outcomes_are_counted_not_swallowed(caplog):
    app = _bare_app()
    replies = iter([True, False, RuntimeError("nats down"), True])

    def publish_overlay(**kw):
        r = next(replies)
        if isinstance(r, Exception):
            raise r
        return r

    app.events = SimpleNamespace(publish_overlay=publish_overlay)
    for _ in range(4):
        app.publish_overlay("cam3", [{"label": "person"}], frame(1))
    o = app.state_snapshot()["overlay"]
    assert (o["published"], o["failed"], o["last_boxes"]) == (2, 2, 1)
    assert "nats down" in o["last_error"]
    assert "overlay publish failed" in caplog.text
