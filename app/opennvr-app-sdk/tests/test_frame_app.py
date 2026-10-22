# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""FrameApp + KaiCClient tests — the poll-tick path and the
contract-v1 base64 infer call, both without a live KAI-C."""
from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import httpx
import pytest

from opennvr_app_sdk import (
    Alert,
    AlertDispatcher,
    FrameApp,
    KaiCClient,
    KaiCError,
)


class _RecorderChannel:
    name = "recorder"

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> bool:
        self.alerts.append(alert)
        return True


class _FakeSource:
    """FrameSource returning canned bytes per camera; None = no frame."""

    def __init__(self, frames: dict[str, bytes | None]) -> None:
        self.frames = frames
        self.calls: list[str] = []

    def get_frame(self, camera_id: str) -> bytes | None:
        self.calls.append(camera_id)
        value = self.frames.get(camera_id)
        if isinstance(value, Exception):
            raise value
        return value


class SpottingApp(FrameApp):
    def on_frame(self, camera_id: str, frame_bytes: bytes):
        if b"person" in frame_bytes:
            yield Alert(title="spotted", description="d", camera_id=camera_id)


def _build(frames: dict, **kwargs) -> tuple[SpottingApp, _FakeSource, _RecorderChannel]:
    recorder = _RecorderChannel()
    source = _FakeSource(frames)
    cfg = SimpleNamespace(poll_interval_seconds=0.01)
    app_obj = SpottingApp(
        cfg,
        AlertDispatcher([recorder]),
        frame_source=source,
        cameras=list(frames.keys()),
        **kwargs,
    )
    return app_obj, source, recorder


# ── handle_tick ────────────────────────────────────────────────────


def test_tick_fetches_every_camera_and_dispatches():
    app_obj, source, recorder = _build({
        "cam-1": b"a person here",
        "cam-2": b"empty porch",
    })
    fired = app_obj.handle_tick()
    assert source.calls == ["cam-1", "cam-2"]
    assert [a.camera_id for a in fired] == ["cam-1"]
    assert recorder.alerts == fired


def test_tick_skips_cameras_without_frames():
    app_obj, _, recorder = _build({"cam-1": None})
    assert app_obj.handle_tick() == []
    assert recorder.alerts == []


def test_tick_isolates_fetch_and_rule_failures():
    app_obj, source, recorder = _build({
        "cam-bad": RuntimeError("rtsp down"),
        "cam-2": b"a person here",
    })
    # cam-bad raises in get_frame; cam-2 must still be processed.
    fired = app_obj.handle_tick()
    assert [a.camera_id for a in fired] == ["cam-2"]

    class BoomApp(SpottingApp):
        def on_frame(self, camera_id, frame_bytes):
            raise RuntimeError("rule kaboom")

    boom = BoomApp(
        SimpleNamespace(poll_interval_seconds=1.0),
        AlertDispatcher([recorder]),
        frame_source=source,
        cameras=["cam-2"],
    )
    assert boom.handle_tick() == []  # MUST NOT raise


def test_interval_must_be_positive():
    with pytest.raises(ValueError, match="poll_interval_seconds"):
        _build({"cam-1": b"x"}, poll_interval_seconds=0)


async def test_run_once_does_one_tick():
    app_obj, source, _ = _build({"cam-1": b"a person here"})
    await app_obj.run(once=True)
    assert source.calls == ["cam-1"]


# ── KaiCClient ─────────────────────────────────────────────────────


class _CapturingTransport(httpx.BaseTransport):
    def __init__(self, status_code: int = 200, body: dict | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._status = status_code
        self._body = body if body is not None else {"result": {"detections": []}}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        return httpx.Response(self._status, json=self._body)


def test_kaic_client_posts_contract_v1_base64_payload():
    transport = _CapturingTransport()
    client = KaiCClient(
        "http://kaic:8000/",  # trailing slash must not double up
        "yolov8",
        api_key="sekrit",
        http_client=httpx.Client(transport=transport),
    )
    frame = b"\xff\xd8jpegbytes"
    response = client.infer(frame, task="object_detection", camera_id="cam-1", correlation_id="corr-9")
    assert response == {"result": {"detections": []}}

    request = transport.requests[0]
    assert str(request.url) == "http://kaic:8000/api/v1/infer/yolov8"
    assert request.headers["x-correlation-id"] == "corr-9"
    assert request.headers["x-internal-api-key"] == "sekrit"
    body = json.loads(request.content)
    assert body["task"] == "object_detection"
    assert body["camera_id"] == "cam-1"
    assert base64.b64decode(body["frame_b64"]) == frame


def test_kaic_client_generates_correlation_id_when_omitted():
    transport = _CapturingTransport()
    client = KaiCClient(
        "http://kaic:8000", "yolov8",
        http_client=httpx.Client(transport=transport),
    )
    client.infer(b"frame", task="object_detection", camera_id="cam-1")
    request = transport.requests[0]
    assert request.headers["x-correlation-id"].startswith("app-")
    assert "x-internal-api-key" not in request.headers  # no key configured


def test_kaic_client_raises_kaic_error_on_non_200():
    transport = _CapturingTransport(status_code=503, body={"detail": "overloaded"})
    client = KaiCClient(
        "http://kaic:8000", "yolov8",
        http_client=httpx.Client(transport=transport),
    )
    with pytest.raises(KaiCError, match="HTTP 503"):
        client.infer(b"frame", task="object_detection", camera_id="cam-1")


def test_kaic_client_wraps_transport_errors():
    class _ExplodingTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

    client = KaiCClient(
        "http://kaic:8000", "yolov8",
        http_client=httpx.Client(transport=_ExplodingTransport()),
    )
    with pytest.raises(KaiCError, match="unreachable"):
        client.infer(b"frame", task="object_detection", camera_id="cam-1")
