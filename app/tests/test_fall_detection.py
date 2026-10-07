"""Tests for FallDetectionDetector state machine and alert generation."""
from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock

from opennvr_app_sdk.alerts import AlertDispatcher

from fall_detection import AppConfig, CameraWatch, FallDetectionDetector


def _make_config(fallen_seconds: float = 2.0) -> AppConfig:
    cam = CameraWatch(camera_id="cam1", frame_width=1920, frame_height=1080)
    return AppConfig(
        nats_url="nats://localhost:4222",
        nats_token=None,
        subject_pattern="opennvr.inference.>",
        cameras={"cam1": cam},
        webhook_url=None,
        fallen_seconds=fallen_seconds,
        alert_cooldown_seconds=0.0,
    )


def test_standing_person_no_alert():
    cfg = _make_config()
    dispatcher = MagicMock(spec=AlertDispatcher)
    detector = FallDetectionDetector(cfg, dispatcher)

    event = {
        "camera_id": "cam1",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "result": {
            "persons": [
                {
                    "track_id": "track-1",
                    "bbox": [500, 200, 600, 700],
                    "fall_state": "standing",
                    "torso_angle_deg": 12.0,
                    "confidence": 0.85,
                }
            ]
        },
    }

    alerts = detector.handle_event(event)
    assert len(alerts) == 0


def test_persistent_fall_generates_alert():
    cfg = _make_config(fallen_seconds=1.0)
    dispatcher = MagicMock(spec=AlertDispatcher)
    detector = FallDetectionDetector(cfg, dispatcher)

    t0 = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)

    # Frame 1: Person falls anywhere in frame
    event1 = {
        "camera_id": "cam1",
        "completed_at": t0.isoformat(),
        "result": {
            "persons": [
                {
                    "track_id": "track-1",
                    "bbox": [500, 700, 900, 850],
                    "fall_state": "fallen",
                    "torso_angle_deg": 75.0,
                    "confidence": 0.92,
                }
            ]
        },
    }
    alerts1 = detector.handle_event(event1)
    assert len(alerts1) == 0  # not yet past threshold

    # Frame 2: Person still fallen after 1.5 seconds (exceeding 1.0s threshold)
    t1 = t0 + dt.timedelta(seconds=1.5)
    event2 = {
        "camera_id": "cam1",
        "completed_at": t1.isoformat(),
        "result": {
            "persons": [
                {
                    "track_id": "track-1",
                    "bbox": [500, 700, 900, 850],
                    "fall_state": "fallen",
                    "torso_angle_deg": 78.0,
                    "confidence": 0.94,
                }
            ]
        },
    }
    alerts2 = detector.handle_event(event2)
    assert len(alerts2) == 1
    assert alerts2[0].alert_type == "person-fallen"
    assert alerts2[0].severity == "critical"
    assert alerts2[0].evidence["track_id"] == "track-1"

    # Frame 3: Person stands back up
    t2 = t1 + dt.timedelta(seconds=2.0)
    event3 = {
        "camera_id": "cam1",
        "completed_at": t2.isoformat(),
        "result": {
            "persons": [
                {
                    "track_id": "track-1",
                    "bbox": [500, 200, 600, 700],
                    "fall_state": "standing",
                    "torso_angle_deg": 10.0,
                    "confidence": 0.88,
                }
            ]
        },
    }
    alerts3 = detector.handle_event(event3)
    assert len(alerts3) == 1
    assert alerts3[0].alert_type == "person-fallen-resolved"