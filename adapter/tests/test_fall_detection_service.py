# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for adapters/fall_detection.

No weights, no network: the pose half is the yolo-pose service (tested
in test_yolo_pose_service.py), so its ``_infer_image_bytes`` is stubbed
here with a scripted persons list, and the LiteRT interpreter is a fake
that returns the mean of its input — enough to tell one window from
another. What is under test is everything this adapter adds: the
feature port, the per-stream windows, the response shape, and the
failure paths.

Parity with the reference Space on real footage is measured, not unit
tested (README → Fidelity): it needs both models.
"""
from __future__ import annotations

import asyncio
import contextvars

import numpy as np
import pytest

from adapters.fall_detection import features as F
from adapters.fall_detection import service as S
from adapters.yolo_pose.service import YoloPoseService
from opennvr_adapter_sdk import ServiceError
from opennvr_adapter_sdk.contract import HealthStatus, InferResponse

W, H = 640, 360


def _person(x1=100.0, y1=50.0, x2=200.0, y2=300.0, conf=0.9):
    kps = [[x1 + i * 5.0, y1 + i * 10.0, conf] for i in range(17)]
    return {"bbox": [x1, y1, x2, y2], "score": 0.9, "keypoints": kps}


class FakeInterpreter:
    """Input [1, 30, 51] → output [[mean]], so the score tracks the window."""

    def __init__(self, shape=(1, 30, 51)):
        self.shape = shape
        self._x = None
        self.invocations = 0

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [{"index": 0, "shape": np.array(self.shape)}]

    def get_output_details(self):
        return [{"index": 1, "shape": np.array([1, 1])}]

    def set_tensor(self, index, value):
        assert value.shape == self.shape and value.dtype == np.float32
        self._x = value

    def invoke(self):
        self.invocations += 1

    def get_tensor(self, index):
        return np.array([[float(np.clip(np.abs(self._x).mean(), 0, 1))]], dtype=np.float32)


@pytest.fixture
def weights(tmp_path):
    (tmp_path / S.POSE_WEIGHTS_FILENAME).write_bytes(b"pose-weights")
    (tmp_path / S.CLASSIFIER_WEIGHTS_FILENAME).write_bytes(b"classifier-weights")
    return tmp_path


@pytest.fixture
def svc(weights, monkeypatch):
    """A loaded service: pose stubbed ready, fake classifier."""
    script: list[list[dict]] = []

    def fake_pose_load(self):
        self._load_state = HealthStatus.OK
        self._fingerprint_cache = self._compute_fingerprint()

    def fake_pose_infer(self, image_bytes, params):
        fake_pose_infer.params.append(dict(params))
        persons = script.pop(0) if script else []
        return InferResponse(model_name="x", model_version="x", inference_ms=5, result={
            "persons": persons, "keypoint_names": [], "frame_dimensions": {"w": W, "h": H}})
    fake_pose_infer.params = []

    monkeypatch.setattr(YoloPoseService, "load", fake_pose_load)
    monkeypatch.setattr(YoloPoseService, "_infer_image_bytes", fake_pose_infer)
    fake = FakeInterpreter()
    monkeypatch.setattr(S, "_make_interpreter", lambda path: fake)
    service = S.FallDetectionService(weights_dir=str(weights))
    service.load()
    service.script = script
    service.fake = fake
    service.pose_params = fake_pose_infer.params
    return service


# ── features: the port of the reference's arithmetic ───────────────


def test_training_order_is_alphabetical_not_coco():
    assert F.TRAINING_NAMES[0] == "Left Ankle"
    assert F.TRAINING_NAMES[-1] == "Right Wrist"
    # COCO 0 is the nose; it lands where "Nose" sorts.
    assert F.COCO_TO_SLOT[0] == F.TRAINING_NAMES.index("Nose")
    assert sorted(F.COCO_TO_SLOT) == list(range(17))


def test_no_person_is_51_zeros():
    out = F.features_for(None, W, H)
    assert out.shape == (51,) and out.dtype == np.float32 and not out.any()


def test_normalized_on_torso_midpoint_and_length():
    kps = [[0.0, 0.0, 0.9]] * 17
    kps = [list(k) for k in kps]
    # shoulders at y=100, hips at y=200, all at x=320 → ref (320,150)/frame, torso 100/H
    for name, y in (("Left Shoulder", 100), ("Right Shoulder", 100),
                    ("Left Hip", 200), ("Right Hip", 200)):
        kps[F.COCO_NAMES.index(name)] = [320.0, float(y), 0.9]
    out = F.features_for(kps, W, H)
    slot = F.TRAINING_NAMES.index("Left Hip")
    torso = 100 / H
    assert out[slot * 3] == pytest.approx(0.0, abs=1e-6)
    assert out[slot * 3 + 1] == pytest.approx((200 / H - 150 / H) / torso, abs=1e-5)
    assert out[slot * 3 + 2] == pytest.approx(0.9)


def test_low_confidence_joints_are_zeroed_but_keep_their_confidence():
    kps = [[100.0, 100.0, 0.9] for _ in range(17)]
    kps[F.COCO_NAMES.index("Left Wrist")] = [500.0, 300.0, 0.29]
    out = F.features_for(kps, W, H)
    slot = F.TRAINING_NAMES.index("Left Wrist")
    assert out[slot * 3] == 0.0 and out[slot * 3 + 1] == 0.0
    assert out[slot * 3 + 2] == pytest.approx(0.29)


def test_exactly_the_floor_counts_as_visible_like_the_reference():
    """float32(0.3) > 0.3: the reference stores features as float32 and
    so keeps a 0.3 joint. Parity, not tidiness."""
    kps = [[100.0, 100.0, 0.9] for _ in range(17)]
    kps[F.COCO_NAMES.index("Left Wrist")] = [500.0, 300.0, 0.3]
    out = F.features_for(kps, W, H)
    slot = F.TRAINING_NAMES.index("Left Wrist")
    assert out[slot * 3] != 0.0


def test_no_torso_means_no_normalization():
    kps = [[64.0, 36.0, 0.2] for _ in range(17)]     # nothing above the floor
    out = F.features_for(kps, W, H)
    assert out[0] == pytest.approx(0.1) and out[1] == pytest.approx(0.1)


def test_subject_is_the_largest_box():
    small, big = _person(0, 0, 10, 10), _person(0, 0, 100, 50)
    assert F.subject_index([small, big, small]) == 1
    assert F.subject_index([]) is None


def test_window_is_bounded():
    w = F.SequenceWindow()
    for i in range(100):
        w.push(np.full(51, i, dtype=np.float32))
    assert w.filled == 30 and w.full
    assert w.batch().shape == (1, 30, 51)
    assert w.batch()[0, 0, 0] == 70.0        # oldest kept frame


# ── the service ────────────────────────────────────────────────────


def test_ready_only_with_both_models(svc):
    assert svc.is_ready()
    assert svc.fake.invocations == 1           # warm-up


def test_missing_classifier_is_not_ready_and_answers_503(weights, monkeypatch):
    (weights / S.CLASSIFIER_WEIGHTS_FILENAME).unlink()
    monkeypatch.setattr(YoloPoseService, "load",
                        lambda self: setattr(self, "_load_state", HealthStatus.OK))
    service = S.FallDetectionService(weights_dir=str(weights))
    service.load()
    assert not service.is_ready()
    with pytest.raises(ServiceError) as err:
        service._infer_image_bytes(b"jpeg", {})
    assert err.value.envelope().error.code == "weights_missing"


def test_wrong_classifier_shape_is_refused(weights, monkeypatch):
    monkeypatch.setattr(YoloPoseService, "load",
                        lambda self: setattr(self, "_load_state", HealthStatus.OK))
    monkeypatch.setattr(S, "_make_interpreter", lambda p: FakeInterpreter(shape=(1, 10, 51)))
    service = S.FallDetectionService(weights_dir=str(weights))
    service.load()
    assert not service.is_ready()
    assert "expects input" in (service._classifier_error or "")


def test_fingerprint_covers_both_files(svc, weights):
    before = svc.fingerprint()
    (weights / S.CLASSIFIER_WEIGHTS_FILENAME).write_bytes(b"retrained")
    assert svc.fingerprint() != before


def test_reference_pose_defaults_reach_the_pose_model(svc):
    svc._infer_image_bytes(b"jpeg", {})
    assert svc.pose_params[-1] == {"conf": 0.25, "imgsz": 640}
    svc._infer_image_bytes(b"jpeg", {"conf": 0.5, "stream_id": "c", "threshold": 0.5})
    assert svc.pose_params[-1] == {"conf": 0.5, "imgsz": 640}   # our params never leak


def test_stateless_call_returns_pose_and_no_score(svc):
    svc.script.append([_person()])
    res = svc._infer_image_bytes(b"jpeg", {}).result
    assert res["persons"] and res["fall"]["score"] is None
    assert res["fall"]["stateful"] is False and res["fall"]["subject_index"] == 0


def test_http_stream_scores_once_the_window_is_full(svc):
    scores = []
    for _ in range(31):
        svc.script.append([_person()])
        scores.append(svc._infer_image_bytes(b"j", {"stream_id": "cam1"}).result["fall"])
    assert all(s["score"] is None for s in scores[:29])
    assert scores[28]["window"] == {"filled": 29, "size": 30}
    assert scores[29]["score"] is not None and scores[30]["score"] is not None


def test_http_streams_do_not_share_a_window(svc):
    for _ in range(30):
        svc.script.append([_person()])
        a = svc._infer_image_bytes(b"j", {"stream_id": "a"}).result["fall"]
    svc.script.append([_person()])
    b = svc._infer_image_bytes(b"j", {"stream_id": "b"}).result["fall"]
    assert a["window"]["filled"] == 30 and b["window"]["filled"] == 1 and b["score"] is None


def test_reset_starts_a_stream_over(svc):
    for _ in range(5):
        svc._infer_image_bytes(b"j", {"stream_id": "a"})
    fall = svc._infer_image_bytes(b"j", {"stream_id": "a", "reset": True}).result["fall"]
    assert fall["window"]["filled"] == 1


def test_http_windows_are_bounded_and_expire(svc, monkeypatch):
    for i in range(S.MAX_HTTP_STREAMS + 10):
        svc._infer_image_bytes(b"j", {"stream_id": f"s{i}"})
    assert len(svc._http_windows) == S.MAX_HTTP_STREAMS
    now = [1e9]
    monkeypatch.setattr(S.time, "monotonic", lambda: now[0])
    svc._infer_image_bytes(b"j", {"stream_id": "fresh"})
    now[0] += S.HTTP_STREAM_IDLE_S + 1
    svc._infer_image_bytes(b"j", {"stream_id": "other"})
    assert list(svc._http_windows) == ["other"]


def test_threshold_decides_detected(svc):
    # Fake score = mean |feature|; a big normalized skeleton scores high.
    for _ in range(30):
        svc.script.append([_person(conf=0.9)])
        fall = svc._infer_image_bytes(b"j", {"stream_id": "a", "threshold": 0.01}).result["fall"]
    assert fall["detected"] is True and fall["threshold"] == 0.01
    svc.script.append([_person(conf=0.9)])
    fall = svc._infer_image_bytes(b"j", {"stream_id": "a", "threshold": 1.0}).result["fall"]
    assert fall["detected"] is False


def test_empty_frames_still_advance_the_window(svc):
    for _ in range(30):
        fall = svc._infer_image_bytes(b"j", {"stream_id": "a"}).result["fall"]
    assert fall["score"] == 0.0 and fall["subject_index"] is None


def test_each_stream_session_owns_its_window(svc):
    """The §6 path: the window rides a ContextVar set per session, and
    asyncio.to_thread (which the inherited loop uses) carries it into
    the worker thread — so two sessions never see each other's frames."""
    async def session(n):
        token = S._SESSION_WINDOW.set(F.SequenceWindow())
        try:
            out = None
            for _ in range(n):
                out = await asyncio.to_thread(
                    lambda: svc._infer_image_bytes(b"j", {}).result["fall"])
            return out
        finally:
            S._SESSION_WINDOW.reset(token)

    async def both():
        return await asyncio.gather(session(30), session(3))

    a, b = asyncio.run(both())
    assert a["window"]["filled"] == 30 and a["stateful"]
    assert b["window"]["filled"] == 3 and b["score"] is None
    # And nothing leaked into the HTTP registry or the outer context.
    assert svc._http_windows == {} and S._SESSION_WINDOW.get() is None
    assert contextvars.copy_context().get(S._SESSION_WINDOW) is None


def test_metrics_count_scored_frames(svc):
    for _ in range(31):
        svc._infer_image_bytes(b"j", {"stream_id": "a"})
    text = svc.metrics.render() if hasattr(svc.metrics, "render") else str(svc.metrics)
    assert "adapter_fall_scores_total" in text


# ── GPU: where the pose model actually ran ──────────────────────────


def test_results_say_which_device_ran_the_pose_model(svc):
    assert svc._infer_image_bytes(b"j", {}).result["fall"]["device"] == "cpu"
    svc._gpu_in_use = True
    assert svc._infer_image_bytes(b"j", {}).result["fall"]["device"] == "cuda"


def test_gpu_build_that_fell_back_to_cpu_is_a_warning(svc, monkeypatch):
    """onnxruntime falls back to CPU without failing; the operator who
    deployed the GPU image must not have to guess."""
    from opennvr_adapter_sdk.contract import HardwareVerdict

    monkeypatch.setattr(S, "_cuda_build", lambda: True)
    report = svc.hardware_evaluation()
    assert report.verdict == HardwareVerdict.WARN and "fell back to CPU" in report.reasoning
    assert report.details["pose_device"] == "cpu"
    svc._gpu_in_use = True
    report = svc.hardware_evaluation()
    assert report.verdict != HardwareVerdict.WARN or "fell back" not in report.reasoning
    assert report.details["pose_device"] == "cuda"


def test_cpu_build_on_cpu_is_not_a_warning(svc, monkeypatch):
    monkeypatch.setattr(S, "_cuda_build", lambda: False)
    assert "fell back" not in (svc.hardware_evaluation().reasoning or "")


def test_cuda_libraries_are_preloaded_only_on_the_gpu_build(monkeypatch):
    import onnxruntime as ort

    calls = []
    monkeypatch.setattr(ort, "preload_dlls", lambda *a, **k: calls.append(1), raising=False)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    assert S._prepare_cuda() is False and calls == []
    monkeypatch.setattr(ort, "get_available_providers",
                        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert S._prepare_cuda() is True and calls == [1]
