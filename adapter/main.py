"""
Fall Detection AI Adapter — OpenNVR Contract-Compliant FastAPI Service.

Exposes:
- GET  /health              (Liveness probe)
- GET  /capabilities        (Supported tasks, GPU status, model info)
- GET  /hardware/evaluation (Operator hardware diagnostic)
- GET  /metrics             (Prometheus metrics)
- POST /infer               (Single-frame fallback inference)
- WS   /infer/stream        (§6 WebSocket streaming inference)

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret \\
    uvicorn adapters.fall_detection.main:app --host 0.0.0.0 --port 9010
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

from adapters.fall_detection.service import (
    MAX_IMAGE_BYTES,
    MODEL_URL_ENV,
    FallDetectionService,
)
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)


def _model_fetch_egress() -> list[str]:
    url = os.getenv(MODEL_URL_ENV, "").strip()
    if not url:
        return []
    host = urlparse(url).hostname
    return [host] if host else []


def _cuda_provider_available() -> bool:
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


_adapter_app = AdapterApp(
    service_factory=FallDetectionService,
    name="fall-detection-adapter",
    version="1.0.0",
    vendor="open-nvr",
    license="AGPL-3.0",
    model_card_url="https://docs.ultralytics.com/tasks/pose/",
    tasks_advertised=["pose_estimation", "fall_detection"],
    body_shape=BodyShape.IMAGE,
    max_body_bytes=MAX_IMAGE_BYTES,
    permissions=Permissions(
        gpu=_cuda_provider_available(),
        network_egress=_model_fetch_egress(),
        host_filesystem=[],
        shared_memory_paths=[],
        host_metadata=False,
    ),
    scheduling=Scheduling(
        max_inflight=1,
        preferred_batch_size=1,
        fair_queuing=FairQueuing.PER_CAMERA,
    ),
    cost=Cost(currency="USD"),
    supports_stream=True,
    stream_max_concurrent=4,
    stream_supports_shared_memory=False,
)

app = _adapter_app.fastapi_app


def __getattr__(name: str):
    if name == "_service":
        return _adapter_app.service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")