# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Fall-detection adapter — contract-compliant FastAPI service.

Declaration only, as §3.7 asks: the two models and the per-stream window
live in ``adapters/fall_detection/service.py`` (which inherits the §6
loop from ``adapters/yolo_pose/``), and the SDK provides the rest.

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret FALL_WEIGHTS_DIR=../models \\
    python -m uvicorn adapters.fall_detection.main:app --host 0.0.0.0 --port 9013

Conformance check:
    python -m conformance http://localhost:9013 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

from adapters.fall_detection.service import MAX_IMAGE_BYTES, FallDetectionService
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)



def _cuda_provider_available() -> bool:
    """True only when the installed onnxruntime ships the CUDA provider
    (a GPU build). Same rule as the yolo-pose adapter: declare the GPU
    scope only when this build can actually use one."""
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


_adapter_app = AdapterApp(
    service_factory=FallDetectionService,
    name="fall-detection",
    version="1.0.0",
    vendor="open-nvr",
    license="AGPL-3.0",
    model_card_url="https://docs.ultralytics.com/tasks/pose/",
    tasks_advertised=["fall_detection"],
    body_shape=BodyShape.IMAGE,
    max_body_bytes=MAX_IMAGE_BYTES,
    permissions=Permissions(
        # Same build-accurate rule as the pose adapter: the stock image
        # is CPU onnxruntime and declares no GPU.
        gpu=_cuda_provider_available(),
        # Both models are mounted; the adapter never fetches anything.
        network_egress=[],
        # A container-owned named volume (fall-detection-weights-init in
        # open-nvr's docker-compose.apps.yml), not a host path.
        host_filesystem=[],
        shared_memory_paths=[],
        host_metadata=False,
    ),
    scheduling=Scheduling(
        max_inflight=1,
        preferred_batch_size=1,
        # Every camera streams continuously; without per-camera fairness
        # the busiest one starves the rest, and a starved camera's window
        # fills with gaps the classifier reads as motion.
        fair_queuing=FairQueuing.PER_CAMERA,
    ),
    cost=Cost(currency="USD"),
    supports_stream=True,
    # ~60-90 ms per 640 px frame on an 8-core CPU (measured), so one
    # instance feeds about one camera at the 10 fps the app defaults to.
    # Same headroom argument as the pose adapter for 4 rather than 1.
    stream_max_concurrent=4,
    stream_supports_shared_memory=False,
)

app = _adapter_app.fastapi_app


def __getattr__(name: str):
    if name == "_service":
        return _adapter_app.service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
