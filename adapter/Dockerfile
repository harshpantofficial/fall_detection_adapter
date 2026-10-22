# syntax=docker/dockerfile:1.7
# Fall-detection adapter — YOLO26n-pose (onnxruntime) + the fall
# transformer (LiteRT), AI Adapter Contract v1. Both models are baked in.
#
# Build context is the bundle root (it needs ../models):
#   docker build -f adapter/Dockerfile -t opennvr/fall-detection-adapter:local .
# docker compose (docker-compose.yml) does this for you.

# ── Stage 1: the two model files ────────────────────────────────────
# models/ must hold fall_detection_transformer.tflite, plus either
# yolo26n-pose.onnx (used as-is) or yolo26n-pose.pt (exported here, which
# downloads CPU torch, ~1 GB, once — the layer is cached afterwards).
FROM python:3.12-slim AS models
WORKDIR /build
COPY models/ ./models/
RUN set -e; \
    mkdir -p /out; \
    if [ ! -s models/fall_detection_transformer.tflite ]; then \
      echo "ERROR: models/fall_detection_transformer.tflite is missing (see README)."; exit 1; \
    fi; \
    if [ -s models/yolo26n-pose.onnx ]; then \
      echo "Using models/yolo26n-pose.onnx as-is."; \
      cp models/yolo26n-pose.onnx /out/; \
      pip install --no-cache-dir onnxruntime==1.23.2; \
    elif [ -s models/yolo26n-pose.pt ]; then \
      echo "Exporting models/yolo26n-pose.pt (dynamic, opset 12, 640px)..."; \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision; \
      pip install --no-cache-dir ultralytics==8.4.183 "onnx>=1.16,<2" onnxslim onnxruntime==1.23.2; \
      # ultralytics pulls opencv-python, which needs libGL; headless does not.
      pip uninstall -y opencv-python && pip install --no-cache-dir opencv-python-headless; \
      cp models/yolo26n-pose.pt ./; \
      yolo export model=yolo26n-pose.pt format=onnx opset=12 imgsz=640 dynamic=True; \
      cp yolo26n-pose.onnx /out/; \
    else \
      echo "ERROR: put yolo26n-pose.pt (or a dynamic yolo26n-pose.onnx) in models/ (see README)."; exit 1; \
    fi; \
    cp models/fall_detection_transformer.tflite /out/; \
    # dynamic=True is what makes YOLO26 export as (1, 56, anchors), the
    # layout the adapter decodes. Fail the build, not every frame later.
    python -c "import onnxruntime as ort; \
o = ort.InferenceSession('/out/yolo26n-pose.onnx', providers=['CPUExecutionProvider']).get_outputs()[0].shape; \
print('pose output', o); \
assert o[1] == 56, f'expected 56 features per anchor, got {o} - export with dynamic=True'"; \
    ls -la /out

# ── Stage 2: the adapter ────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The yolo-pose adapter's pin set (its code runs here) plus LiteRT for the
# classifier, the runtime the reference Space uses, so the .tflite runs
# unconverted. websockets: without it /infer/stream 404s while /health
# stays green.
RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        websockets==17.1 \
        numpy==1.26.4 \
        opencv-python-headless==4.10.0.84 \
        onnxruntime==1.23.2 \
        ai-edge-litert==2.3.0

COPY adapter/opennvr_adapter_sdk/ ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk

# GPU build only (docker-compose.gpu.yml sets ORT_GPU_PACKAGE): swap in
# onnxruntime-gpu with the CUDA 12 + cuDNN 9 libraries as pip wheels
# (~3 GB), so no CUDA base image is needed — the host's NVIDIA driver is
# injected by Docker when the GPU is requested. It is a step of its own,
# after the layers above, so the CPU build keeps their cache and needs no
# network on a rebuild.
ARG ORT_GPU_PACKAGE=""
RUN if [ -n "$ORT_GPU_PACKAGE" ]; then \
        pip uninstall -y onnxruntime && \
        pip install --no-cache-dir --default-timeout=120 --retries 10 "$ORT_GPU_PACKAGE"; \
    fi

COPY adapter/adapters/ ./adapters/
COPY --from=models /out/ /weights/

ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN="" \
    FALL_WEIGHTS_DIR=/weights

EXPOSE 9013

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9013/health', timeout=2).status_code == 200 else 1)"

CMD ["uvicorn", "adapters.fall_detection.main:app", "--host", "0.0.0.0", "--port", "9013"]
