# Fall Detection AI Adapter — OpenNVR Contract-Compliant (NVIDIA GPU target)
#
# Build:
#   docker build -f Dockerfile -t opennvr/fall-detection-adapter:1.0.0 .
#
# Run with GPU (NVIDIA Container Toolkit):
#   docker run --rm -p 9010:9010 --gpus all \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     -v C:/models:/weights:ro \
#     opennvr/fall-detection-adapter:1.0.0

FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin core dependencies; onnxruntime-gpu for NVIDIA GPU inference
RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        websockets==17.1 \
        numpy==1.26.4 \
        opencv-python-headless==4.10.0.84 \
        onnxruntime-gpu==1.20.1

# Install OpenNVR Adapter SDK
COPY opennvr_adapter_sdk/ ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk

# Copy adapter source
COPY adapters/fall_detection/ ./adapters/fall_detection/

ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN="" \
    YOLO_POSE_WEIGHTS_DIR=/weights \
    YOLO_POSE_MODEL_URL="" \
    FALL_TRANSFORMER_TFLITE_PATH=/weights/fall_detection_transformer.tflite

EXPOSE 9010

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9010/health', timeout=2).status_code == 200 else 1)"

CMD ["uvicorn", "adapters.fall_detection.main:app", "--host", "0.0.0.0", "--port", "9010"]