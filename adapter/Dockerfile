# Fall Detection AI Adapter — OpenNVR Contract-Compliant (NVIDIA GPU / CPU target)
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
        libgl1 \
        libglib2.0-0 \
        patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin core dependencies; PyTorch + Ultralytics + ai-edge-litert
RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        websockets==17.1 \
        numpy==1.26.4 \
        opencv-python-headless==4.10.0.84 \
        torch==2.5.1 \
        ultralytics>=8.4.0 \
        "ai-edge-litert>=2.1.4"

# Clear executable stack flag from LiteRT binaries so ld.so loads it cleanly on modern kernels/Docker
RUN find /usr/local/lib/python3.11/site-packages/ai_edge_litert -name "*.so" -exec patchelf --clear-execstack {} +

# Smoke test: fail early during build if LiteRT native library fails to load
RUN python -c "from ai_edge_litert.interpreter import Interpreter; print('LiteRT interpreter import OK')"

# Install OpenNVR Adapter SDK
COPY opennvr_adapter_sdk/ ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk

# Copy adapter source into adapters.fall_detection package
RUN mkdir -p ./adapters/fall_detection && touch ./adapters/__init__.py
COPY coco_keypoints.py fall_transformer.py fall_detection_core.py main.py service.py __init__.py ./adapters/fall_detection/

ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN="" \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    YOLO_POSE_WEIGHTS_DIR=/weights \
    YOLO_POSE_MODEL_URL="" \
    FALL_TRANSFORMER_TFLITE_PATH=/weights/fall_detection_transformer.tflite

EXPOSE 9010

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9010/health', timeout=2).status_code == 200 else 1)"

CMD ["uvicorn", "adapters.fall_detection.main:app", "--host", "0.0.0.0", "--port", "9010"]