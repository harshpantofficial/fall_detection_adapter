# Fall Detection AI Adapter for OpenNVR

An OpenNVR AI Adapter that runs YOLO pose estimation on ONNX Runtime with NVIDIA CUDA acceleration, paired with a Fall Transformer analyzing posture and torso dynamics.

## Features
- **Contract Compliant**: Implements the OpenNVR Adapter Contract v1 (`/health`, `/capabilities`, `/hardware/evaluation`, `/metrics`, `/infer`, `/infer/stream`).
- **High-Performance Streaming**: Fully implements §6 WebSocket streaming (`05_streaming.py` protocol) off the event loop via `asyncio.to_thread`.
- **Accurate Posture Analysis**: Classifies `"standing"`, `"falling"`, and `"fallen"` via torso angle geometry and optional sequence Transformer (`fall_detection_transformer.tflite`).
- **NVIDIA GPU Acceleration**: Runs with `onnxruntime-gpu` and `CUDAExecutionProvider`.

## Deployment on Windows PC (with NVIDIA GPU)

### 1. Build Docker Image
Copy this directory into your `ai-adapter/adapters/fall_detection/` on the target PC, or build directly:
```bash
docker build -f Dockerfile -t opennvr/fall-detection-adapter:1.0.0 .
```

### 2. Run Container with GPU
```bash
docker run -d --name opennvr-fall-adapter \
  --restart unless-stopped \
  --gpus all \
  --network opennvr_internal \
  -p 9010:9010 \
  -e OPENNVR_ADAPTER_TOKEN="secret" \
  -v C:/models:/weights:ro \
  opennvr/fall-detection-adapter:1.0.0
```

Where `C:/models` contains `yolo11n-pose.onnx` (and optionally `fall_detection_transformer.tflite`).

### 3. Verify Health & Capabilities
```bash
curl http://localhost:9010/health
curl http://localhost:9010/capabilities
```