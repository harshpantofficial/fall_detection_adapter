# Fall Detection AI Adapter for OpenNVR

An OpenNVR AI Adapter running YOLO26 pose estimation (`yolo26n-pose.pt` via Ultralytics) combined with the sequence-based Fall Transformer (`fall_detection_transformer.tflite` via TFLite/LiteRT) for human fall classification.

## Features
- **Contract Compliant**: Implements the OpenNVR Adapter Contract v1 (`/health`, `/capabilities`, `/hardware/evaluation`, `/metrics`, `/infer`, `/infer/stream`).
- **High-Performance Streaming**: Fully implements §6 WebSocket streaming off the event loop via `asyncio.to_thread`.
- **Accurate Sequence Classification**: Extracts 17 COCO keypoints with YOLO26-pose, normalises them relative to the torso centre and torso height scale, and feeds a 30-timestep sliding sequence into the trained TFLite Transformer.
- **Hardware Acceleration**: Automatic CUDA acceleration on NVIDIA GPU systems with CPU fallback.

## Model Files
- Pose Estimation: `yolo26n-pose.pt`
- Fall Transformer: `fall_detection_transformer.tflite`

## Deployment on Host / Docker

### 1. Build Docker Image
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
  -v /path/to/weights:/weights:ro \
  opennvr/fall-detection-adapter:1.0.0
```

Where `/weights` contains `yolo26n-pose.pt` and `fall_detection_transformer.tflite`.

### 3. Verify Health & Capabilities
```bash
curl http://localhost:9010/health
curl http://localhost:9010/capabilities
```
```