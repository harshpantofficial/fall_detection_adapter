# Models

Put here before `docker compose up -d --build`:

- `fall_detection_transformer.tflite` (required)
- `yolo26n-pose.pt`, or a `yolo26n-pose.onnx` exported with `dynamic=True` (skips the torch download)

Not committed; baked into the adapter image at build time.
