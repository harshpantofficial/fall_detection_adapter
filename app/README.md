# Fall Detection App for OpenNVR

An OpenNVR-native monitoring app built on `opennvr-app-sdk` using the `Detector` archetype, following the pattern of `open-nvr/examples/abandoned-object`.

## How It Works
1. Subscribes to NATS event bus (`opennvr.inference.>`).
2. Consumes pose and fall posture detections emitted by the Fall Detection AI Adapter.
3. Tracks individuals over time using `keyed_state(ttl=10.0)`.
4. Evaluates persistence: a fall must persist for `fallen_seconds` (default: 3.0s) to trigger an alert.
5. Emits `person-fallen` (critical severity) alerts with bounding box, torso angle, and snapshot evidence.
6. Automatically fires `person-fallen-resolved` (info severity) when the person stands back up.
7. Exposes contract surfaces (`GET /health`, `/manifest`, `/state`, `/ui`) on `:9210`.

## Deployment on Windows PC

### 1. Build Docker Image
From the open-nvr directory:
```bash
docker build -f Dockerfile -t opennvr/fall-detection:1.0.0 .
```

### 2. Run Container
```bash
docker run -d --name opennvr-fall-app \
  --restart unless-stopped \
  --network opennvr_internal \
  -v C:/opennvr/config/fall_detection.yml:/app/config.yml:ro \
  opennvr/fall-detection:1.0.0
```

### 3. Open Web UI
Visit `http://localhost:9210/ui` in your browser (or access via OpenNVR App Catalog).