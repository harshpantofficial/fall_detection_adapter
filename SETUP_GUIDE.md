# Fall Detection Setup Guide with OpenNVR

This guide walks you through connecting the **Fall Detection AI Adapter** and **Fall Detection Monitoring App** to a running OpenNVR system on your Windows PC (NVIDIA GPU).

---

## Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │        Windows PC (OpenNVR Host)       │
                      │                                        │
                      │   ┌──────────────┐   ┌─────────────┐   │
                      │   │ OpenNVR Core │   │    NATS     │   │
                      │   │   (:8000)    │   │   (:4222)   │   │
                      │   └──────▲───────┘   └──────▲──────┘   │
                      │          │                  │          │
                      │    POST /register     Subscribes to    │
                      │          │          inference broadcast│
┌─────────────────────┼──────────┴──────────────────┴──────────┼─────────────────────┐
│                     │                                        │                     │
│ ┌───────────────────┴──────────┐               ┌─────────────┴───────────────────┐ │
│ │   fall-detection-adapter     │               │        fall-detection-app       │ │
│ │  - Runs ONNX Runtime (CUDA)  │               │ - Subscribes: opennvr.inference │ │
│ │  - Exposed on port :9010     │               │ - UI Dashboard on port :9210    │ │
│ └──────────────────────────────┘               └─────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Option 1: Docker Compose Overlay (Recommended)

If OpenNVR on the Windows PC is running via Docker Compose:

### 1. Copy Files to Windows PC
Transfer `fall_detection_adapter/` to the PC where OpenNVR is running. You can place it directly in the OpenNVR folder or alongside it.

### 2. Put Weights in Place
Make sure `yolo11n-pose.onnx` is located in the weights directory/volume (or in `C:\models\yolo11n-pose.onnx`).

### 3. Start the Overlay
Run from your terminal / PowerShell on the Windows PC:
```powershell
docker compose -f docker-compose.yml -f docker-compose.fall.yml up -d
```

This starts:
1. `fall-detection-adapter` with NVIDIA GPU acceleration enabled (`--gpus all`).
2. `fall-detection-register` which registers the adapter with KAI-C at `http://opennvr-core:8100/api/v1/adapters/register`.
3. `fall-detection-app` which connects to NATS and self-registers with OpenNVR Core at `http://opennvr-core:8000/api/v1/apps/register`.

---

## Option 2: Step-by-Step Manual Setup (Across LAN or Distributed)

If OpenNVR is on another device or you prefer to run services manually:

### Step 1: Ensure Model Weights & Environment
On the machine with the NVIDIA GPU:
- Place `yolo11n-pose.onnx` in a folder (e.g. `C:\models`).
- Note the IP address of the OpenNVR host (e.g. `192.168.1.100`) and the `INTERNAL_API_KEY` from OpenNVR's `.env`.

### Step 2: Build and Run the Adapter
```powershell
cd adapter
docker build -t opennvr/fall-detection-adapter:1.0.0 .

docker run -d --name fall-adapter `
  --gpus all `
  -p 9010:9010 `
  -e OPENNVR_ADAPTER_TOKEN="<INTERNAL_API_KEY>" `
  -v C:\models:/weights:ro `
  opennvr/fall-detection-adapter:1.0.0
```

Verify it is running:
```powershell
curl http://localhost:9010/health
curl http://localhost:9010/capabilities
```

### Step 3: Register the Adapter with OpenNVR
Tell OpenNVR's inference router (KAI-C) about this adapter.

**Via OpenNVR Web UI:**
1. Open OpenNVR Web UI in your browser (`http://<opennvr-ip>:8000`).
2. Go to **Settings → AI Models / Adapters → Add Adapter**.
3. Name: `fall-detection`
4. URL: `http://<adapter-ip>:9010` (or `http://host.docker.internal:9010` if running on same Docker host).
5. Click **Save** and approve permissions.

**Or via cURL:**
```bash
curl -X POST http://<opennvr-ip>:8100/api/v1/adapters/register \
  -H "Content-Type: application/json" \
  -H "X-Internal-Api-Key: <INTERNAL_API_KEY>" \
  -d '{"name": "fall-detection", "url": "http://<adapter-ip>:9010"}'

curl -X POST http://<opennvr-ip>:8100/api/v1/adapters/fall-detection/permissions/approve-all \
  -H "X-Internal-Api-Key: <INTERNAL_API_KEY>"
```

### Step 4: Configure and Run the App
1. Edit `app/config.example.yml` and set the target OpenNVR details:
```yaml
nats_url: "nats://<opennvr-ip>:4222"
nats_token: "<INTERNAL_API_KEY>"
subject_pattern: "opennvr.inference.>"

contract_port: 9210
contract_host: "<app-ip-or-hostname>"
opennvr_url: "http://<opennvr-ip>:8000"
opennvr_token: "<INTERNAL_API_KEY>"

fallen_seconds: 3.0
alert_severity: "critical"

cameras:
  - camera_id: "cam1"
```

2. Run the App:
```powershell
cd app
docker build -t opennvr/fall-detection-app:1.0.0 .

docker run -d --name fall-app `
  -p 9210:9210 `
  -v ${PWD}/config.example.yml:/app/config.yml:ro `
  opennvr/fall-detection-app:1.0.0
```

---

## Step 5: Verification in OpenNVR

1. **App Catalog**: Go to **Settings → App Catalog** in the OpenNVR web UI.
   - You will see **Fall Detection** with a green status indicator.
2. **Assign Cameras**:
   - Click Configure on the Fall Detection card.
   - Select the cameras you want to monitor.
3. **Live UI Dashboard**:
   - View real-time fallen/standing counts and posture angles by opening `http://<app-ip>:9210/ui` in your browser.
4. **Alerts**:
   - When a person falls and stays down for $\ge$ `fallen_seconds`, a `critical` alert (`person-fallen`) will appear in OpenNVR's Alerts timeline with a snapshot and metadata.
   - When they stand back up, an `info` alert (`person-fallen-resolved`) clears the incident.