# Fall detection for OpenNVR

A standalone add-on: two containers that plug into an already running OpenNVR stack, the same way OpenNVR's own apps and adapters do. Nothing in the OpenNVR checkout changes, and the ai-adapter repo is not needed.

```
camera ─► OpenNVR MediaMTX ─► fall-detection (app)  ── one session per camera ──►  KAI-C ─► fall-detection-adapter
                                   │   ▲                                                        YOLO26n-pose (ONNX)
                                   │   └──── persons + keypoints + fall score ◄──────────────── fall transformer (TFLite)
                                   │                                                             30-frame window per camera
                                   ├─► "Person fallen" alert + annotated frame ─► OpenNVR alerts inbox (bell)
                                   └─► overlay.boxes.v1 (box, label, score, keypoints) ─► OpenNVR Live View
```

| Container | Network | Does |
|---|---|---|
| `fall-detection-adapter` | `<project>_opennvr_internal` | Holds **both** models, baked into the image. Answers KAI-C as `fall-detection`. |
| `fall-detection-register` | `<project>_opennvr_internal` | One-shot job: registers the adapter with KAI-C and grants its declared permissions. |
| `fall-detection` | `<project>_opennvr_apps` | The app. Reads each picked camera **live** from OpenNVR (10 fps), raises alarms, draws on the Live View. |
| `fall-testcam` (opt-in) | `<project>_opennvr_internal` | Loops the videos in `test-videos/` as RTSP cameras, for testing. |

The app works on live camera streams. Your videos are only for testing, and the test path turns them into cameras so they go through the same live pipeline.

## Setup (Windows, PowerShell)

Put this folder anywhere, for example next to your `open-nvr` folder. OpenNVR must already be running.

**1. Models.** Copy into `models\`:

```powershell
copy <your-space>\fall_detection_transformer.tflite models\
copy <your-space>\yolo26n-pose.pt models\
```

The first build exports `yolo26n-pose.pt` to ONNX, which downloads CPU torch (about 1 GB, once; Docker caches the layer). To skip that, put an already-exported `yolo26n-pose.onnx` (`dynamic=True`) in `models\` instead of the `.pt`.

**2. Settings.**

```powershell
copy .env.example .env
docker network ls            # find <project>_opennvr_internal
notepad .env
```

- `INTERNAL_API_KEY`: the same value as in OpenNVR's `.env`.
- `OPENNVR_PROJECT`: the prefix of those network names, e.g. `open-nvr` for `open-nvr_opennvr_internal`.

**3. Start.**

```powershell
docker compose up -d --build
docker compose ps                                  # adapter healthy, register exited 0
docker compose logs fall-detection-register        # "Registered 'fall-detection'."
docker compose logs -f fall-detection              # "watching camN" once cameras are picked
```

**4. In OpenNVR's UI.**

1. **App Catalog → Fall Detection.** The app registers itself.
   - **Configure → Cameras**: pick the cameras to watch.
   - Under **Overlay**, press **Allow drawing**.
2. **AI models / adapters page**: `fall-detection` is listed and healthy.
3. **Live View**: switch the **Detections** button (scan-eye icon) on. Each watched person gets a box labelled `person (warming n/30)`, then `Upright 4%` / `Fall risk 63%`, and `FALLEN 97%` for 5 s after an alarm.
4. **Alerts (bell)**: one **Person fallen** alert per fall, with the frame and the skeleton drawn on it.
5. **App Catalog → Fall Detection** (the app's page) shows per-camera status: frame rate, current score, window fill, last fall, errors.

## Testing with your videos

```powershell
copy <your-space>\fall_example_*.mp4 test-videos\
docker compose --profile testcam up -d fall-testcam

# register each video as an OpenNVR camera (runs inside OpenNVR's core container)
Get-Content -Raw testcam\register_test_cameras.py | docker exec -i `
  -e FAKECAM_HOST=fall-testcam:9997 -e FAKECAM_IP=172.28.90.20 -e FAKECAM_PREFIX=fall-test- `
  opennvr_core python -
```

The videos appear as cameras named `fall-test-fall_example_1` and so on. They play in Live View like any camera. Pick them for the app (step 4.1) and every loop of a fall clip raises an alarm, at most once per `cooldown_s` per camera. The clips are short, so one alarm per loop is expected. `fall_example_4` is the no-fall control and must never alarm.

If `TESTCAM_IP` is taken, or you changed OpenNVR's `OPENNVR_DOCKER_SUBNET`, set it in `.env` and pass the same address as `FAKECAM_IP`.

Remove the test cameras in OpenNVR's Cameras page, then run `docker compose --profile testcam rm -sf fall-testcam`.

## Checking each piece on its own

```powershell
# the adapter answers the contract (both models loaded)
docker exec opennvr_fall_detection_adapter python -m opennvr_adapter_sdk.conformance http://127.0.0.1:9013 --token <INTERNAL_API_KEY>
# the app's live state
docker exec opennvr_fall_detection python -c "import httpx; print(httpx.get('http://127.0.0.1:9212/state').text)"
```

## What you get on a stock (pulled) OpenNVR core

Everything above works against the core image you pull (`ghcr.io/open-nvr/core`): alarms with pictures, Live View boxes and labels, catalog configuration, and per-camera state.

Two extras need OpenNVR core built with `opennvr-ui-patch/fall-detection-ui.patch`:

- the **skeleton lines** drawn on Live View (stock core drops the keypoints the app sends);
- a dedicated **Applications → Falls** page: live video with the skeleton, per-camera status and the fall alarms on one screen.

The patch applies cleanly to and typechecks on both `v0.1.8` and `main`. It is optional and independent of this bundle.

```powershell
cd <open-nvr>
git apply <this folder>\opennvr-ui-patch\fall-detection-ui.patch
docker build -t ghcr.io/open-nvr/core:fall-local .
# then in open-nvr\.env: CORE_TAG=fall-local, and recreate core:
docker compose up -d opennvr-core
```

Undo: set `CORE_TAG` back and run `git checkout -- .` in the open-nvr checkout.

## Settings

Change the app's settings in the App Catalog form; they apply live. The adapter's settings go in `.env` and need `docker compose up -d`.

| Setting | Default | Where | |
|---|---|---|---|
| `threshold` | 0.90 | App Catalog | Fall score that raises an alarm (the model's validated value). |
| `cooldown_s` | 10 | App Catalog | Seconds between alarms on one camera. |
| `min_hits` | 1 | App Catalog | Hot frames in a row before alarming. |
| `alert_severity` | critical | App Catalog | |
| `fps` | 10 | App Catalog | Frames analysed per second per camera. |
| `hold_s` | 5 | App Catalog | Seconds `FALLEN` stays on the live view. |
| `FALL_POSE_IMGSZ` | 640 | `.env` | 448 is faster but moves fall onset by up to 7 frames. |
| `FALL_POSE_CONF` | 0.25 | `.env` | Person floor (ultralytics' default, what the classifier was fed). |

## How faithful it is to your Space

The feature code is a line-for-line port of `app.py`: alphabetical joint order, torso normalization, largest person as subject. It matched the Space's own functions on 3,000 randomized frames with zero mismatches.

End to end, comparing the Space (`.pt`, CPU) with this adapter (ONNX, JPEG frames over the live-stream protocol, two cameras interleaved):

| Clip | Space: first frame ≥ 0.90 / peak | This adapter |
|---|---|---|
| 1 (fall) | 113 / 0.974 | 113 / 0.975 |
| 2 (fall) | 88 / 0.952 | 88 / 0.943 |
| 3 (fall) | 48 / 0.990 | 48 / 0.993 |
| 4 (no fall) | — / 0.090 | — / 0.092 |

At the app's 10 fps, the full app and adapter raised exactly one alarm on each of clips 1–3 (at 5.2 s, 3.1 s and 2.9 s into the clip) and none on clip 4.

## Limits

- **One person per camera.** The model scores the largest person in view; others are drawn as plain `person` boxes.
- **Frame rate.** It was trained on 24–30 fps video. 10–15 fps is measured to work; below about 6 fps a fall is too few frames of the window, and the app's state says so.
- **3 s blind after a (re)connect.** At 10 fps the window needs 30 frames before the first score.
- **CPU.** 60–90 ms a frame at 640 px on an 8-core laptop, so about one camera at 10 fps per adapter. More cameras means more adapter instances, or lower `fps`.

## Remove

```powershell
docker compose --profile testcam down     # containers only; OpenNVR keeps running
```

Then delete the app from OpenNVR's App Catalog and the adapter from the AI models page.

## Development

```powershell
cd adapter; python -m pytest tests -q     # needs numpy 1.26, fastapi, pydantic, ./opennvr_adapter_sdk
cd app;     python -m pytest tests -q     # needs numpy, opencv-python-headless, ./opennvr-app-sdk
```

`VENDORED.md` lists the code copied in from ai-adapter and open-nvr.
