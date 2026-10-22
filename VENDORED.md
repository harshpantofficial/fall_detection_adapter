# Vendored code

This bundle builds without the `ai-adapter` checkout and without the `open-nvr` source tree, so it carries copies of what it needs. Re-copy them to pick up upstream fixes.

| Here | Copied from | At |
|---|---|---|
| `adapter/adapters/yolo_pose/` (`__init__.py`, `service.py`, `coco_keypoints.py`) | ai-adapter `adapters/yolo_pose/` | `40a6119` |
| `adapter/opennvr_adapter_sdk/` | ai-adapter `opennvr_adapter_sdk/` | `40a6119` |
| `app/opennvr-app-sdk/` | open-nvr `sdk/opennvr-app-sdk/` | `b01b2a23` |
| `testcam/entrypoint.sh` | open-nvr `scripts/fakecams/entrypoint.sh` | `b01b2a23` |
| `testcam/register_test_cameras.py` | open-nvr `scripts/fakecams/register_fake_cameras.py` | `b01b2a23` |

Local changes to the copies:

- `adapter/opennvr_adapter_sdk/conformance/runner.py`: sample payloads for the `fall_detection` task, so conformance exercises `/infer` and the §6 stream instead of warning.
- `app/opennvr-app-sdk/opennvr_app_sdk/validate.py`: `fall_detection` added to `KNOWN_TASKS`, which silences the "unknown task" warning.

Everything under `adapter/adapters/fall_detection/` and `app/` (other than the SDK) is this bundle's own code.
