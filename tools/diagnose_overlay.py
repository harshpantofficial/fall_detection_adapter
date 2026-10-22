"""Where do the fall-detection app's live-view boxes stop?

Runs INSIDE OpenNVR's core container (it needs core's settings, database
and loopback API), reads only, changes nothing:

    Get-Content -Raw tools\\diagnose_overlay.py | docker exec -i opennvr_core python -

The app's boxes travel four hops, and a break at any one of them looks
the same in the browser — no boxes:

  1. app      publishes overlay.boxes.v1 on the apps bus
  2. bus      the apps bus forwards it to the platform bus
  3. core     tier0_track_consumer: app enabled + "Allow drawing" + camera
              picked for the app → a `tracks` event, task=overlay
  4. browser  /api/v1/events/ws streams it to the Live View

This checks 2-4 directly and core's switches for 3, and says which hop
broke. Hop 1 is the app's own /state (`overlay.published` / `failed`).

Environment knobs: FALL_APP_ID (default fall-detection), DIAG_SECONDS
(default 10). Prints no keys or tokens.
"""
import asyncio
import json
import os
import sys
import urllib.request
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/app/server")
os.chdir("/app/server")

from core.config import settings  # noqa: E402
from core.database import SessionLocal  # noqa: E402
from models import Camera, InstalledApp, User  # noqa: E402

APP_ID = os.environ.get("FALL_APP_ID", "fall-detection")
SECONDS = float(os.environ.get("DIAG_SECONDS", "10"))
API = "http://127.0.0.1:8000/api/v1"


def say(ok, text):
    print(f"  [{'OK ' if ok else 'BAD'}] {text}")
    return ok


def check_core_switches():
    print("core switches (hop 3)")
    good = say(bool(getattr(settings, "detection_overlay_enabled", True)),
               "DETECTION_OVERLAY_ENABLED is on (off = no boxes reach any browser)")
    good &= say(bool((getattr(settings, "nats_url", "") or "").strip()),
                "core has a NATS_URL, so its overlay consumer is running")
    db = SessionLocal()
    try:
        row = db.query(InstalledApp).filter(InstalledApp.id == APP_ID).first()
        if not say(row is not None, f"app '{APP_ID}' is registered with core"):
            return False, set(), {}
        good &= say(bool(row.enabled), "app is enabled")
        good &= say(bool(getattr(row, "overlay_enabled", False)),
                    "drawing is allowed (App Catalog -> app -> Overlay -> Allow drawing)")
        try:
            from services.skill_assignments import picked_camera_ids
            picked = set(picked_camera_ids(db, APP_ID))
        except Exception as exc:  # older core
            print(f"  [?? ] could not read the app's camera picks: {exc}")
            picked = set()
        names = {c.id: c.name for c in db.query(Camera).filter(Camera.id.in_(picked)).all()} if picked else {}
        good &= say(bool(picked), "cameras picked for the app: "
                    + (", ".join(f"cam{i} ({names.get(i, '?')})" for i in sorted(picked)) or "none"))
        return good, picked, names
    finally:
        db.close()


def admin_token():
    db = SessionLocal()
    try:
        user = (db.query(User).filter(User.is_active.is_(True), User.is_superuser.is_(True))
                .order_by(User.id).first())
        if user is None:
            return None
        from core.auth import create_access_token
        return create_access_token({"sub": user.username})
    finally:
        db.close()


async def watch_bus(counts):
    """Hop 2: the overlay events as they arrive on the PLATFORM bus."""
    import nats

    url = (getattr(settings, "nats_url", "") or "").strip()
    token = (getattr(settings, "internal_api_key", "") or "").strip() or None
    try:
        nc = await nats.connect(url, token=token, connect_timeout=5)
    except Exception as exc:
        counts["error"] = f"could not connect to the platform bus: {exc}"
        return

    async def on_msg(msg):
        try:
            env = json.loads(msg.data)
        except ValueError:
            return
        producer = str(env.get("producer") or "")
        if producer.removeprefix("app:") != APP_ID:
            counts["other"] = counts.get("other", 0) + 1
            return
        cam = msg.subject.rsplit(".", 1)[-1]
        counts[cam] = counts.get(cam, 0) + 1
        counts.setdefault("sample", (env.get("payload") or {}).get("boxes"))

    await nc.subscribe("opennvr.events.overlay.boxes.v1.>", cb=on_msg)
    await asyncio.sleep(SECONDS)
    await nc.drain()


async def watch_ws(counts):
    """Hop 4: what the Live View's socket actually receives."""
    import websockets

    token = admin_token()
    if token is None:
        counts["error"] = "no active superuser to test the browser socket as"
        return
    req = urllib.request.Request(f"{API}/events/ws-ticket", data=b"", method="POST",
                                 headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            ticket = json.loads(resp.read())["ticket"]
    except Exception as exc:
        counts["error"] = f"could not get a websocket ticket: {exc}"
        return
    url = f"ws://127.0.0.1:8000/api/v1/events/ws?ticket={ticket}&task=tier0&task=overlay"
    try:
        async with websockets.connect(url, open_timeout=10) as ws:
            loop = asyncio.get_running_loop()
            end = loop.time() + SECONDS
            while (left := end - loop.time()) > 0:
                try:
                    evt = json.loads(await asyncio.wait_for(ws.recv(), timeout=left))
                except asyncio.TimeoutError:
                    break
                if evt.get("event_type") != "tracks":
                    continue
                key = f"{evt.get('task')}:cam{evt.get('camera_id')}"
                counts[key] = counts.get(key, 0) + 1
                if evt.get("task") == "overlay":
                    counts.setdefault("sample", [(t.get("label"), t.get("score"))
                                                 for t in (evt.get("payload") or {}).get("tracks", [])])
    except Exception as exc:
        counts["error"] = f"websocket failed: {str(exc).split('?')[0]}"


async def main():
    print(f"fall-detection live-view check (app '{APP_ID}', {SECONDS:.0f}s)\n")
    switches_ok, picked, names = check_core_switches()

    bus, ws = {}, {}
    print(f"\nlistening for {SECONDS:.0f}s ...")
    await asyncio.gather(watch_bus(bus), watch_ws(ws))

    print("\nplatform bus (hop 2)")
    if "error" in bus:
        say(False, bus["error"])
    per_cam = {k: v for k, v in bus.items() if k.startswith("cam")}
    say(bool(per_cam), "overlay events from the app: "
        + (", ".join(f"{k}={v}" for k, v in sorted(per_cam.items())) or "none"))
    if per_cam and picked:
        stray = [k for k in per_cam if int(k[3:]) not in picked]
        if stray:
            say(False, f"events for cameras NOT picked for the app (core drops them): {stray}")
    if bus.get("sample"):
        print(f"        sample boxes: {[(b.get('label'), b.get('score')) for b in bus['sample']][:4]}")

    print("\nbrowser socket (hop 4)")
    if "error" in ws:
        say(False, ws["error"])
    overlay = {k: v for k, v in ws.items() if k.startswith("overlay:")}
    tier0 = {k: v for k, v in ws.items() if k.startswith("tier0:")}
    say(bool(overlay), "app boxes reaching the Live View: "
        + (", ".join(f"{k[8:]}={v}" for k, v in sorted(overlay.items())) or "none"))
    print(f"        (Tier-0 person boxes on the same socket: "
          f"{', '.join(f'{k[6:]}={v}' for k, v in sorted(tier0.items())) or 'none'})")
    if ws.get("sample"):
        print(f"        sample: {ws['sample'][:4]}")

    print("\nverdict")
    if overlay:
        print("  Boxes reach the browser. In Live View, turn the Detections button (scan-eye")
        print("  icon) on, and watch one of the cameras listed above. Hard-refresh the page.")
    elif not per_cam:
        print("  The app's boxes never reach the platform bus (hop 1-2). Check the app's")
        print("  /state 'overlay' block and `docker logs opennvr_fall_detection` for")
        print("  'overlay publish failed' and NATS lines.")
    elif not switches_ok:
        print("  The boxes arrive, and core drops them: fix the BAD line(s) under 'core switches'.")
    else:
        print("  The boxes arrive and every switch is on, but nothing reaches the socket.")
        print("  Check `docker logs opennvr_core` for 'overlay consumer' / 'tier0 track consumer' lines.")


asyncio.run(main())
