# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
The AlertSubscriber base — the pass-through archetype (App SDK spec §02).

Where a :class:`~.detector.Detector` rides ``opennvr.inference.*`` and
a :class:`~.frame_app.FrameApp` drives inference itself, an
AlertSubscriber rides the other bus: it subscribes to the §11.5 alert
fan-out (``opennvr.alerts.>``) that Detectors / FrameApps publish onto,
and forwards each envelope to a sink — the operator-UI inbox, a SIEM
bridge, a Slack/PagerDuty bot, Home Assistant. It consumes alerts; it
does not emit them, so there is no ``AlertDispatcher`` in this base.

What the base owns (ported from the alerts-subscriber example, the
canonical subscriber-side template):

* the NATS connect / subscribe / drain loop — shared verbatim with
  ``Detector`` via :class:`~.nats_loop.NatsSubscriberMixin`;
* per-message JSON decoding + exception isolation (one bad envelope or
  one raising handler never takes down a long-lived bridge) — factored
  into :meth:`AlertSubscriber._handle_raw` so tests can exercise the
  full path without a broker;
* the §03 contract surface via :class:`~.contract.ContractMixin`
  (``/health`` counts every decoded alert as an event);
* the CLI / logging / SIGINT+SIGTERM lifecycle behind
  ``alert_app(MySubscriber).run()``.

What the app writes: an optional ``manifest``, an optional ``setup()``,
and ``on_alert(alert, subject)`` — the sink.

``alert`` is the JSON-decoded §11.5 Alert envelope as a plain dict
(``alert_id``, ``fired_at``, ``title``, ``description``, ``severity``,
``source``, ``camera_id``, ``correlation_id``, ``evidence``, ``tags``)
— defensively: it is whatever JSON arrived on the subject, so sinks
should ``.get()`` rather than index.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from typing import Any, Callable

from .contract import ContractMixin
from .manifest import AppManifest
from .nats_loop import NatsSubscriberMixin

logger = logging.getLogger(__name__)


class AlertSubscriber(ContractMixin, NatsSubscriberMixin):
    """Base class for NATS alert-consuming apps.

    Subclasses optionally set a class-level ``manifest``
    (:class:`AppManifest`), optionally override :meth:`setup` to
    allocate state, and implement :meth:`on_alert`. ``cfg`` is the
    app-parsed config object; the NATS loop reads ``cfg.nats_url``,
    ``cfg.nats_token`` (optional) and ``cfg.subject_pattern`` from it
    (apps default the pattern to ``"opennvr.alerts.>"``).
    """

    manifest: AppManifest | None = None

    def __init__(self, config: Any) -> None:
        self.cfg = config
        # Compat alias — pre-SDK subscribers (and their tests) used
        # ``self._config``.
        self._config = config
        self._stop_event = asyncio.Event()
        self._nc: Any = None
        self._contract_init()
        self.setup()

    # ── App surface ────────────────────────────────────────────────

    def setup(self) -> None:
        """Optional hook — allocate per-app state (counters, HTTP
        clients, …). Runs once at construction, after ``cfg`` is set."""

    def on_alert(self, alert: dict[str, Any], subject: str) -> None:
        """The sink. Called once per JSON-decoded alert envelope with
        the raw dict and the NATS subject it arrived on. Forward it,
        store it, page someone — whatever the bridge is for."""
        raise NotImplementedError

    # ── Per-message handling (testable without NATS) ───────────────

    def _handle_raw(self, data: bytes, *, subject: str = "") -> bool:
        """Decode one raw NATS message and run it through the sink.

        Isolation contract (ported from the pre-SDK loop): non-JSON
        payloads are logged + skipped; a raising ``on_alert`` is logged
        + swallowed. A long-lived bridge never dies to one bad message.
        Returns ``True`` iff the handler ran without raising."""
        try:
            payload = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            # UnicodeDecodeError too: ``json.loads(bytes)`` BOM-sniffs
            # the encoding and can fail before JSON parsing starts.
            logger.warning("skipping non-JSON message on %r: %s", subject, exc)
            return False
        # Contract counters (spec §03): every decoded envelope counts
        # as "seen" — /health's last_event_age_s is stall detection for
        # the alert bus.
        self._contract_note_event()
        try:
            self.on_alert(payload, subject)
        except Exception:
            # No single alert-handler failure should kill the
            # subscriber. The operator sees the traceback and the next
            # alert is still processed.
            logger.exception("on_alert failed for subject=%s", subject)
            return False
        return True

    # ── NATS loop ──────────────────────────────────────────────────

    async def run(self, *, once: bool = False) -> None:
        """Connect to NATS, subscribe, drive the sink on every received
        alert. Returns when ``stop()`` is called or when ``once=True``
        and one message has been processed.

        Also owns the app-contract lifecycle (spec §03), same as the
        other archetypes: starts the ``/health`` / ``/manifest`` /
        ``/state`` server when ``cfg.contract_port`` is set and
        self-registers when ``cfg.opennvr_url`` is set — both
        best-effort no-ops otherwise. The connect / subscribe / drain
        machinery lives on :class:`~.nats_loop.NatsSubscriberMixin`,
        shared with ``Detector``."""
        self.start_contract_server()
        self.register_with_opennvr()
        self.start_config_poll()
        try:
            await self._run_nats_loop(once=once)
        finally:
            self.stop_config_poll()
            self.stop_contract_server()


# ── CLI runner ──────────────────────────────────────────────────────


class AlertSubscriberRunner:
    """The ``alert_app(MySubscriber)`` return value — owns argparse,
    logging setup, and the signal-driven lifecycle. Mirrors the
    Detector's :class:`~.detector.AppRunner` minus the dispatcher: an
    AlertSubscriber consumes alerts rather than emitting them."""

    def __init__(
        self,
        subscriber_cls: type[AlertSubscriber],
        *,
        load_config: Callable[[str], Any] | None = None,
    ) -> None:
        loader = load_config or getattr(subscriber_cls, "load_config", None)
        if loader is None:
            raise TypeError(
                f"alert_app({subscriber_cls.__name__}): pass load_config= or "
                f"define a load_config classmethod on the subscriber class"
            )
        self._subscriber_cls = subscriber_cls
        self._load_config = loader

    def run(self, argv: list[str] | None = None) -> int:
        manifest = self._subscriber_cls.manifest
        parser = argparse.ArgumentParser(
            prog=manifest.id if manifest else self._subscriber_cls.__name__,
            description=(manifest.summary or manifest.name) if manifest else None,
        )
        parser.add_argument("--config", required=True, help="Path to config.yml")
        parser.add_argument(
            "--once",
            action="store_true",
            help="Process one alert then exit (smoke testing).",
        )
        parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        )
        args = parser.parse_args(argv)

        logging.basicConfig(
            level=args.log_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        try:
            config = self._load_config(args.config)
        except (ValueError, OSError) as exc:
            print(f"config error: {exc}", file=sys.stderr)
            return 2

        subscriber = self._subscriber_cls(config)

        loop = asyncio.new_event_loop()

        def _handle_signal(_signum: int, _frame: Any) -> None:
            logger.info("signal received, stopping…")
            loop.call_soon_threadsafe(subscriber.stop)

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
        try:
            loop.run_until_complete(subscriber.run(once=args.once))
        finally:
            loop.close()
        return 0


def alert_app(
    subscriber_cls: type[AlertSubscriber],
    *,
    load_config: Callable[[str], Any] | None = None,
) -> AlertSubscriberRunner:
    """Wrap an AlertSubscriber subclass in a CLI runner::

        if __name__ == "__main__":
            raise SystemExit(
                alert_app(MyBridge, load_config=load_config).run()
            )
    """
    return AlertSubscriberRunner(subscriber_cls, load_config=load_config)


__all__ = ["AlertSubscriber", "AlertSubscriberRunner", "alert_app"]
