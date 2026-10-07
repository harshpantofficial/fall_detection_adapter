"""
Alerts compatibility shim — re-exports from opennvr_app_sdk.alerts.
Matches OpenNVR example app architecture.
"""
from __future__ import annotations

import httpx  # noqa: F401

from opennvr_app_sdk.alerts import (  # noqa: F401
    DEFAULT_ALERT_SUBJECT_PREFIX,
    Alert,
    AlertChannel,
    AlertDispatcher,
    AlertSource,
    NatsAlertChannel,
    StdoutChannel,
    WebhookChannel,
    alert_subject,
    build_dispatcher,
    get_default_source,
    set_default_source,
)

# Identify this app process
set_default_source(kind="app", name="fall-detection", version="1.0.0")

