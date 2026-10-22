# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
Shared fixtures for the SDK test suite.

``test_alerts.py`` is ported verbatim from the loitering-detection
example (the parity reference), and its assertions pin the default
§11.5 ``source.name`` to ``"loitering-detection"`` — in the example
that default was hardcoded; in the SDK it's the process-wide default
an app sets via ``set_default_source``. This autouse fixture plays the
role the app plays in production, and restores the SDK default after
each test so ordering can't leak state.
"""
from __future__ import annotations

import pytest

from opennvr_app_sdk import alerts as alerts_mod


@pytest.fixture(autouse=True)
def _loitering_default_source():
    saved = alerts_mod.get_default_source()
    alerts_mod.set_default_source(
        kind="app", name="loitering-detection", version="1.0.0",
    )
    yield
    alerts_mod.set_default_source(**saved)
