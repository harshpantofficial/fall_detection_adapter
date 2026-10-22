# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The alarm rule on a synthetic clock: a run of scores → one alarm."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fall_rules import FallMonitor, FallPolicy  # noqa: E402


def feed(monitor, scores, start=0.0, dt=0.1):
    alarms = []
    for i, s in enumerate(scores):
        a = monitor.observe(s, start + i * dt)
        if a is not None:
            alarms.append(a)
    return alarms


def test_one_fall_is_one_alarm_with_its_peak():
    m = FallMonitor()
    alarms = feed(m, [0.1, 0.5, 0.91, 0.97, 0.95, 0.93, 0.4])
    assert len(alarms) == 1
    assert alarms[0].score == 0.91 and alarms[0].hits == 1
    assert m.peak == 0.0                          # the run ended at 0.4


def test_reference_default_fires_on_the_first_frame_over_threshold():
    assert FallPolicy().threshold == 0.90 and FallPolicy().min_hits == 1
    assert feed(FallMonitor(), [0.89, 0.90])[0].at == 0.1   # >= , as the reference


def test_cooldown_is_seconds_not_frames():
    m = FallMonitor(FallPolicy(cooldown_s=10.0))
    first = feed(m, [0.95], start=0.0)
    # A second fall 5 s later: inside the cooldown.
    assert feed(m, [0.1, 0.95], start=5.0) == []
    # And one 11 s later: a new alarm.
    assert len(feed(m, [0.1, 0.95], start=11.0)) == 1
    assert first


def test_min_hits_needs_consecutive_frames():
    m = FallMonitor(FallPolicy(min_hits=3))
    assert feed(m, [0.95, 0.95, 0.2, 0.95, 0.95]) == []
    assert len(feed(m, [0.95], start=1.0)) == 1


def test_warming_up_neither_builds_nor_breaks_a_run():
    m = FallMonitor(FallPolicy(min_hits=2))
    assert feed(m, [0.95, None, 0.95])[0].hits == 2


def test_reset_breaks_a_run_across_a_feed_gap():
    m = FallMonitor(FallPolicy(min_hits=2))
    feed(m, [0.95])
    m.reset()
    assert feed(m, [0.95], start=1.0) == []


def test_fallen_holds_for_hold_s():
    m = FallMonitor(FallPolicy(hold_s=5.0))
    feed(m, [0.95], start=100.0)
    assert m.fallen(104.9) and not m.fallen(105.1)
    assert not FallMonitor().fallen(0.0)


def test_policy_from_config_clamps_and_survives_junk():
    p = FallPolicy.from_config({"threshold": "1.7", "min_hits": "x", "cooldown_s": -3})
    assert p.threshold == 1.0 and p.min_hits == 1 and p.cooldown_s == 0.0
