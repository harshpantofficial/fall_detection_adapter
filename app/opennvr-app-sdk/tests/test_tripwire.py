# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""Tripwire geometry tests — promoted alongside the implementation
from ``examples/line-crossing`` (whose own suite keeps the app-level
crossing state machine pinned)."""
from __future__ import annotations

import pytest

from opennvr_app_sdk.geometry import Point, Tripwire


def _vertical_mid(count_direction: str = "both") -> Tripwire:
    # Vertical wire down the middle of a 1000-wide frame, A=top B=bottom.
    return Tripwire.from_config(
        "mid", a=[500, 0], b=[500, 1000], count_direction=count_direction,
    )


def test_detects_crossings_both_ways():
    wire = _vertical_mid()
    assert wire.crossing(Point(400, 500), Point(600, 500)) == "a_to_b"
    assert wire.crossing(Point(600, 500), Point(400, 500)) == "b_to_a"


def test_moving_along_one_side_is_not_a_crossing():
    wire = _vertical_mid()
    assert wire.crossing(Point(400, 100), Point(400, 900)) is None


def test_respects_count_direction():
    a_to_b_only = _vertical_mid("a_to_b")
    assert a_to_b_only.crossing(Point(400, 500), Point(600, 500)) == "a_to_b"
    assert a_to_b_only.crossing(Point(600, 500), Point(400, 500)) is None
    b_to_a_only = _vertical_mid("b_to_a")
    assert b_to_a_only.crossing(Point(400, 500), Point(600, 500)) is None
    assert b_to_a_only.crossing(Point(600, 500), Point(400, 500)) == "b_to_a"


def test_grazing_the_line_is_not_a_crossing():
    wire = _vertical_mid()
    # Ends exactly on the line → not a committed crossing.
    assert wire.crossing(Point(400, 500), Point(500, 500)) is None
    # Starts exactly on the line → same.
    assert wire.crossing(Point(500, 500), Point(600, 500)) is None


def test_side_flip_without_segment_intersection_is_not_a_crossing():
    wire = _vertical_mid()
    # Both points are beyond the wire's B end: the infinite line is
    # crossed, the segment is not.
    assert wire.crossing(Point(400, 1500), Point(600, 1500)) is None


def test_side_sign_convention():
    wire = _vertical_mid()
    assert wire.side(Point(400, 500)) > 0   # left of A→B (y-down)
    assert wire.side(Point(600, 500)) < 0   # right of A→B
    assert wire.side(Point(500, 250)) == 0  # on the line


def test_degenerate_wire_rejected():
    with pytest.raises(ValueError, match="must differ"):
        Tripwire.from_config("dot", a=[10, 10], b=[10, 10])


def test_bad_count_direction_rejected():
    with pytest.raises(ValueError, match="count_direction"):
        Tripwire.from_config("mid", a=[0, 0], b=[1, 1], count_direction="sideways")
