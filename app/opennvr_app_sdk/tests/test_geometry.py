# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""Zone geometry tests — point-in-polygon + bbox_center scaling."""
from __future__ import annotations

import pytest

from opennvr_app_sdk.geometry import Point, Zone, bbox_center


def _square() -> Zone:
    return Zone.from_config("sq", [[0, 0], [100, 0], [100, 100], [0, 100]])


def test_zone_contains_interior_point():
    assert _square().contains(Point(50, 50))


def test_zone_excludes_exterior_point():
    zone = _square()
    assert not zone.contains(Point(150, 50))
    assert not zone.contains(Point(-1, 50))
    assert not zone.contains(Point(50, 101))


def test_zone_edge_points_count_as_inside():
    """Points exactly on an edge are INSIDE — the safer choice for a
    monitoring app (track the borderline person, don't miss them)."""
    zone = _square()
    assert zone.contains(Point(0, 50))     # left edge
    assert zone.contains(Point(100, 100))  # corner
    assert zone.contains(Point(50, 0))     # top edge


def test_concave_polygon():
    # L-shape: the notch at top-right is outside.
    zone = Zone.from_config(
        "L", [[0, 0], [50, 0], [50, 50], [100, 50], [100, 100], [0, 100]],
    )
    assert zone.contains(Point(25, 25))    # in the vertical arm
    assert zone.contains(Point(75, 75))    # in the horizontal arm
    assert not zone.contains(Point(75, 25))  # in the notch


def test_zone_rejects_degenerate_polygon():
    with pytest.raises(ValueError, match="at least 3 vertices"):
        Zone(name="line", polygon=[Point(0, 0), Point(1, 1)])
    with pytest.raises(ValueError, match="at least 3 vertices"):
        Zone.from_config("dot", [[5, 5]])


def test_zone_from_config_coerces_floats():
    zone = Zone.from_config("z", [["0", "0"], [10, 0], [10, 10]])
    assert zone.polygon[0] == Point(0.0, 0.0)


def test_bbox_center_scales_normalized_bbox_to_pixels():
    # bbox (0.45, 0.45, 0.1, 0.1) on 1920x1080 → center (960, 540)...
    center = bbox_center({"x": 0.45, "y": 0.45, "w": 0.1, "h": 0.1}, 1920, 1080)
    assert center == Point(0.5 * 1920, 0.5 * 1080)


def test_bbox_center_missing_keys_default_to_zero():
    center = bbox_center({}, 1920, 1080)
    assert center == Point(0.0, 0.0)
    center = bbox_center({"x": 0.5, "y": 0.5}, 100, 100)  # no w/h
    assert center == Point(50.0, 50.0)


def test_bbox_center_non_numeric_values_default_to_zero():
    center = bbox_center({"x": "junk", "y": None, "w": 0.2, "h": 0.2}, 100, 100)
    assert center == Point(10.0, 10.0)


# ── scale_vertices: the App Catalog editor's normalized→pixel bridge ───


def test_scale_vertices_normalized_scales_to_pixels():
    from opennvr_app_sdk.geometry import scale_vertices

    out = scale_vertices([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]], 1920, 1080)
    assert out == [[192.0, 216.0], [960.0, 540.0], [1728.0, 864.0]]


def test_scale_vertices_pixels_pass_through():
    from opennvr_app_sdk.geometry import scale_vertices

    # Any coord > 1 marks the whole set as literal pixels (legacy config).
    out = scale_vertices([[200, 400], [1720, 400], [100, 900]], 1920, 1080)
    assert out == [[200.0, 400.0], [1720.0, 400.0], [100.0, 900.0]]


def test_scale_vertices_empty_is_empty():
    from opennvr_app_sdk.geometry import scale_vertices

    assert scale_vertices([], 1920, 1080) == []
