# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
Zone + tripwire geometry — the per-camera shapes apps evaluate
detections against.

The polygon half was promoted verbatim from
``examples/loitering-detection/zone.py`` (itself a copy of
``examples/intrusion-detection/zone.py``); the tripwire half from
``examples/line-crossing/line.py``. The old READMEs framed these as
"per-camera business logic, not framework-y enough to factor into a
library" — the App SDK spec (§04) reverses that call: every zone- or
line-shaped app copies exactly these files, so they are the definition
of framework-y.

A :class:`Zone` is a closed polygon in pixel coordinates of the camera
frame. Detections whose bbox center falls inside the polygon contribute
to the app's per-zone logic (dwell timers, intrusion checks, counting).
Ray-casting algorithm: cast a horizontal ray from the test point to
the right and count intersections with polygon edges. Odd = inside,
even = outside. Standard textbook implementation.

A :class:`Tripwire` is an oriented line segment A→B. A tracked object
crosses it when the segment between its previous and current center
positions intersects the tripwire AND the object genuinely changed
sides. See the class docstring for the direction convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Point:
    """2D point in pixel coordinates (origin top-left, y-down)."""

    x: float
    y: float


@dataclass
class Zone:
    """A named polygonal zone in pixel coordinates.

    Polygon vertices are pixel coords on the camera frame. The polygon
    is implicitly closed (we connect the last vertex back to the
    first). A degenerate zone with < 3 vertices is rejected.
    """

    name: str
    polygon: list[Point]

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(
                f"Zone {self.name!r} requires at least 3 vertices; got {len(self.polygon)}"
            )

    def contains(self, point: Point) -> bool:
        """True if the point lies inside the polygon (ray-casting)."""
        return _point_in_polygon(point, self.polygon)

    @classmethod
    def from_config(cls, name: str, vertices: Sequence[Sequence[float]]) -> "Zone":
        """Build a Zone from config-style vertex list ``[[x, y], [x, y], ...]``."""
        return cls(
            name=name,
            polygon=[Point(float(v[0]), float(v[1])) for v in vertices],
        )


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting point-in-polygon. Points exactly on an edge are
    treated as INSIDE — the safer choice for a monitoring app (we'd
    rather track a borderline person than miss them)."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        pi, pj = polygon[i], polygon[j]
        if _point_on_segment(point, pi, pj):
            return True
        if (pi.y > point.y) != (pj.y > point.y):
            slope_x = (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x
            if point.x < slope_x:
                inside = not inside
        j = i
    return inside


def _point_on_segment(point: Point, a: Point, b: Point, *, eps: float = 1e-9) -> bool:
    cross = (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x)
    if abs(cross) > eps:
        return False
    dot = (point.x - a.x) * (b.x - a.x) + (point.y - a.y) * (b.y - a.y)
    if dot < -eps:
        return False
    sq_len = (b.x - a.x) ** 2 + (b.y - a.y) ** 2
    return dot - sq_len <= eps


# ── Tripwire (directional line crossing) ───────────────────────────
#
# Promoted verbatim from ``examples/line-crossing/line.py``. A tripwire
# is an oriented line segment A→B in pixel coordinates of the camera
# frame. The *direction* of a crossing (which way it went) is
# determined by which side of the line the object started on:
#
#     side(P) = sign of the 2D cross product (B - A) × (P - A)
#
#     side > 0  → "left" of A→B
#     side < 0  → "right" of A→B
#
# By convention we name the two directions ``a_to_b`` and ``b_to_a`` so
# an operator can label them meaningfully ("enter" / "exit", "in" /
# "out"). A configured ``count_direction`` of ``both`` counts either way.


@dataclass
class Tripwire:
    """An oriented line segment A→B with a counted direction.

    ``count_direction`` is one of ``"a_to_b"``, ``"b_to_a"``, ``"both"``.
    """

    name: str
    a: Point
    b: Point
    count_direction: str = "both"

    def __post_init__(self) -> None:
        if self.a.x == self.b.x and self.a.y == self.b.y:
            raise ValueError(f"Tripwire {self.name!r}: A and B must differ")
        if self.count_direction not in ("a_to_b", "b_to_a", "both"):
            raise ValueError(
                f"Tripwire {self.name!r}: count_direction must be "
                f"'a_to_b', 'b_to_a' or 'both'; got {self.count_direction!r}"
            )

    @classmethod
    def from_config(
        cls, name: str, a: Sequence[float], b: Sequence[float],
        count_direction: str = "both",
    ) -> "Tripwire":
        return cls(
            name=name,
            a=Point(float(a[0]), float(a[1])),
            b=Point(float(b[0]), float(b[1])),
            count_direction=count_direction,
        )

    def side(self, p: Point) -> float:
        """Signed side of point ``p`` relative to the oriented line A→B.
        Positive = left, negative = right, ~0 = on the line."""
        return (self.b.x - self.a.x) * (p.y - self.a.y) - (
            self.b.y - self.a.y
        ) * (p.x - self.a.x)

    def crossing(self, prev: Point, curr: Point) -> str | None:
        """Return the crossing direction (``"a_to_b"`` / ``"b_to_a"``) if
        the movement ``prev → curr`` crosses this tripwire in a counted
        direction, else ``None``.

        Two conditions must both hold:
        1. the segment prev→curr intersects the segment A→B
           (so the object physically traversed the wire), and
        2. the side sign flipped from prev to curr (so it genuinely
           changed sides, not merely touched the line).
        """
        side_prev = self.side(prev)
        side_curr = self.side(curr)
        # Must end up on opposite sides (strict) — grazing the line
        # (one side == 0) is not a committed crossing.
        if side_prev == 0 or side_curr == 0:
            return None
        if (side_prev > 0) == (side_curr > 0):
            return None
        if not _segments_intersect(prev, curr, self.a, self.b):
            return None
        # side_prev > 0 means it started on the LEFT of A→B and ended on
        # the right → it moved across in the A→B-rightward sense, which
        # we name "a_to_b". The opposite is "b_to_a".
        direction = "a_to_b" if side_prev > 0 else "b_to_a"
        if self.count_direction in (direction, "both"):
            return direction
        return None


def _orientation(p: Point, q: Point, r: Point) -> int:
    """0 = collinear, 1 = clockwise, 2 = counter-clockwise."""
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if val == 0:
        return 0
    return 1 if val > 0 else 2


def _on_segment(p: Point, q: Point, r: Point) -> bool:
    """True if q lies on segment p–r (assuming the three are collinear)."""
    return (
        min(p.x, r.x) <= q.x <= max(p.x, r.x)
        and min(p.y, r.y) <= q.y <= max(p.y, r.y)
    )


def _segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Standard segment-intersection test (p1–p2 vs p3–p4), handling
    the collinear-overlap special cases."""
    o1 = _orientation(p1, p2, p3)
    o2 = _orientation(p1, p2, p4)
    o3 = _orientation(p3, p4, p1)
    o4 = _orientation(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p3, p2):
        return True
    if o2 == 0 and _on_segment(p1, p4, p2):
        return True
    if o3 == 0 and _on_segment(p3, p1, p4):
        return True
    if o4 == 0 and _on_segment(p3, p2, p4):
        return True
    return False


def bbox_center(bbox_normalized: dict, frame_width: int, frame_height: int) -> Point:
    """Convert a §5.1 ``NormalizedBBox`` (x/y/w/h in [0, 1]) into a
    pixel-space center point given the camera's actual frame size.
    Defensive against partial bboxes: missing keys default to 0."""
    def _coerce(key: str) -> float:
        value = bbox_normalized.get(key, 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    x = _coerce("x")
    y = _coerce("y")
    w = _coerce("w")
    h = _coerce("h")
    return Point(
        x=(x + w / 2.0) * frame_width,
        y=(y + h / 2.0) * frame_height,
    )


def scale_vertices(
    vertices: Sequence[Sequence[float]], frame_width: int, frame_height: int
) -> list[list[float]]:
    """Return pixel-space ``[[x, y], …]`` for geometry that may be either
    NORMALIZED (0–1) or already in pixels.

    The App Catalog geometry editor emits normalized 0–1 coordinates
    (resolution-independent). Hand-written legacy config uses literal
    pixels. This resolves both: if EVERY coordinate is within [0, 1] the
    input is treated as normalized and scaled by the frame dims;
    otherwise it's taken as literal pixels and passed through. The
    ambiguous all-in-[0,1] pixel case (a zone in the top-left few pixels)
    is not a real ROI, so treating it as normalized is safe.
    """
    pts = [(float(v[0]), float(v[1])) for v in vertices]
    if pts and all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in pts):
        return [[x * frame_width, y * frame_height] for x, y in pts]
    return [[x, y] for x, y in pts]
