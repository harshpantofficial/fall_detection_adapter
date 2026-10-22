# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""keyed_state tests — TTL expiry, the alerted latch, GC semantics,
and the two archetypal consumers (loitering-style dwell and
package-delivery-style per-track lifecycle)."""
from __future__ import annotations

import pytest

from opennvr_app_sdk.state import KeyedState, StateRecord, keyed_state


# ── Basics ─────────────────────────────────────────────────────────


def test_keyed_state_factory_returns_keyed_state():
    states = keyed_state(5.0)
    assert isinstance(states, KeyedState)
    assert states.ttl == 5.0
    assert len(states) == 0


def test_ttl_must_be_positive():
    with pytest.raises(ValueError, match="ttl"):
        keyed_state(0)
    with pytest.raises(ValueError, match="ttl"):
        keyed_state(-1.0)


def test_first_touch_creates_record():
    states = keyed_state(5.0)
    rec = states.touch("k", at=100.0)
    assert rec.first_seen == 100.0
    assert rec.last_seen == 100.0
    assert rec.age == 0.0
    assert rec.alerted is False
    assert "k" in states
    assert states.get("k") is rec
    assert states["k"] is rec


def test_repeated_touch_refreshes_last_seen_and_grows_age():
    states = keyed_state(5.0)
    states.touch("k", at=100.0)
    rec = states.touch("k", at=103.0)
    assert rec.first_seen == 100.0
    assert rec.last_seen == 103.0
    assert rec.age == 3.0


def test_touch_uses_wall_clock_when_at_omitted():
    ticks = iter([100.0, 107.0])
    states = keyed_state(5.0, clock=lambda: next(ticks))
    rec = states.touch("k")
    assert rec.first_seen == 100.0
    rec = states.touch("k")
    assert rec.last_seen == 107.0


# ── Latch ──────────────────────────────────────────────────────────


def test_alerted_latch_is_settable_and_persists_across_touches():
    states = keyed_state(5.0)
    rec = states.touch("k", at=0.0)
    assert rec.alerted is False
    rec.alerted = True
    rec = states.touch("k", at=1.0)
    assert rec.alerted is True  # same episode — the latch holds


def test_latch_resets_when_key_expires_and_reappears():
    states = keyed_state(5.0, auto_gc=False)
    rec = states.touch("k", at=0.0)
    rec.alerted = True
    states.gc(100.0)  # long past TTL — episode over
    rec2 = states.touch("k", at=100.0)
    assert rec2.alerted is False  # fresh episode, fresh latch
    assert rec2.first_seen == 100.0


# ── GC ─────────────────────────────────────────────────────────────


def test_gc_prunes_only_stale_keys_and_returns_them():
    states = keyed_state(5.0, auto_gc=False)
    states.touch("stale", at=0.0)
    states.touch("fresh", at=8.0)
    pruned = states.gc(10.0)  # cutoff = 5.0
    assert [k for k, _ in pruned] == ["stale"]
    assert isinstance(pruned[0][1], StateRecord)
    assert states.get("stale") is None
    assert "fresh" in states


def test_gc_boundary_is_strict():
    """A record exactly at the TTL boundary survives — matches the
    ``last_seen < cutoff`` grace-period semantics the loitering
    detector shipped with."""
    states = keyed_state(5.0, auto_gc=False)
    states.touch("edge", at=5.0)
    assert states.gc(10.0) == []  # last_seen == cutoff → keep
    assert "edge" in states
    assert [k for k, _ in states.gc(10.001)] == ["edge"]


def test_gc_respects_exclude():
    states = keyed_state(5.0, auto_gc=False)
    states.touch("a", at=0.0)
    states.touch("b", at=0.0)
    pruned = states.gc(100.0, exclude=("a",))
    assert [k for k, _ in pruned] == ["b"]
    assert "a" in states


def test_touch_auto_gc_prunes_other_stale_keys():
    states = keyed_state(5.0)  # auto_gc defaults on
    states.touch("old", at=0.0)
    states.touch("new", at=100.0)
    assert states.get("old") is None
    assert "new" in states


def test_touch_never_prunes_its_own_key():
    """A touch means "present now" — even after a gap far beyond the
    TTL, the touched key is refreshed, not restarted. This is what
    keeps a continuous-but-sparse presence stream (0.1 fps with a 5 s
    TTL) accruing one dwell episode instead of many."""
    states = keyed_state(5.0)
    states.touch("k", at=0.0)
    rec = states.touch("k", at=100.0)
    assert rec.first_seen == 0.0  # episode NOT restarted
    assert rec.age == 100.0


def test_pop_and_items():
    states = keyed_state(5.0)
    states.touch(("cam-1", "person"), at=0.0)
    states.touch(("cam-1", "car"), at=1.0)
    keys = [k for k, _ in states.items()]
    assert set(keys) == {("cam-1", "person"), ("cam-1", "car")}
    # items() is a snapshot — popping while iterating is safe.
    for key, rec in states.items():
        if key[1] == "car":
            assert states.pop(key) is rec
    assert len(states) == 1
    assert states.pop("missing") is None
    assert states.pop("missing", "sentinel") == "sentinel"


def test_record_factory_allows_typed_subclasses():
    class DwellRecord(StateRecord):
        @property
        def present_since(self) -> float:
            return self.first_seen

    states = keyed_state(5.0, record_factory=DwellRecord)
    rec = states.touch("k", at=42.0)
    assert isinstance(rec, DwellRecord)
    assert rec.present_since == 42.0


def test_record_data_scratchpad():
    states = keyed_state(5.0)
    rec = states.touch("trk_1", at=0.0)
    rec.data["phase"] = "arrived"
    assert states.touch("trk_1", at=1.0).data["phase"] == "arrived"


# ── Archetype scenario: package-delivery per-track lifecycle ───────


def test_package_delivery_track_lifecycle():
    """Per-track arrival → linger → disappear with a one-shot latch,
    the shape ``examples/package-delivery`` needs from keyed_state:

    * arrival: first touch of a track_id starts the clock
    * linger: the alert fires exactly once when age crosses the
      threshold (the ``alerted`` latch), no matter how many more
      frames the package sits there
    * disappear: no touches for > TTL → ``gc`` prunes the track and
      hands the record back so the app can emit a "package gone"
      event; a later same-id arrival is a fresh episode
    """
    linger_threshold = 2.0
    gone_ttl = 3.0
    states = keyed_state(gone_ttl, auto_gc=False)
    linger_alerts: list[tuple[str, float]] = []
    gone_events: list[str] = []

    def tick(now: float, visible_tracks: list[str]) -> None:
        # What a poll-loop tick does: touch what's visible, GC what's not.
        for track_id in visible_tracks:
            rec = states.touch(track_id, at=now)
            if rec.age >= linger_threshold and not rec.alerted:
                rec.alerted = True
                linger_alerts.append((track_id, now))
        for track_id, _rec in states.gc(now):
            gone_events.append(track_id)

    # Arrival at t=100; box sits on the porch through t=104.
    for t in (100.0, 101.0, 102.0, 103.0, 104.0):
        tick(t, ["trk_1"])
    # Latch: fired exactly once, at the threshold crossing.
    assert linger_alerts == [("trk_1", 102.0)]

    # Box picked up — track vanishes. Within TTL: still tracked.
    tick(106.0, [])
    assert gone_events == []
    # Beyond TTL: pruned, "gone" event emitted once.
    tick(108.0, [])
    assert gone_events == ["trk_1"]
    assert states.get("trk_1") is None
    tick(109.0, [])
    assert gone_events == ["trk_1"]  # no re-fire after prune

    # A new delivery reusing the id is a FRESH episode: alert re-arms.
    for t in (200.0, 201.0, 202.0):
        tick(t, ["trk_1"])
    assert linger_alerts == [("trk_1", 102.0), ("trk_1", 202.0)]


# ── Archetype scenario: loitering-style dwell with grace period ────


def test_loitering_dwell_with_grace_reset():
    """Absence beyond the grace period (= TTL) resets the dwell so a
    fresh arrival doesn't inherit the earlier episode's clock."""
    states = keyed_state(5.0, auto_gc=False)
    key = ("cam-1", "person")

    # Brief presence t=0..2.
    for t in (0.0, 1.0, 2.0):
        states.touch(key, at=t)
    # Absent frames drive gc; state survives the grace window…
    assert states.gc(6.0) == []          # 6 - 2 = 4 < 5
    # …and resets once the gap exceeds it.
    assert [k for k, _ in states.gc(8.0)] == [key]  # 8 - 2 = 6 > 5

    # Fresh arrival at t=11 → age counts from 11, not 0.
    rec = states.touch(key, at=11.0)
    assert rec.first_seen == 11.0
    rec = states.touch(key, at=20.0)
    assert rec.age == 9.0
