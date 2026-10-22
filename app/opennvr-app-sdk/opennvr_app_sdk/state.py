# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
Keyed TTL state — the "presence bookkeeping" every stateful app rewrote.

Two archetypal consumers drove the design (App SDK spec §04):

* **loitering-detection** — per-``(camera_id, label)`` dwell: first
  presence starts the clock, continuous presence refreshes it, the
  threshold-crossing fires ONCE (the ``alerted`` latch), and absence
  beyond a grace period resets the episode.

* **package-delivery** — per-``track_id`` lifecycle: a track *arrives*
  (first touch), *lingers* (age grows across touches, latch fires the
  one-shot alert), and *disappears* (no touches for > TTL → GC'd; the
  pruned records let the app emit "package gone" events).

The core object is a dict of :class:`StateRecord` keyed by anything
hashable, with TTL-based garbage collection of keys that have not been
touched recently.

Semantics that matter (and are tested):

* ``touch(key, at)`` NEVER prunes the key being touched, even when its
  ``last_seen`` is stale — a touch means "present now", so a sparse
  event stream (0.1 fps with a 5 s TTL) keeps one continuous episode
  rather than restarting it. Staleness only ever removes keys that
  are NOT being refreshed.
* ``touch`` with ``auto_gc=True`` (the default) prunes *other* stale
  keys as a side effect, so a naive app never leaks state. Apps that
  need finer-grained reset rules (loitering only resets keys for the
  camera the current event belongs to) construct with
  ``auto_gc=False`` and drive :meth:`KeyedState.gc` / :meth:`pop`
  themselves.
* ``gc(now)`` returns the pruned ``(key, record)`` pairs so lifecycle
  apps can react to disappearance (package-delivery's "gone" event).
* Out-of-order timestamps are the CALLER's problem: ``touch`` writes
  ``at`` into ``last_seen`` verbatim. Guard with ``get(key)`` first if
  your event bus can reorder (see loitering's out-of-order skip).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterator


@dataclass
class StateRecord:
    """Bookkeeping for one key.

    ``first_seen`` is the timestamp of the first touch since the record
    was created (or re-created after a GC) — loitering reads it as
    ``present_since``. ``last_seen`` is the most recent touch.
    ``alerted`` is a caller-settable latch so a threshold-crossing
    alert fires once per episode. ``data`` is a free-form scratchpad
    for app-specific flags (state-machine phase, counters, …);
    subclassing and passing ``record_factory`` works too when you want
    typed fields.
    """

    first_seen: float
    last_seen: float
    alerted: bool = False
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def age(self) -> float:
        """Seconds between the first and the most recent touch —
        i.e. the dwell time as of the last ``touch``."""
        return self.last_seen - self.first_seen


class KeyedState:
    """A TTL-pruned mapping of hashable keys to :class:`StateRecord`.

    Build via :func:`keyed_state`. Dict-like surface: ``get`` / ``pop``
    / ``items`` / ``in`` / ``len`` / ``[key]``.
    """

    def __init__(
        self,
        ttl: float,
        *,
        auto_gc: bool = True,
        record_factory: Callable[..., StateRecord] = StateRecord,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl <= 0:
            raise ValueError(f"keyed_state: ttl must be > 0, got {ttl!r}")
        self.ttl = float(ttl)
        self._auto_gc = auto_gc
        self._factory = record_factory
        self._clock = clock
        self._records: dict[Hashable, StateRecord] = {}

    # ── Core ────────────────────────────────────────────────────────

    def touch(self, key: Hashable, at: float | None = None) -> StateRecord:
        """Record a presence ping for ``key`` at time ``at`` (defaults
        to wall clock). Creates the record on first touch (fresh
        ``first_seen``, ``alerted=False``); refreshes ``last_seen`` on
        subsequent touches. With ``auto_gc`` enabled, prunes OTHER
        stale keys first — never the touched key itself."""
        now = self._clock() if at is None else float(at)
        if self._auto_gc:
            self.gc(now, exclude=(key,))
        record = self._records.get(key)
        if record is None:
            record = self._factory(first_seen=now, last_seen=now)
            self._records[key] = record
        else:
            record.last_seen = now
        return record

    def gc(
        self,
        now: float | None = None,
        *,
        exclude: Any = (),
    ) -> list[tuple[Hashable, StateRecord]]:
        """Prune every key whose ``last_seen`` is more than ``ttl``
        seconds before ``now`` (strictly older — a record exactly at
        the TTL boundary survives, matching the grace-period semantics
        the loitering detector shipped with). Keys in ``exclude`` are
        kept regardless. Returns the pruned ``(key, record)`` pairs so
        lifecycle apps can emit disappearance events."""
        now = self._clock() if now is None else float(now)
        cutoff = now - self.ttl
        excluded = set(exclude)
        pruned = [
            (key, record)
            for key, record in self._records.items()
            if key not in excluded and record.last_seen < cutoff
        ]
        for key, _record in pruned:
            del self._records[key]
        return pruned

    # ── Dict-like surface ───────────────────────────────────────────

    def get(self, key: Hashable, default: Any = None) -> StateRecord | Any:
        return self._records.get(key, default)

    def pop(self, key: Hashable, default: Any = None) -> StateRecord | Any:
        return self._records.pop(key, default)

    def items(self) -> list[tuple[Hashable, StateRecord]]:
        """Snapshot list — safe to ``pop`` while iterating."""
        return list(self._records.items())

    def clear(self) -> None:
        self._records.clear()

    def __getitem__(self, key: Hashable) -> StateRecord:
        return self._records[key]

    def __contains__(self, key: Hashable) -> bool:
        return key in self._records

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._records)


def keyed_state(
    ttl: float,
    *,
    auto_gc: bool = True,
    record_factory: Callable[..., StateRecord] = StateRecord,
    clock: Callable[[], float] = time.time,
) -> KeyedState:
    """Build a :class:`KeyedState` — TTL + latch + GC per §04 of the
    App SDK spec. ``ttl`` is in seconds of *event time* (whatever
    timeline you pass to ``touch(at=...)``)."""
    return KeyedState(ttl, auto_gc=auto_gc, record_factory=record_factory, clock=clock)
