# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later
"""When a run of fall scores becomes ONE alarm.

The adapter scores every frame: ``fall.score`` is the probability that
the person it watches fell, over the last 30 frames. A fall stays above
the threshold for a second or more, so a frame-by-frame rule would raise
a dozen alarms for one fall. This is the rule that turns that run into
one alarm, per camera, and nothing else: no platform, no camera, no
model, so it is tested on a synthetic clock.

The defaults are the reference Space's: alarm on the first frame at or
above 0.90, then nothing for 10 seconds (its FALL_EVENT_COOLDOWN, which
it counted in frames; here it is seconds, so the rule means the same
thing at 10 fps as at 30).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FallPolicy:
    #: The score a frame needs. The reference's FALL_CONFIDENCE_THRESHOLD.
    threshold: float = 0.90
    #: Consecutive hot frames before the alarm. 1 is the reference;
    #: raise it on a camera that alarms on a single noisy frame.
    min_hits: int = 1
    #: Seconds after an alarm before the same camera can raise another.
    cooldown_s: float = 10.0
    #: How long the person stays labelled FALLEN on the live view after
    #: an alarm — long enough for whoever is watching to find them.
    hold_s: float = 5.0

    @classmethod
    def from_config(cls, cfg: dict) -> "FallPolicy":
        def num(key, default, lo, hi, cast=float):
            try:
                v = cast(cfg.get(key, default))
            except (TypeError, ValueError):
                return default
            return min(hi, max(lo, v))
        return cls(
            threshold=num("threshold", cls.threshold, 0.0, 1.0),
            min_hits=num("min_hits", cls.min_hits, 1, 100, int),
            cooldown_s=num("cooldown_s", cls.cooldown_s, 0.0, 3600.0),
            hold_s=num("hold_s", cls.hold_s, 0.0, 600.0),
        )


@dataclass
class FallAlarm:
    """One fall, the moment it is ruled one."""

    at: float               # wall clock of the frame that decided it
    score: float            # that frame's score
    peak: float             # highest score in the run so far
    hits: int               # consecutive hot frames it took


@dataclass
class FallMonitor:
    """One camera's run of scores."""

    policy: FallPolicy = field(default_factory=FallPolicy)
    hits: int = 0
    peak: float = 0.0
    last_alarm_at: float | None = None
    last_score: float | None = None

    def observe(self, score: float | None, now: float) -> FallAlarm | None:
        """Feed one frame's score; returns the alarm when this frame
        decides one. ``None`` (the adapter's window not full yet) is
        not evidence either way: it neither builds nor breaks a run."""
        self.last_score = score
        if score is None:
            return None
        if score < self.policy.threshold:
            self.hits = 0
            self.peak = 0.0
            return None
        self.hits += 1
        self.peak = max(self.peak, score)
        if self.hits < self.policy.min_hits:
            return None
        if (self.last_alarm_at is not None
                and now - self.last_alarm_at < self.policy.cooldown_s):
            return None
        self.last_alarm_at = now
        return FallAlarm(at=now, score=score, peak=self.peak, hits=self.hits)

    def fallen(self, now: float) -> bool:
        """Should the live view still say FALLEN?"""
        return (self.last_alarm_at is not None
                and now - self.last_alarm_at <= self.policy.hold_s)

    def reset(self) -> None:
        """The feed broke: the adapter starts a new window, so a run that
        straddled the gap is not one run."""
        self.hits = 0
        self.peak = 0.0
        self.last_score = None
