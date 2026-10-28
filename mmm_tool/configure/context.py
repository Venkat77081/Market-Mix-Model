"""State shared by Phase D steps (21-27).

Phase D builds Meridian's ModelSpec from sensible defaults and asks only for
what the user knows better than the data: business calendars and past
experiment results. Its output is `config`, a plain JSON description of the
model, which spec.py turns into Meridian objects.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..ingest.pairing import Channel

CALENDARS = Path(__file__).with_name("calendars.yaml")

# Roadmap defaults and limits.
DEFAULT_MAX_LAG, MAX_LAG_RANGE = 8, (2, 13)
WEEKS_PER_KNOT = 13
# Step 24: knots above weeks/8 bend the baseline more than every 2 months (amber);
# above weeks/6, likely to absorb campaign effects (red). The slider stops at weeks/4.
KNOTS_AMBER_WEEKS, KNOTS_RED_WEEKS, KNOTS_MAX_WEEKS = 8, 6, 4
DEFAULT_HOLDOUT, HOLDOUT_RANGE = 13, (8, 26)
MIN_TRAIN_WEEKS = 104
# Meridian's default ROI prior: LogNormal(0.2, 0.9), median about 1.2.
DEFAULT_ROI_MU, DEFAULT_ROI_SIGMA = 0.2, 0.9
Z90 = 1.6449  # a 90% range spans +/- 1.645 standard deviations


def user_calendars_file() -> Path:
    return Path(os.environ.get("MMM_CALENDARS", "calendars.yaml"))


def load_calendars() -> dict:
    """Built-in presets (none applied by default) with the project's own file on top
    (MMM_CALENDARS, default ./calendars.yaml): its presets are added, its
    default_preset is the one step 21 suggests, its direction words are added."""
    cal = yaml.safe_load(CALENDARS.read_text(encoding="utf-8")) or {}
    cal.setdefault("presets", {})
    cal.setdefault("default_preset", "none")
    user = user_calendars_file()
    if user.is_file() and user.resolve() != CALENDARS.resolve():
        mine = yaml.safe_load(user.read_text(encoding="utf-8")) or {}
        cal["presets"].update(mine.get("presets") or {})
        cal["default_preset"] = mine.get("default_preset", cal["default_preset"])
        for k in ("binomial_words",):
            cal[k] = list(cal.get(k) or []) + list(mine.get(k) or [])
        for way, words in (mine.get("control_direction_words") or {}).items():
            cal.setdefault("control_direction_words", {}).setdefault(way, []).extend(words or [])
    if cal["default_preset"] not in cal["presets"]:
        cal["default_preset"] = "none"
    return cal


def lognormal_from_range(low: float, high: float) -> tuple[float, float]:
    """LogNormal(mu, sigma) whose 90% interval is [low, high]."""
    mu = (math.log(low) + math.log(high)) / 2
    sigma = (math.log(high) - math.log(low)) / (2 * Z90)
    return mu, sigma


def lognormal_summary(mu: float, sigma: float) -> dict:
    return {"mu": round(mu, 4), "sigma": round(sigma, 4), "median": round(math.exp(mu), 3),
            "low": round(math.exp(mu - Z90 * sigma), 3), "high": round(math.exp(mu + Z90 * sigma), 3)}


@dataclass
class DCtx:
    df: pd.DataFrame
    roles: dict[str, str]
    channels: list[Channel]
    kpi: dict[str, Any]
    granularity: dict[str, Any]
    badges: dict[str, list[str]]
    config: dict[str, Any] = field(default_factory=dict)
    date: str = ""
    geo: str | None = None

    def __post_init__(self):
        role = lambda r: next((c for c, v in self.roles.items() if v == r), None)  # noqa: E731
        self.date = role("date")
        self.geo = role("geo") if self.granularity.get("geo_mode") == "regional" and role("geo") in self.df.columns else None

    @property
    def weeks(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(self.df[self.date].unique()))

    def badge(self, channel: str, badge: str) -> None:
        self.badges.setdefault(channel, [])
        if badge not in self.badges[channel]:
            self.badges[channel].append(badge)
