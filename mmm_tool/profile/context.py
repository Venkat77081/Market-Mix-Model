"""State shared by Phase C steps (14-20).

Phase C decides whether the data can support an MMM at all and warns about
each channel before any fitting. Every finding also goes into the run-level
warnings, and channel findings become badges (hard to measure, rarely on,
moves with another channel, timed with a season) that later screens show
beside that channel's ROI.

Checks run on national weekly totals: spend is summed across regions,
controls are averaged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..ingest.pairing import Channel

# Roadmap thresholds, one place to change them.
MIN_WEEKS, ADEQUATE_WEEKS = 104, 156
MIN_OBS_PER_PARAM, COMFORT_OBS_PER_PARAM = 3, 6
RARE_ACTIVE_SHARE, RARE_SPEND_SHARE = 0.10, 0.01
LOW_SPEND_CV = 0.3
COLLINEAR_CORR, COLLINEAR_VIF = 0.7, 10
MEDIATOR_LAG_CORR, MEDIATOR_LAGS = 0.6, (1, 2)
SEASON_CORR = 0.6

# Parameters Meridian estimates for each part of the model (step 14).
PARAMS_PER_MEDIA = 3      # effect (ROI), carryover (adstock), saturation (half point)
PARAMS_PER_CONTROL = 1
WEEKS_PER_KNOT = 13       # baseline flexibility: about one knot per quarter (step 24 default)

BADGE_TEXT = {
    "hard_to_measure": "hard to measure",
    "rarely_on": "rarely on",
    "moves_with": "moves with another channel",
    "season_timed": "timed with a season",
    "calibrated": "calibrated",      # Phase D step 26: a lift test sets its prior
    "model_only": "model only",      # no experiment behind it
}


@dataclass
class CCtx:
    df: pd.DataFrame
    roles: dict[str, str]
    channels: list[Channel]
    kpi: dict[str, Any]
    granularity: dict[str, Any]
    badges: dict[str, list[str]] = field(default_factory=dict)
    date: str = ""
    geo: str | None = None
    kpi_col: str = ""

    def __post_init__(self):
        role = lambda r: next((c for c, v in self.roles.items() if v == r), None)  # noqa: E731
        self.date, self.kpi_col = role("date"), role("kpi")
        self.geo = role("geo") if self.granularity.get("geo_mode") == "regional" and role("geo") in self.df.columns else None

    def cols(self, *roles: str) -> list[str]:
        return [c for c in self.df.columns if self.roles.get(c) in roles]

    def weekly_sum(self, cols: list[str]) -> pd.DataFrame:
        return self.df.groupby(self.date)[cols].sum().sort_index()

    def weekly_mean(self, cols: list[str]) -> pd.DataFrame:
        return self.df.groupby(self.date)[cols].mean().sort_index()

    def spend(self) -> pd.DataFrame:
        """National weekly spend, one column per channel name."""
        w = self.weekly_sum([c.spend for c in self.channels])
        return w.rename(columns={c.spend: c.name for c in self.channels})

    def badge(self, channel: str, badge: str) -> None:
        self.badges.setdefault(channel, [])
        if badge not in self.badges[channel]:
            self.badges[channel].append(badge)

    def drop_channel(self, ch: Channel) -> None:
        for col in (ch.spend, ch.exposure, ch.reach, ch.frequency):
            if col and col in self.roles:
                self.roles[col] = "excluded"
        self.df = self.df.drop(columns=[c for c in (ch.spend, ch.exposure, ch.reach, ch.frequency) if c and c in self.df.columns])
        self.channels = [c for c in self.channels if c.name != ch.name]
        self.badges.pop(ch.name, None)

    def merge_channels(self, chans: list[Channel], name: str) -> Channel:
        """Replace channels with one whose spend is their sum. Exposures are in
        different units, so the merged channel uses spend as its exposure."""
        col = f"{name}_spend"
        i = 2
        while col in self.df.columns:
            col, i = f"{name}_spend_{i}", i + 1
        total = sum(self.df[c.spend] for c in chans)
        for c in chans:
            self.drop_channel(c)
        self.df[col] = total
        self.roles[col] = "media_spend"
        merged = Channel(name, "media", col, exposure=col, exposure_is_spend=True)
        self.channels.append(merged)
        return merged


def cv(x: pd.Series) -> float:
    m = float(x.mean())
    return float(x.std() / m) if m else float("inf")


def vif(frame: pd.DataFrame) -> dict[str, float]:
    """Variance inflation factor of each column against all the others."""
    out = {}
    X = frame.to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1, X.std(axis=0))
    for j, col in enumerate(frame.columns):
        y, others = X[:, j], np.delete(X, j, axis=1)
        if others.shape[1] == 0 or not y.any():
            out[col] = 1.0
            continue
        coef, *_ = np.linalg.lstsq(others, y, rcond=None)
        resid = y - others @ coef
        r2 = 1 - resid.var() / y.var() if y.var() else 0.0
        out[col] = float(1 / (1 - r2)) if r2 < 1 else float("inf")
    return out
