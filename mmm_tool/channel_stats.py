"""Per-channel spend statistics for the channels as they are now.

Later steps must not reuse an earlier step's per-channel numbers by name:
step 15 can drop a channel and step 17 can combine two (TV + YouTube) after
step 16 measured them. So step 24's burst channels and step 35's placebo
channel are computed here, from the current channels and the current model
table (the same table the preview regression and the fit see).

  cv     sample standard deviation of weekly spend ÷ its mean (step 16's rule:
         in a regional model, the median of each region's CV)
  share  the channel's total spend ÷ all paid spend
  active the share of weeks with any spend (step 15's rule: summed over regions)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class ChannelStatsError(ValueError):
    """A current channel whose spend isn't in the table."""


def _cv(x: pd.Series) -> float:
    m = float(x.mean())
    return float(x.std() / m) if m else float("inf")


def channel_stats(result) -> dict[str, dict[str, Any]]:
    """{channel: {cv, share, spend, sources}} for every current paid channel of a Phase C/D result."""
    cols = result.coord_to_columns()["coord_to_columns"]
    date, geo, t = cols["time"], cols.get("geo"), result.table
    missing = [c.name for c in result.channels if c.spend not in t.columns]
    if missing:
        raise ChannelStatsError(f"No spend column in the model table for channel(s) {missing}; "
                                "the channel list and the table are out of step.")
    totals = {c.name: float(t[c.spend].sum()) for c in result.channels}
    paid = sum(totals.values()) or 1.0
    out = {}
    for c in result.channels:
        if geo:
            cv = float(np.median([_cv(part) for _, part in t.groupby(geo)[c.spend]]))
        else:
            cv = _cv(t.groupby(date)[c.spend].sum())
        weekly = t.groupby(date)[c.spend].sum()
        out[c.name] = {"cv": cv, "share": totals[c.name] / paid, "spend": totals[c.name], "sources": c.sources(),
                       "active": float((weekly > 0).mean()) if len(weekly) else 0.0}
    return out
