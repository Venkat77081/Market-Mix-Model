"""State passed between Phase B steps, and the per-column aggregation rules.

Phase B turns Phase A's table into one complete row per week (per region in
a regional model), because Meridian refuses missing cells and carryover needs
an unbroken run of weeks. Two rules from the forecast tool carry over:
binary columns never default to sum, and a zero is never treated as missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..ingest.pairing import Channel

SETTINGS_FILE = Path(__file__).with_name("cleaning.yaml")


@lru_cache(maxsize=1)
def settings() -> dict[str, Any]:
    """Phase B thresholds from cleaning.yaml."""
    import yaml

    return yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))


RULES = ("sum", "mean", "max")
MEDIA_ROLES = {"media", "media_spend", "rf_spend", "reach", "organic_media"}
SPEND_ROLES = {"media_spend", "rf_spend"}


@dataclass
class Ctx:
    df: pd.DataFrame
    roles: dict[str, str]
    channels: list[Channel]
    kpi: dict[str, Any]
    granularity: dict[str, Any]
    date: str = ""
    geo: str | None = None  # None once regions are added up to national
    kpi_col: str = ""
    freq: str = "weekly"
    rules: dict[str, str] = field(default_factory=dict)
    weights: dict[str, str] = field(default_factory=dict)  # frequency column → impressions (or reach) column
    currency_symbols: dict[str, list[str]] = field(default_factory=dict)
    imputed_kpi: pd.Series | None = None  # True where step 10 filled the KPI

    def __post_init__(self):
        role = lambda r: next((c for c, v in self.roles.items() if v == r), None)  # noqa: E731
        self.date = role("date")
        self.kpi_col = role("kpi")
        self.geo = role("geo") if self.granularity.get("geo_mode") == "regional" else None

    @property
    def keys(self) -> list[str]:
        return [self.date] + ([self.geo] if self.geo else [])

    def cols(self, *roles: str) -> list[str]:
        return [c for c in self.df.columns if self.roles.get(c) in roles]

    @property
    def value_cols(self) -> list[str]:
        return [c for c in self.df.columns if c not in self.keys and self.roles.get(c) not in ("date", "geo", "excluded")]

    def drop_weight_columns(self) -> None:
        """Impressions kept only to weight frequency while rows are combined (steps 8-9)."""
        for c in [c for c in self.df.columns if self.roles.get(c) == "excluded"]:
            self.df = self.df.drop(columns=c)

    def add_control(self, name: str, values: pd.Series | np.ndarray) -> None:
        self.df[name] = values
        self.roles[name] = "control"

    def by_geo(self):
        """Yield (geo value or None, row index) per series, in date order."""
        df = self.df.sort_values(self.keys)
        if self.geo:
            for g, part in df.groupby(self.geo, sort=True):
                yield g, part.index
        else:
            yield None, df.index


def is_binary(s: pd.Series) -> bool:
    v = pd.unique(s.dropna())
    return len(v) > 0 and set(np.asarray(v, dtype=float)) <= {0.0, 1.0}


def aggregate(df: pd.DataFrame, keys: list[str], rules: dict[str, str],
              weights: dict[str, str] | None = None) -> pd.DataFrame:
    """Group by keys and combine each column by its rule. An all-missing group
    stays missing (a sum of nothing is not zero). "first" keeps a value that
    every merged row repeats. weights maps a per-unit column to the column it
    is weighted by (frequency → impressions): it is averaged weighted by that
    column, so the merged value is total ÷ total, not an average of ratios."""
    funcs = {
        "sum": lambda x: x.sum(min_count=1),
        "mean": "mean",
        "max": "max",
        "first": "first",
    }
    work = df
    weighted = {c: w for c, w in (weights or {}).items() if c in df.columns and w in df.columns and c in rules}
    if weighted:
        work = df.copy()
        for c, w in weighted.items():
            ok = work[c].notna() & work[w].notna()
            work[f"_num_{c}"] = (work[c] * work[w]).where(ok)
            work[f"_den_{c}"] = work[w].where(ok)
    agg = {c: funcs[rules[c]] for c in df.columns if c not in keys and c in rules}
    for c in weighted:
        agg[f"_num_{c}"] = agg[f"_den_{c}"] = funcs["sum"]
    out = work.groupby(keys, as_index=False, sort=True).agg(agg)
    for c in weighted:
        den = out.pop(f"_den_{c}")
        num = out.pop(f"_num_{c}")
        out[c] = (num / den.where(den > 0)).fillna(out[c])
    return out[[c for c in df.columns if c in out.columns]]


def cell_key(date: pd.Timestamp, geo: Any = None) -> str:
    d = pd.Timestamp(date).strftime("%Y-%m-%d")
    return f"{geo}|{d}" if geo is not None else d
