"""State passed between Phase B steps, and the per-column aggregation rules.

Phase B turns Phase A's table into one complete row per week (per region in
a regional model), because Meridian refuses missing cells and carryover needs
an unbroken run of weeks. Two rules from the forecast tool carry over:
binary columns never default to sum, and a zero is never treated as missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..ingest.keywords import load_keywords, words_of
from ..ingest.pairing import Channel

RULES = ("sum", "mean", "max")
SUMMED_ROLES = {"kpi", "media", "media_spend", "rf_spend", "reach", "organic_media"}
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


def default_rule(col: str, role: str, s: pd.Series) -> tuple[str | None, str]:
    """(rule, reason); rule is None when the column can't be classified."""
    if role in SUMMED_ROLES:
        return "sum", "spend, exposure and KPI add up"
    if role == "frequency":
        return "mean", "frequency is an average per person"
    if role == "revenue_per_kpi":
        return "mean", "value per unit is averaged"
    if role == "population":
        return "max", "one value per region"
    if is_binary(s):
        return "max", "0/1 flag: takes the maximum, never the sum"
    kw, w = load_keywords(), words_of(col)
    if kw.agg_mean is not None and kw.agg_mean.search(w):
        return "mean", "index, price or rate by name"
    if kw.agg_sum is not None and kw.agg_sum.search(w):
        return "sum", "an amount or count by name"
    return None, "can't tell whether to add or average this column"


def aggregate(df: pd.DataFrame, keys: list[str], rules: dict[str, str]) -> pd.DataFrame:
    """Group by keys and combine each column by its rule. An all-missing group
    stays missing (a sum of nothing is not zero)."""
    funcs = {
        "sum": lambda x: x.sum(min_count=1),
        "mean": "mean",
        "max": "max",
    }
    agg = {c: funcs[rules[c]] for c in df.columns if c not in keys and c in rules}
    out = df.groupby(keys, as_index=False, sort=True).agg(agg)
    return out[[c for c in df.columns if c in out.columns]]


def cell_key(date: pd.Timestamp, geo: Any = None) -> str:
    d = pd.Timestamp(date).strftime("%Y-%m-%d")
    return f"{geo}|{d}" if geo is not None else d
