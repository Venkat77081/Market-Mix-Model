"""Tolerant readers for raw cell values.

Phase A never rewrites values (type coercion is step 7). These helpers only
give steps a numeric view of a column so they can suggest roles.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "--", "#n/a", "nil"}
# A lone dash is how accounting formats show zero ("₹ -"); it is a real zero, not missing.
ZERO_TOKENS = {"-", "–", "—"}

_CURRENCY = re.compile(r"(₹|rs\.?|inr|\$|€|£)", re.IGNORECASE)
_WITH_UNIT = re.compile(r"^([+-]?\d*\.?\d+)([a-z]+)\.?$")


def is_null_token(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return isinstance(value, str) and value.strip().lower() in NULL_TOKENS


def value_units() -> dict[str, float]:
    """Unit words allowed after a number in a cell ('4.5 Cr', '12.6 L'), from role_keywords.yaml."""
    from .keywords import load_keywords

    return load_keywords().value_units


def to_number(value, units: dict[str, float] | None = None) -> float:
    """Parse '1,20,000', '₹ 4.5', '(120)', '12%', '₹1.17 Cr', '12.6 L' and '-' (zero); NaN otherwise."""
    if is_null_token(value):
        return np.nan
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if str(value).strip() in ZERO_TOKENS:
        return 0.0
    s = _CURRENCY.sub("", str(value)).strip().replace(",", "").replace(" ", "")
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()").rstrip("%")
    try:
        x = float(s)
    except ValueError:
        m = _WITH_UNIT.match(s.lower())
        units = value_units() if units is None else units
        if not m or m[2] not in units:
            return np.nan
        x = float(m[1]) * units[m[2]]
    return -x if negative else x


def numeric_view(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    units = value_units()
    return series.map(lambda v: to_number(v, units)).astype(float)


def numeric_share(series: pd.Series) -> float:
    """Share of non-null cells that read as numbers."""
    present = series[~series.map(is_null_token)]
    if present.empty:
        return 0.0
    return float(numeric_view(present).notna().mean())
