"""Tolerant readers for raw cell values.

Phase A never rewrites values (type coercion is step 7). These helpers only
give steps a numeric view of a column so they can suggest roles.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--", "#n/a", "nil"}

_CURRENCY = re.compile(r"(₹|rs\.?|inr|\$|€|£)", re.IGNORECASE)


def is_null_token(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return isinstance(value, str) and value.strip().lower() in NULL_TOKENS


def to_number(value) -> float:
    """Parse '1,20,000', '₹ 4.5', '(120)' and '12%' style cells; NaN otherwise."""
    if is_null_token(value):
        return np.nan
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = _CURRENCY.sub("", str(value)).strip().replace(",", "").replace(" ", "")
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()").rstrip("%")
    try:
        x = float(s)
    except ValueError:
        return np.nan
    return -x if negative else x


def numeric_view(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    return series.map(to_number).astype(float)


def numeric_share(series: pd.Series) -> float:
    """Share of non-null cells that read as numbers."""
    present = series[~series.map(is_null_token)]
    if present.empty:
        return 0.0
    return float(numeric_view(present).notna().mean())
