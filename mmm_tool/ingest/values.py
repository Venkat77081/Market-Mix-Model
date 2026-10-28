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

# Any currency sign (A$, R$, US$ included) or code, wherever it sits in the cell.
_CURRENCY = re.compile(r"(?:(?<![a-z])(?:us|[acsr])\$|[$€£¥₹₩₽₺₫₱₦₪฿]|(?<![a-z])(?:rs\.?|inr|usd|eur|gbp|jpy|aud|cad|sgd|aed|brl)(?![a-z]))",
                       re.IGNORECASE)
_GROUP_SPACE = re.compile(r"(?<=\d)[\s\u00a0\u202f'’](?=\d)")     # 1 200 / 1'200: a thousands space
_THOUSANDS = {",": re.compile(r"^\d{1,3}(,\d{3})+$|^\d{1,2}(,\d\d)+,\d{3}$"),   # 1,234,567 / 1,20,000
              ".": re.compile(r"^\d{1,3}(\.\d{3})+$")}
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


def _plain(value) -> str:
    """The cell's number text: currency, spaces between digits and signs taken off."""
    s = _CURRENCY.sub("", str(value)).strip()
    return _GROUP_SPACE.sub("", s).replace(" ", "")


def decimal_mark(values) -> str:
    """The column's decimal mark, ',' or '.', from its text cells: '12,5' or '1.234,56' mean ','.

    A value with both marks decides (the last is the decimal); otherwise a mark not followed
    by groups of three digits ('12,5', '0.75') is the decimal, and repeated dots ('1.234.567')
    are thousands. With no sign either way, '.'."""
    dot = comma = 0
    for v in values:
        if not isinstance(v, str):
            continue
        t = re.sub(r"[^\d.,]", "", _plain(v))
        if "," in t and "." in t:
            if t.rfind(",") > t.rfind("."):
                comma += 1
            else:
                dot += 1
        elif "," in t:
            comma += 0 if _THOUSANDS[","].match(t) else 1
        elif "." in t:
            if t.count(".") > 1:
                comma += 1
            elif not _THOUSANDS["."].match(t):
                dot += 1
    return "," if comma > dot else "."


def to_number(value, units: dict[str, float] | None = None, decimal: str = ".") -> float:
    """Parse '1,20,000', '₹ 4.5', '(120)', '12%', '₹1.17 Cr', '12.6 L' and '-' (zero); NaN otherwise.
    decimal is the column's decimal mark (decimal_mark): with ',', '1.234,56' is 1234.56."""
    if is_null_token(value):
        return np.nan
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if str(value).strip() in ZERO_TOKENS:
        return 0.0
    s = _plain(value)
    s = s.replace(".", "").replace(",", ".") if decimal == "," else s.replace(",", "")
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
    units, mark = value_units(), decimal_mark(series.dropna().head(500))
    return series.map(lambda v: to_number(v, units, mark)).astype(float)


def numeric_share(series: pd.Series) -> float:
    """Share of non-null cells that read as numbers."""
    present = series[~series.map(is_null_token)]
    if present.empty:
        return 0.0
    return float(numeric_view(present).notna().mean())
