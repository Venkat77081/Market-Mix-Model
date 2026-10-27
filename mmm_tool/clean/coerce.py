"""Step 7: coerce types.

Text nulls (N/A, NULL, -, blank) become missing; number formats such as
1,20,000, ₹4,500 and (250) become numbers; yes/no flags become 1/0. Nothing
for the user to decide. Cells that still can't be read are listed.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..ingest.types import StepResult, Verdict
from ..ingest.values import is_null_token, to_number
from .context import Ctx

STEP = 7
NAME = "Coerce types"
YES_NO = {"yes": 1.0, "y": 1.0, "true": 1.0, "no": 0.0, "n": 0.0, "false": 0.0}
CURRENCIES = [("₹", re.compile(r"₹|\brs\b\.?|\binr\b", re.I)), ("$", re.compile(r"\$|\busd\b", re.I)),
              ("€", re.compile(r"€|\beur\b", re.I)), ("£", re.compile(r"£|\bgbp\b", re.I))]


def currency_of(text: str) -> str | None:
    return next((sym for sym, pat in CURRENCIES if pat.search(text)), None)


def coerce_types(ctx: Ctx) -> StepResult:
    df = ctx.df
    report = []
    warnings = []
    for col in list(df.columns):
        role = ctx.roles.get(col)
        if role == "date":
            continue
        s = df[col]
        if role == "geo":
            df[col] = s.map(lambda v: None if is_null_token(v) else str(v).strip())
            continue
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            df[col] = s.astype(float)
            report.append({"column": col, "role": role, "text_nulls": 0, "formatted": 0, "yes_no": 0,
                           "blank": int(s.isna().sum()), "unreadable": 0, "examples": []})
            continue

        text = s.map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip())
        blank = text.isna() | (text == "")
        text_null = ~blank & text.map(lambda v: v is not None and is_null_token(v))
        low = text.str.lower()
        yes_no = ~blank & ~text_null & low.isin(YES_NO)
        numbers = s.map(to_number)
        numbers[yes_no] = low[yes_no].map(YES_NO)
        formatted = ~blank & ~text_null & ~yes_no & numbers.notna() & text.str.contains(r"[,₹$€£()%]|rs|inr", case=False, regex=True, na=False)
        unreadable = ~blank & ~text_null & numbers.isna()
        symbols = sorted({sym for v in text.dropna() if (sym := currency_of(v))})
        if symbols:
            ctx.currency_symbols[col] = symbols
        examples = sorted(set(text[unreadable]))[:5]
        if unreadable.any():
            warnings.append(f"'{col}': {int(unreadable.sum())} cell(s) aren't numbers and are treated as missing, e.g. {examples}.")
        df[col] = numbers.astype(float)
        report.append({
            "column": col, "role": role,
            "text_nulls": int(text_null.sum()), "formatted": int(formatted.sum()), "yes_no": int(yes_no.sum()),
            "blank": int(blank.sum()), "unreadable": int(unreadable.sum()), "examples": examples,
        })

    totals = {k: sum(r[k] for r in report) for k in ("text_nulls", "formatted", "yes_no", "unreadable")}
    parts = [f"{len(report)} columns are numbers now"]
    if totals["formatted"]:
        parts.append(f"{totals['formatted']:,} formatted numbers read (e.g. 1,20,000)")
    if totals["text_nulls"]:
        parts.append(f"{totals['text_nulls']:,} text nulls marked missing")
    if totals["yes_no"]:
        parts.append(f"{totals['yes_no']:,} yes/no values read as 1/0")
    if totals["unreadable"]:
        parts.append(f"{totals['unreadable']:,} unreadable cells marked missing")
    details = {"columns": report, "totals": totals, "currency_symbols": ctx.currency_symbols}
    return StepResult(STEP, NAME, Verdict.WARN if warnings else Verdict.PASS, "; ".join(parts) + ".", details, warnings)
