"""Step 2: parse dates.

The format is detected per row, so a column that mixes dd/mm/yyyy and ISO
dates parses correctly. A slash date such as 03/04/2024 is ambiguous on its
own; it is settled by the other rows in the column (any row with a day above
12 fixes the order). Only when every slash date is ambiguous does the user
choose, with a suggestion from which reading gives the more regular spacing.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from dateutil import parser as du_parser

from .types import Answers, Decision, StepResult, Verdict
from .values import is_null_token, to_number

STEP = 2
NAME = "Parse dates"

ISO = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[ T].*)?$")
SLASH = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2}|\d{4})(?:[ T].*)?$")
HAS_MONTH_NAME = re.compile(r"[A-Za-z]{3,}")
DATE_NAME = re.compile(r"date|week|time|period|day|month", re.IGNORECASE)
EXCEL_EPOCH = datetime(1899, 12, 30)


def _year(y: str) -> int:
    n = int(y)
    return n + 2000 if len(y) == 2 and n < 70 else n + 1900 if len(y) == 2 else n


def _safe(y: int, m: int, d: int):
    try:
        return pd.Timestamp(date(y, m, d))
    except ValueError:
        return None


def classify(value) -> tuple[str, object]:
    """Return (kind, parsed). kind is one of: datetime, excel_serial, iso,
    dayfirst, monthfirst, same_either_way, ambiguous, text_month, unparseable. For ambiguous
    rows `parsed` is (dayfirst_reading, monthfirst_reading)."""
    if is_null_token(value):
        return "unparseable", None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return "datetime", pd.Timestamp(value).normalize()
    if isinstance(value, (int, float, np.integer, np.floating)):
        if 20000 <= value <= 80000:
            return "excel_serial", pd.Timestamp(EXCEL_EPOCH + timedelta(days=int(value)))
        return "unparseable", None
    s = str(value).strip()
    m = ISO.match(s)
    if m:
        ts = _safe(int(m[1]), int(m[2]), int(m[3]))
        return ("iso", ts) if ts is not None else ("unparseable", None)
    m = SLASH.match(s)
    if m:
        a, b, y = int(m[1]), int(m[2]), _year(m[3])
        day_first, month_first = _safe(y, b, a), _safe(y, a, b)
        if day_first is not None and month_first is not None:
            if day_first == month_first:
                return "same_either_way", day_first
            return "ambiguous", (day_first, month_first)
        if day_first is not None:
            return "dayfirst", day_first
        if month_first is not None:
            return "monthfirst", month_first
        return "unparseable", None
    if HAS_MONTH_NAME.search(s):
        try:
            return "text_month", pd.Timestamp(du_parser.parse(s)).normalize()
        except (ValueError, OverflowError):
            return "unparseable", None
    n = to_number(s)
    if not np.isnan(n) and 20000 <= n <= 80000:
        return "excel_serial", pd.Timestamp(EXCEL_EPOCH + timedelta(days=int(n)))
    return "unparseable", None


def parse_rate(series: pd.Series) -> float:
    sample = series.head(200)
    if sample.empty:
        return 0.0
    return float(np.mean([classify(v)[0] != "unparseable" for v in sample]))


def find_date_column(df: pd.DataFrame) -> str | None:
    """The column whose values parse as dates, preferring date-like names."""
    scored = []
    for col in df.columns:
        rate = parse_rate(df[col])
        if rate >= 0.8:
            scored.append((bool(DATE_NAME.search(col)), rate, -list(df.columns).index(col), col))
    return max(scored)[3] if scored else None


def regularity(dates: list[pd.Timestamp]) -> float:
    """Share of gaps between consecutive distinct dates that sit within
    three days of the median gap. Weekly data read the right way scores 1."""
    uniq = sorted(set(dates))
    if len(uniq) < 3:
        return 0.0
    gaps = np.diff([d.value for d in uniq]) / 86_400e9
    med = np.median(gaps)
    return float(np.mean(np.abs(gaps - med) <= 3))


def parse_dates(df: pd.DataFrame, answers: Answers, date_col: str | None = None) -> tuple[pd.DataFrame | None, str | None, StepResult]:
    date_col = answers.values.get("date_column", date_col) or find_date_column(df)
    if date_col is None or date_col not in df.columns:
        return None, None, StepResult(
            STEP, NAME, Verdict.FAIL,
            "No column parses as dates. Check the file has a date or week column.",
            {"columns": list(df.columns)},
        )

    kinds, parsed = zip(*(classify(v) for v in df[date_col])) if len(df) else ((), ())
    kinds, parsed = list(kinds), list(parsed)
    counts = {k: kinds.count(k) for k in sorted(set(kinds))}
    details: dict = {"date_column": date_col, "formats": counts}
    decisions: list[Decision] = []
    warnings: list[str] = []
    notes: list[str] = []

    if counts.get("unparseable", 0) > 0.5 * len(df):
        return None, date_col, StepResult(
            STEP, NAME, Verdict.FAIL,
            f"Only {len(df) - counts['unparseable']} of {len(df)} values in '{date_col}' read as dates.",
            details,
        )

    # Settle ambiguous slash dates.
    n_amb = counts.get("ambiguous", 0)
    if n_amb:
        n_day, n_month = counts.get("dayfirst", 0), counts.get("monthfirst", 0)
        if n_day and not n_month:
            order, why = "dayfirst", f"{n_day} other row(s) have a day above 12"
        elif n_month and not n_day:
            order, why = "monthfirst", f"{n_month} other row(s) have a day above 12 in second place"
        else:
            fixed = [p for k, p in zip(kinds, parsed) if k not in ("ambiguous", "unparseable")]
            amb = [p for k, p in zip(kinds, parsed) if k == "ambiguous"]
            score_day = regularity(fixed + [a[0] for a in amb])
            score_month = regularity(fixed + [a[1] for a in amb])
            suggestion = "monthfirst" if score_month > score_day else "dayfirst"
            d = Decision(
                id="date_order",
                step=STEP,
                question=(
                    f"{n_amb} dates in '{date_col}' could be read either way (for example "
                    f"'{df[date_col].iloc[kinds.index('ambiguous')]}'). Are they day-first or month-first?"
                ),
                options=["dayfirst", "monthfirst"],
                default=suggestion,
                blocking=True,
                context={"regularity": {"dayfirst": round(score_day, 3), "monthfirst": round(score_month, 3)}},
            )
            decisions.append(d)
            order, why = answers.ask(d), "answered at the prompt"
        if order is not None:
            idx = 0 if order == "dayfirst" else 1
            for i, k in enumerate(kinds):
                if k == "ambiguous":
                    parsed[i] = parsed[i][idx]
            details["ambiguous_resolved_as"] = order
            notes.append(f"{n_amb} ambiguous date(s) read {order.replace('first', '-first')} ({why})")

    if counts.get("dayfirst") and counts.get("monthfirst"):
        warnings.append(
            f"'{date_col}' has both day-first ({counts['dayfirst']}) and month-first "
            f"({counts['monthfirst']}) dates. Each was read in the only way that is valid; check the source export."
        )

    # Rows whose date can't be read.
    bad = [i for i, k in enumerate(kinds) if k == "unparseable"]
    drop = False
    if bad:
        examples = [str(df[date_col].iloc[i]) for i in bad[:5]]
        details["unparseable_rows"] = [int(i) for i in bad]
        d = Decision(
            id="unparseable_dates",
            step=STEP,
            question=f"{len(bad)} row(s) have a date that can't be read (e.g. {examples}). Drop those rows?",
            options=["drop_rows"],
            default=None,
            blocking=True,
            context={"examples": examples},
        )
        decisions.append(d)
        drop = answers.ask(d) == "drop_rows"
        if drop:
            warnings.append(f"Dropped {len(bad)} row(s) with unreadable dates: {examples}.")

    if any(d.pending for d in decisions):
        pending = [d.id for d in decisions if d.pending]
        return None, date_col, StepResult(
            STEP, NAME, Verdict.NEEDS_INPUT,
            f"Dates in '{date_col}' need a decision before the run can continue ({', '.join(pending)}).",
            details, warnings, decisions,
        )

    out = df.copy()
    out[date_col] = pd.to_datetime(pd.Series(parsed, index=df.index, dtype=object))
    if drop:
        out = out.drop(index=out.index[bad]).reset_index(drop=True)
    details.update(first=str(out[date_col].min().date()), last=str(out[date_col].max().date()))
    summary = f"'{date_col}' parsed: {len(out):,} rows from {details['first']} to {details['last']}"
    if len([k for k in counts if k != "unparseable"]) > 1:
        summary += " (mixed formats: " + ", ".join(f"{v} {k}" for k, v in counts.items() if k != "unparseable") + ")"
    if notes:
        summary += "; " + "; ".join(notes)
    verdict = Verdict.WARN if warnings else Verdict.PASS
    return out, date_col, StepResult(STEP, NAME, verdict, summary + ".", details, warnings, decisions)
