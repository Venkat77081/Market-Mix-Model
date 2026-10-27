"""Step 2: parse dates.

The format is detected per row, so a column that mixes dd/mm/yyyy and ISO
dates parses correctly. A slash date such as 03/04/2024 is ambiguous on its
own; it is settled by the other rows in the column (any row with a day above
12 fixes the order). Only when every slash date is ambiguous does the user
choose, with a suggestion from which reading gives the more regular spacing.
"""
from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from dateutil import parser as du_parser

from .types import Answers, Decision, StepResult, Verdict
from .values import is_null_token, numeric_share, to_number

STEP = 2
NAME = "Parse dates"

ISO = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[ T].*)?$")
SLASH = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2}|\d{4})(?:[ T].*)?$")
# Month-only dates read as the first of the month. A 4-digit year makes
# MM/YYYY unambiguous, so it never goes through the day-first logic.
YEAR_MONTH = re.compile(r"^(\d{4})[-/.](\d{1,2})$")
MONTH_YEAR = re.compile(r"^(\d{1,2})[-/.](\d{4})$")
MONTH_NAME_YEAR = re.compile(r"^([A-Za-z]{3,9})\.?[\s\-/,']*(\d{4})$")
MONTHS = {name.lower(): i for i in range(1, 13) for name in (calendar.month_name[i], calendar.month_abbr[i])} | {"sept": 9}
ISO_WEEK = re.compile(r"^(\d{4})[-/ ]?W(\d{1,2})(?:[-/ ]?([1-7]))?$", re.IGNORECASE)
HAS_MONTH_NAME = re.compile(r"[A-Za-z]{3,}")
# Readings that are right when a whole column uses them, but suspicious when
# a few rows of an ordinary date column do: they are offered, not applied.
OFFERED = {"excel_serial": "an Excel serial date", "iso_week": "an ISO week"}
NO_DAY = datetime(1900, 1, 1)  # dateutil fills missing parts from this, never from today
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
    """Return (kind, parsed). kind is one of: datetime, excel_serial, iso, month_only, iso_week,
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
    for pattern, ym in ((YEAR_MONTH, (1, 2)), (MONTH_YEAR, (2, 1))):
        m = pattern.match(s)
        if m:
            ts = _safe(int(m[ym[0]]), int(m[ym[1]]), 1)
            return ("month_only", ts) if ts is not None else ("unparseable", None)
    m = ISO_WEEK.match(s)
    if m:
        try:
            return "iso_week", pd.Timestamp(date.fromisocalendar(int(m[1]), int(m[2]), int(m[3] or 1)))
        except ValueError:
            return "unparseable", None
    m = MONTH_NAME_YEAR.match(s)
    if m and m[1].lower() in MONTHS:
        return "month_only", _safe(int(m[2]), MONTHS[m[1].lower()], 1)
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
            parsed = du_parser.parse(s, default=NO_DAY)
            if parsed.year == NO_DAY.year and str(NO_DAY.year) not in s:
                return "unparseable", None  # no year in the text
            return "text_month", pd.Timestamp(parsed).normalize()
        except (ValueError, OverflowError):
            return "unparseable", None
    n = to_number(s)
    if not np.isnan(n) and 20000 <= n <= 80000:
        return "excel_serial", pd.Timestamp(EXCEL_EPOCH + timedelta(days=int(n)))
    return "unparseable", None


def explaining_region(df: pd.DataFrame, date_col: str) -> str | None:
    """A text column with a few repeating values that makes every (date, value)
    pair unique, as in long-format regional data."""
    for col in df.columns:
        if col == date_col or numeric_share(df[col]) >= 0.8:
            continue
        n = df[col].nunique()
        if 2 <= n <= 60 and not df.duplicated([date_col, col]).any():
            return col
    return None


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


def parse_dates(df: pd.DataFrame, answers: Answers, date_col: str | None = None,
                lines: list[int] | None = None) -> tuple[pd.DataFrame | None, str | None, StepResult]:
    """lines: the file line of each row (from step 1), for messages."""
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

    # Rows whose date can't be read, each listed with its file line. A few
    # Excel serials or ISO weeks in an ordinary date column are offered as fixes.
    line = (lambda i: lines[i]) if lines and len(lines) == len(df) else (lambda i: i + 2)
    readable = sum(1 for k in kinds if k != "unparseable")
    offered = [i for i, k in enumerate(kinds) if k in OFFERED and counts[k] < 0.5 * readable]
    raw = lambda i: "" if is_null_token(df[date_col].iloc[i]) and str(df[date_col].iloc[i]).strip() == "" else str(df[date_col].iloc[i])  # noqa: E731
    unreadable = [{"row": i, "line": line(i), "value": raw(i),
                   "suggestion": (f"{OFFERED[kinds[i]]}: {parsed[i].date()}" if i in offered else None)}
                  for i, k in enumerate(kinds) if k == "unparseable" or i in offered]
    details["unreadable"] = unreadable
    taken = {parsed[i] for i, k in enumerate(kinds) if k != "unparseable" and i not in offered}
    for u in unreadable:
        if u["suggestion"] and parsed[u["row"]] in taken:
            u["suggestion"] += " (that week is already in the file)"
    if offered:
        d = Decision(
            id="date_fixes",
            step=STEP,
            question="These can be read another way: " + "; ".join(
                f"'{u['value']}' (line {u['line']}) as {u['suggestion']}" for u in unreadable if u["suggestion"]) + ". Use these readings?",
            options=["apply", "no"],
            default="apply",
            blocking=True,
            context={"rows": [u for u in unreadable if u["suggestion"]]},
        )
        decisions.append(d)
        if answers.ask(d) == "apply":
            notes.append(f"{len(offered)} date(s) read as suggested ({', '.join(sorted({kinds[i].replace('_', ' ') for i in offered}))})")
        for i in offered:
            if d.answer != "apply":
                kinds[i], parsed[i] = "unparseable", None
    bad = [i for i, k in enumerate(kinds) if k == "unparseable"]
    drop = False
    if bad:
        listed = [f"line {line(i)}: '{raw(i)}'" for i in bad]
        details["unparseable_rows"] = [int(i) for i in bad]
        d = Decision(
            id="unparseable_dates",
            step=STEP,
            question=(f"{len(bad)} row(s) have a date that can't be read: {', '.join(listed[:20])}"
                      f"{'…' if len(listed) > 20 else ''}. Fix them in the file, or drop those rows?"),
            options=["drop_rows"],
            default=None,
            blocking=True,
            context={"rows": [u for u in unreadable if u["row"] in bad]},
        )
        decisions.append(d)
        drop = answers.ask(d) == "drop_rows"
        if drop:
            warnings.append(f"Dropped {len(bad)} row(s) with unreadable dates: {', '.join(listed[:10])}.")

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

    # Every row should have its own date unless a region column explains the repeats.
    n_rows, n_unique = len(out), int(out[date_col].nunique())
    details.update(rows=n_rows, unique_dates=n_unique)
    counted = f"{n_rows:,} rows, {n_unique:,} unique dates"
    if n_unique < n_rows:
        region = explaining_region(out, date_col)
        if region:
            details["repeats_explained_by"] = region
            counted += f" ({n_unique:,} dates × {out[region].nunique()} values of '{region}')"
        else:
            repeated = out[date_col][out[date_col].duplicated(keep=False)]
            examples = sorted({str(x.date()) for x in repeated})[:5]
            d = Decision(
                id="duplicate_dates",
                step=STEP,
                question=(f"{n_rows - n_unique} row(s) repeat a date already in the file (e.g. {examples}) and no region "
                          "column explains it. If the export has duplicate rows, Phase B step 8 merges them. "
                          "If not, a date is being read wrongly. Merge them in Phase B?"),
                options=["merge"],
                default=None,
                blocking=True,
                context={"examples": examples, "rows": n_rows, "unique_dates": n_unique},
            )
            decisions.append(d)
            if answers.ask(d) == "merge":
                warnings.append(f"{n_rows - n_unique} repeated date(s), e.g. {examples}; Phase B step 8 merges them.")
            else:
                return None, date_col, StepResult(
                    STEP, NAME, Verdict.NEEDS_INPUT,
                    f"Stopped: {counted}. Repeated dates need a decision.", details, warnings, decisions,
                )
    summary = f"'{date_col}' parsed: {counted}, from {details['first']} to {details['last']}"
    if len([k for k in counts if k != "unparseable"]) > 1:
        summary += " (mixed formats: " + ", ".join(f"{v} {k}" for k, v in counts.items() if k != "unparseable") + ")"
    if notes:
        summary += "; " + "; ".join(notes)
    verdict = Verdict.WARN if warnings else Verdict.PASS
    return out, date_col, StepResult(STEP, NAME, verdict, summary + ".", details, warnings, decisions)
