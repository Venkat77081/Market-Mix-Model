"""Step 10: label every gap.

Weeks absent from the export are added, then every empty cell is labelled:
not started (before a column's first value), ended (after its last value),
or missing (a hole in between). A zero is a real zero and is never touched.

  KPI gaps        the user chooses: interpolate and add a kpi_imputed flag,
                  or trim to the longest stretch with no KPI gaps. Blocks.
  Media           not started or ended is no spend (0); holes are 0 by
                  default, or interpolated.
  Everything else interpolated between known values, carried to the edges;
                  0/1 flags are 0 where missing.
"""
from __future__ import annotations

import pandas as pd

from ..ingest.types import Answers, Decision, StepResult, Verdict
from .context import MEDIA_ROLES, Ctx, is_binary

STEP = 10
NAME = "Label every gap"


def _calendar(dates: pd.Series, freq: str) -> pd.DatetimeIndex:
    lo, hi = dates.min(), dates.max()
    if freq == "weekly":
        return pd.date_range(lo, hi, freq="7D")
    if freq == "monthly" and (dates.dt.day == 1).all():
        return pd.date_range(lo, hi, freq="MS")
    return pd.DatetimeIndex(sorted(dates.unique()))


def _edges(s: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Masks for (not started, ended, missing in between) on one series."""
    valid = s.notna().to_numpy()
    n = len(s)
    if not valid.any():
        none = pd.Series(False, index=s.index)
        return pd.Series(True, index=s.index), none, none.copy()
    first, last = valid.argmax(), n - 1 - valid[::-1].argmax()
    pos = pd.Series(range(n), index=s.index)
    na = s.isna()
    return na & (pos < first), na & (pos > last), na & (pos > first) & (pos < last)


def _longest_run(ok: pd.Series) -> pd.Series:
    """Mask of the longest run of consecutive True values."""
    best, start, best_span = 0, None, (0, -1)
    vals = ok.to_numpy()
    for i, v in enumerate(list(vals) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best:
                best, best_span = i - start, (start, i - 1)
            start = None
    keep = pd.Series(False, index=ok.index)
    keep.iloc[best_span[0]: best_span[1] + 1] = True
    return keep


def label_gaps(ctx: Ctx, answers: Answers) -> StepResult:
    df = ctx.df.sort_values(ctx.keys).reset_index(drop=True)
    cal = _calendar(df[ctx.date], ctx.freq)
    if ctx.geo:
        full = pd.MultiIndex.from_product([cal, sorted(df[ctx.geo].unique())], names=ctx.keys).to_frame(index=False)
    else:
        full = pd.DataFrame({ctx.date: cal})
    before = len(df)
    df = full.merge(df, on=ctx.keys, how="left", indicator=True)
    inserted = df["_merge"] == "left_only"
    missing_weeks = sorted({str(d.date()) for d in df.loc[inserted, ctx.date]})
    df = df.drop(columns="_merge").sort_values(ctx.keys).reset_index(drop=True)
    inserted = inserted.reset_index(drop=True)
    ctx.df = df

    groups = list(ctx.by_geo())
    value_cols = ctx.value_cols
    labels = {}
    for col in value_cols:
        ns, en, mi = (pd.Series(False, index=df.index) for _ in range(3))
        for _, idx in groups:
            a, b, c = _edges(df.loc[idx, col])
            ns[idx], en[idx], mi[idx] = a, b, c
        labels[col] = (ns, en, mi)

    decisions: list[Decision] = []
    warnings: list[str] = []
    notes: list[str] = []

    # KPI gaps first: trimming changes which weeks the other columns cover.
    kpi = ctx.kpi_col
    ns, en, mi = labels[kpi]
    kpi_gaps = int((ns | en | mi).sum())
    trimmed: list[str] = []
    if kpi_gaps:
        default = "interpolate" if mi.any() else "trim"
        d = Decision(
            id="kpi_gaps",
            step=STEP,
            question=(f"'{kpi}' is empty in {kpi_gaps} week(s) ({int(mi.sum())} in the middle, "
                      f"{int((ns | en).sum())} at the start or end). Interpolate them and add a flag, or trim the period?"),
            options=["interpolate", "trim"],
            default=default,
            blocking=True,
            context={"weeks": sorted({str(x.date()) for x in df.loc[ns | en | mi, ctx.date]})[:50]},
        )
        decisions.append(d)
        choice = answers.ask(d)
        if choice is None:
            return StepResult(STEP, NAME, Verdict.NEEDS_INPUT, f"'{kpi}' has {kpi_gaps} empty week(s); choose how to handle them.",
                              {"missing_weeks": missing_weeks}, warnings, decisions)
        if choice == "trim":
            present = df.groupby(ctx.date)[kpi].apply(lambda x: x.notna().all()).sort_index()
            keep_dates = present.index[_longest_run(present)]
            trimmed = sorted(str(x.date()) for x in present.index.difference(keep_dates))
            keep = df[ctx.date].isin(keep_dates)
            df = df[keep].reset_index(drop=True)
            inserted = inserted[keep.to_numpy()].reset_index(drop=True)
            ctx.df = df
            groups = list(ctx.by_geo())
            for col in value_cols:  # relabel on the trimmed frame
                ns, en, mi = (pd.Series(False, index=df.index) for _ in range(3))
                for _, idx in groups:
                    a, b, c = _edges(df.loc[idx, col])
                    ns[idx], en[idx], mi[idx] = a, b, c
                labels[col] = (ns, en, mi)
            notes.append(f"trimmed {len(trimmed)} week(s); {df[ctx.date].min().date()} to {df[ctx.date].max().date()} kept")
        elif choice == "interpolate":
            was_na = df[kpi].isna()
            for _, idx in groups:
                df.loc[idx, kpi] = df.loc[idx, kpi].interpolate(limit_area="inside").ffill().bfill()
            ctx.imputed_kpi = was_na
            ctx.add_control("kpi_imputed", was_na.astype(float))
            notes.append(f"{int(was_na.sum())} KPI week(s) interpolated and flagged in kpi_imputed")
        else:
            d.answered = False
            return StepResult(STEP, NAME, Verdict.NEEDS_INPUT, f"'{choice}' isn't an option; choose interpolate or trim.", {}, [], decisions)

    media_holes = sum(int(labels[c][2].sum()) for c in value_cols if ctx.roles.get(c) in MEDIA_ROLES)
    media_mode = "zero"
    if media_holes:
        d = Decision(
            id="media_gaps",
            step=STEP,
            question=f"{media_holes} media cell(s) are empty between weeks that have values. Read them as no spend (0) or interpolate?",
            options=["zero", "interpolate"],
            default="zero",
        )
        decisions.append(d)
        media_mode = answers.ask(d) or "zero"

    report = []
    for col in value_cols:
        if col == kpi:
            ns, en, mi = labels[col]
            report.append({"column": col, "role": "kpi", "not_started": int(ns.sum()), "ended": int(en.sum()),
                           "missing": int(mi.sum()), "zeros": int((df[col] == 0).sum()),
                           "fill": "trimmed" if trimmed else "interpolated, flagged" if (ns | en | mi).any() else "none needed"})
            continue
        role = ctx.roles.get(col)
        ns, en, mi = labels[col]
        n_gaps = int((ns | en | mi).sum())
        zeros = int((df[col] == 0).sum())
        if role in MEDIA_ROLES:
            fill = "0 outside its active weeks"
            df.loc[ns | en, col] = 0.0
            if mi.any():
                if media_mode == "interpolate":
                    for _, idx in groups:
                        df.loc[idx, col] = df.loc[idx, col].interpolate(limit_area="inside")
                    fill += "; holes interpolated"
                else:
                    df.loc[mi, col] = 0.0
                    fill += "; holes read as 0"
        elif is_binary(df[col]):
            df[col] = df[col].fillna(0.0)
            fill = "0 where missing (flag off)"
        else:
            for _, idx in groups:
                df.loc[idx, col] = df.loc[idx, col].interpolate(limit_area="inside").ffill().bfill()
            fill = "interpolated, carried to the edges"
        if df[col].isna().any():
            n = int(df[col].isna().sum())
            df[col] = df[col].fillna(0.0 if role in MEDIA_ROLES else df[col].median())
            warnings.append(f"'{col}' is empty for a whole region; filled {n} cell(s) with {'0' if role in MEDIA_ROLES else 'its median'}.")
        report.append({"column": col, "role": role, "not_started": int(ns.sum()), "ended": int(en.sum()),
                       "missing": int(mi.sum()), "zeros": zeros, "fill": fill if n_gaps else "none needed"})
    ctx.df = df

    if missing_weeks:
        warnings.append(f"{len(missing_weeks)} week(s) were absent from the export and added, e.g. {missing_weeks[:3]}.")
    labelled = sum(r["not_started"] + r["ended"] + r["missing"] for r in report)
    parts = [f"{len(missing_weeks)} missing week(s) added" if missing_weeks else "no missing weeks",
             f"{labelled:,} empty cell(s) labelled and filled",
             f"{sum(r['zeros'] for r in report):,} zeros kept as real zeros"]
    summary = "; ".join(parts + notes) + "."
    details = {"columns": report, "missing_weeks": missing_weeks, "rows_added": int(inserted.sum()),
               "rows_before": before, "rows_after": len(df), "trimmed_weeks": trimmed}
    return StepResult(STEP, NAME, Verdict.WARN if warnings else Verdict.PASS, summary, details, warnings, decisions)
