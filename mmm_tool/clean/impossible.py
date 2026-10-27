"""Step 12: check impossible values.

  Negative spend or exposure (refunds, reversals). Blocks the run until each
      cell is set to zero, kept, or corrected to a number.
  KPI of zero in a trading week. Interpolated by default, or kept, or corrected.
  An index, price or rate that jumps 10x from one week to the next (a premium
      index from 102 to 1,020 is usually a unit change), or an amount (spend,
      exposure, a summed control) 10x both of its neighbouring weeks (a slipped
      unit in one cell); at the edge of a burst, where one neighbour is 0, 10x
      the other and 10x the column's typical spending week. Kept by default, or interpolated, or corrected. The
      ratio is set in cleaning.yaml.

Each answer maps a cell key ("col|2024-03-04", with the region first in a
regional model) to "zero", "keep", "interpolate" or a corrected number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..ingest.types import Answers, Decision, StepResult, Verdict
from .context import MEDIA_ROLES, SPEND_ROLES, Ctx, cell_key, is_binary, settings

STEP = 12
NAME = "Check impossible values"
NON_NEGATIVE = SPEND_ROLES | {"media", "reach", "frequency", "organic_media", "kpi", "population", "revenue_per_kpi", "revenue_for_kpi"}


def _issue(ctx: Ctx, col: str, i: int, g, extra: dict) -> dict:
    return {"key": f"{col}|{cell_key(ctx.df.at[i, ctx.date], g)}", "column": col, "date": str(ctx.df.at[i, ctx.date].date()),
            "geo": g, "value": float(ctx.df.at[i, col]), "row": int(i), **extra}


def _apply(ctx: Ctx, issues: list[dict], chosen: dict, default: str) -> list[str]:
    """Apply each cell's answer; returns problems with the answers."""
    df, problems = ctx.df, []
    interp: dict[str, list[int]] = {}
    for it in issues:
        a = chosen.get(it["key"], default)
        it["action"] = a
        if a == "keep":
            continue
        if a == "zero":
            df.at[it["row"], it["column"]] = 0.0
        elif a == "interpolate":
            df.at[it["row"], it["column"]] = np.nan
            interp.setdefault(it["column"], []).append(it["row"])
        else:
            try:
                df.at[it["row"], it["column"]] = float(a)
                it["action"] = "corrected"
                it["corrected_to"] = float(a)
            except (TypeError, ValueError):
                problems.append(f"{it['key']}: '{a}' isn't zero, keep, interpolate or a number")
    for col in interp:
        for _, idx in ctx.by_geo():
            df.loc[idx, col] = df.loc[idx, col].interpolate(limit_area="inside").ffill().bfill()
    return problems


def check_impossible(ctx: Ctx, answers: Answers) -> StepResult:
    df = ctx.df
    JUMP, NEAR = float(settings()["jumps"]["ratio"]), float(settings()["jumps"]["near"])
    negatives, zeros, jumps = [], [], []
    groups = list(ctx.by_geo())
    imputed = ctx.imputed_kpi.reindex(df.index, fill_value=False) if ctx.imputed_kpi is not None else pd.Series(False, index=df.index)

    for col in df.columns:
        role = ctx.roles.get(col)
        if role in NON_NEGATIVE:
            for g, idx in groups:
                for i in idx[df.loc[idx, col].to_numpy() < 0]:
                    negatives.append(_issue(ctx, col, i, g, {"role": role}))
        # Only levels (indexes, prices, rates: averaged in step 8) should never jump 10x;
        # amounts such as competitor spend swing on their own.
        if role in ("control", "non_media_treatment") and ctx.rules.get(col) == "mean" and not is_binary(df[col]):
            for g, idx in groups:
                s = df.loc[idx, col]
                prev = s.shift(1)
                ratio = (s / prev).where((prev > 0) & (s > 0))
                for i in s.index[(ratio >= JUMP * NEAR) | (ratio <= 1 / (JUMP * NEAR))]:
                    jumps.append(_issue(ctx, col, i, g, {"kind": "level", "previous": float(prev[i]), "ratio": round(float(ratio[i]), 1)}))
        # Amounts swing on their own (bursts, launches), so only a single week far
        # above both neighbours counts: one cell in the wrong unit.
        elif col != ctx.kpi_col and (role in MEDIA_ROLES or (
                role in ("control", "non_media_treatment") and ctx.rules.get(col) == "sum")) and not is_binary(df[col]):
            for g, idx in groups:
                s = df.loc[idx, col]
                prev, nxt = s.shift(1), s.shift(-1)
                both = (prev > 0) & (nxt > 0)
                # At the edge of a burst one neighbour is 0: compare with the other,
                # and also with the column's typical spending week.
                one = ((prev > 0) ^ (nxt > 0)) & (s >= JUMP * NEAR * s[s > 0].median())
                ratio = (s / pd.concat([prev, nxt], axis=1).max(axis=1)).where((both | one) & (s > 0))
                for i in s.index[ratio >= JUMP * NEAR]:
                    jumps.append(_issue(ctx, col, i, g, {"kind": "spike", "previous": float(prev[i]), "next": float(nxt[i]),
                                                         "ratio": round(float(ratio[i]), 1)}))

    kpi = ctx.kpi_col
    for g, idx in groups:
        s = df.loc[idx, kpi]
        if s.median() > 0:
            for i in s.index[(s == 0) & ~imputed[idx]]:
                zeros.append(_issue(ctx, kpi, i, g, {}))

    decisions, warnings, problems = [], [], []
    specs = [
        ("negative_values", negatives, "zero", True,
         "These cells are negative, which spend, exposure and the KPI can't be (often a refund). Set to zero, keep, or enter the right value?"),
        ("kpi_zero_weeks", zeros, "interpolate", False,
         f"'{kpi}' is zero in these weeks although it is normally positive. Interpolate, keep the zero, or enter the right value?"),
        ("control_jumps", jumps, "keep", False,
         f"These values are {JUMP:g}x or more off the weeks around them (a level against the week before, an amount "
         f"against both neighbours), often a unit slip. Keep, interpolate, or enter the right value?"),
    ]
    for did, issues, default, blocking, question in specs:
        if not issues:
            continue
        d = Decision(id=did, step=STEP, question=question, options=["zero", "keep", "interpolate"],
                     default={it["key"]: default for it in issues}, blocking=blocking, context={"cells": issues})
        decisions.append(d)
        chosen = answers.ask(d)
        if blocking and chosen is None:
            continue
        problems += _apply(ctx, issues, {**d.default, **(chosen or {})}, default)

    if problems:
        for d in decisions:
            d.answered, d.blocking = False, True
        return StepResult(STEP, NAME, Verdict.NEEDS_INPUT, "; ".join(problems) + ".", {}, [], decisions)
    details = {"negative": negatives, "kpi_zero": zeros, "jumps": jumps}
    if any(d.pending for d in decisions):
        cols = sorted({n["column"] for n in negatives})
        return StepResult(STEP, NAME, Verdict.NEEDS_INPUT,
                          f"{len(negatives)} negative value(s) in {', '.join(cols)}; decide each one to continue.",
                          details, warnings, decisions)

    if zeros:
        warnings.append(f"'{kpi}' was zero in {len(zeros)} trading week(s).")
    if jumps:
        warnings.append(f"{len(jumps)} value(s) are {JUMP:g}x or more off the weeks around them; check their units.")
    if not (negatives or zeros or jumps):
        return StepResult(STEP, NAME, Verdict.PASS, f"No negative spend, no zero-KPI weeks, no {JUMP:g}x jumps.", details)
    parts = []
    if negatives:
        parts.append(f"{len(negatives)} negative value(s) fixed")
    if zeros:
        parts.append(f"{len(zeros)} zero-KPI week(s)")
    if jumps:
        parts.append(f"{len(jumps)} {JUMP:g}x jump(s)")
    return StepResult(STEP, NAME, Verdict.WARN if warnings else Verdict.PASS, "; ".join(parts) + ".", details, warnings, decisions)
