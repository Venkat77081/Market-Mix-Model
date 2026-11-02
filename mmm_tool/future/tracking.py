"""Tracking: real weeks against the tracked plan.

Actuals come in the training file's raw format. Columns are matched to the training file's by
name; a renamed one is offered by similarity and confirmed once (remembered for later uploads).
Phases A and B then read them exactly like the training file (the run's answers and confirmed
roles, its reading settings), and each model channel's spend is the sum of its source columns, so
merged and combined channels match the model's. Only plan weeks are compared; every new week is
kept for the next refit.

Per plan week with actuals:
  spend vs plan         each channel's actual - planned spend (amount and %), tolerance ±10%
  predicted at actual   forecast + (media outcome at the actual spend - the reference's), on the
                        model, with its 90% range (the forecast's bounds, else ± step 31's error)
  execution gap         predicted at actual - the plan's frozen prediction ("didn't spend as planned")
  model / outside gap   actual - predicted at actual ("spend was on plan, the result differed")
"""
from __future__ import annotations

import difflib
from typing import Any

import numpy as np
import pandas as pd


def _norm(name: str) -> str:
    from ..ingest.keywords import words_of

    return words_of(name)


def match_columns(new_cols: list[str], training_cols: list[str], renames: dict[str, str], threshold: float) -> dict[str, Any]:
    """Which uploaded columns are the training file's: same name, confirmed rename, a likely rename
    (offered, by similarity), or new. Training columns not found are listed as missing."""
    train = list(training_cols)
    same = [c for c in new_cols if c in train]
    confirmed = {n: o for n, o in renames.items() if n in new_cols and o in train and o not in same}
    taken = set(same) | set(confirmed.values())
    missing = [c for c in train if c not in taken]
    offered, new = {}, []
    for c in new_cols:
        if c in same or c in confirmed:
            continue
        scored = sorted(((difflib.SequenceMatcher(None, _norm(c), _norm(m)).ratio(), m) for m in missing), reverse=True)
        if scored and scored[0][0] >= threshold and scored[0][1] not in offered.values():
            offered[c] = {"column": scored[0][1], "similarity": round(scored[0][0], 2)}
        else:
            new.append(c)
    return {"same": same, "confirmed": confirmed, "offered": offered, "new": new,
            "missing": [c for c in missing if c not in [o["column"] for o in offered.values()]]}


def actuals_from_table(table: pd.DataFrame, date_col: str, geo_col: str | None, kpi_col: str, channels) -> dict[str, Any]:
    """National weekly KPI, and each model channel's spend (sum of its source columns) and media
    units (its exposure column, when it has one and the upload does), from a Phase B table."""
    t = table.copy()
    t[date_col] = pd.to_datetime(t[date_col])
    g = t.groupby(date_col)
    weeks = [str(w.date()) for w in sorted(t[date_col].unique())]
    out = {"weeks": weeks, "kpi": [float(x) for x in g[kpi_col].sum().reindex(pd.to_datetime(weeks)).values],
           "spend": {}, "units": {}, "missing_channels": []}
    for ch in channels:
        srcs = [c for c in ch.sources() if c in t.columns]
        if not srcs:
            out["missing_channels"].append(ch.name)
            continue
        out["spend"][ch.name] = [float(x) for x in g[srcs].sum().sum(axis=1).reindex(pd.to_datetime(weeks)).fillna(0).values]
        unit_col = ch.reach if ch.kind == "rf" else (None if ch.exposure_is_spend or ch.spend_sources else ch.exposure)
        if unit_col and unit_col in t.columns:
            out["units"][ch.name] = [float(x) for x in g[unit_col].sum().reindex(pd.to_datetime(weeks)).fillna(0).values]
    return out


def on_plan(actual: dict, plan_weeks: list[str]) -> dict[str, Any]:
    """The actuals for the plan's weeks only, matched to the nearest plan week within 3 days."""
    keep, idx = [], []
    for i, w in enumerate(actual["weeks"]):
        near = min(plan_weeks, key=lambda p: abs((pd.Timestamp(p) - pd.Timestamp(w)).days))
        if abs((pd.Timestamp(near) - pd.Timestamp(w)).days) <= 3:
            keep.append(near)
            idx.append(i)
    pick = lambda v: [v[i] for i in idx]  # noqa: E731
    return {"weeks": keep, "kpi": pick(actual["kpi"]), "spend": {c: pick(v) for c, v in actual["spend"].items()},
            "units": {c: pick(v) for c, v in actual["units"].items()}}


def merge(earlier: dict | None, new: dict, plan_weeks: list[str]) -> dict[str, Any]:
    """Add a new upload's plan weeks to the earlier uploads' (a week in both: the new upload's), so
    actuals can come a few weeks at a time."""
    if not earlier or not earlier.get("weeks"):
        return new
    rows: dict[str, dict] = {}
    for src in (earlier, new):
        for i, w in enumerate(src["weeks"]):
            rows[w] = {"kpi": src["kpi"][i], "spend": {c: v[i] for c, v in src["spend"].items()},
                       "units": {c: v[i] for c, v in (src.get("units") or {}).items()}}
    weeks = [w for w in plan_weeks if w in rows]
    chans = sorted({c for r in rows.values() for c in r["spend"]})
    uchans = sorted({c for r in rows.values() for c in r["units"]})
    return {"weeks": weeks, "kpi": [rows[w]["kpi"] for w in weeks],
            "spend": {c: [rows[w]["spend"].get(c, 0.0) for w in weeks] for c in chans},
            "units": {c: [rows[w]["units"].get(c) for w in weeks] for c in uchans}}


def spend_for_model(on: dict, targets: dict, plan_weeks: list[str], channels: list[str]) -> dict[str, list[float]]:
    """Spend for every plan week: actual where there is one, the tracked plan's elsewhere."""
    pos = {w: i for i, w in enumerate(on["weeks"])}
    return {c: [on["spend"][c][pos[w]] if (w in pos and c in on["spend"]) else float(targets[w]["spend"].get(c, 0.0))
                for w in plan_weeks] for c in channels}


def _streak(flags: list[bool]) -> int:
    n = 0
    for f in reversed(flags):
        if not f:
            break
        n += 1
    return n


def compare(on: dict, targets: dict, at_actual: dict, settings: dict, holdout_error: float | None,
            plan_by_channel: dict | None = None) -> dict[str, Any]:
    """The Tracking screen's numbers: rows per week, tiles, alerts."""
    tol = float(settings["spend_tolerance_pct"]) / 100
    pw = at_actual["per_week"]
    plan_weeks = at_actual["weeks"]
    pos = {w: i for i, w in enumerate(plan_weeks)}
    has_total = pw.get("total") is not None
    bounds_from = None
    rows = []
    for k, w in enumerate(on["weeks"]):
        i, tgt = pos[w], targets.get(w)
        if tgt is None:
            continue
        planned = {c: float(v) for c, v in tgt["spend"].items()}
        actual_spend = {c: float(on["spend"].get(c, [0.0] * len(on["weeks"]))[k]) for c in planned}
        off = {c: (actual_spend[c] - planned[c]) / planned[c] if planned[c] else (0.0 if actual_spend[c] == 0 else float("inf"))
               for c in planned}
        row = {"week": w, "plan_id": tgt.get("plan_id"), "planned_spend": sum(planned.values()), "actual_spend": sum(actual_spend.values()),
               "channels": {c: {"planned": planned[c], "actual": actual_spend[c], "off": off[c] if off[c] != float("inf") else None,
                                "outside": abs(off[c]) > tol} for c in planned},
               "spend_off_plan": any(abs(v) > tol for v in off.values()), "actual": float(on["kpi"][k]),
               "plan_predicted": tgt.get("total")}
        if has_total:
            pred, lo, hi = pw["total"]["mean"][i], pw["total"]["lo"][i], pw["total"]["hi"][i]
            if pw.get("forecast_lower") is None and holdout_error is not None:
                lo, hi = pred * (1 - holdout_error), pred * (1 + holdout_error)
                bounds_from = f"± the model's holdout error ({holdout_error:.1%}, step 31)"
            elif pw.get("forecast_lower") is not None:
                bounds_from = "the model's posterior range combined with the forecast's bounds"
            else:
                bounds_from = "the model's posterior range"
            row.update(predicted=pred, lo=lo, hi=hi, gap=row["actual"] - pred, gap_pct=(row["actual"] - pred) / pred if pred else None,
                       execution_gap=pred - tgt["total"] if tgt.get("total") is not None else None, model_gap=row["actual"] - pred)
            row["range"] = "below" if row["actual"] < lo else "above" if row["actual"] > hi else "inside"
            row["status"] = row["range"] if row["range"] != "inside" else ("spend off plan" if row["spend_off_plan"] else "inside")
        else:
            row["status"] = "spend off plan" if row["spend_off_plan"] else "spend on plan"
        rows.append(row)
    tiles = {"weeks_done": len(rows), "weeks_total": len(plan_weeks), "to_week": rows[-1]["week"] if rows else None,
             "actual": float(sum(r["actual"] for r in rows)),
             "planned_spend": float(sum(r["planned_spend"] for r in rows)), "actual_spend": float(sum(r["actual_spend"] for r in rows))}
    tiles["spend_vs_plan"] = tiles["actual_spend"] / tiles["planned_spend"] - 1 if tiles["planned_spend"] else None
    worst = None
    if rows:
        per_c = {c: (sum(r["channels"][c]["actual"] for r in rows), sum(r["channels"][c]["planned"] for r in rows)) for c in rows[0]["channels"]}
        devs = {c: (a / p - 1 if p else None) for c, (a, p) in per_c.items()}
        cand = [(abs(v), c, v) for c, v in devs.items() if v is not None]
        worst = max(cand)[1:] if cand else None
        tiles["channels_within"] = all(abs(v) <= tol for v in devs.values() if v is not None)
        tiles["worst_channel"] = {"channel": worst[0], "off": worst[1]} if worst else None
    alerts = []
    if has_total and rows:
        tiles.update(predicted=float(sum(r["predicted"] for r in rows)),
                     inside=sum(r["range"] == "inside" for r in rows))
        tiles["gap"] = tiles["actual"] - tiles["predicted"]
        tiles["gap_pct"] = tiles["gap"] / tiles["predicted"] if tiles["predicted"] else None
        # Red: outside the range for N weeks in a row, or the cumulative model gap beyond the limit.
        streak_lo = _streak([r["range"] == "below" for r in rows])
        streak_hi = _streak([r["range"] == "above" for r in rows])
        cum = tiles["gap_pct"] or 0.0
        n_need = int(settings["outside_range_weeks"])
        if streak_lo >= n_need or streak_hi >= n_need or abs(cum) > float(settings["cumulative_gap_pct"]) / 100:
            side = "below" if (streak_lo >= streak_hi and (streak_lo or cum < 0)) else "above"
            weeks_out = rows[-max(streak_lo, streak_hi):] if max(streak_lo, streak_hi) else rows
            on_plan_spend = not any(r["spend_off_plan"] for r in weeks_out)
            alerts.append({"level": "red", "kind": "range", "side": side,
                           "title": f"{'Below' if side == 'below' else 'Above'} the predicted range",
                           "weeks": [r["week"] for r in weeks_out] if max(streak_lo, streak_hi) else [],
                           "text": ("Spend matched the plan, so the gap is not explained by media. Check outside causes first."
                                    if on_plan_spend else "Spend was also off plan in these weeks; see the execution gap."),
                           "streak": max(streak_lo, streak_hi), "cumulative_gap_pct": cum})
        # Amber: the gap mostly comes from spending differently from the plan.
        exe = sum(r["execution_gap"] or 0 for r in rows)
        tot = sum((r["actual"] - r["plan_predicted"]) for r in rows if r["plan_predicted"] is not None)
        if tot and abs(exe) >= float(settings["execution_share"]) * abs(tot) and abs(exe) > 0:
            drivers = []
            if plan_by_channel and at_actual.get("by_channel"):
                contrib = {c: at_actual["by_channel"][c]["media"] - plan_by_channel[c]["media"] for c in at_actual["by_channel"] if c in plan_by_channel}
                drivers = [{"channel": c, "effect": v} for c, v in sorted(contrib.items(), key=lambda kv: kv[1]) if abs(v) >= 0.1 * abs(exe)]
            alerts.append({"level": "amber", "kind": "execution", "title": "The gap comes mostly from spending differently from the plan",
                           "execution_gap": exe, "total_gap": tot, "channels": drivers})
    # Amber: a channel off its spend tolerance for N weeks in a row.
    if rows:
        need = int(settings["spend_off_weeks"])
        off_ch = [c for c in rows[0]["channels"] if _streak([r["channels"][c]["outside"] for r in rows]) >= need]
        if off_ch:
            alerts.append({"level": "amber", "kind": "spend", "title": "Spend off plan", "channels": off_ch,
                           "text": f"{', '.join(off_ch)}: outside ±{tol:.0%} of the plan for {need}+ weeks in a row."})
    return {"rows": rows, "tiles": tiles, "alerts": alerts, "bounds_from": bounds_from, "has_forecast": has_total}
