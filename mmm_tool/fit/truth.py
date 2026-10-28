"""Compare a fitted model with a test file's known truth (step 32, test files only).

Real data has no truth, so nothing here runs unless the user loads an answer
key: a JSON file shaped like a synthetic test set's truth_for_tool.json

    {"kpi_column": "...", "assumes_decisions": "...",
     "channels": [{"source_columns": [...], "true_roi": x, "true_share_of_kpi_pct": x,
                   "true_policies_per_lakh": x, "true_marginal_roi": x}, ...],
     "true_non_media_share_pct": x}

It is kept with the run and only read when results are shown: it never goes
to the model and never changes a step's verdict.

A modelled channel matches the truth entry whose source_columns are exactly
the raw columns the channel is made of, after step 15 drops and step 17
combinations (Channel.sources(); its exposure columns count too). Channels
are never matched by name; a channel with no exact match has no truth.
"""
from __future__ import annotations

from typing import Any

from .. import currency as money_fmt

PER_LAKH = 100_000          # the answer key's KPI-per-spend figures are per 1,00,000 of spend


class TruthError(ValueError):
    """An answer key that can't be used for this run."""


def validate(key: Any, kpi_column: str) -> dict:
    if not isinstance(key, dict) or not isinstance(key.get("channels"), list):
        raise TruthError("An answer key is a JSON object with 'kpi_column' and a 'channels' list.")
    if key.get("kpi_column") != kpi_column:
        raise TruthError(f"This answer key is for the KPI column '{key.get('kpi_column')}', but this run's KPI "
                         f"column is '{kpi_column}'. Load the answer key that belongs to this file.")
    for i, ch in enumerate(key["channels"]):
        if not isinstance(ch, dict) or not isinstance(ch.get("source_columns"), list) or not ch["source_columns"]:
            raise TruthError(f"Answer key channel {i + 1} needs a non-empty 'source_columns' list.")
    return key


def match(channels, key: dict) -> dict[str, dict | None]:
    """{channel name: its truth entry or None}, by the exact set of raw columns."""
    entries = [(frozenset(map(str, e["source_columns"])), e) for e in key["channels"]]
    out = {}
    for ch in channels:
        mine = {frozenset(ch.sources()), frozenset(ch.all_sources())}
        out[ch.name] = next((e for cols, e in entries if cols in mine), None)
    return out


def compare(result: dict, channels, key: dict, kpi: dict) -> dict:
    """The truth columns for step 32 and the true marginal ROI for later screens."""
    s32 = next((s for s in result.get("steps", []) if s["step"] == 32), None)
    money = bool(s32 and s32["details"].get("roi_in_money"))
    cur = money_fmt.of(kpi)
    found = match(channels, key)
    rows, errors, inside = [], [], 0
    share_sum = 0.0
    for c in (x for x in result.get("channels", []) if x.get("paid")):
        e = found.get(c["channel"])
        share_sum += c.get("share_of_kpi") or 0.0
        row = {"channel": c["channel"], "matched": e is not None,
               "sources": next((list(ch.sources()) for ch in channels if ch.name == c["channel"]), [])}
        if e is None:
            rows.append(row)
            continue
        scale = 1.0 if money else PER_LAKH
        model, lo, hi = (c[f"roi_posterior_{k}"] * scale for k in ("mean", "ci_lo", "ci_hi"))
        true = e.get("true_roi") if money else e.get("true_policies_per_lakh")
        if true is None:
            row.update(model=model, lo=lo, hi=hi, true=None, inside=None)
        else:
            ok = lo <= true <= hi
            inside += ok
            if true:
                errors.append(abs(model - true) / abs(true))
            row.update(model=model, lo=lo, hi=hi, true=true, inside=ok)
        row.update(true_share=(e.get("true_share_of_kpi_pct") or 0) / 100 if e.get("true_share_of_kpi_pct") is not None else None,
                   true_marginal_roi=e.get("true_marginal_roi"), source_columns=e["source_columns"])
        rows.append(row)
    compared = [r for r in rows if r.get("inside") is not None]
    nm = key.get("true_non_media_share_pct")
    label = "ROI" if money else f"{kpi.get('kpi') or 'KPI'} per {money_fmt.money_short(PER_LAKH, cur)}"
    return {
        "assumes_decisions": key.get("assumes_decisions"), "kpi_column": key.get("kpi_column"),
        "measure": "roi" if money else "per_lakh", "label": label, "rows": rows,
        "non_media": {"model_share": 1 - share_sum, "true_share": nm / 100 if nm is not None else None},
        "inside": inside, "compared": len(compared), "channels": len(rows),
        "mean_abs_error": sum(errors) / len(errors) if errors else None,
        "true_marginal_roi": {r["channel"]: r["true_marginal_roi"] for r in rows if r.get("true_marginal_roi") is not None},
    }
