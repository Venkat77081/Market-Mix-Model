"""Phase F: everything the Model Summary screen shows (steps 37-42), computed
once from the accepted final model and stored as summary.json, so the screen
opens instantly.

37 Fit health        weekly actual vs predicted with a 90% band
                     (Analyzer.expected_vs_actual_data); train and test MAPE
                     and R² from the holdout fit (step 31); R-hat from the
                     final fit (Analyzer.get_rhat)
38 Sales breakdown   baseline and each channel, in total and week by week
                     (Analyzer.summary_metrics, baseline_summary_metrics); the
                     parts add up to the predicted KPI every week
39 ROI table         ROI, marginal ROI, cost per KPI unit and KPI driven per
                     paid channel, each with its 90% range (summary_metrics),
                     for all weeks and for the last 52
40 Trust badges      per channel, from Phases C, D and E, with the reason
41 Response curves   KPI driven against spend, current spend at 1x
                     (Analyzer.response_curves)
42 Carryover         share of a week's effect left after each week
                     (Analyzer.adstock_decay), with its half-life

The ordering rule on the screen is trust before results: fit health and
badges sit above the ROI table.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .. import currency as money_fmt
AGGREGATES = ("All Paid Channels", "All Channels")
CONFIDENCE = 0.9
SPEND_MULTIPLIERS = [round(x * 0.1, 1) for x in range(0, 21)] + [2.5, 3.0]

BADGE_TEXT = {
    "calibrated": "calibrated", "model_only": "model only", "hard_to_measure": "hard to measure",
    "prior_driven": "prior-driven", "unstable": "unstable", "rarely_on": "rarely on",
    "moves_with": "moves with another channel", "season_timed": "timed with a season",
}


def _f(x) -> float:
    return float(np.asarray(x).ravel()[0]) if np.ndim(x) else float(x)


def _range(da, dist: str = "posterior") -> dict[str, float]:
    s = da.sel(distribution=dist)
    return {"mean": _f(s.sel(metric="mean")), "lo": _f(s.sel(metric="ci_lo")), "hi": _f(s.sel(metric="ci_hi"))}


def _names(ds) -> list[str]:
    return [str(c) for c in ds.channel.values if str(c) not in AGGREGATES]


def fit_health(an, mmm, e_steps: dict[int, dict], config: dict) -> dict[str, Any]:
    ev = an.expected_vs_actual_data(use_kpi=True, aggregate_geos=True, confidence_level=CONFIDENCE)
    weeks = [str(t)[:10] for t in ev.time.values]
    rh = {}
    for k, v in an.get_rhat().items():
        a = np.asarray(v, float)
        if a.size and not np.isnan(a).all():
            rh[k] = float(np.nanmax(a))
    worst = max(rh, key=rh.get) if rh else None
    s31, s30 = e_steps.get(31, {}), e_steps.get(30, {})
    return {
        "weeks": weeks,
        "actual": np.asarray(ev["actual"], float).ravel().round(2).tolist(),
        "expected": np.asarray(ev["expected"].sel(metric="mean"), float).ravel().round(2).tolist(),
        "expected_lo": np.asarray(ev["expected"].sel(metric="ci_lo"), float).ravel().round(2).tolist(),
        "expected_hi": np.asarray(ev["expected"].sel(metric="ci_hi"), float).ravel().round(2).tolist(),
        "holdout": config.get("holdout"),
        "accuracy": s31.get("accuracy", {}),            # from the fit that held the last weeks back (step 31)
        "naive_mape": s31.get("naive_mape"),
        "test_mape_max": s31.get("test_mape_max"),
        "rhat": {"worst": {"parameter": worst, "rhat": rh.get(worst)} if worst else None, "limit": s30.get("rhat_max", 1.1)},
    }


def breakdown(an, weeks: list[str], last52: list[str] | None) -> dict[str, Any]:
    """Baseline and every channel (paid, unpaid and non-media levers), in KPI units."""
    def totals(selected):
        sm = an.summary_metrics(confidence_level=CONFIDENCE, use_kpi=True, include_non_paid_channels=True, selected_times=selected)
        b = an.baseline_summary_metrics(use_kpi=True, confidence_level=CONFIDENCE, selected_times=selected)
        ch = {c: _range(sm["incremental_outcome"].sel(channel=c)) for c in _names(sm)}
        base = _range(b["baseline_outcome"])
        total = base["mean"] + sum(v["mean"] for v in ch.values())
        return {"baseline": base, "channels": ch, "total": total}

    w = an.summary_metrics(confidence_level=CONFIDENCE, use_kpi=True, aggregate_times=False, include_non_paid_channels=True)
    wb = an.baseline_summary_metrics(use_kpi=True, aggregate_times=False, confidence_level=CONFIDENCE)
    weekly = {c: np.asarray(w["incremental_outcome"].sel(channel=c, distribution="posterior", metric="mean"), float).round(2).tolist()
              for c in _names(w)}
    return {
        "all": totals(None),
        "last_52": totals(last52) if last52 else None,
        "weekly": {"weeks": weeks, "baseline": np.asarray(wb["baseline_outcome"].sel(distribution="posterior", metric="mean"),
                                                          float).round(2).tolist(), "channels": weekly},
    }


def roi_table(an, money: bool, selected: list[str] | None) -> dict[str, Any]:
    kpi = an.summary_metrics(confidence_level=CONFIDENCE, use_kpi=True, selected_times=selected)
    rev = an.summary_metrics(confidence_level=CONFIDENCE, use_kpi=False, selected_times=selected) if money else kpi
    rows = []
    present = [str(x) for x in kpi.channel.values]
    total = next((a for a in AGGREGATES if a in present), None)      # Meridian names the paid total either way
    for c in _names(kpi) + ([total] if total else []):
        rows.append({
            "channel": "all paid" if c == total else c,
            "spend": _f(kpi["spend"].sel(channel=c)),
            "share_of_spend": _f(kpi["pct_of_spend"].sel(channel=c)) / 100,
            "roi": _range(rev["roi"].sel(channel=c)),
            "mroi": _range(rev["mroi"].sel(channel=c)),
            "kpi_driven": _range(kpi["incremental_outcome"].sel(channel=c)),
            "revenue_driven": _range(rev["incremental_outcome"].sel(channel=c)) if money else None,
            "cost_per_kpi": _range(kpi["cpik"].sel(channel=c)),
            "share_of_kpi": _f(kpi["pct_of_contribution"].sel(channel=c, distribution="posterior", metric="mean")) / 100,
        })
    return {"rows": rows}


def response_curves(an, money: bool) -> dict[str, Any]:
    rc = an.response_curves(spend_multipliers=SPEND_MULTIPLIERS, confidence_level=CONFIDENCE, use_kpi=True)
    out = {}
    for c in [str(x) for x in rc.channel.values]:
        inc = rc["incremental_outcome"].sel(channel=c)
        out[c] = {
            "multiplier": [float(x) for x in rc.spend_multiplier.values],
            "spend": np.asarray(rc["spend"].sel(channel=c), float).round(0).tolist(),
            "kpi": np.asarray(inc.sel(metric="mean"), float).round(2).tolist(),
            "kpi_lo": np.asarray(inc.sel(metric="ci_lo"), float).round(2).tolist(),
            "kpi_hi": np.asarray(inc.sel(metric="ci_hi"), float).round(2).tolist(),
        }
    return out


def carryover(an, paid: list[str]) -> dict[str, Any]:
    ad = an.adstock_decay(confidence_level=CONFIDENCE)
    ad = ad[(ad["distribution"] == "posterior") & ad["is_int_time_unit"]]
    out = {}
    for c in paid:
        rows = ad[ad["channel"] == c].sort_values("time_units")
        if rows.empty:
            continue
        mean = rows["mean"].to_numpy(float)
        # Half-life: the first week the effect falls to half of week 0's, interpolated between weeks.
        half = None
        for i in range(1, len(mean)):
            if mean[i] <= 0.5 * mean[0]:
                half = (i - 1) + (mean[i - 1] - 0.5 * mean[0]) / (mean[i - 1] - mean[i]) if mean[i - 1] != mean[i] else float(i)
                break
        out[c] = {"weeks": rows["time_units"].astype(int).tolist(), "mean": mean.round(4).tolist(),
                  "lo": rows["ci_lo"].to_numpy(float).round(4).tolist(), "hi": rows["ci_hi"].to_numpy(float).round(4).tolist(),
                  "half_life": None if half is None else round(float(half), 2)}
    return out


def trust_badges(badges: dict[str, list[str]], e_steps: dict[int, dict], config: dict, paid: list[str]) -> dict[str, Any]:
    """Each paid channel's badges with the reason behind each."""
    narrow = {r["channel"]: r for r in e_steps.get(33, {}).get("channels", [])}
    stab = {r["channel"]: r for r in e_steps.get(34, {}).get("channels", [])} if not e_steps.get(34, {}).get("skipped") else {}
    tests = {}
    for t in (config.get("roi_priors") or {}).items():
        if t[1].get("source") == "lift_test":
            tests[t[0]] = t[1].get("tests", [])
    out = {}
    for c in paid:
        rows = []
        for b in badges.get(c, []):
            if b == "calibrated":
                ts = tests.get(c, [])
                why = ("Prior set from " + "; ".join(f"the {t['start']} to {t['end']} lift test (ROI {t['roi']}, range {t['low']}–{t['high']})" for t in ts)
                       + " (step 26).") if ts else "A lift test sets this channel's prior (step 26)."
            elif b == "model_only":
                why = "No experiment behind it; the estimate comes from the data and the default prior."
            elif b == "hard_to_measure":
                why = "Spend barely varies from week to week, so the data says little about its effect (step 16)."
            elif b == "prior_driven" and "calibrated" in badges.get(c, []):
                why = "Its ROI comes mostly from its lift test, which set the prior (steps 26 and 33)."
            elif b == "prior_driven":
                k = narrow.get(c, {}).get("narrowing")
                why = (f"The data narrowed its ROI range by only {k:.0%} (30% needed), so its ROI mostly repeats the prior (step 33)."
                       if k is not None else "Its ROI mostly repeats the prior (step 33).")
            elif b == "unstable":
                cv = stab.get(c, {}).get("cv")
                why = (f"Its ROI moves when the first or last 26 weeks are dropped (CV {cv:.2f}, step 34)." if cv is not None
                       else "Its ROI moves when the first or last 26 weeks are dropped (step 34).")
            elif b == "rarely_on":
                why = "It spends in few weeks, so there is little to learn from (step 15)."
            elif b == "moves_with":
                why = "Its spend moves with another channel's, so the two are hard to tell apart (step 17)."
            elif b == "season_timed":
                why = "It spends mostly in a season, so the season and the channel compete for the same sales (step 19)."
            else:
                why = ""
            rows.append({"badge": b, "text": BADGE_TEXT.get(b, b), "why": why})
        out[c] = rows
    return out


def build_summary(mmm, d, e_steps: dict[int, dict], badges: dict[str, list[str]], model_info: dict[str, Any]) -> dict[str, Any]:
    """The stored Model Summary for an accepted final fit."""
    from ..fit import model as m

    an = m.analyzer(mmm)
    money = m.uses_revenue(mmm)
    paid = [str(c) for c in mmm.input_data.get_all_paid_channels()]
    weeks = [str(t)[:10] for t in mmm.input_data.time.values]
    last52 = weeks[-52:] if len(weeks) >= 104 else None
    geos = int(np.asarray(mmm.input_data.kpi).shape[0])
    kpi_name = d.kpi.get("kpi")
    cur = money_fmt.of(d.kpi)
    out = {
        "version": {
            "data_version": model_info.get("data_version"), "saved_at": model_info.get("saved_at"),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "meridian": model_info.get("meridian"),
            "sampler": model_info.get("sampler"), "first_week": weeks[0], "last_week": weeks[-1], "weeks": len(weeks),
            "geos": geos, "kpi": kpi_name, "kpi_type": d.kpi.get("kpi_type"), "money": money,
            "roi_unit": money_fmt.per(cur) if money else f"{kpi_name} per {money_fmt.unit(cur)}", "currency": cur,
            "not_converged": bool(model_info.get("not_converged")), "seed": model_info.get("seed"),
        },
        "fit": fit_health(an, mmm, e_steps, d.config),
        "badges": trust_badges(badges, e_steps, d.config, paid),
        "roi": {"all": roi_table(an, money, None), "last_52": roi_table(an, money, last52) if last52 else None},
        "breakdown": breakdown(an, weeks, last52),
        "curves": response_curves(an, money),
        "carryover": carryover(an, paid),
        "paid": paid,
        "decision_log": list(getattr(d, "decision_log", None) or []),
    }
    return _jsonable(out)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def meridian_html(mmm, folder, filename: str = "model_results_summary.html") -> str:
    """Meridian's own HTML results summary, kept as a download."""
    from meridian.analysis import summarizer

    from .offline import make_offline

    summarizer.Summarizer(mmm).output_model_results_summary(filename, str(folder))
    return str(make_offline(folder / filename))           # charts work without reaching gstatic.com
