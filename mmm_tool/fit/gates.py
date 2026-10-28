"""Phase E pass rules as plain functions of numbers, so they can be checked
without fitting a model. Thresholds come from fit.yaml."""
from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np


def interval_width(lo: float, hi: float) -> float:
    return float(hi - lo)


def prior_check(q_lo: float, q_hi: float, actual: float, max_width: float) -> dict:
    """Step 28. Does the prior's 90% range of total KPI cover the actual total
    without being more than max_width times as wide as it?"""
    covers = q_lo <= actual <= q_hi
    width = (q_hi - q_lo) / actual if actual else math.inf
    return {"covers": bool(covers), "width_ratio": float(width), "ok": bool(covers and width <= max_width)}


def converged(rhat: Mapping[str, float], rhat_max: float) -> dict:
    """Step 30. Every parameter's largest R-hat below rhat_max. A NaN R-hat never counts as a
    failure: Meridian returns NaN for parameters a model doesn't use (eta_m, slope_m, tau_g,
    xi_c on a national model)."""
    real = {k: float(v) for k, v in rhat.items() if v == v}
    bad = {k: v for k, v in real.items() if not v < rhat_max}
    worst = max(real.items(), key=lambda kv: kv[1]) if real else (None, math.nan)
    return {"ok": not bad, "bad": bad, "worst": {"parameter": worst[0], "rhat": float(worst[1])}}


def mape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    a, p = np.asarray(list(actual), float), np.asarray(list(predicted), float)
    m = a != 0
    return float(np.mean(np.abs(a[m] - p[m]) / np.abs(a[m]))) if m.any() else math.nan


def seasonal_naive_mape(series: Iterable[float], test: Iterable[bool], season: int) -> float:
    """MAPE on the test weeks of predicting each week by the same week a season earlier."""
    y, t = np.asarray(list(series), float), np.asarray(list(test), bool)
    idx = np.flatnonzero(t)
    idx = idx[idx >= season]
    if not len(idx):
        return math.nan
    return mape(y[idx], y[idx - season])


def holdout_ok(test_mape: float, naive_mape: float, mape_max: float) -> dict:
    """Step 31. Test MAPE below mape_max and better than the seasonal-naive forecast."""
    under = test_mape < mape_max
    beats = not (naive_mape == naive_mape) or test_mape < naive_mape  # no naive forecast possible: not held against it
    return {"ok": bool(under and beats), "under_max": bool(under), "beats_naive": bool(beats)}


def plausible(roi: Mapping[str, float], baseline_share: float, reconcile_error: float, *, roi_min: float,
              roi_max: float, share_range: tuple[float, float], tolerance: float, roi_in_money: bool = True) -> dict:
    """Step 32. ROI per channel inside [roi_min, roi_max] (when ROI is money per
    money), baseline share inside share_range, contributions reconcile with the fit."""
    out_of_range = {c: float(v) for c, v in roi.items() if roi_in_money and not (roi_min <= v <= roi_max)}
    share_ok = share_range[0] <= baseline_share <= share_range[1]
    reconciles = abs(reconcile_error) <= tolerance
    return {"ok": bool(not out_of_range and share_ok and reconciles), "roi_out_of_range": out_of_range,
            "baseline_ok": bool(share_ok), "reconciles": bool(reconciles)}


def narrowing(prior: tuple[float, float], posterior: tuple[float, float]) -> float:
    """Step 33. How much narrower the posterior 90% range is than the prior's (0 = not at all)."""
    pw, qw = interval_width(*prior), interval_width(*posterior)
    return float(1 - qw / pw) if pw > 0 else 0.0


def cv(values: Iterable[float]) -> float:
    """Step 34. Coefficient of variation of a channel's ROI across fits."""
    v = np.asarray(list(values), float)
    m = v.mean()
    return float(v.std(ddof=1) / abs(m)) if len(v) > 1 and m else math.nan


def placebo_credited(prior_median: float, post_median: float, post_lo: float, roi_min: float) -> bool:
    """Step 35. A random series should earn nothing. Meridian's ROI priors are
    LogNormal, so its range can never include zero itself; the placebo counts as
    'not credited' when the low end of its range is under roi_min (as good as
    zero) or the data pulled its ROI down from the prior, not up."""
    return not (post_lo < roi_min or post_median <= prior_median)


def exact_prior_range(entry: dict | None, confidence: float) -> tuple[float, float] | None:
    """Step 33. The prior's 90% range from its distribution, exactly (no sampling noise):
    LogNormal(mu, sigma) or Beta(a, b) as configured in step 25. None without a closed form."""
    from scipy import stats

    if not entry or entry.get("family") not in ("lognormal", "beta"):
        return None
    q = [(1 - confidence) / 2, (1 + confidence) / 2]
    if entry["family"] == "lognormal":
        lo, hi = stats.lognorm(s=float(entry["p2"]), scale=math.exp(float(entry["p1"]))).ppf(q)
    else:
        lo, hi = stats.beta(float(entry["p1"]), float(entry["p2"])).ppf(q)
    return float(lo), float(hi)


def placebo_passes(placebo_mean: float, placebo_low: float, real_mean: float,
                   max_share: float, max_low: float, roi_in_money: bool = True) -> dict:
    """Step 35. A random series shaped like a real channel should earn much less: its mean ROI
    under max_share of the real channel's, and the low end of its range under max_low (as good
    as nothing; checked when ROI is money per money)."""
    below_real = placebo_mean < max_share * real_mean
    low_ok = placebo_low < max_low if roi_in_money else True
    return {"ok": bool(below_real and low_ok), "below_real": bool(below_real), "low_ok": bool(low_ok)}
