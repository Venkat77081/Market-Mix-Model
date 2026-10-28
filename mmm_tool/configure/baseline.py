"""Step 24's baseline preview: what a number of knots does, before choosing it.

Knot placement and weights always come from Meridian (knots.get_knot_info),
never from a copy of its rules. The preview fits, for every number of knots K
from 1 to weeks/4,

    KPI_t = knots_t (weights.T @ levels) + controls_t
            + sum over paid channels of beta_c * log1p(adstock_c,t / typical spend_c)

with ridge regression (alpha 1 on standardised inputs) and media coefficients
held at zero or above. The baseline drawn is the knot part only. It is a
preview: Meridian estimates the real baseline in step 29.

The trade-off is shown three ways, never as a fit error (more knots always fit
the sales line better):
  left for media      media's share of the preview's explained sales
  baseline follows    correlation of the baseline's week-to-week change with
                      the adstocked spend of each burst channel (CV > 0.5 in
                      step 16); the highest is reported
  trend left out      with K = 1, a linear trend in the KPI after controls (|t| > 2)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .context import KNOTS_AMBER_WEEKS, KNOTS_MAX_WEEKS, KNOTS_RED_WEEKS, WEEKS_PER_KNOT

RIDGE_ALPHA = 1.0
PREVIEW_DECAY = 0.5        # geometric adstock decay used by the preview only
BURST_CV = 0.5             # step 16 CV above this: a burst channel, drawn as a strip
FOLLOW_AMBER, FOLLOW_RED = 0.3, 0.5
TREND_T = 2.0


def meridian_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("meridian") is not None


def knot_info(n_weeks: int, knots: int, n_geos: int = 1):
    """Meridian's own knot locations and weights (needs google-meridian)."""
    from meridian.model import knots as mk

    return mk.get_knot_info(n_times=n_weeks, knots=knots, is_national=n_geos == 1)


def spacing(locations) -> int | None:
    """Typical weeks between knots (the median gap), from the real locations."""
    locs = sorted(int(x) for x in locations)
    if len(locs) < 2:
        return None
    gaps = sorted(b - a for a, b in zip(locs, locs[1:]))
    return gaps[len(gaps) // 2]


def limits(n_weeks: int) -> dict[str, Any]:
    """Default, slider range, quick choices and warning thresholds for a number of weeks."""
    return {
        "default": max(1, math.ceil(n_weeks / WEEKS_PER_KNOT)),
        "max": max(1, n_weeks // KNOTS_MAX_WEEKS),
        "amber": n_weeks / KNOTS_AMBER_WEEKS,
        "red": n_weeks / KNOTS_RED_WEEKS,
        "quick": [{"k": 1, "label": "flat"}, {"k": max(1, n_weeks // 26), "label": "half-yearly"},
                  {"k": max(1, n_weeks // 13), "label": "quarterly"}, {"k": max(1, n_weeks // 8), "label": "every 2 months"},
                  {"k": max(1, n_weeks // 4), "label": "monthly"}],
    }


def adstock(x: np.ndarray, max_lag: int, decay: float = PREVIEW_DECAY) -> np.ndarray:
    w = decay ** np.arange(max_lag + 1)
    w = w / w.sum()
    out = np.zeros(len(x))
    for i, wi in enumerate(w):
        out[i:] += wi * x[: len(x) - i]
    return out


def _std(a: np.ndarray) -> np.ndarray:
    sd = a.std(axis=0)
    return np.where(sd > 0, sd, 1.0)


def national(result) -> tuple[pd.DataFrame, list[str], int]:
    """The model table as one row per week: KPI and spend summed over regions,
    controls averaged. Returns (table, controls, number of regions)."""
    m = result.coord_to_columns()["coord_to_columns"]
    date, geo = m["time"], m.get("geo")
    t = result.table
    controls = [c for c, r in result.roles.items() if r in ("control", "non_media_treatment") and c in t.columns]
    sums = [m["kpi"]] + [c.spend for c in result.channels]
    if geo:
        g = t.groupby(date)
        out = pd.concat([g[sums].sum(), g[controls].mean()], axis=1).reset_index()
        return out.sort_values(date).reset_index(drop=True), controls, int(t[geo].nunique())
    return t[[date] + sums + controls].sort_values(date).reset_index(drop=True), controls, 1


def preview(result, cvs: dict[str, float], outlier_dates: list[str] | None = None) -> dict[str, Any]:
    """Everything step 24's chart and diagnostics need, for every K (needs google-meridian and scipy)."""
    from scipy.optimize import lsq_linear

    m = result.coord_to_columns()["coord_to_columns"]
    date, kpi = m["time"], m["kpi"]
    t, controls, n_geos = national(result)
    n = len(t)
    lim = limits(n)
    y = t[kpi].to_numpy(float)
    max_lag = int(result.config.get("max_lag", 8))

    C = t[controls].to_numpy(float) if controls else np.zeros((n, 0))
    C = (C - C.mean(axis=0)) / _std(C) if controls else C
    feats, ad = {}, {}
    for ch in result.channels:
        x = t[ch.spend].to_numpy(float)
        typical = x[x > 0].mean() if (x > 0).any() else 1.0
        ad[ch.name] = adstock(x, max_lag)
        feats[ch.name] = np.log1p(ad[ch.name] / typical)
    M = np.column_stack(list(feats.values())) if feats else np.zeros((n, 0))
    M = M / _std(M) if feats else M
    ym, ys = y.mean(), y.std() or 1.0
    yz = (y - ym) / ys
    burst = [c for c in feats if cvs.get(c, 0) > BURST_CV]

    fits, share, follow, follow_name = {}, [], [], []
    for K in range(1, lim["max"] + 1):
        info = knot_info(n, K, n_geos)
        W = np.asarray(info.weights, float).T            # weeks x K
        X = np.hstack([W, C, M])
        p = X.shape[1]
        A = np.vstack([X, math.sqrt(RIDGE_ALPHA) * np.eye(p)])
        b = np.r_[yz, np.zeros(p)]
        lower = np.r_[np.full(K + C.shape[1], -np.inf), np.zeros(M.shape[1])]
        sol = lsq_linear(A, b, bounds=(lower, np.full(p, np.inf)), method="bvls")
        levels = ym + ys * sol.x[:K]                     # each knot's level, in KPI units
        base = ym + ys * (W @ sol.x[:K])                 # intercept + knot part only
        media = ys * (M @ sol.x[K + C.shape[1]:]) if M.shape[1] else np.zeros(n)
        fitted = base + (ys * (C @ sol.x[K:K + C.shape[1]]) if C.shape[1] else 0) + media
        share.append(float(media.sum() / fitted.sum()) if fitted.sum() else 0.0)
        db = np.diff(base)
        per = {}
        if K > 1 and db.std() > 0:
            for c in burst:
                a = ad[c][1:]
                if a.std() > 0:
                    per[c] = float(np.corrcoef(db, a)[0, 1])
        name = max(per, key=per.get) if per else None
        follow.append(per[name] if name else None)
        follow_name.append(name)
        fits[K] = {"baseline": np.round(base, 2).tolist(), "locations": [int(i) for i in info.knot_locations],
                   "levels": np.round(levels, 2).tolist()}

    # Trend left out (shown with K = 1): KPI after controls, against time.
    X0 = np.column_stack([np.ones(n), C])
    resid = y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]
    tt = np.arange(n) - (n - 1) / 2
    slope = float(tt @ resid / (tt @ tt))
    e = resid - resid.mean() - slope * tt
    se = math.sqrt(float(e @ e) / (n - 2) / float(tt @ tt)) if n > 2 else math.inf
    trend_t = slope / se if se else 0.0

    weeks = [str(pd.Timestamp(d).date()) for d in t[date]]
    out_idx = [weeks.index(d) for d in (outlier_dates or []) if d in weeks]
    strips = {}
    for c in burst:
        spend = t[next(ch.spend for ch in result.channels if ch.name == c)].to_numpy(float)
        strips[c] = np.round(spend / spend.max(), 3).tolist() if spend.max() > 0 else [0.0] * n
    return {
        "weeks": weeks, "kpi": np.round(y, 2).tolist(), "kpi_name": kpi, "geos": n_geos, **lim,
        "fits": fits, "share": share, "follow": follow, "follow_name": follow_name,
        "follow_amber": FOLLOW_AMBER, "follow_red": FOLLOW_RED,
        "trend": {"t": float(trend_t), "slope": slope, "warn": abs(trend_t) > TREND_T},
        "strips": strips, "outliers": out_idx, "max_lag": max_lag,
    }


def suggest(result) -> dict[str, Any]:
    """Meridian's automatic knot selection (AKS) on the model data. Shown, never applied."""
    from meridian.model import knots as mk

    from .spec import build_input_data

    aks = mk.AKS(build_input_data(result)).automatic_knot_selection()
    locs = [int(k) for k in aks.knots]
    return {"knots": len(locs), "locations": locs}
