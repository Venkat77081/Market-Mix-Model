"""Step 24's baseline preview: what a number of knots does, before choosing it.

Knot placement and weights always come from Meridian (knots.get_knot_info),
never from a copy of its rules. For every number of knots K the preview fits

    KPI_t = knots_t (weights.T @ levels) + controls_t + sum_c beta_c * x_c,t

where x_c = log1p(adstock of channel c's spend / its typical spending week),
centred on its own mean. Knot levels are not penalised (Meridian's knot prior
is very wide); controls and media get a light ridge, and media coefficients
are held at zero or above. Because media and controls are centred, the knots
own the average level: the split of that level between the baseline and
always-on media can't be learned from the data (only Meridian's ROI priors
settle it), so the preview doesn't try. The chart's baseline is the knot part.
It is a preview: Meridian estimates the real baseline in step 29.

Three cards show the trade-off, never a fit error (more knots always fit the
sales line better):
  1 media's share of the explained ups and downs   a guide, no colour
  2 burst credit kept    the burst channels' total contribution at K, against
                         a straight-line baseline (K = 2); green / amber / red
  3 trend left out       with K = 1, a linear trend in the KPI after controls
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .context import KNOTS_AMBER_WEEKS, KNOTS_MAX_WEEKS, KNOTS_RED_WEEKS, WEEKS_PER_KNOT

RIDGE_ALPHA = 1.0          # on standardised controls and media; knot levels are not penalised
PREVIEW_DECAY = 0.5        # geometric adstock decay used by the preview only
BURST_CV = 0.5             # step 16 spend CV at or above this...
BURST_MIN_SPEND = 0.05     # ...and at least this share of total paid spend: a burst channel
CREDIT_GOOD, CREDIT_OK = 0.90, 0.75   # card 2: green at 90%+, amber 75-90%, red below
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


def burst_group(result, cvs: dict[str, float], spend: dict[str, float] | None = None) -> list[str]:
    """Burst channels: spend CV >= 0.5 in step 16 and at least 5% of total paid spend."""
    if spend is None:
        spend = {c.name: float(result.table[c.spend].sum()) for c in result.channels}
    total = sum(spend.values()) or 1.0
    return [c for c in spend if cvs.get(c, 0) >= BURST_CV and spend[c] / total >= BURST_MIN_SPEND]


def tone(kept: float | None) -> str | None:
    if kept is None:
        return None
    return "good" if kept >= CREDIT_GOOD else "warn" if kept >= CREDIT_OK else "fail"


@dataclass
class Design:
    """The preview regression's inputs, built once and fitted for any K."""
    n: int
    n_geos: int
    y_mean: float
    y_sd: float
    yz: np.ndarray        # standardised KPI
    C: np.ndarray         # controls, centred and standardised
    M: np.ndarray         # media, standardised, NOT centred (for contributions)
    Mc: np.ndarray        # media, centred (what the regression sees)
    channels: list[str]
    group: list[str]
    adstocked: dict[str, np.ndarray]

    @classmethod
    def build(cls, result, cvs: dict[str, float]) -> "Design":
        m = result.coord_to_columns()["coord_to_columns"]
        t, controls, n_geos = national(result)
        n = len(t)
        y = t[m["kpi"]].to_numpy(float)
        max_lag = int(result.config.get("max_lag", 8))
        C = t[controls].to_numpy(float) if controls else np.zeros((n, 0))
        if controls:
            C = (C - C.mean(axis=0)) / _std(C)
        names, feats, ad, spend = [], [], {}, {}
        for ch in result.channels:
            x = t[ch.spend].to_numpy(float)
            typical = x[x > 0].mean() if (x > 0).any() else 1.0
            ad[ch.name] = adstock(x, max_lag)
            feats.append(np.log1p(ad[ch.name] / typical))
            names.append(ch.name)
            spend[ch.name] = float(x.sum())
        M = np.column_stack(feats) if feats else np.zeros((n, 0))
        if feats:
            M = M / _std(M)
        sd = float(y.std()) or 1.0
        return cls(n, n_geos, float(y.mean()), sd, (y - y.mean()) / sd, C, M, M - M.mean(axis=0) if feats else M,
                   names, burst_group(result, cvs, spend), ad)

    def fit(self, K: int) -> dict[str, Any]:
        from scipy.optimize import lsq_linear

        info = knot_info(self.n, K, self.n_geos)
        W = np.asarray(info.weights, float).T                 # weeks x K; rows sum to 1, so the knots hold the level
        nc, nm = self.C.shape[1], self.Mc.shape[1]
        X = np.hstack([W, self.C, self.Mc])
        pen = np.r_[np.zeros(K), np.full(nc + nm, math.sqrt(RIDGE_ALPHA))]
        A = np.vstack([X, np.diag(pen)])
        b = np.r_[self.yz, np.zeros(len(pen))]
        lower = np.r_[np.full(K + nc, -np.inf), np.zeros(nm)]
        x = lsq_linear(A, b, bounds=(lower, np.full(len(pen), np.inf)), method="bvls").x
        levels = self.y_mean + self.y_sd * x[:K]
        base = W @ levels                                      # intercept + knot part (weights sum to 1)
        controls = self.y_sd * (self.C @ x[K:K + nc]) if nc else np.zeros(self.n)
        beta = x[K + nc:]
        media_c = self.y_sd * (self.Mc @ beta) if nm else np.zeros(self.n)
        fitted = base + controls + media_c
        # Card 1: share of the explained ups and downs, as a covariance split of the
        # fitted values' variance: Cov(part, fitted) / Var(fitted). The baseline,
        # controls and media shares add up to exactly 1.
        vf = float(fitted.var())
        ups = float(np.cov(media_c, fitted, bias=True)[0, 1] / vf) if vf > 0 else 0.0
        # Card 2: each channel's uncentred contribution, beta x transformed spend summed over
        # weeks. Spend is the same at every K, so the ratio between two K is the ratio of betas.
        contrib = {c: float(self.y_sd * beta[j] * self.M[:, j].sum()) for j, c in enumerate(self.channels)}
        return {"baseline": np.round(base, 2).tolist(), "locations": [int(i) for i in info.knot_locations],
                "levels": np.round(levels, 2).tolist(), "controls_mean": float(controls.mean()),
                "media_ups_downs": ups, "contrib": contrib}


def _reference_k(max_k: int, trend_warn: bool) -> int | None:
    """K = 2 (a straight-line baseline), or the smallest K that doesn't trigger the trend warning."""
    if max_k >= 2:
        return 2
    return None if trend_warn else 1


def trend(design: Design) -> dict[str, Any]:
    """With a flat baseline: a linear trend in the KPI after controls."""
    n, C = design.n, design.C
    y = design.y_mean + design.y_sd * design.yz
    X0 = np.column_stack([np.ones(n), C])
    resid = y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]
    tt = np.arange(n) - (n - 1) / 2
    slope = float(tt @ resid / (tt @ tt))
    e = resid - resid.mean() - slope * tt
    se = math.sqrt(float(e @ e) / (n - 2) / float(tt @ tt)) if n > 2 else math.inf
    t_stat = slope / se if se else 0.0
    return {"t": float(t_stat), "slope": slope, "warn": abs(t_stat) > TREND_T}


def credit(fit: dict[str, Any], ref: dict[str, Any] | None, group: list[str]) -> dict[str, Any]:
    """Card 2 for one K: the group's credit kept against the reference K, and each channel's."""
    if not group or ref is None:
        return {"kept": None, "tone": None, "by_channel": {}}
    total_ref = sum(ref["contrib"][c] for c in group)
    by = {c: (fit["contrib"][c] / ref["contrib"][c] if ref["contrib"][c] > 0 else None) for c in group}
    kept = sum(fit["contrib"][c] for c in group) / total_ref if total_ref > 0 else None
    return {"kept": kept, "tone": tone(kept), "by_channel": by}


def _entry(fit: dict[str, Any], ref: dict[str, Any] | None, group: list[str]) -> dict[str, Any]:
    out = {k: fit[k] for k in ("baseline", "locations", "levels", "controls_mean", "media_ups_downs")}
    return {**out, **credit(fit, ref, group)}


def preview(result, cvs: dict[str, float], outlier_dates: list[str] | None = None) -> dict[str, Any]:
    """Everything step 24's chart and cards need, for every K from 1 to weeks/4 (needs google-meridian and scipy)."""
    m = result.coord_to_columns()["coord_to_columns"]
    t, _, _ = national(result)
    design = Design.build(result, cvs)
    lim = limits(design.n)
    tr = trend(design)
    fits = {K: design.fit(K) for K in range(1, lim["max"] + 1)}
    ref_k = _reference_k(lim["max"], tr["warn"])
    ref = fits.get(ref_k) if ref_k else None
    weeks = [str(pd.Timestamp(d).date()) for d in t[m["time"]]]
    strips = {}
    for c in design.group:
        spend = t[next(ch.spend for ch in result.channels if ch.name == c)].to_numpy(float)
        strips[c] = np.round(spend / spend.max(), 3).tolist() if spend.max() > 0 else [0.0] * design.n
    return {
        "weeks": weeks, "kpi": np.round(design.y_mean + design.y_sd * design.yz, 2).tolist(), "kpi_name": m["kpi"],
        "geos": design.n_geos, **lim, "fits": {K: _entry(f, ref, design.group) for K, f in fits.items()},
        "group": design.group, "ref_k": ref_k, "credit_good": CREDIT_GOOD, "credit_ok": CREDIT_OK,
        "burst_cv": BURST_CV, "burst_min_spend": BURST_MIN_SPEND,
        "group_adstock": np.round(sum((design.adstocked[c] for c in design.group), np.zeros(design.n)), 1).tolist(),
        "trend": tr, "strips": strips, "outliers": [weeks.index(d) for d in (outlier_dates or []) if d in weeks],
        "max_lag": int(result.config.get("max_lag", 8)),
    }


def suggest(result, cvs: dict[str, float]) -> dict[str, Any]:
    """Meridian's automatic knot selection (AKS), with the preview for that K so it can be
    previewed and applied like any other. Shown, never applied on its own."""
    from meridian.model import knots as mk

    from .spec import build_input_data

    aks = mk.AKS(build_input_data(result)).automatic_knot_selection()
    locs = [int(k) for k in aks.knots]
    design = Design.build(result, cvs)
    k = len(locs)
    ref_k = _reference_k(limits(design.n)["max"], trend(design)["warn"])
    ref = design.fit(ref_k) if ref_k else None
    fit = _entry(design.fit(k), ref, design.group) if 1 <= k < design.n else None
    return {"knots": k, "locations": locs, "fit": fit}
