"""Everything Phase E asks of Meridian, in one place (needs google-meridian).

The rest of Phase E works with plain numbers and dicts from these helpers, so
the gates can be read and tested without a fitted model.
"""
from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np
import pandas as pd

from ..configure.spec import build_input_data, build_model_spec
from ..ingest.pairing import Channel

AGGREGATES = ("All Paid Channels", "All Channels")


def meridian_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("meridian") is not None


def variant(result, table: pd.DataFrame | None = None, channels: list[Channel] | None = None,
            roles: dict[str, str] | None = None):
    """A copy of a Phase D result with another table (a window of weeks, an extra column)."""
    return dataclasses.replace(result, table=result.table if table is None else table,
                               channels=result.channels if channels is None else channels,
                               roles=result.roles if roles is None else roles)


def window_config(config: dict[str, Any], weeks: pd.DatetimeIndex) -> dict[str, Any]:
    """The model config for a window of weeks: no holdout, and lift-test periods
    kept only where they fall inside the window (Meridian refuses others)."""
    cfg = {**config, "holdout": None}
    lo, hi = weeks.min(), weeks.max()
    cal = {}
    for ch, c in (config.get("calibration") or {}).items():
        periods = [p for p in c["periods"] if lo <= pd.Timestamp(p["start"]) and pd.Timestamp(p["end"]) <= hi]
        if periods:
            cal[ch] = {**c, "periods": periods}
    cfg["calibration"] = cal
    return cfg


def build(result, config: dict[str, Any]):
    from meridian.model import model

    data = build_input_data(result)
    return model.Meridian(input_data=data, model_spec=build_model_spec(config, data))


def sample_posterior(mmm, sampler: dict[str, int], seed: int) -> None:
    mmm.sample_posterior(n_chains=sampler["n_chains"], n_adapt=sampler["n_adapt"], n_burnin=sampler["n_burnin"],
                         n_keep=sampler["n_keep"], seed=seed)


def analyzer(mmm):
    from meridian.analysis import analyzer as an

    return an.Analyzer(mmm)


def uses_revenue(mmm) -> bool:
    """ROI in money per money: the KPI is revenue, or there is a revenue per KPI."""
    data = mmm.input_data
    return data.kpi_type == "revenue" or getattr(data, "revenue_per_kpi", None) is not None


def actual_kpi(mmm) -> np.ndarray:
    """National KPI per week."""
    return np.asarray(mmm.input_data.kpi.values, float).sum(axis=0)


def weeks(mmm) -> list[str]:
    return [str(t)[:10] for t in mmm.input_data.time.values]


def prior_total_kpi(mmm, confidence: float) -> tuple[float, float, float]:
    """The prior's (low, median, high) total KPI over the whole period."""
    draws = np.asarray(analyzer(mmm).expected_outcome(use_posterior=False, use_kpi=True), float).ravel()
    a = (1 - confidence) / 2
    return float(np.quantile(draws, a)), float(np.median(draws)), float(np.quantile(draws, 1 - a))


def rhat(mmm) -> dict[str, float]:
    """Largest R-hat per parameter (NaN where Meridian masks a parameter it doesn't use)."""
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k, v in analyzer(mmm).get_rhat().items():
            arr = np.asarray(v, float)
            if arr.size and not np.isnan(arr).all():
                out[k] = float(np.nanmax(arr))
    return out


def accuracy(mmm) -> dict[str, dict[str, float]]:
    """National R-squared and MAPE for Train, Test and All Data."""
    df = analyzer(mmm).predictive_accuracy(use_kpi=True).to_dataframe().reset_index()
    df = df[df["geo_granularity"] == "national"]
    out: dict[str, dict[str, float]] = {}
    for _, r in df.iterrows():
        out.setdefault(str(r["evaluation_set"]), {})[str(r["metric"])] = float(r["value"])
    return out


def fitted_kpi(mmm) -> np.ndarray:
    """Posterior mean expected KPI per week (national)."""
    draws = np.asarray(analyzer(mmm).expected_outcome(use_kpi=True, aggregate_times=False), float)
    return draws.reshape(-1, draws.shape[-1]).mean(axis=0)


def channel_metrics(mmm, confidence: float) -> pd.DataFrame:
    """Per paid channel: ROI (money if there is revenue, else KPI per unit of
    spend) and incremental KPI, each with prior and posterior mean, median and
    range."""
    an = analyzer(mmm)
    money = uses_revenue(mmm)
    roi = an.summary_metrics(confidence_level=confidence, use_kpi=not money)["roi"]
    inc = an.summary_metrics(confidence_level=confidence, use_kpi=True, include_non_paid_channels=True)
    rows = []
    for ch in [str(c) for c in inc.channel.values if str(c) not in AGGREGATES]:
        row = {"channel": ch, "paid": ch in [str(c) for c in roi.channel.values]}
        for dist in ("prior", "posterior"):
            if row["paid"]:
                r = roi.sel(channel=ch, distribution=dist)
                for m in ("mean", "median", "ci_lo", "ci_hi"):
                    row[f"roi_{dist}_{m}"] = float(r.sel(metric=m))
            row[f"inc_{dist}_mean"] = float(inc["incremental_outcome"].sel(channel=ch, distribution=dist, metric="mean"))
        row["share_of_kpi"] = float(inc["pct_of_contribution"].sel(channel=ch, distribution="posterior", metric="mean")) / 100
        rows.append(row)
    return pd.DataFrame(rows).set_index("channel")


def contributions(mmm, confidence: float) -> dict[str, Any]:
    """Baseline, channel contributions and the fitted total, in KPI units (posterior means)."""
    an = analyzer(mmm)
    base = an.baseline_summary_metrics(use_kpi=True, confidence_level=confidence)
    baseline = float(base["baseline_outcome"].sel(distribution="posterior", metric="mean"))
    share = float(base["pct_of_contribution"].sel(distribution="posterior", metric="mean")) / 100
    inc = an.summary_metrics(confidence_level=confidence, use_kpi=True, include_non_paid_channels=True)
    by_channel = {str(c): float(inc["incremental_outcome"].sel(channel=c, distribution="posterior", metric="mean"))
                  for c in inc.channel.values if str(c) not in AGGREGATES}
    total = float(np.asarray(an.expected_outcome(use_kpi=True), float).mean())
    error = (total - baseline - sum(by_channel.values())) / total if total else float("nan")
    return {"baseline": baseline, "baseline_share": share, "channels": by_channel, "fitted_total": total,
            "reconcile_error": float(error)}


def save(mmm, path) -> None:
    """save_mmm needs a path with a folder in it; Meridian marks it deprecated
    in favour of its schema serde, which needs extra packages, so the warning is silenced."""
    from meridian.model import model

    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model.save_mmm(mmm, str(path))


def load(path):
    from meridian.model import model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return model.load_mmm(str(path))


# What each Meridian parameter means, for step 30's plain-words list. {name} is the channel or control.
_STEMS = {
    "alpha": "{name} carryover", "ec": "{name} diminishing returns (half-saturation)",
    "slope": "{name} diminishing returns (slope)", "beta": "{name} effect", "eta": "{name} variation across regions",
    "roi": "{name} return", "mroi": "{name} return on the next rupee", "contribution": "{name} share of sales",
    "xi": "{name} variation across regions",
}
_WHOLE = {"knot_values": "baseline over time (knots)", "mu_t": "baseline over time (knots)",
          "tau_g": "regional baseline", "sigma": "unexplained noise"}
_SUFFIX = {"m": "media_channel", "rf": "rf_channel", "om": "organic_media_channel", "orf": "organic_rf_channel",
           "c": "control_variable", "n": "non_media_channel"}


def _words(param: str, mmm) -> tuple[str, list[str]]:
    """(template, names along the parameter's last axis), e.g. alpha_m -> ("{name} carryover",
    media channels); beta_gm (by region) -> ("{name} effect by region", media channels)."""
    if param in _WHOLE:
        return _WHOLE[param], []
    stem, _, suffix = param.rpartition("_")
    regional = False
    if suffix.startswith("g") and suffix[1:] in _SUFFIX:          # beta_gm, gamma_gc, gamma_gn
        suffix, regional = suffix[1:], True
    if not stem or suffix not in _SUFFIX:
        return param + " {name}", []
    coord = getattr(mmm.input_data, _SUFFIX[suffix], None)
    names = [str(x) for x in coord.values] if coord is not None else []
    if stem == "gamma":
        template = "control effects ({name})" if suffix == "c" else "{name} effect (non-media lever)"
    else:
        template = _STEMS.get(stem, "{name} " + stem)
    return template + (" by region" if regional and stem != "gamma" else ""), names


def rhat_detail(mmm, limit: float) -> list[dict[str, Any]]:
    """Every parameter element above the limit, in plain words: [{text, parameter, rhat}]. NaN is skipped."""
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k, v in analyzer(mmm).get_rhat().items():
            a = np.asarray(v, float)
            if not a.size or np.isnan(a).all():
                continue
            template, names = _words(k, mmm)
            per = np.nanmax(a.reshape(-1, a.shape[-1]), axis=0) if a.ndim >= 1 and names and a.shape[-1] == len(names) else None
            if per is not None:
                for name, r in zip(names, per):
                    if r == r and r > limit:
                        out.append({"text": f"{template.format(name=name)} ({k})", "parameter": k, "rhat": float(r)})
            elif np.nanmax(a) > limit:
                out.append({"text": f"{template.format(name='all')} ({k})", "parameter": k, "rhat": float(np.nanmax(a))})
    return sorted(out, key=lambda x: -x["rhat"])
