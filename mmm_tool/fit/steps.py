"""Steps 28-36: fit the model, then check it before anyone reads its numbers.

28 Prior check       does the prior imply believable sales? (warning)
29 Fit with holdout  the posterior, with the last weeks held back
30 Convergence       R-hat below 1.1 for every parameter; one rerun with
                     twice the draws, then the gate fails (gate)
31 Holdout accuracy  test MAPE below 10% and better than a seasonal-naive
                     forecast (gate)
32 Plausibility      ROI 0.05-20, baseline 20-95% of the KPI, contributions
                     reconcile with the fit (gate)
33 Data vs prior     ROI range at least 30% narrower than the prior's, or the
                     channel is badged "prior-driven"
34 Stability         optional: refits that drop the first or last 26 weeks;
                     ROI varying more than 0.3 (CV) badges "unstable"
35 Placebo           optional: a random series shaped like real spend should
                     earn nothing (warning)
36 Final fit         refit on every week and save, if every gate passed

Thresholds are in fit.yaml.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..configure.context import DEFAULT_ROI_MU, DEFAULT_ROI_SIGMA, lognormal_summary
from ..ingest.pairing import Channel
from ..ingest.types import StepResult, Verdict
from . import gates
from . import model as m
from .context import ECtx

NAMES = {28: "Prior check", 29: "Fit with holdout", 30: "Convergence", 31: "Holdout accuracy", 32: "Plausibility",
         33: "Did the data teach anything?", 34: "Stability", 35: "Placebo channel", 36: "Final fit and save"}
GATES = (30, 31, 32)


def _step(n: int, verdict: Verdict, summary: str, details: dict | None = None, warnings: list[str] | None = None) -> StepResult:
    return StepResult(n, NAMES[n], verdict, summary, details or {}, warnings or [])


def _fmt_sampler(s: dict) -> str:
    return f"{s['n_chains']} chains, {s['n_adapt']} adapt, {s['n_burnin']} burn-in, {s['n_keep']} kept"


def _fit(ctx: ECtx, mmm, sampler: dict, label: str) -> float:
    t = time.time()
    m.sample_posterior(mmm, sampler, ctx.seed)
    ctx.timings[label] = round(time.time() - t, 1)
    return ctx.timings[label]


def prior_check(ctx: ECtx) -> StepResult:
    ctx.progress(28, "Sampling the prior", 0.0)
    ctx.mmm = m.build(ctx.d, ctx.config)
    ctx.mmm.sample_prior(int(ctx.cfg["prior_draws"]), seed=ctx.seed)
    lo, med, hi = m.prior_total_kpi(ctx.mmm, ctx.confidence)
    actual = float(m.actual_kpi(ctx.mmm).sum())
    res = gates.prior_check(lo, hi, actual, float(ctx.cfg["prior_check"]["max_width"]))
    details = {"actual_total": actual, "prior_low": lo, "prior_median": med, "prior_high": hi, **res}
    rng = f"{lo:,.0f} to {hi:,.0f}"
    if res["ok"]:
        return _step(28, Verdict.PASS, f"The prior's 90% range of total KPI, {rng}, covers the actual {actual:,.0f} "
                     f"and is {res['width_ratio']:.1f}x as wide.", details)
    why = []
    if not res["covers"]:
        why.append(f"misses the actual {actual:,.0f}")
    if res["width_ratio"] > ctx.cfg["prior_check"]["max_width"]:
        why.append(f"is {res['width_ratio']:.0f}x as wide as actual sales")
    summary = f"The prior's 90% range of total KPI, {rng}, " + " and ".join(why) + "."
    warning = ("Before seeing the data, the model expects sales it shouldn't. Check the ROI priors (step 25) "
               "and lift tests (step 26); the fit can still correct this.")
    return _step(28, Verdict.WARN, summary, details, [warning])


def fit_holdout(ctx: ECtx) -> StepResult:
    hold = ctx.config.get("holdout")
    ctx.progress(29, f"Fitting ({_fmt_sampler(ctx.sampler)})", 0.0)
    secs = _fit(ctx, ctx.mmm, ctx.sampler, "fit_holdout")
    where = f"weeks {hold['start']} to {hold['end']} held back" if hold else "no holdout"
    return _step(29, Verdict.PASS, f"Fitted with {_fmt_sampler(ctx.sampler)}; {where}; {secs:,.0f} s.",
                 {"sampler": ctx.sampler, "seconds": secs, "holdout": hold})


def convergence(ctx: ECtx) -> StepResult:
    limit = float(ctx.cfg["convergence"]["rhat_max"])
    res = gates.converged(m.rhat(ctx.mmm), limit)
    details = {"rhat_max": limit, "first": res, "rerun": None}
    if not res["ok"]:
        f = int(ctx.cfg["convergence"]["rerun_factor"])
        more = {**ctx.sampler, **{k: ctx.sampler[k] * f for k in ("n_adapt", "n_burnin", "n_keep")}}
        ctx.progress(30, f"R-hat {res['worst']['rhat']:.2f} on {res['worst']['parameter']}: refitting with {_fmt_sampler(more)}", 0.0)
        ctx.mmm = m.build(ctx.d, ctx.config)
        ctx.mmm.sample_prior(int(ctx.cfg["prior_draws"]), seed=ctx.seed)
        _fit(ctx, ctx.mmm, more, "fit_rerun")
        ctx.sampler = more
        res = gates.converged(m.rhat(ctx.mmm), limit)
        details["rerun"] = {"sampler": more, **res}
    ctx.gates[30] = res["ok"]
    worst = res["worst"]
    again = " after one rerun with twice the draws" if details["rerun"] else ""
    if res["ok"]:
        return _step(30, Verdict.PASS, f"Every R-hat is below {limit}{again} (largest {worst['rhat']:.3f}, {worst['parameter']}).", details)
    bad = ", ".join(f"{k} {v:.2f}" for k, v in sorted(res["bad"].items(), key=lambda kv: -kv[1] if kv[1] == kv[1] else 0)[:5])
    return _step(30, Verdict.FAIL, f"Failed: the chains disagree{again} (R-hat above {limit}: {bad}).", details,
                 ["The sampler hasn't settled, so none of this fit's numbers can be trusted. Try the full sampler, "
                  "fewer knots (step 24), or tighter ROI priors (steps 25-26)."])


def holdout_accuracy(ctx: ECtx) -> StepResult:
    hold = ctx.config.get("holdout")
    acc = m.accuracy(ctx.mmm)
    week_list = m.weeks(ctx.mmm)
    actual = m.actual_kpi(ctx.mmm)
    fitted = m.fitted_kpi(ctx.mmm)
    test = np.array([bool(hold) and hold["start"] <= w <= hold["end"] for w in week_list])
    chart = {"weeks": week_list, "actual": actual.tolist(), "fitted": fitted.tolist(), "test": test.tolist()}
    if not hold or "Test" not in acc:
        ctx.gates[31] = False
        return _step(31, Verdict.FAIL, "Failed: there is no holdout to test the model on (step 27).",
                     {"accuracy": acc, "chart": chart})
    cfg = ctx.cfg["holdout"]
    naive = gates.seasonal_naive_mape(actual, test, int(cfg["season_weeks"]))
    t_mape, t_r2 = acc["Test"]["MAPE"], acc["Test"].get("R_Squared")
    res = gates.holdout_ok(t_mape, naive, float(cfg["test_mape_max"]))
    ctx.gates[31] = res["ok"]
    details = {"accuracy": acc, "naive_mape": naive, "test_mape_max": cfg["test_mape_max"], **res, "chart": chart}
    head = (f"Test MAPE {t_mape:.1%} (R² {t_r2:.2f}) on {int(test.sum())} held-back weeks; "
            f"train MAPE {acc['Train']['MAPE']:.1%}; "
            + (f"same-week-last-year forecast {naive:.1%}" if naive == naive
               else "no same-week-last-year forecast (under a year of history before the test weeks)"))
    if res["ok"]:
        return _step(31, Verdict.PASS, head + ".", details)
    why = []
    if not res["under_max"]:
        why.append(f"test MAPE is not below {cfg['test_mape_max']:.0%}")
    if not res["beats_naive"]:
        why.append("it doesn't beat repeating last year's sales")
    return _step(31, Verdict.FAIL, f"Failed: {' and '.join(why)}. {head}.", details,
                 ["A model that can't predict weeks it hasn't seen can't be trusted to split sales between channels."])


def plausibility(ctx: ECtx) -> StepResult:
    cfg = ctx.cfg["plausibility"]
    metrics = m.channel_metrics(ctx.mmm, ctx.confidence)
    ctx.metrics = metrics
    money = m.uses_revenue(ctx.mmm)
    paid = metrics[metrics["paid"]]
    roi = paid["roi_posterior_mean"].to_dict()
    con = m.contributions(ctx.mmm, ctx.confidence)
    res = gates.plausible(roi, con["baseline_share"], con["reconcile_error"], roi_min=float(cfg["roi_min"]),
                          roi_max=float(cfg["roi_max"]), share_range=tuple(cfg["baseline_share"]),
                          tolerance=float(cfg["reconcile_tolerance"]), roi_in_money=money)
    ctx.gates[32] = res["ok"]
    details = {"roi": roi, "roi_in_money": money, "contributions": con, **res,
               "limits": {"roi": [cfg["roi_min"], cfg["roi_max"]], "baseline_share": cfg["baseline_share"]}}
    warnings = []
    if not money:
        warnings.append("ROI is in KPI units per ₹ (no revenue per KPI, step 5), so the 0.05-20 ROI range isn't checked.")
    lo, hi = cfg["baseline_share"]
    parts = [f"baseline {con['baseline_share']:.0%} of the KPI",
             f"contributions reconcile with the fit to {abs(con['reconcile_error']):.2%}"]
    if money:
        parts.insert(0, f"ROI {min(roi.values()):.2f} to {max(roi.values()):.2f}")
    if res["ok"]:
        text = "; ".join(parts)
        return _step(32, Verdict.WARN if warnings else Verdict.PASS, text[0].upper() + text[1:] + ".", details, warnings)
    why = []
    if res["roi_out_of_range"]:
        why.append("ROI outside " + f"{cfg['roi_min']}-{cfg['roi_max']}: "
                   + ", ".join(f"{c} {v:.2f}" for c, v in res["roi_out_of_range"].items()))
    if not res["baseline_ok"]:
        why.append(f"baseline is {con['baseline_share']:.0%} of the KPI (expected {lo:.0%}-{hi:.0%})")
    if not res["reconciles"]:
        why.append(f"baseline plus contributions miss the fit by {con['reconcile_error']:.2%}")
    return _step(32, Verdict.FAIL, "Failed: " + "; ".join(why) + ".", details, warnings)


def prior_driven(ctx: ECtx) -> StepResult:
    need = float(ctx.cfg["prior_driven"]["min_narrowing"])
    paid = ctx.metrics[ctx.metrics["paid"]]
    rows, driven = [], []
    for ch, r in paid.iterrows():
        k = gates.narrowing((r["roi_prior_ci_lo"], r["roi_prior_ci_hi"]), (r["roi_posterior_ci_lo"], r["roi_posterior_ci_hi"]))
        rows.append({"channel": ch, "narrowing": k, "prior": [r["roi_prior_ci_lo"], r["roi_prior_ci_hi"]],
                     "posterior": [r["roi_posterior_ci_lo"], r["roi_posterior_ci_hi"]]})
        if k < need:
            driven.append(ch)
            ctx.badge(ch, "prior_driven")
    details = {"channels": rows, "min_narrowing": need, "prior_driven": driven}
    if not driven:
        return _step(33, Verdict.PASS, f"Every channel's ROI range is at least {need:.0%} narrower than its prior.", details)
    return _step(33, Verdict.WARN, f"{len(driven)} channel(s) badged prior-driven: {', '.join(driven)}.", details,
                 [f"'{c}': the data narrowed its ROI range by only "
                  f"{next(r['narrowing'] for r in rows if r['channel'] == c):.0%}; its ROI mostly repeats the prior. "
                  "A lift test (step 26) would anchor it." for c in driven])


def _date_col(ctx: ECtx) -> str:
    return ctx.d.coord_to_columns()["coord_to_columns"]["time"]


def stability(ctx: ECtx) -> StepResult:
    if not ctx.options.stability:
        return _step(34, Verdict.PASS, "Not run. Turn on the stability check for three extra fits.", {"skipped": True})
    cfg = ctx.cfg["stability"]
    drop, min_weeks = int(cfg["drop_weeks"]), int(cfg["min_weeks"])
    date = _date_col(ctx)
    all_weeks = pd.DatetimeIndex(sorted(ctx.d.table[date].unique()))
    windows = {f"without the first {drop} weeks": all_weeks[drop:], f"without the last {drop} weeks": all_weeks[:-drop],
               f"without the first and last {drop} weeks": all_weeks[drop:-drop]}
    paid = list(ctx.metrics[ctx.metrics["paid"]].index)
    rois = {"all weeks (step 29)": ctx.metrics.loc[paid, "roi_posterior_mean"].to_dict()}
    skipped = []
    for i, (label, wk) in enumerate(windows.items()):
        if len(wk) < min_weeks:
            skipped.append(f"{label} ({len(wk)} weeks, under {min_weeks})")
            continue
        ctx.progress(34, f"Refit {i + 1} of {len(windows)}: {label}", i / len(windows))
        part = m.variant(ctx.d, table=ctx.d.table[ctx.d.table[date].isin(wk)].reset_index(drop=True))
        mmm = m.build(part, m.window_config(ctx.config, wk))
        mmm.sample_prior(int(ctx.cfg["prior_draws"]), seed=ctx.seed)
        _fit(ctx, mmm, ctx.sampler, f"stability_{i + 1}")
        rois[label] = m.channel_metrics(mmm, ctx.confidence).loc[paid, "roi_posterior_mean"].to_dict()
    limit = float(cfg["cv_max"])
    table = [{"channel": ch, "roi": {k: v[ch] for k, v in rois.items()}, "cv": gates.cv([v[ch] for v in rois.values()])}
             for ch in paid]
    unstable = [r["channel"] for r in table if not (r["cv"] < limit)]
    for ch in unstable:
        ctx.badge(ch, "unstable")
    details = {"windows": list(rois), "channels": table, "cv_max": limit, "unstable": unstable, "skipped": skipped}
    warnings = [f"Window skipped: {s}." for s in skipped]
    if len(rois) < 2:
        return _step(34, Verdict.WARN, "No window long enough to refit.", details, warnings)
    if not unstable:
        return _step(34, Verdict.WARN if warnings else Verdict.PASS,
                     f"ROI holds across {len(rois)} fits: every channel's CV is below {limit}.", details, warnings)
    warnings += [f"'{r['channel']}' ROI moves with the weeks used (CV {r['cv']:.2f}): "
                 + ", ".join(f"{v:.2f}" for v in r["roi"].values()) for r in table if r["channel"] in unstable]
    return _step(34, Verdict.WARN, f"{len(unstable)} channel(s) badged unstable: {', '.join(unstable)}.", details, warnings)


def placebo(ctx: ECtx) -> StepResult:
    if not ctx.options.placebo:
        return _step(35, Verdict.PASS, "Not run. Turn on the placebo check for one extra fit.", {"skipped": True})
    name = ctx.cfg["placebo"]["name"]
    media = [c for c in ctx.d.channels if c.kind == "media"]
    if not media:
        return _step(35, Verdict.PASS, "Not run: there is no media channel to shape a placebo on.", {"skipped": True})
    table = ctx.d.table.copy()
    totals = {c.name: float(table[c.spend].sum()) for c in media}
    model_on = sorted(totals, key=totals.get)[len(totals) // 2]  # the median channel by spend
    src = next(c for c in media if c.name == model_on)
    col = f"{name}_spend"
    rng = np.random.default_rng(ctx.seed)
    geo = ctx.d.coord_to_columns()["coord_to_columns"].get("geo")
    groups = table.groupby(geo).groups.values() if geo else [table.index]
    table[col] = 0.0
    for idx in groups:  # same weekly amounts, random weeks: shaped like real spend, unrelated to sales
        table.loc[idx, col] = rng.permutation(table.loc[idx, src.spend].to_numpy())
    channel = Channel(name=name, kind="media", spend=col, exposure=col, exposure_is_spend=True)
    variant = m.variant(ctx.d, table=table, channels=[*ctx.d.channels, channel], roles={**ctx.d.roles, col: "media_spend"})
    cfg = {**ctx.config, "holdout": None, "adstock": {**ctx.config["adstock"], name: "geometric"},
           "roi_priors": {**ctx.config["roi_priors"], name: {**lognormal_summary(DEFAULT_ROI_MU, DEFAULT_ROI_SIGMA), "source": "default"}}}
    ctx.progress(35, f"Fitting with a placebo shaped like {model_on}", 0.0)
    mmm = m.build(variant, cfg)
    mmm.sample_prior(int(ctx.cfg["prior_draws"]), seed=ctx.seed)
    _fit(ctx, mmm, ctx.sampler, "placebo")
    r = m.channel_metrics(mmm, ctx.confidence).loc[name]
    lo, hi = r["roi_posterior_ci_lo"], r["roi_posterior_ci_hi"]
    credited = gates.placebo_credited(r["roi_prior_median"], r["roi_posterior_median"], lo, float(ctx.cfg["plausibility"]["roi_min"]))
    details = {"shaped_like": model_on, "roi_posterior": [lo, r["roi_posterior_median"], hi],
               "roi_prior": [r["roi_prior_ci_lo"], r["roi_prior_median"], r["roi_prior_ci_hi"]], "credited": credited}
    head = (f"A placebo with {model_on}'s weekly amounts in random weeks gets ROI {r['roi_posterior_median']:.2f} "
            f"(90% {lo:.2f}-{hi:.2f}; prior median {r['roi_prior_median']:.2f})")
    if not credited:
        return _step(35, Verdict.PASS, head + ": not credited.", details)
    return _step(35, Verdict.WARN, head + ": credited.", details,
                 ["The model credits a random series with sales above what its prior expects, so real channels' "
                  "ROI may be inflated the same way. Tighter priors or lift tests (steps 25-26) help."])


def data_version(table: pd.DataFrame) -> str:
    return hashlib.sha256(table.to_csv(index=False).encode()).hexdigest()[:12]


def final_fit(ctx: ECtx) -> StepResult:
    failed = [n for n in GATES if not ctx.gates.get(n, False)]
    if failed:
        return _step(36, Verdict.WARN, f"Not run: the model failed step(s) {', '.join(map(str, failed))}, so nothing is saved.",
                     {"skipped": True, "failed_gates": failed})
    ctx.progress(36, "Refitting on every week", 0.0)
    cfg = {**ctx.config, "holdout": None}
    mmm = m.build(ctx.d, cfg)
    mmm.sample_prior(int(ctx.cfg["prior_draws"]), seed=ctx.seed)
    secs = _fit(ctx, mmm, ctx.sampler, "final")
    res = gates.converged(m.rhat(mmm), float(ctx.cfg["convergence"]["rhat_max"]))
    import meridian

    version = data_version(ctx.d.table)
    info = {"data_version": version, "weeks": len(m.weeks(mmm)), "sampler": ctx.sampler, "seconds": secs,
            "rhat_worst": res["worst"], "meridian": getattr(meridian, "__version__", "?"),
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "path": None}
    if ctx.out_dir is not None:
        path = ctx.out_dir / "model.pkl"
        ctx.progress(36, "Saving the model", 0.9)
        m.save(mmm, path)
        info["path"] = str(path)
        meta = {**info, "config": ctx.config, "gates": {str(k): v for k, v in ctx.gates.items()}, "badges": ctx.badges,
                "options": ctx.options.to_dict()}
        (ctx.out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    ctx.model = info
    ctx.final = mmm
    warnings = [] if res["ok"] else [f"The final fit's largest R-hat is {res['worst']['rhat']:.2f} ({res['worst']['parameter']}); "
                                     "refit with the full sampler before relying on it."]
    where = f"saved as {info['path']}" if info["path"] else "not saved (no folder given)"
    return _step(36, Verdict.WARN if warnings else Verdict.PASS,
                 f"Refitted on all {info['weeks']} weeks in {secs:,.0f} s; {where}; data version {version}.", info, warnings)


STEPS = [prior_check, fit_holdout, convergence, holdout_accuracy, plausibility, prior_driven, stability, placebo, final_fit]
