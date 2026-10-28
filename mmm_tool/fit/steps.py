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
from typing import Any

import numpy as np
import pandas as pd

from ..configure.choices import meridian_defaults, prior_summary
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
    """sample_posterior with the run's seed; every fit's time goes into this machine's log (estimate.py)."""
    from . import estimate

    t = time.time()
    m.sample_posterior(mmm, sampler, ctx.seed)
    secs = time.time() - t
    ctx.timings[label] = round(secs, 1)
    try:
        estimate.record(*estimate.model_size(mmm), sampler, secs)
    except OSError:
        pass                                        # a read-only home folder never stops a fit
    return ctx.timings[label]


def _retry_sampler(sampler: dict) -> dict:
    """Step 30's one automatic retry: twice the adapt and burn-in; same chains, kept draws and seed."""
    return {**sampler, "label": f"{sampler.get('label', 'Custom')}, longer warm-up",
            "n_adapt": 2 * int(sampler["n_adapt"]), "n_burnin": 2 * int(sampler["n_burnin"])}


def _about(ctx: ECtx, mmm, sampler: dict) -> str:
    from . import estimate

    try:
        e = estimate.estimate(*estimate.model_size(mmm), sampler)
    except Exception:  # noqa: BLE001 - an estimate is a courtesy
        e = None
    return f" (about {max(1, round(e['seconds'] / 60))} min)" if e else ""


def _fit_converged(ctx: ECtx, build, sampler: dict, label: str, step: int) -> tuple[Any, list[dict], dict]:
    """Fit; if the largest R-hat (NaN ignored) is over the limit, refit once with a longer warm-up.
    Returns (model, attempts, final check)."""
    limit = float(ctx.cfg["convergence"]["rhat_max"])
    attempts, mmm = [], None
    for i, smp in enumerate((sampler, _retry_sampler(sampler))):
        if i:
            ctx.progress(step, "Retrying with longer warm-up" + _about(ctx, mmm, smp), 0.5)
        mmm = build()
        mmm.sample_prior(int(ctx.cfg["prior_draws"]), seed=ctx.seed)
        secs = _fit(ctx, mmm, smp, label if not i else label + "_retry")
        res = gates.converged(m.rhat(mmm), limit)
        attempts.append({"attempt": i + 1, "sampler": smp, "rhat": res["worst"]["rhat"],
                         "parameter": res["worst"]["parameter"], "seconds": secs, "ok": res["ok"]})
        if res["ok"]:
            break
    res["above"] = [] if res["ok"] else m.rhat_detail(mmm, limit)
    return mmm, attempts, res


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


SUGGESTIONS = [
    "Use more chains and more kept draws (Custom sampler, step 29).",
    "Use fewer knots (step 24).",
    "Combine channels that move together (step 17).",
    "Set tighter priors for prior-driven channels (steps 23 and 25).",
]


def convergence(ctx: ECtx) -> StepResult:
    """Step 30: the largest R-hat over all parameters (NaN ignored) must be under the limit. Above it,
    one automatic retry with a longer warm-up; then the step fails and steps 31+ wait for the user."""
    limit = float(ctx.cfg["convergence"]["rhat_max"])
    first = gates.converged(m.rhat(ctx.mmm), limit)
    attempts = [{"attempt": 1, "sampler": ctx.sampler, "rhat": first["worst"]["rhat"], "parameter": first["worst"]["parameter"],
                 "seconds": ctx.timings.get("fit_holdout"), "ok": first["ok"]}]
    res = {**first, "above": []}
    if not first["ok"]:
        smp = _retry_sampler(ctx.sampler)
        ctx.progress(30, "Retrying with longer warm-up" + _about(ctx, ctx.mmm, smp), 0.0)
        ctx.mmm = m.build(ctx.d, ctx.config)
        ctx.mmm.sample_prior(int(ctx.cfg["prior_draws"]), seed=ctx.seed)
        secs = _fit(ctx, ctx.mmm, smp, "fit_holdout_retry")
        res = gates.converged(m.rhat(ctx.mmm), limit)
        attempts.append({"attempt": 2, "sampler": smp, "rhat": res["worst"]["rhat"], "parameter": res["worst"]["parameter"],
                         "seconds": secs, "ok": res["ok"]})
        ctx.fit_sampler = smp                       # later refits (steps 34-35) use the settings that converged
        res["above"] = [] if res["ok"] else m.rhat_detail(ctx.mmm, limit)
    ctx.gates[30] = res["ok"]
    details = {"rhat_max": limit, "attempts": attempts, "above": res["above"], "suggestions": [],
               "continue_anyway": ctx.options.continue_unconverged}
    worst = res["worst"]
    again = " after one retry with a longer warm-up" if len(attempts) > 1 else ""
    if res["ok"]:
        return _step(30, Verdict.PASS, f"Every R-hat is below {limit}{again} (largest {worst['rhat']:.3f}, {worst['parameter']}).", details)
    details["suggestions"] = SUGGESTIONS
    n = len(res["above"])
    names = (f"{n} parameter{'s' if n != 1 else ''} above {limit}, the worst {res['above'][0]['text']}" if n
             else f"largest R-hat {worst['rhat']}")
    if ctx.options.continue_unconverged:
        ctx.not_converged = True
        ctx.log.append({"step": 30, "text": "continue although the model did not converge", "source": "user"})
        return _step(30, Verdict.FAIL, f"Did not converge{again}: {names}. You chose to continue: every later result "
                     "carries a 'did not converge' badge.", details,
                     ["The sampler hasn't settled, so these numbers can't be fully trusted."])
    return _step(30, Verdict.FAIL, f"Failed: the chains disagree{again}: {names}.", details,
                 ["Steps 31 onward wait: none of this fit's numbers can be trusted yet. See the list of what to try, "
                  "or choose 'Continue anyway'."])


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
    directions = direction_check(ctx)
    details = {"roi": roi, "roi_in_money": money, "contributions": con, **res, "directions": directions,
               "limits": {"roi": [cfg["roi_min"], cfg["roi_max"]], "baseline_share": cfg["baseline_share"]}}
    warnings = [f"'{x['control']}' was set to {x['label']}, but {x['share_at_limit']:.0%} of its posterior sits at zero, "
                "against that limit: the data may point the other way. Check the direction in step 21." for x in directions if x["flagged"]]
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


def direction_check(ctx: ECtx) -> list[dict[str, Any]]:
    """Step 32: each control with a direction set (step 21). Its prior is truncated at zero, so the
    posterior can't cross it; when the data points the other way, the draws pile up at zero instead.
    Flagged when most of the posterior sits within a small distance of the limit."""
    from ..configure.choices import DIRECTION_LABEL, meridian_defaults

    dirs = {c: v for c, v in (ctx.config.get("control_direction") or {}).items() if v != "either"}
    post = ctx.mmm.inference_data.posterior
    if not dirs or "gamma_c" not in post:
        return []
    cfg = ctx.cfg["direction_check"]
    near = float(cfg["near_zero"]) * float(meridian_defaults()["gamma_c_scale"])
    names = [str(x) for x in ctx.mmm.input_data.control_variable.values]
    g = np.asarray(post["gamma_c"]).reshape(-1, len(names))
    out = []
    for c, v in dirs.items():
        if c not in names:
            continue
        draws = g[:, names.index(c)]
        wrong = float(np.mean(draws < 0)) if v == "raises" else float(np.mean(draws > 0))
        at_limit = float(np.mean(np.abs(draws) < near))
        out.append({"control": c, "direction": v, "label": DIRECTION_LABEL[v], "median": float(np.median(draws)),
                    "share_at_limit": at_limit, "share_wrong_side": wrong,
                    "flagged": wrong > 0 or at_limit > float(cfg["max_share_at_limit"])})
    return out


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
        _fit(ctx, mmm, ctx.fit_sampler, f"stability_{i + 1}")
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
           "roi_priors": {**ctx.config["roi_priors"], name: _default_prior(ctx.config.get("prior_type", "roi"))},
           "carryover_prior": {**(ctx.config.get("carryover_prior") or {}), name: {"speed": "dont_know", "beta": [1.0, 1.0]}},
           "saturation": {**(ctx.config.get("saturation") or {}), name: "hill"}}
    ctx.progress(35, f"Fitting with a placebo shaped like {model_on}", 0.0)
    mmm = m.build(variant, cfg)
    mmm.sample_prior(int(ctx.cfg["prior_draws"]), seed=ctx.seed)
    _fit(ctx, mmm, ctx.fit_sampler, "placebo")
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


def _default_prior(kind: str) -> dict:
    family, p1, p2 = meridian_defaults()[kind]
    out = {**prior_summary(family, p1, p2), "family": family, "p1": p1, "p2": p2, "source": "default"}
    return {**out, "mu": p1, "sigma": p2} if family == "lognormal" else out


def data_version(table: pd.DataFrame) -> str:
    return hashlib.sha256(table.to_csv(index=False).encode()).hexdigest()[:12]


def final_fit(ctx: ECtx) -> StepResult:
    """Step 36: refit on every week with the chosen settings (and the same one-retry rule as
    step 30), then save. A model the user continued with after step 30 is saved, badged."""
    failed = [n for n in GATES if not ctx.gates.get(n, False) and not (n == 30 and ctx.not_converged)]
    if failed:
        return _step(36, Verdict.WARN, f"Not run: the model failed step(s) {', '.join(map(str, failed))}, so nothing is saved.",
                     {"skipped": True, "failed_gates": failed})
    ctx.progress(36, "Refitting on every week", 0.0)
    cfg = {**ctx.config, "holdout": None}
    mmm, attempts, res = _fit_converged(ctx, lambda: m.build(ctx.d, cfg), ctx.sampler, "final", 36)
    secs = sum(a["seconds"] for a in attempts)
    if not res["ok"]:
        ctx.not_converged = True
    import meridian

    version = data_version(ctx.d.table)
    info = {"data_version": version, "weeks": len(m.weeks(mmm)), "sampler": attempts[-1]["sampler"], "seconds": secs,
            "rhat_worst": res["worst"], "attempts": attempts, "seed": ctx.seed, "not_converged": ctx.not_converged,
            "meridian": getattr(meridian, "__version__", "?"),
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "path": None}
    if ctx.out_dir is not None:
        path = ctx.out_dir / "model.pkl"
        ctx.progress(36, "Saving the model", 0.9)
        m.save(mmm, path)
        info["path"] = str(path)
        meta = {**info, "config": ctx.config, "gates": {str(k): v for k, v in ctx.gates.items()}, "badges": ctx.badges,
                "options": ctx.options.to_dict(), "decision_log": list(getattr(ctx.d, "decision_log", None) or []) + ctx.log}
        (ctx.out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    ctx.model = info
    ctx.final = mmm
    warnings = []
    if not res["ok"]:
        warnings.append(f"The final fit did not converge either (largest R-hat {res['worst']['rhat']:.2f}, {res['worst']['parameter']}); "
                        "it is saved with a 'did not converge' badge.")
    elif ctx.not_converged:
        warnings.append("Saved with a 'did not converge' badge: step 30 failed and you chose to continue.")
    where = f"saved as {info['path']}" if info["path"] else "not saved (no folder given)"
    again = " after one retry with a longer warm-up" if len(attempts) > 1 else ""
    return _step(36, Verdict.WARN if warnings else Verdict.PASS,
                 f"Refitted on all {info['weeks']} weeks{again} in {secs:,.0f} s; {where}; data version {version}.", info, warnings)


STEPS = [prior_check, fit_holdout, convergence, holdout_accuracy, plausibility, prior_driven, stability, placebo, final_fit]
