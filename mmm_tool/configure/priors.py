"""Steps 25 and 26: ROI priors and lift-test calibration.

25 Every paid channel starts from Meridian's default ROI prior, LogNormal(0.2, 0.9):
   median about 1.2, 90% between about 0.28 and 5.4. The user can give an
   expected ROI range from benchmarks; it becomes a LogNormal whose 90%
   interval is that range.
26 A past lift test (channel, test weeks, measured ROI and its range) becomes
   that channel's prior, and roi_calibration limits the prior to the test
   weeks. Several tests of one channel are combined, the more precise counting
   more. Channels with a test show "calibrated"; the rest "model only".
"""
from __future__ import annotations

import math

import pandas as pd

from ..ingest.types import Answers, Decision, StepResult, Verdict
from .context import DCtx, lognormal_from_range, lognormal_summary


def week_period(weeks: pd.DatetimeIndex, start: str, end: str) -> dict:
    """The data weeks a test covers: from the week containing its first day to
    the week containing its last. Meridian needs exact week starts."""
    first = weeks[max(0, weeks.searchsorted(pd.Timestamp(start), side="right") - 1)]
    last = weeks[max(0, weeks.searchsorted(pd.Timestamp(end), side="right") - 1)]
    return {"start": str(first.date()), "end": str(last.date()), "entered": f"{start} to {end}"}


def _range(v) -> tuple[float, float] | None:
    try:
        lo, hi = float(v["low"]), float(v["high"])
    except (KeyError, TypeError, ValueError):
        return None
    return (lo, hi) if 0 < lo < hi else None


def _money(ctx: DCtx) -> bool:
    """A money value per KPI unit exists (step 5), so returns can be stated in money per money."""
    return ctx.kpi.get("kpi_type") == "revenue" or bool(ctx.kpi.get("revenue_per_kpi_column") or ctx.kpi.get("revenue_per_kpi_value"))


def roi_priors(ctx: DCtx, answers: Answers) -> StepResult:
    """Step 25: what the user knows about each paid channel, as one prior type for all of
    them (Meridian's media_prior_type and rf_prior_type) and a 90% range per channel."""
    from .. import currency as money_fmt
    from .choices import meridian_defaults, prior_from_range, prior_labels, prior_summary

    cur = money_fmt.of(ctx.kpi)
    PRIOR_TYPES = prior_labels(cur)  # noqa: N806 - the labels, in the file's currency

    names = [c.name for c in ctx.channels]
    hard = [c for c in names if "hard_to_measure" in ctx.badges.get(c, [])]
    money = _money(ctx)
    allowed = list(PRIOR_TYPES) if money else ["contribution"]
    d_type = Decision(
        id="prior_type", step=25,
        question="I know about each channel's…",
        options=allowed, default="roi" if money else "contribution",
        context={"labels": PRIOR_TYPES, "money": money,
                 "why_disabled": None if money else ("There is no money value per KPI unit (step 5), so returns can't be "
                                                     "stated in money; Meridian uses the share-of-sales prior in that case.")},
    )
    kind = answers.ask(d_type) or d_type.default
    if kind not in allowed:
        d_type.blocking, d_type.answered = True, False
        why = d_type.context["why_disabled"] or ""
        return StepResult(25, "ROI priors", Verdict.NEEDS_INPUT, f"'{kind}' isn't available here. {why}".strip(), {}, [], [d_type])
    ctx.config["prior_type"] = kind
    answers.describe(d_type, [f"priors set on {PRIOR_TYPES[kind]}"])
    unit = "in %" if kind == "contribution" else money_fmt.returned_per(cur)
    d = Decision(
        id="roi_ranges", step=25,
        question=(f"For any channel you know about, the range you're 90% sure of ({PRIOR_TYPES[kind]}, {unit}). "
                  "Leave a channel blank to use Meridian's default."),
        options=names, default={},
        context={"prior_type": kind, "hard_to_measure": hard},
    )
    given = answers.ask(d) or {}
    family, p1, p2 = meridian_defaults()[kind]
    default = {**prior_summary(family, p1, p2), "family": family, "p1": p1, "p2": p2}
    if family == "lognormal":
        default.update(mu=p1, sigma=p2)
    d.context["default"] = default
    priors, bad = {}, []
    for c in names:
        if c not in given:
            priors[c] = {**default, "source": "default"}
            continue
        try:
            lo, hi = float(given[c]["low"]), float(given[c]["high"])
            fam, a, b = prior_from_range(kind, lo, hi)
        except (KeyError, TypeError, ValueError):
            bad.append(c)
            continue
        priors[c] = {**prior_summary(fam, a, b), "family": fam, "p1": a, "p2": b, "source": "benchmark", "entered": [lo, hi]}
        if fam == "lognormal":
            priors[c].update(mu=a, sigma=b)
    bad += [c for c in given if c not in names]
    if bad:
        d.blocking, d.answered = True, False
        rule = "both between 0 and 1, low below high" if kind == "contribution" else "both ends above zero, low below high"
        return StepResult(25, "ROI priors", Verdict.NEEDS_INPUT,
                          f"Check the range for {', '.join(map(str, bad))}: {rule}.", {}, [], [d_type, d])
    ctx.config["roi_priors"] = priors
    fmt = (lambda v: f"{v:.1%}") if kind == "contribution" else (lambda v: f"{v:.2f}")
    answers.describe(d, [f"{c} {PRIOR_TYPES[kind]} {fmt(given[c]['low'])}-{fmt(given[c]['high'])}" for c in names if c in given])
    warnings = []
    unguided = [c for c in hard if c not in given]
    if unguided:
        warnings.append(f"{', '.join(unguided)} {'is' if len(unguided) == 1 else 'are'} hard to measure (step 16) and "
                        "on the default prior; a range here or a lift test (step 26) would anchor it.")
    n = sum(1 for p in priors.values() if p["source"] == "benchmark")
    on_default = (f"Meridian's default {PRIOR_TYPES[kind]} prior (median {fmt(default['median'])}, "
                  f"90% {fmt(default['low'])}–{fmt(default['high'])})")
    summary = (f"Prior on {PRIOR_TYPES[kind]}: {n} channel(s) from your ranges, {len(names) - n} on {on_default}" if n
               else f"Prior on {PRIOR_TYPES[kind]}: all {len(names)} channels on {on_default}")
    return StepResult(25, "ROI priors", Verdict.WARN if warnings else Verdict.PASS, summary + ".",
                      {"priors": priors, "prior_type": kind, "labels": PRIOR_TYPES, "money": money}, warnings, [d_type, d])


def lift_tests(ctx: DCtx, answers: Answers) -> StepResult:
    names = {c.name for c in ctx.channels}
    weeks = ctx.weeks
    d = Decision(
        id="lift_tests",
        step=26,
        question=("Past lift tests: channel, test start and end dates, the measured ROI and its 90% range. "
                  "Each becomes that channel's ROI prior."),
        options=sorted(names),
        default=[],
        context={"first_week": str(weeks[0].date()), "last_week": str(weeks[-1].date())},
    )
    tests = answers.ask(d) or []
    if tests and ctx.config.get("prior_type", "roi") != "roi":
        for c in ctx.channels:
            ctx.badge(c.name, "model_only")
        ctx.config["calibration"] = {}
        return StepResult(26, "Lift-test calibration", Verdict.WARN,
                          f"{len(tests)} lift test(s) not used: lift tests set ROI priors, and step 25 uses another prior type.",
                          {"tests": []}, ["Switch step 25 to 'average return (ROI)' to calibrate with your lift tests."], [d])
    problems, clean = [], []
    for i, t in enumerate(tests, 1):
        try:
            ch = t["channel"]
            start, end = pd.Timestamp(t["start"]), pd.Timestamp(t["end"])
            roi, lo, hi = float(t["roi"]), float(t["low"]), float(t["high"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"test {i} needs channel, start, end, roi, low and high")
            continue
        if ch not in names:
            problems.append(f"test {i}: '{ch}' isn't a channel in the model")
        elif start > end:
            problems.append(f"test {i}: the start is after the end")
        elif end < weeks[0] or start > weeks[-1] + pd.Timedelta(days=6):
            problems.append(f"test {i}: {start.date()} to {end.date()} is outside the data")
        elif not 0 < lo <= roi <= hi or lo == hi:
            problems.append(f"test {i}: the ROI must sit inside its range, all above zero")
        else:
            clean.append({"channel": ch, "start": str(start.date()), "end": str(end.date()), "roi": roi, "low": lo, "high": hi})
    if problems:
        d.blocking, d.answered = True, False
        return StepResult(26, "Lift-test calibration", Verdict.NEEDS_INPUT, "Lift tests: " + "; ".join(problems) + ".",
                          {}, [], [d])

    priors = ctx.config["roi_priors"]
    calibrated = {}
    for ch in sorted({t["channel"] for t in clean}):
        own = [t for t in clean if t["channel"] == ch]
        # Combine tests on the log scale, weighting each by its precision.
        parts = [(math.log(t["roi"]), lognormal_from_range(t["low"], t["high"])[1]) for t in own]
        w = [1 / s ** 2 for _, s in parts]
        mu = sum(m * wi for (m, _), wi in zip(parts, w)) / sum(w)
        sigma = math.sqrt(1 / sum(w))
        replaced = priors[ch]["source"]
        priors[ch] = {**lognormal_summary(mu, sigma), "family": "lognormal", "p1": mu, "p2": sigma,
                      "source": "lift_test", "tests": own}
        calibrated[ch] = {"periods": [week_period(weeks, t["start"], t["end"]) for t in own], "replaced": replaced}
    ctx.config["calibration"] = calibrated
    for c in ctx.channels:
        ctx.badge(c.name, "calibrated" if c.name in calibrated else "model_only")
    warnings = [f"'{c}' had a benchmark range; the lift test replaces it." for c, v in calibrated.items() if v["replaced"] == "benchmark"]
    if not clean:
        return StepResult(26, "Lift-test calibration", Verdict.PASS,
                          f"No lift tests entered; all {len(names)} channels are 'model only'.", {"tests": []}, [], [d])
    return StepResult(26, "Lift-test calibration", Verdict.WARN if warnings else Verdict.PASS,
                      f"{len(clean)} lift test(s) calibrate {', '.join(calibrated)}; "
                      f"{len(names) - len(calibrated)} channel(s) stay 'model only'.",
                      {"tests": clean, "calibrated": calibrated}, warnings, [d])
