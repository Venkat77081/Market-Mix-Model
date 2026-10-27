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
from .context import DEFAULT_ROI_MU, DEFAULT_ROI_SIGMA, DCtx, lognormal_from_range, lognormal_summary


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


def roi_priors(ctx: DCtx, answers: Answers) -> StepResult:
    names = [c.name for c in ctx.channels]
    hard = [c for c in names if "hard_to_measure" in ctx.badges.get(c, [])]
    d = Decision(
        id="roi_ranges",
        step=25,
        question=("Expected ROI range for any channel you have benchmarks for (₹ returned per ₹, the range you'd be "
                  "90% sure of). Leave a channel blank to use Meridian's default."),
        options=names,
        default={},
        context={"default": lognormal_summary(DEFAULT_ROI_MU, DEFAULT_ROI_SIGMA), "hard_to_measure": hard},
    )
    given = answers.ask(d) or {}
    bad = [c for c, v in given.items() if c not in names or _range(v) is None]
    if bad:
        d.blocking, d.answered = True, False
        return StepResult(25, "ROI priors", Verdict.NEEDS_INPUT,
                          f"Check the ROI range for {', '.join(map(str, bad))}: both ends above zero, low below high.",
                          {}, [], [d])
    priors = {}
    for c in names:
        if c in given:
            mu, sigma = lognormal_from_range(*_range(given[c]))
            priors[c] = {**lognormal_summary(mu, sigma), "source": "benchmark"}
        else:
            priors[c] = {**lognormal_summary(DEFAULT_ROI_MU, DEFAULT_ROI_SIGMA), "source": "default"}
    ctx.config["roi_priors"] = priors
    warnings = []
    unguided = [c for c in hard if c not in given]
    if unguided:
        warnings.append(f"{', '.join(unguided)} {'is' if len(unguided) == 1 else 'are'} hard to measure (step 16) and "
                        "on the default prior; a benchmark range or a lift test (step 26) would anchor the ROI.")
    n = sum(1 for p in priors.values() if p["source"] == "benchmark")
    dflt = lognormal_summary(DEFAULT_ROI_MU, DEFAULT_ROI_SIGMA)
    on_default = f"Meridian's default ROI prior (median {dflt['median']:.2f}, 90% {dflt['low']:.2f}–{dflt['high']:.2f})"
    summary = (f"{n} channel(s) from your benchmarks, {len(names) - n} on {on_default}" if n
               else f"All {len(names)} channels on {on_default}")
    return StepResult(25, "ROI priors", Verdict.WARN if warnings else Verdict.PASS, summary + ".",
                      {"priors": priors}, warnings, [d])


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
        priors[ch] = {**lognormal_summary(mu, sigma), "source": "lift_test", "tests": own}
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
