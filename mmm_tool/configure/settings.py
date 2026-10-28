"""Steps 22, 23, 24 and 27: the model's structural settings.

22 Carryover length   max_lag: how many weeks an ad keeps working. 8 by default (2-13).
23 Carryover shape    per channel: adstock_decay_spec, geometric (the same share lost
                      every week) by default or binomial (slow at first, then dropping
                      off by the end of the carryover length; Meridian's binomial has
                      no delayed peak); how fast the effect fades, a Beta prior on
                      alpha (choices.py); and diminishing returns (Hill) or none.
24 Baseline flexibility  knots: about one per quarter. More knots let the baseline bend
                      more, and a very flexible baseline absorbs media effects. The
                      UI previews every choice first (baseline.py).
27 Holdout            the last 13 weeks (8-26) are held out to test the fit.
"""
from __future__ import annotations

import re


from .. import currency as money_fmt
from ..ingest.keywords import words_of
from ..ingest.types import Answers, Decision, StepResult, Verdict
from .context import DEFAULT_HOLDOUT, DEFAULT_MAX_LAG, HOLDOUT_RANGE, MAX_LAG_RANGE, MIN_TRAIN_WEEKS, DCtx, load_calendars

SHAPES = ("geometric", "binomial")


def _int_answer(answers: Answers, d: Decision, lo: int, hi: int) -> tuple[int | None, str | None]:
    v = answers.ask(d)
    v = d.default if v is None else v
    try:
        v = int(v)
    except (TypeError, ValueError):
        return None, f"'{v}' isn't a whole number"
    if not lo <= v <= hi:
        return None, f"{v} is outside {lo}-{hi}"
    return v, None


def _blocked(step: int, name: str, d: Decision, problem: str) -> StepResult:
    d.blocking, d.answered = True, False
    return StepResult(step, name, Verdict.NEEDS_INPUT, problem[0].upper() + problem[1:] + ".", {}, [], [d])


def carryover_length(ctx: DCtx, answers: Answers) -> StepResult:
    d = Decision(id="max_lag", step=22, question="How many weeks can an ad keep working after it runs?",
                 options=list(range(MAX_LAG_RANGE[0], MAX_LAG_RANGE[1] + 1)), default=DEFAULT_MAX_LAG)
    lag, problem = _int_answer(answers, d, *MAX_LAG_RANGE)
    if problem:
        return _blocked(22, "Carryover length", d, problem)
    ctx.config["max_lag"] = lag
    note = "the default" if lag == DEFAULT_MAX_LAG else f"default {DEFAULT_MAX_LAG}"
    return StepResult(22, "Carryover length", Verdict.PASS, f"Effects can last up to {lag} weeks ({note}).",
                      {"max_lag": lag, "range": MAX_LAG_RANGE}, [], [d])


def carryover_shape(ctx: DCtx, answers: Answers) -> StepResult:
    """Step 23: per channel, the decay shape, how fast the effect fades (a prior on
    alpha), and whether returns diminish (Hill) or not (saturation "none")."""
    from .choices import NO_SATURATION_WARNING, SPEED_LABEL, SPEEDS, carryover_beta, carryover_summary, saturation_hint

    cal = load_calendars()
    words = cal.get("binomial_words", cal.get("delayed_peak_words", []))
    pattern = re.compile(r"(?<![a-z0-9])(" + "|".join(map(re.escape, words)) + r")(?![a-z0-9])") if words else None
    suggest = [c.name for c in ctx.channels if pattern and pattern.search(words_of(c.name))]
    names = [c.name for c in ctx.channels]
    d = Decision(
        id="adstock_shape",
        step=23,
        question=("How does each channel's effect fade? Geometric loses the same share every week; binomial fades "
                  "slowly at first, then drops off by the end of the carryover length (step 22)."),
        options=list(SHAPES),
        default={c: "geometric" for c in names},
        context={"suggest_binomial": suggest},
    )
    chosen = {**d.default, **(answers.ask(d) or {})}
    bad = {k: v for k, v in chosen.items() if k not in d.default or v not in SHAPES}
    if bad:
        return _blocked(23, "Carryover shape", d, f"unknown channel or shape in {bad}; use geometric or binomial")
    ctx.config["adstock"] = chosen
    max_lag = int(ctx.config.get("max_lag", DEFAULT_MAX_LAG))

    d_speed = Decision(
        id="carryover_speed", step=23,
        question="How fast does each channel's effect fade? Share of the effect still working next week.",
        options=list(SPEEDS), default={c: "dont_know" for c in names},
    )
    speeds = {**d_speed.default, **(answers.ask(d_speed) or {})}
    bad = {k: v for k, v in speeds.items() if k not in d_speed.default or v not in SPEEDS}
    if bad:
        return _blocked(23, "Carryover shape", d_speed, f"unknown channel or speed in {bad}; use dont_know, fast, medium or slow")
    prior = {}
    for c in names:
        a, b = carryover_beta(speeds[c], chosen[c], max_lag)
        prior[c] = {"speed": speeds[c], "beta": [a, b], "next_week": carryover_summary(a, b, chosen[c], max_lag)}
    ctx.config["carryover_prior"] = prior
    answers.describe(d_speed, [f"{c} fades {speeds[c]}" for c in names if speeds[c] != "dont_know"])

    spend = {c.name: ctx.df.groupby(ctx.date)[c.spend].sum().to_numpy(float) for c in ctx.channels}
    total = sum(float(v.sum()) for v in spend.values())
    hints = [c for c in names if saturation_hint(spend[c], total)]
    d_sat = Decision(
        id="diminishing_returns", step=23,
        question=f"Does each extra {money_fmt.of(ctx.kpi)['unit']} on a channel bring a little less than the one before (diminishing returns)?",
        options=["yes", "no"], default={c: "yes" for c in names}, context={"suggest_no": hints},
    )
    sat = {**d_sat.default, **(answers.ask(d_sat) or {})}
    bad = {k: v for k, v in sat.items() if k not in d_sat.default or v not in ("yes", "no")}
    if bad:
        return _blocked(23, "Carryover shape", d_sat, f"unknown channel or answer in {bad}; use yes or no")
    ctx.config["saturation"] = {c: ("hill" if sat[c] == "yes" else "none") for c in names}
    answers.describe(d_sat, [f"{c} has no diminishing returns" for c in names if sat[c] == "no"])
    off = [c for c in names if sat[c] == "no"]
    warnings = [f"'{c}': {NO_SATURATION_WARNING}" for c in off]

    binomial = [c for c, v in chosen.items() if v == "binomial"]
    summary = (f"Geometric for all {len(chosen)} channels" if not binomial
               else f"Binomial for {', '.join(binomial)}; geometric for the rest")
    set_speed = [c for c in names if speeds[c] != "dont_know"]
    if set_speed:
        summary += "; fade speed set for " + ", ".join(f"{c} ({speeds[c]})" for c in set_speed)
    if off:
        summary += "; no diminishing returns for " + ", ".join(off)
    elif suggest and not set(suggest) & set(binomial):
        summary += f". {', '.join(suggest)} often fade slowly at first; consider binomial"
    details = {"shapes": chosen, "suggest_binomial": suggest, "carryover_prior": prior, "speed_labels": SPEED_LABEL,
               "saturation": ctx.config["saturation"], "suggest_no_saturation": hints,
               "share_of_spend": {c: (float(spend[c].sum()) / total if total else 0.0) for c in names}}
    return StepResult(23, "Carryover shape", Verdict.WARN if warnings else Verdict.PASS, summary + ".", details, warnings,
                      [d, d_speed, d_sat])


def baseline_flexibility(ctx: DCtx, answers: Answers) -> StepResult:
    from .baseline import knot_info, limits, meridian_installed, spacing

    weeks = len(ctx.weeks)
    lim = limits(weeks)
    default = lim["default"]
    d = Decision(id="knots", step=24,
                 question=("How flexible should the baseline be? It can bend at each knot; about one knot per quarter "
                           "is the default. Fewer is steadier; more follows sales closely and can take media's credit."),
                 options=[], default=default,
                 context={"weeks": weeks, "max": lim["max"], "quick": lim["quick"], "amber": lim["amber"], "red": lim["red"]})
    knots, problem = _int_answer(answers, d, 1, weeks - 1)
    if problem:
        return _blocked(24, "Baseline flexibility", d, problem)
    ctx.config["knots"] = knots
    # Spacing from Meridian's own knot locations, not weeks / knots.
    geos = ctx.df[ctx.geo].nunique() if ctx.geo else 1
    locations = [int(i) for i in knot_info(weeks, knots, geos).knot_locations] if meridian_installed() else None
    every = spacing(locations) if locations else None
    warnings, level = [], None
    if knots > lim["red"]:
        level = "red"
        warnings.append(f"{knots} knots bend the baseline about every {every or round(weeks / knots)} weeks: "
                        "likely to absorb campaign effects.")
    elif knots > lim["amber"]:
        level = "amber"
        warnings.append(f"{knots} knots bend the baseline about every {every or round(weeks / knots)} weeks, "
                        "more than every 2 months.")
    if knots == 1:
        head = "1 knot: a flat baseline"
    elif every:
        head = f"{knots} knots, one about every {every} weeks"
    else:
        head = f"{knots} knots over {weeks} weeks"
    return StepResult(24, "Baseline flexibility", Verdict.WARN if warnings else Verdict.PASS,
                      head + (" (the default)." if knots == default else f" (default {default})."),
                      {"knots": knots, "default": default, "weeks": weeks, "locations": locations, "spacing": every,
                       "level": level, **{k: lim[k] for k in ("max", "amber", "red", "quick")}}, warnings, [d])


def holdout(ctx: DCtx, answers: Answers) -> StepResult:
    weeks = ctx.weeks
    d = Decision(id="holdout_weeks", step=27, question="How many of the last weeks should be held out to test the fit?",
                 options=list(range(HOLDOUT_RANGE[0], HOLDOUT_RANGE[1] + 1)), default=DEFAULT_HOLDOUT)
    n, problem = _int_answer(answers, d, *HOLDOUT_RANGE)
    if problem:
        return _blocked(27, "Holdout", d, problem)
    start, end = weeks[-n], weeks[-1]
    train = len(weeks) - n
    ctx.config["holdout"] = {"weeks": n, "start": str(start.date()), "end": str(end.date()), "train_weeks": train}
    warnings = []
    if train < MIN_TRAIN_WEEKS:
        warnings.append(f"Only {train} weeks remain to fit on while testing (step 29); "
                        "the final fit (step 36) uses all weeks.")
    return StepResult(27, "Holdout", Verdict.WARN if warnings else Verdict.PASS,
                      f"Last {n} weeks held out ({start.date()} to {end.date()}); {train} weeks to fit on.",
                      ctx.config["holdout"], warnings, [d])
