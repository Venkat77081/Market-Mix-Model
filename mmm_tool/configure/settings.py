"""Steps 22, 23, 24 and 27: the model's structural settings.

22 Carryover length   max_lag: how many weeks an ad keeps working. 8 by default (2-13).
23 Carryover shape    adstock_decay_spec per channel: geometric (strongest the week it
                      runs, then fading) by default; binomial (a delayed peak) is
                      suggested for channels like TV and outdoor.
24 Baseline flexibility  knots: about one per quarter. More knots let the trend bend
                      more, and a very flexible trend absorbs media effects.
27 Holdout            the last 13 weeks (8-26) are held out to test the fit.
"""
from __future__ import annotations

import math
import re

from ..ingest.keywords import words_of
from ..ingest.types import Answers, Decision, StepResult, Verdict
from .context import (DEFAULT_HOLDOUT, DEFAULT_MAX_LAG, HOLDOUT_RANGE, MAX_LAG_RANGE, MIN_TRAIN_WEEKS, WEEKS_PER_KNOT,
                      DCtx, load_calendars)

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
    words = load_calendars().get("delayed_peak_words", [])
    pattern = re.compile(r"(?<![a-z0-9])(" + "|".join(map(re.escape, words)) + r")(?![a-z0-9])") if words else None
    suggest = [c.name for c in ctx.channels if pattern and pattern.search(words_of(c.name))]
    d = Decision(
        id="adstock_shape",
        step=23,
        question=("How does each channel's effect build? Geometric is strongest the week it runs, then fades; "
                  "binomial peaks a week or two later, as TV and outdoor often do."),
        options=list(SHAPES),
        default={c.name: "geometric" for c in ctx.channels},
        context={"suggest_binomial": suggest},
    )
    chosen = {**d.default, **(answers.ask(d) or {})}
    bad = {k: v for k, v in chosen.items() if k not in d.default or v not in SHAPES}
    if bad:
        return _blocked(23, "Carryover shape", d, f"unknown channel or shape in {bad}; use geometric or binomial")
    ctx.config["adstock"] = chosen
    binomial = [c for c, v in chosen.items() if v == "binomial"]
    summary = (f"Geometric for all {len(chosen)} channels" if not binomial
               else f"Binomial (delayed peak) for {', '.join(binomial)}; geometric for the rest")
    if suggest and not set(suggest) & set(binomial):
        summary += f". {', '.join(suggest)} often peak later; consider binomial"
    return StepResult(23, "Carryover shape", Verdict.PASS, summary + ".", {"shapes": chosen, "suggest_binomial": suggest}, [], [d])


def baseline_flexibility(ctx: DCtx, answers: Answers) -> StepResult:
    weeks = len(ctx.weeks)
    default = max(1, math.ceil(weeks / WEEKS_PER_KNOT))
    d = Decision(id="knots", step=24,
                 question=("How flexible should the baseline trend be? About one knot per quarter is the default; "
                           "fewer is more stable, more follows structural breaks."),
                 options=[], default=default, context={"weeks": weeks})
    knots, problem = _int_answer(answers, d, 1, weeks)
    if problem:
        return _blocked(24, "Baseline flexibility", d, problem)
    ctx.config["knots"] = knots
    warnings = []
    if knots > 1.5 * default:
        warnings.append(f"{knots} knots is a flexible baseline (default {default}); a trend that bends this much "
                        "can absorb media effects and push ROI down.")
    every = weeks / knots
    return StepResult(24, "Baseline flexibility", Verdict.WARN if warnings else Verdict.PASS,
                      f"{knots} knots, one every {every:.0f} weeks" + (" (the default)." if knots == default else f" (default {default})."),
                      {"knots": knots, "default": default, "weeks": weeks}, warnings, [d])


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
