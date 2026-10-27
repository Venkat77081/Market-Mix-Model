"""Step 14: data sufficiency.

Weeks of history and observations per parameter. Under 104 weeks or under
3 observations per parameter the run is refused; 104-155 weeks is thin and
3-6 observations per parameter needs strong priors, both with a warning.
"""
from __future__ import annotations

import math

from ..ingest.types import StepResult, Verdict
from .context import (ADEQUATE_WEEKS, COMFORT_OBS_PER_PARAM, MIN_OBS_PER_PARAM, MIN_WEEKS, PARAMS_PER_CONTROL,
                      PARAMS_PER_MEDIA, WEEKS_PER_KNOT, CCtx)

STEP = 14
NAME = "Data sufficiency"


def parameter_count(ctx: CCtx, weeks: int, geos: int) -> dict[str, int]:
    """Roughly how many parameters Meridian estimates for this setup."""
    return {
        "media channels": PARAMS_PER_MEDIA * len(ctx.channels),
        "unpaid media": PARAMS_PER_MEDIA * len(ctx.cols("organic_media")),
        "controls": PARAMS_PER_CONTROL * len(ctx.cols("control")),
        "non-media levers": PARAMS_PER_CONTROL * len(ctx.cols("non_media_treatment")),
        "baseline knots": max(1, math.ceil(weeks / WEEKS_PER_KNOT)),
        "intercepts": max(1, geos),
    }


def check_sufficiency(ctx: CCtx) -> StepResult:
    weeks = int(ctx.df[ctx.date].nunique())
    geos = int(ctx.df[ctx.geo].nunique()) if ctx.geo else 1
    obs = weeks * geos
    params = parameter_count(ctx, weeks, geos)
    n_params = sum(params.values())
    ratio = obs / n_params if n_params else float("inf")
    details = {"weeks": weeks, "regions": geos, "observations": obs, "parameters": params,
               "parameter_total": n_params, "obs_per_parameter": round(ratio, 2),
               "thresholds": {"min_weeks": MIN_WEEKS, "adequate_weeks": ADEQUATE_WEEKS,
                              "min_obs_per_parameter": MIN_OBS_PER_PARAM, "comfortable": COMFORT_OBS_PER_PARAM}}
    freq = ctx.granularity.get("frequency")
    unit = "weeks" if freq in ("weekly", "daily", None) else f"{freq} periods"

    problems = []
    if weeks < MIN_WEEKS:
        problems.append(f"{weeks} {unit} of history; an MMM needs at least {MIN_WEEKS} weekly observations (two years)")
    if ratio < MIN_OBS_PER_PARAM:
        biggest = max(params, key=params.get)
        problems.append(f"{ratio:.1f} observations per parameter ({obs} observations for about {n_params} parameters); "
                        f"at least {MIN_OBS_PER_PARAM} are needed. The largest share is {biggest} ({params[biggest]}): "
                        "merge or drop channels, drop controls in step 3, or model regions together")
    if problems:
        return StepResult(STEP, NAME, Verdict.FAIL, "Refused: " + "; ".join(problems) + ".", details)

    warnings = []
    if weeks < ADEQUATE_WEEKS:
        warnings.append(f"Thin history: {weeks} weeks (156 or more is adequate). Expect wide ROI ranges.")
    if ratio < COMFORT_OBS_PER_PARAM:
        warnings.append(f"{ratio:.1f} observations per parameter: strong priors are needed (step 25), "
                        "and lift tests (step 26) will matter.")
    level = "adequate" if weeks >= ADEQUATE_WEEKS else "thin"
    summary = (f"{weeks} weeks{f' × {geos} regions' if geos > 1 else ''} ({level}); {obs:,} observations for about "
               f"{n_params} parameters = {ratio:.1f} per parameter")
    return StepResult(STEP, NAME, Verdict.WARN if warnings else Verdict.PASS, summary + ".", details, warnings)
