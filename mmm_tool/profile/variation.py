"""Step 16: spend variation.

A channel whose weekly spend barely changes (coefficient of variation under
0.3) gives the model no contrast to learn from: its ROI comes mostly from the
prior. In a regional model the median of the regions' CVs is used. It gets a
"hard to measure" badge on every later screen, and the
warning suggests a lift test or an informed prior.
"""
from __future__ import annotations

from ..ingest.types import StepResult, Verdict
from .context import LOW_SPEND_CV, CCtx, cv

STEP = 16
NAME = "Spend variation"


def channel_cv(ctx: CCtx) -> dict[str, float]:
    """CV of weekly spend. In a regional model each region's own series is what
    the model learns from (adding regions together smooths spend), so the
    median CV across regions is used."""
    if not ctx.geo:
        spend = ctx.spend()
        return {c: cv(spend[c]) for c in spend.columns}
    out = {}
    for ch in ctx.channels:
        per_geo = ctx.df.groupby(ctx.geo)[ch.spend].agg(cv)
        out[ch.name] = float(per_geo.median())
    return out


def check_variation(ctx: CCtx) -> StepResult:
    rows = [{"channel": c, "cv": round(v, 3)} for c, v in channel_cv(ctx).items()]
    flat = [r for r in rows if r["cv"] < LOW_SPEND_CV]
    for r in flat:
        ctx.badge(r["channel"], "hard_to_measure")
    warnings = [f"'{r['channel']}' spend varies little (CV {r['cv']:.2f}, under {LOW_SPEND_CV}): hard to measure. "
                "It needs a lift test or an informed prior (steps 25-26)." for r in flat]
    details = {"channels": rows, "hard_to_measure": [r["channel"] for r in flat], "threshold": LOW_SPEND_CV}
    if flat:
        summary = f"{len(flat)} of {len(rows)} channel(s) hard to measure: {', '.join(r['channel'] for r in flat)}"
    else:
        summary = f"Every channel's spend varies enough (CV {min(r['cv'] for r in rows):.2f} or more)"
    return StepResult(STEP, NAME, Verdict.WARN if flat else Verdict.PASS, summary + ".", details, warnings)
