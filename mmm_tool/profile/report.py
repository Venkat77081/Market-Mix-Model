"""Step 20: profile report.

Spend share per channel, the KPI with total spend overlaid, and an activity
heatmap (each channel's weekly spend as a share of its own peak). Report only.
"""
from __future__ import annotations

from ..ingest.types import StepResult, Verdict
from .context import BADGE_TEXT, CCtx

STEP = 20
NAME = "Profile report"


def profile_report(ctx: CCtx) -> StepResult:
    spend = ctx.spend()
    total = float(spend.to_numpy().sum())
    share = sorted(({"channel": c, "spend": float(spend[c].sum()),
                     "share": round(float(spend[c].sum() / total), 4) if total else 0.0,
                     "badges": [BADGE_TEXT[b] for b in ctx.badges.get(c, [])]} for c in spend.columns),
                   key=lambda r: -r["spend"])
    kpi = ctx.weekly_sum([ctx.kpi_col])[ctx.kpi_col]
    weeks = [str(d.date()) for d in spend.index]
    peak = spend.max().replace(0, 1)
    heat = {c: [round(float(v), 3) for v in (spend[c] / peak[c])] for c in spend.columns}
    details = {
        "spend_share": share,
        "weeks": weeks,
        "kpi": [float(v) for v in kpi.reindex(spend.index)],
        "total_spend": [float(v) for v in spend.sum(axis=1)],
        "heatmap": heat,
        "badges": {c: [BADGE_TEXT[b] for b in bs] for c, bs in ctx.badges.items()},
    }
    top = share[0] if share else None
    summary = (f"{len(share)} channels, ₹{total / 1e7:,.1f} Cr total spend over {len(weeks)} weeks"
               + (f"; largest: {top['channel']} ({top['share']:.0%})" if top else ""))
    return StepResult(STEP, NAME, Verdict.PASS, summary + ".", details)
