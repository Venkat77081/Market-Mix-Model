"""Step 19: spend timed with seasons.

When a channel spends mostly in the weeks a season flag is on (tax season,
festive weeks), the season and the channel compete for the same sales and the
model can split them either way. Warning only, with a badge on the channel.
"""
from __future__ import annotations

from ..ingest.types import StepResult, Verdict
from .context import SEASON_CORR, CCtx


def _is_flag(s) -> bool:
    v = set(s.dropna().unique())
    return len(v) == 2 and v <= {0.0, 1.0}


STEP = 19
NAME = "Spend timed with seasons"


def check_seasons(ctx: CCtx) -> StepResult:
    flags = [c for c in ctx.cols("control", "non_media_treatment") if _is_flag(ctx.df[c]) and c != "kpi_imputed"
             and not c.startswith("event_")]
    spend = ctx.spend()
    found = []
    if flags:
        season = ctx.weekly_mean(flags)
        for ch in spend.columns:
            for f in flags:
                r = spend[ch].corr(season[f])
                if r == r and r > SEASON_CORR:
                    found.append({"channel": ch, "flag": f, "correlation": round(float(r), 3)})
                    ctx.badge(ch, "season_timed")
    warnings = [f"'{x['channel']}' spends mostly when '{x['flag']}' is on (correlation {x['correlation']:.2f}): "
                "the season and the channel will compete for the same sales." for x in found]
    details = {"flags": flags, "pairs": found, "threshold": SEASON_CORR}
    if not flags:
        return StepResult(STEP, NAME, Verdict.PASS, "No season flags to check against.", details)
    if not found:
        return StepResult(STEP, NAME, Verdict.PASS, f"No channel's spend follows the {len(flags)} season flag(s).", details)
    return StepResult(STEP, NAME, Verdict.WARN,
                      f"{len(found)} channel(s) timed with a season: "
                      + ", ".join(f"{x['channel']} with {x['flag']}" for x in found) + ".", details, warnings)
