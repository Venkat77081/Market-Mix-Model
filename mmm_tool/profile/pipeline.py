"""Phase C: profile and sufficiency gates (steps 14-20).

Takes a complete Phase B result. Step 14 can refuse the run; steps 15, 17
and 18 ask before merging, combining or excluding; 16 and 19 warn and badge
channels; 20 reports. The result keeps the badges for later screens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..clean.pipeline import PhaseBResult
from ..ingest.pipeline import PhaseAResult
from ..ingest.types import Answers, StepResult, Verdict
from .activity import check_activity
from .context import BADGE_TEXT, CCtx
from .mediators import check_mediators
from .report import profile_report
from .seasons import check_seasons
from .sufficiency import check_sufficiency
from .together import check_together
from .variation import check_variation

STEPS = [
    lambda ctx, ans: check_sufficiency(ctx),
    check_activity,
    lambda ctx, ans: check_variation(ctx),
    check_together,
    check_mediators,
    check_seasons,
    lambda ctx, ans: profile_report(ctx),
]


@dataclass
class PhaseCResult(PhaseAResult):
    badges: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        out["badges"] = {c: [BADGE_TEXT[b] for b in bs] for c, bs in self.badges.items()}
        return out


def run_phase_c(b: PhaseBResult, answers: Answers) -> PhaseCResult:
    if b.status != "complete":
        raise ValueError("Phase C needs a complete Phase B result.")
    ctx = CCtx(df=b.table.copy(), roles=dict(b.roles), channels=[*b.channels], kpi=dict(b.kpi),
               granularity=dict(b.granularity))
    steps: list[StepResult] = []
    for fn in STEPS:
        step = fn(ctx, answers)
        steps.append(step)
        if step.blocks:
            status = "failed" if step.verdict == Verdict.FAIL else "needs_input"
            return PhaseCResult(status, steps, decision_log=answers.log, badges=ctx.badges)
    roles = {c: r for c, r in ctx.roles.items() if c in ctx.df.columns or r == "excluded"}
    return PhaseCResult("complete", steps, ctx.df.reset_index(drop=True), roles, ctx.channels, ctx.kpi,
                        ctx.granularity, answers.log, ctx.badges)
