"""Phase D: configure the model (steps 21-27).

Takes a complete Phase C result. Every step has a sensible default, so the
phase completes on its own; the user adds calendars and past lift tests, and
can change any setting. The result carries `config`, a JSON description of
Meridian's ModelSpec (spec.py builds the real objects), and the table with
calendar flags added.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ..ingest.types import Answers, StepResult, Verdict
from ..profile.context import BADGE_TEXT
from ..profile.pipeline import PhaseCResult
from .calendar import add_calendar_flags
from .context import DCtx
from .priors import lift_tests, roi_priors
from .settings import baseline_flexibility, carryover_length, carryover_shape, holdout

STEPS = [add_calendar_flags, carryover_length, carryover_shape, baseline_flexibility, roi_priors, lift_tests, holdout]


@dataclass
class PhaseDResult(PhaseCResult):
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        if self.status == "complete":
            out["config"] = self.config
        return out


def run_phase_d(c: PhaseCResult, answers: Answers) -> PhaseDResult:
    if c.status != "complete":
        raise ValueError("Phase D needs a complete Phase C result.")
    ctx = DCtx(df=c.table.copy(), roles=dict(c.roles), channels=[*c.channels], kpi=dict(c.kpi),
               granularity=dict(c.granularity), badges=copy.deepcopy(c.badges))
    steps: list[StepResult] = []
    for fn in STEPS:
        step = fn(ctx, answers)
        steps.append(step)
        if step.blocks:
            status = "failed" if step.verdict == Verdict.FAIL else "needs_input"
            return PhaseDResult(status, steps, decision_log=answers.log, badges=ctx.badges)
    ctx.config["channels"] = [c.name for c in ctx.channels]
    roles = {col: r for col, r in ctx.roles.items() if col in ctx.df.columns or r == "excluded"}
    return PhaseDResult("complete", steps, ctx.df, roles, ctx.channels, ctx.kpi, ctx.granularity,
                        answers.log, ctx.badges, ctx.config)
