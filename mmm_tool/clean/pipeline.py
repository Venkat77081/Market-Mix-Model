"""Phase B: clean and align to weeks (steps 7-13).

Takes a complete Phase A result and produces one complete row per week (per
region in a regional model) with no missing cells. Like Phase A, it stops at
the first step with a pending blocking decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ingest.pipeline import PhaseAResult
from ..ingest.types import Answers, StepResult, Verdict
from .coerce import coerce_types
from .context import Ctx
from .duplicates import collapse_duplicates
from .gaps import label_gaps
from .impossible import check_impossible
from .money import normalise_money
from .outliers import flag_outliers
from .resample import resample_weeks
from .revenue import add_revenue_per_kpi, per_unit_to_total

STEPS = [
    lambda ctx, ans: coerce_types(ctx),
    collapse_duplicates,
    resample_weeks,
    label_gaps,
    flag_outliers,
    check_impossible,
    normalise_money,
]


@dataclass
class PhaseBResult(PhaseAResult):
    """Same shape as Phase A's result; the table is now clean and weekly."""

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        if self.status == "complete":
            out["periods"] = int(self.table[self.coord_to_columns()["coord_to_columns"]["time"]].nunique())
        return out


def run_phase_b(a: PhaseAResult, answers: Answers) -> PhaseBResult:
    if a.status != "complete":
        raise ValueError("Phase B needs a complete Phase A result.")
    ctx = Ctx(
        df=a.table.copy(),
        roles=dict(a.roles),
        channels=list(a.channels),
        kpi=dict(a.kpi),
        granularity=dict(a.granularity),
    )
    steps: list[StepResult] = []
    for fn in STEPS:
        step = fn(ctx, answers)
        steps.append(step)
        if step.blocks:
            status = "failed" if step.verdict == Verdict.FAIL else "needs_input"
            return PhaseBResult(status, steps, decision_log=answers.log)
        if fn is STEPS[0]:                      # numbers now: a value per unit becomes a revenue total
            note = per_unit_to_total(ctx)
            if note:
                step.summary = step.summary.rstrip(".") + f"; {note}."
        if fn is resample_weeks:
            ctx.drop_weight_columns()

    # Revenue per KPI, once, from the cleaned columns (shown on step 13).
    notes, warns, info = add_revenue_per_kpi(ctx)
    last = steps[-1]
    if notes:
        last.summary = last.summary.rstrip(".") + "; " + "; ".join(notes) + "."
    last.warnings += warns
    last.details["revenue_per_kpi"] = info
    if warns and last.verdict == Verdict.PASS:
        last.verdict = Verdict.WARN

    df = ctx.df
    left = df[ctx.value_cols].isna().sum()
    if left.any():
        steps[-1].warnings.append(f"Cells still empty after cleaning: {left[left > 0].to_dict()}.")
    first = [c for c in ctx.keys if c in df.columns]
    df = df[first + [c for c in df.columns if c not in first]].reset_index(drop=True)
    roles = {c: r for c, r in ctx.roles.items() if c in df.columns or r == "excluded"}
    return PhaseBResult("complete", steps, df, roles, ctx.channels, ctx.kpi, ctx.granularity, answers.log)
