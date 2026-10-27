"""Run the pipeline phases in order on one file with one set of answers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .clean import PhaseBResult, run_phase_b
from .configure import PhaseDResult, run_phase_d
from .profile import PhaseCResult, run_phase_c
from .ingest import Answers, PhaseAResult, run_phase_a
from .ingest.memory import RoleMemory


@dataclass
class PipelineResult:
    a: PhaseAResult
    b: PhaseBResult | None = None
    c: PhaseCResult | None = None
    d: PhaseDResult | None = None

    @property
    def last(self) -> PhaseAResult:
        return self.d or self.c or self.b or self.a

    @property
    def phase(self) -> str:
        """The phase the run has reached: each starts once the one before is complete."""
        return "D" if self.d is not None else "C" if self.c is not None else "B" if self.b is not None else "A"

    @property
    def status(self) -> str:
        return self.last.status

    @property
    def decision_log(self) -> list[dict[str, Any]]:
        return self.last.decision_log


def run_pipeline(path: str | Path, answers: dict[str, Any] | None = None, accept_defaults: bool = False,
                 memory: RoleMemory | None = None) -> PipelineResult:
    ans = Answers(answers, accept_defaults)
    a = run_phase_a(path, ans, memory=memory)
    if a.status != "complete":
        return PipelineResult(a)
    b = run_phase_b(a, ans)
    if b.status != "complete":
        return PipelineResult(a, b)
    c = run_phase_c(b, ans)
    if c.status != "complete":
        return PipelineResult(a, b, c)
    return PipelineResult(a, b, c, run_phase_d(c, ans))
