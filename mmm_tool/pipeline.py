"""Run the pipeline phases in order on one file with one set of answers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .clean import PhaseBResult, run_phase_b
from .ingest import Answers, PhaseAResult, run_phase_a
from .ingest.memory import RoleMemory


@dataclass
class PipelineResult:
    a: PhaseAResult
    b: PhaseBResult | None = None

    @property
    def phase(self) -> str:
        """The phase the run has reached: 'A' until A is complete, then 'B'."""
        return "B" if self.b is not None else "A"

    @property
    def status(self) -> str:
        return (self.b or self.a).status

    @property
    def decision_log(self) -> list[dict[str, Any]]:
        return (self.b or self.a).decision_log


def run_pipeline(path: str | Path, answers: dict[str, Any] | None = None, accept_defaults: bool = False,
                 memory: RoleMemory | None = None) -> PipelineResult:
    ans = Answers(answers, accept_defaults)
    a = run_phase_a(path, ans, memory=memory)
    if a.status != "complete":
        return PipelineResult(a)
    return PipelineResult(a, run_phase_b(a, ans))
