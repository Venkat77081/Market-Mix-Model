"""Shared result types for pipeline steps.

Every step reports a verdict, a one-line summary, details for its chart or
table, run-level warnings, and the decision prompts it needs answered. This
mirrors the forecast tool: steps run on their own and only stop at a marked
decision point.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    PASS = "pass"
    WARN = "warn"
    NEEDS_INPUT = "needs_input"
    FAIL = "fail"


# Column roles. Each maps one-to-one to a Meridian CoordToColumns field
# (see coord_to_columns in pipeline.py); "excluded" drops the column.
ROLES = (
    "date",
    "geo",
    "population",
    "kpi",
    "revenue_per_kpi",
    "media",
    "media_spend",
    "reach",
    "frequency",
    "rf_spend",
    "control",
    "non_media_treatment",
    "organic_media",
    "revenue_for_kpi",  # total money for the KPI (premium for policies); step 5 turns it into revenue per KPI
    "needs_role",       # step 3 couldn't tell; the run waits until the user assigns one
    "excluded",
)


@dataclass
class Decision:
    """A prompt the user answers. `default` is what the tool suggests."""

    id: str
    step: int
    question: str
    options: list[Any]
    default: Any = None
    blocking: bool = False
    context: dict[str, Any] = field(default_factory=dict)
    answered: bool = False
    answer: Any = None

    @property
    def pending(self) -> bool:
        return self.blocking and not self.answered


@dataclass
class StepResult:
    step: int
    name: str
    verdict: Verdict
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)

    @property
    def blocks(self) -> bool:
        return self.verdict == Verdict.FAIL or any(d.pending for d in self.decisions)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        d["blocks"] = self.blocks
        for dd, dec in zip(d["decisions"], self.decisions):
            dd["pending"] = dec.pending
        return d


class Answers:
    """User answers to decision prompts, keyed by decision id.

    With accept_defaults=True every unanswered decision takes its default,
    which is how tests and unattended refits run. Every lookup is logged so
    the bundle export (step 55) can show who decided what.
    """

    def __init__(self, values: dict[str, Any] | None = None, accept_defaults: bool = False):
        self.values = dict(values or {})
        self.accept_defaults = accept_defaults
        self.log: list[dict[str, Any]] = []

    def ask(self, decision: Decision) -> Any:
        """Record the answer on the decision and return it (None if unanswered)."""
        answered, value = self.resolve(decision)
        decision.answered, decision.answer = answered, value
        return value

    def resolve(self, decision: Decision) -> tuple[bool, Any]:
        """Return (answered, value). Unanswered decisions stay pending."""
        if decision.id in self.values:
            value = self.values[decision.id]
            source = "user"
        elif self.accept_defaults and decision.default is not None:
            value = decision.default
            source = "default"
        else:
            return False, None
        self.log.append(
            {
                "decision": decision.id,
                "step": decision.step,
                "question": decision.question,
                "answer": value,
                "source": source,
            }
        )
        return True, value
