"""Phase A: ingest (steps 1-6)."""
from .pipeline import PhaseAResult, run_phase_a
from .types import Answers, Decision, StepResult, Verdict

__all__ = ["Answers", "Decision", "PhaseAResult", "StepResult", "Verdict", "run_phase_a"]
