"""Phase E: fit and validate (steps 28-36).

Takes a complete Phase D result. Fitting takes minutes, so this runs as a
background job (job.py), never inside a request. A model is accepted only if
it passes every gate (steps 30, 31 and 32); a failed gate doesn't stop the
other checks, so every reason shows, but the final fit (step 36) only runs
for an accepted model.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..ingest.types import StepResult, Verdict
from ..profile.context import BADGE_TEXT
from .context import ECtx, FitOptions, Progress
from .steps import GATES, NAMES, STEPS

# Rough share of the run each step takes, for the progress bar (fits dominate).
WEIGHTS = {28: 1, 29: 6, 30: 1, 31: 1, 32: 1, 33: 0.2, 34: 18, 35: 6, 36: 6}


@dataclass
class PhaseEResult:
    status: str  # "accepted", "rejected" or "failed"
    steps: list[StepResult]
    options: dict[str, Any]
    sampler: dict[str, Any]
    gates: dict[int, bool] = field(default_factory=dict)
    badges: dict[str, list[str]] = field(default_factory=dict)
    channels: list[dict[str, Any]] = field(default_factory=list)
    model: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "steps": [s.to_dict() for s in self.steps],
            "gates": [{"step": n, "name": NAMES[n], "passed": self.gates.get(n)} for n in GATES],
            "badges": {c: [BADGE_TEXT.get(b, b) for b in bs] for c, bs in self.badges.items()},
            "channels": self.channels,
            "model": self.model,
            "options": self.options,
            "sampler": self.sampler,
            "timings": self.timings,
            "error": self.error,
        }


def run_phase_e(d, options: FitOptions | dict | None = None, out_dir: Path | None = None,
                progress: Progress | None = None) -> PhaseEResult:
    if d is None or d.status != "complete":
        raise ValueError("Phase E needs a complete Phase D result.")
    options = options if isinstance(options, FitOptions) else FitOptions.from_dict(options)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    weights = {**WEIGHTS, 34: WEIGHTS[34] if options.stability else 0, 35: WEIGHTS[35] if options.placebo else 0}
    total = sum(weights.values())
    done = {"w": 0.0}

    def report(step: int, message: str, within: float) -> None:
        if progress:
            progress(step, message, min(1.0, (done["w"] + weights[step] * within) / total))

    ctx = ECtx(d=d, options=options, out_dir=out_dir, progress=report)
    steps: list[StepResult] = []
    error = None
    for fn in STEPS:
        n = 28 + len(steps)
        report(n, NAMES[n], 0.0)
        try:
            steps.append(fn(ctx))
        except Exception as e:  # a Meridian error: show it on the step instead of losing the run
            error = f"Step {n} ({NAMES[n]}): {type(e).__name__}: {e}"
            steps.append(StepResult(n, NAMES[n], Verdict.FAIL, f"Stopped: {type(e).__name__}: {e}",
                                    {"traceback": traceback.format_exc(limit=8)}))
            break
        done["w"] += weights[n]
    if error:
        status = "failed"
    else:
        status = "accepted" if all(ctx.gates.get(n) for n in GATES) else "rejected"
        report(36, "Done", 1.0)
    channels = []
    if ctx.metrics is not None:
        for ch, r in ctx.metrics.iterrows():
            channels.append({"channel": ch, **{k: (None if v != v else v) for k, v in r.to_dict().items()},
                             "badges": [BADGE_TEXT.get(b, b) for b in ctx.badges.get(ch, [])]})
    return PhaseEResult(status, steps, options.to_dict(), ctx.sampler, dict(ctx.gates), ctx.badges, channels,
                        ctx.model, ctx.timings, error)
