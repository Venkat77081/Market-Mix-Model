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
            "waiting": getattr(self, "waiting", False),            # step 30 failed: steps 31+ wait for "Continue anyway"
            "not_converged": getattr(self, "not_converged", False),
            "decision_log": getattr(self, "decision_log", []),
            "seed": getattr(self, "seed", None),
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


def write_summary(mmm, d, steps: list[StepResult], badges: dict, model: dict, out_dir: Path) -> None:
    """Phase F's stored numbers (summary.json) and Meridian's own HTML report. A failure
    here never rejects the model: the reason is kept in the model info instead."""
    import json

    from ..summary import build_summary, meridian_html

    try:
        summary = build_summary(mmm, d, {s.step: s.details for s in steps}, badges, model)
        (out_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        model["summary"] = True
    except Exception as e:  # noqa: BLE001 - shown on the Phase F screen
        model["summary_error"] = f"{type(e).__name__}: {e}"
    try:
        path = Path(meridian_html(mmm, out_dir))
        if model.get("not_converged"):              # the export carries the badge too
            html = path.read_text(encoding="utf-8")
            note = ('<div style="padding:12px 16px;background:#fdecea;color:#8a1c12;font:14px sans-serif">'
                    '<b>Did not converge.</b> This model was kept with "Continue anyway" after step 30 failed: '
                    "the sampler's chains hadn't settled, so its numbers can't be fully trusted.</div>")
            path.write_text(html.replace("<body>", "<body>" + note, 1) if "<body>" in html else note + html, encoding="utf-8")
        model["meridian_html"] = True
    except Exception as e:  # noqa: BLE001
        model["meridian_html_error"] = f"{type(e).__name__}: {e}"


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
    smp = ctx.sampler
    ctx.log += [{"step": 29, "text": f"sampler {smp.get('label', 'Custom')} ({smp['n_chains']} chains, {smp['n_adapt']} adapt, "
                                     f"{smp['n_burnin']} burn-in, {smp['n_keep']} kept)", "source": "user"},
                {"step": 29, "text": f"seed {ctx.seed}", "source": options.seed_source}]
    steps: list[StepResult] = _resume(ctx, d) if options.continue_unconverged else []
    for w in range(28, 28 + len(steps)):
        done["w"] += weights[w]
    error, waiting = None, False
    for fn in STEPS[len(steps):]:
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
        if n == 30 and not ctx.gates[30] and not ctx.not_converged:
            waiting = True                      # steps 31+ wait for "Continue anyway"
            _save_for_resume(ctx, d, steps)
            break
    if error:
        status = "failed"
    else:
        passed = all(ctx.gates.get(n) or (n == 30 and ctx.not_converged) for n in GATES)
        status = "accepted" if passed else "rejected"
        if status == "accepted" and ctx.final is not None and out_dir is not None:
            report(36, "Computing the Model Summary (Phase F)", 0.97)
            write_summary(ctx.final, d, steps, ctx.badges, ctx.model, out_dir)
        report(36, "Done", 1.0)
    channels = []
    if ctx.metrics is not None:
        for ch, r in ctx.metrics.iterrows():
            channels.append({"channel": ch, **{k: (None if v != v else v) for k, v in r.to_dict().items()},
                             "badges": [BADGE_TEXT.get(b, b) for b in ctx.badges.get(ch, [])]})
    result = PhaseEResult(status, steps, options.to_dict(), ctx.sampler, dict(ctx.gates), ctx.badges, channels,
                          ctx.model, ctx.timings, error)
    result.waiting = waiting
    result.not_converged = ctx.not_converged
    result.decision_log = ctx.log
    result.seed = ctx.seed
    return result


def _state_key(ctx: ECtx, d) -> str:
    """What a saved holdout fit was built from: data, model settings, seed and sampler."""
    import hashlib
    import json

    h = hashlib.sha256(d.table.to_csv(index=False).encode())
    h.update(json.dumps([d.config, ctx.seed, ctx.sampler], sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def _save_for_resume(ctx: ECtx, d, steps: list[StepResult]) -> None:
    """Keep the holdout fit after step 30 fails, so "Continue anyway" carries on without refitting."""
    import json

    from . import model as m

    if ctx.out_dir is None:
        return
    m.save(ctx.mmm, ctx.out_dir / "holdout_model.pkl")
    state = {"key": _state_key(ctx, d), "fit_sampler": ctx.fit_sampler, "timings": ctx.timings,
             "steps": [s.to_dict() for s in steps]}
    (ctx.out_dir / "resume.json").write_text(json.dumps(state, default=str), encoding="utf-8")


def _resume(ctx: ECtx, d) -> list[StepResult]:
    """Steps 28-30 from the saved holdout fit, if it was built from the same inputs; else [] (fit again)."""
    import json

    from . import model as m

    if ctx.out_dir is None or not (ctx.out_dir / "resume.json").exists() or not (ctx.out_dir / "holdout_model.pkl").exists():
        return []
    state = json.loads((ctx.out_dir / "resume.json").read_text(encoding="utf-8"))
    if state.get("key") != _state_key(ctx, d):
        return []
    ctx.mmm = m.load(ctx.out_dir / "holdout_model.pkl")
    ctx.fit_sampler = state["fit_sampler"]
    ctx.timings = dict(state["timings"])
    ctx.gates[30] = False
    ctx.not_converged = True
    ctx.log.append({"step": 30, "text": "continue although the model did not converge", "source": "user"})
    steps = []
    for s in state["steps"]:
        st = StepResult(s["step"], s["name"], Verdict(s["verdict"]), s["summary"], s["details"], s.get("warnings", []))
        if st.step == 30:
            st.details["continue_anyway"] = True
            st.summary = st.summary.replace("Failed: the chains disagree", "Did not converge") + \
                " You chose to continue: every later result carries a 'did not converge' badge."
            st.warnings = ["The sampler hasn't settled, so these numbers can't be fully trusted."]
        steps.append(st)
    return steps
