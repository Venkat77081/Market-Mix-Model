"""Fit and validate from the command line (Phase E).

    python -m mmm_tool.fit data.csv --answers answers.json --sampler quick --out fit/
    python -m mmm_tool.fit data.csv --defaults --stability --placebo

Phases A-D run first with the given answers (--defaults takes every default).
--run-dir is how the web UI starts a fit in the background.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m mmm_tool.fit", description="Fit and validate a Meridian model (Phase E)")
    p.add_argument("file", nargs="?", help="CSV or Excel export")
    p.add_argument("--answers", help="JSON file of answers to the phase A-D questions")
    p.add_argument("--defaults", action="store_true", help="take the default for every unanswered question")
    p.add_argument("--sampler", default="quick", help="quick or full (fit.yaml)")
    p.add_argument("--stability", action="store_true", help="step 34: three extra fits on shorter windows")
    p.add_argument("--placebo", action="store_true", help="step 35: one extra fit with a random channel")
    p.add_argument("--out", default="fit", help="folder for model.pkl and result.json (default ./fit)")
    p.add_argument("--run-dir", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    if args.run_dir:
        from .job import run_job

        return run_job(Path(args.run_dir))
    if not args.file:
        p.error("give a file, or --run-dir")

    from ..pipeline import run_pipeline
    from .job import _jsonable
    from .model import meridian_installed
    from .pipeline import run_phase_e

    if not meridian_installed():
        print("Phase E needs google-meridian: pip install -r requirements-model.txt", file=sys.stderr)
        return 2
    answers = json.loads(Path(args.answers).read_text(encoding="utf-8")) if args.answers else {}
    run = run_pipeline(Path(args.file), answers, accept_defaults=args.defaults)
    if run.d is None or run.d.status != "complete":
        last = run.last
        print(f"Phase {run.phase} needs answers first: {last.steps[-1].summary}", file=sys.stderr)
        for d in last.pending:
            print(f"  {d['id']}: {d['question']}", file=sys.stderr)
        return 1
    out = Path(args.out)
    options = {"sampler": args.sampler, "stability": args.stability, "placebo": args.placebo}
    result = run_phase_e(run.d, options, out_dir=out,
                         progress=lambda step, msg, frac: print(f"[{frac:4.0%}] step {step}: {msg}", flush=True))
    (out / "result.json").write_text(json.dumps(_jsonable(result.to_dict()), indent=2), encoding="utf-8")
    print()
    for s in result.steps:
        print(f"{s.step:>3} {s.name:<30} {s.verdict.value:<5} {s.summary}")
    print(f"\nModel {result.status.upper()}" + (f": {result.error}" if result.error else ""))
    return 0 if result.status == "accepted" else 1


if __name__ == "__main__":
    sys.exit(main())
