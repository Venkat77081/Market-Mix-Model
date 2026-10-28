"""Rebuild the Model Summary (Phase F) for a run whose model was accepted.

    python -m mmm_tool.summary --run-dir runs/<id>

Phase E writes fit/summary.json itself after an accepted fit; this rebuilds it
from fit/model.pkl, for instance for a model fitted before Phase F existed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..profile.context import BADGE_TEXT as BADGE_CODES  # badge code -> the text result.json stores


def rebuild(run_dir: Path) -> Path:
    from ..fit import model as m
    from ..fit.pipeline import write_summary
    from ..ingest.memory import RoleMemory
    from ..ingest.types import StepResult, Verdict
    from ..pipeline import run_pipeline

    fd = run_dir / "fit"
    result = json.loads((fd / "result.json").read_text(encoding="utf-8"))
    if result.get("status") != "accepted" or not (fd / "model.pkl").exists():
        raise SystemExit("No accepted model in this run: fit it in Phase E first.")
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    req = json.loads((fd / "request.json").read_text(encoding="utf-8")) if (fd / "request.json").exists() else {}
    memory = RoleMemory(req["memory"]) if req.get("memory") else None
    run = run_pipeline(run_dir / meta["file"], meta["answers"], memory=memory)
    if run.d is None or run.d.status != "complete":
        raise SystemExit("Phase D isn't complete for this run's current answers.")
    steps = [StepResult(s["step"], s["name"], Verdict(s["verdict"]), s["summary"], s["details"], s.get("warnings", []))
             for s in result["steps"]]
    badges = {c: [k for k, v in BADGE_CODES.items() if v in texts] for c, texts in result.get("badges", {}).items()}
    model = dict(result.get("model") or {})
    write_summary(m.load(fd / "model.pkl"), run.d, steps, badges, model, fd)
    if model.get("summary_error"):
        raise SystemExit(model["summary_error"])
    return fd / "summary.json"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m mmm_tool.summary", description="Rebuild a run's Model Summary (Phase F)")
    p.add_argument("--run-dir", required=True)
    args = p.parse_args(argv)
    print(rebuild(Path(args.run_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
