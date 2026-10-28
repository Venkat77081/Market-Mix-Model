"""Phase G jobs, run in the background by the web UI (Meridian stays out of the server).

    python -m mmm_tool.whatif --run-dir runs/<id> --curves all        step 44's curves for a period
    python -m mmm_tool.whatif --run-dir runs/<id> --exact <key>       step 45 for a stored request

Results go to runs/<id>/fit/whatif/; a failure is written to the same file as {"error": ...}.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from . import folder, model_key, write


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m mmm_tool.whatif")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--curves")
    p.add_argument("--exact")
    a = p.parse_args(argv)
    run_dir = Path(a.run_dir)
    wd = folder(run_dir)
    key = model_key(run_dir)
    out = wd / (f"curves_{a.curves}.json" if a.curves else f"exact_{a.exact}.json")
    try:
        from ..fit import model as m
        from .scenario import curves, exact

        mmm = m.load(run_dir / "fit" / "model.pkl")
        if a.curves:
            res = curves(mmm, a.curves)
        else:
            req = json.loads((wd / f"request_{a.exact}.json").read_text(encoding="utf-8"))
            res = exact(mmm, req["multipliers"], req["period"])
        write(out, {**res, "model_key": key})
        return 0
    except Exception as e:  # noqa: BLE001 - shown on the What-If screen
        write(out, {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=6), "model_key": key})
        return 1


if __name__ == "__main__":
    sys.exit(main())
