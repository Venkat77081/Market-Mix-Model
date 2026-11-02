"""Future-planning jobs, run in the background by the web UI (Meridian stays out of the server).

    python -m mmm_tool.future --run-dir <run> --key <key>

Reads <run>/fit/future/request_<key>.json ({"kind": curves | evaluate | optimize, ...}), writes
result_<key>.json with the model key it was computed on, or {"error": ...}.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from ..whatif import model_key, write


def folder(run_dir: Path) -> Path:
    return Path(run_dir) / "fit" / "future"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m mmm_tool.future")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--key", required=True)
    a = p.parse_args(argv)
    run_dir = Path(a.run_dir)
    fd = folder(run_dir)
    out, mk = fd / f"result_{a.key}.json", model_key(run_dir)
    try:
        from ..fit import model as m
        from . import engine

        req = json.loads((fd / f"request_{a.key}.json").read_text(encoding="utf-8"))
        mmm = m.load(run_dir / "fit" / "model.pkl")
        kind, R = req["kind"], req["resolved"]
        if kind == "curves":
            res = engine.curves(mmm, R, req.get("cost"), req.get("rpk"), req["grid"])
        elif kind == "evaluate":
            res = engine.evaluate(mmm, R, req.get("forecast"), mult=req.get("mult"), spend=req.get("spend"),
                                  cost=req.get("cost"), rpk=req.get("rpk"), units=req.get("units"))
        elif kind == "optimize":
            res = engine.optimize(mmm, R, req.get("forecast"), req.get("cost"), req.get("rpk"), req["checked"])
        else:
            raise ValueError(f"Unknown job '{kind}'.")
        write(out, {**res, "kind": kind, "model_key": mk, "key": a.key})
        return 0
    except Exception as e:  # noqa: BLE001 - shown on the screen that asked
        write(out, {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=6), "model_key": mk, "key": a.key})
        return 1


if __name__ == "__main__":
    sys.exit(main())
