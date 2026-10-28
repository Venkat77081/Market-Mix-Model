"""Phase H's optimization job, run in the background by the web UI.

    python -m mmm_tool.plan --run-dir runs/<id> --key <key>

Reads runs/<id>/fit/plan/request_<key>.json (the checked plan), writes result_<key>.json,
or {"error": ...} in the same file.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from ..whatif import model_key, write


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m mmm_tool.plan")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--key", required=True)
    a = p.parse_args(argv)
    run_dir = Path(a.run_dir)
    pd_ = run_dir / "fit" / "plan"
    out = pd_ / f"result_{a.key}.json"
    mk = model_key(run_dir)
    try:
        from ..fit import model as m
        from .optimize import optimize

        checked = json.loads((pd_ / f"request_{a.key}.json").read_text(encoding="utf-8"))
        res = optimize(m.load(run_dir / "fit" / "model.pkl"), checked, pd_ / f"report_{a.key}")
        write(out, {**res, "model_key": mk, "key": a.key})
        return 0
    except Exception as e:  # noqa: BLE001 - shown on the Optimize screen
        write(out, {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=6), "model_key": mk, "key": a.key})
        return 1


if __name__ == "__main__":
    sys.exit(main())
