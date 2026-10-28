"""Check a model bundle reproduces (phases A-D rebuild the same model table and configuration).

    python -m mmm_tool.export --verify bundle.zip
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m mmm_tool.export")
    p.add_argument("--verify", required=True, help="a bundle zip from step 55")
    a = p.parse_args(argv)
    from .bundle import verify

    with tempfile.TemporaryDirectory() as tmp:
        res = verify(Path(a.verify), Path(tmp))
    print(json.dumps(res, indent=1, default=str))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
