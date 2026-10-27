"""Run Phase A from the command line.

    mmm-ingest data.xlsx                      # stops at the first decision
    mmm-ingest data.xlsx --answers a.json     # answers keyed by decision id
    mmm-ingest data.xlsx --accept-defaults    # take every suggestion
"""
from __future__ import annotations

import argparse
import json
import sys

from .pipeline import run_phase_a

MARK = {"pass": "✓", "warn": "!", "needs_input": "?", "fail": "✗"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mmm-ingest", description="Phase A: ingest (steps 1-6)")
    p.add_argument("file")
    p.add_argument("--answers", help="JSON file of answers keyed by decision id")
    p.add_argument("--accept-defaults", action="store_true")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.add_argument("--out", help="write the ingested table to this CSV")
    args = p.parse_args(argv)

    answers = json.load(open(args.answers)) if args.answers else {}
    result = run_phase_a(args.file, answers, accept_defaults=args.accept_defaults)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        for s in result.steps:
            print(f"[{MARK[s.verdict.value]}] {s.step}. {s.name}: {s.summary}")
            for w in s.warnings:
                print(f"      ! {w}")
        for d in result.pending:
            print(f"\nDecision needed ({d['id']}): {d['question']}")
            print(f"  suggested: {json.dumps(d['default'], default=str)}")
        if result.status == "complete":
            print("\nMeridian CoordToColumns:")
            print(json.dumps(result.coord_to_columns(), indent=2, default=str))
    if args.out and result.table is not None:
        result.table.to_csv(args.out, index=False)
    return {"complete": 0, "needs_input": 2, "failed": 1}[result.status]


if __name__ == "__main__":
    sys.exit(main())
