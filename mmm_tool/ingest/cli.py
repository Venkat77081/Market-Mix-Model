"""Run Phase A from the command line.

    mmm-ingest data.xlsx                      # stops at the first decision
    mmm-ingest data.xlsx --answers a.json     # answers keyed by decision id
    mmm-ingest data.xlsx --accept-defaults    # take every suggestion
    mmm-ingest --forget                       # clear remembered column roles

Roles you confirm (answer "roles") are remembered in .mmm/role_memory.json
(MMM_ROLE_MEMORY) and suggested first next time. Your own role keywords go
in roles.yaml (MMM_ROLES_FILE).
"""
from __future__ import annotations

import argparse
import json
import sys

from .keywords import KeywordError
from .memory import RoleMemory
from .pipeline import run_phase_a

MARK = {"pass": "✓", "warn": "!", "needs_input": "?", "fail": "✗"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mmm-ingest", description="Phase A: ingest (steps 1-6)")
    p.add_argument("file", nargs="?")
    p.add_argument("--answers", help="JSON file of answers keyed by decision id")
    p.add_argument("--accept-defaults", action="store_true")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.add_argument("--out", help="write the ingested table to this CSV")
    p.add_argument("--no-memory", action="store_true", help="don't use or update remembered column roles")
    p.add_argument("--forget", action="store_true", help="clear remembered column roles and exit")
    args = p.parse_args(argv)

    memory = None if args.no_memory else RoleMemory()
    if args.forget:
        n = RoleMemory().forget()
        print(f"Forgot {n} remembered column role(s).")
        return 0
    if not args.file:
        p.error("give a file to ingest")

    answers = json.load(open(args.answers)) if args.answers else {}
    try:
        result = run_phase_a(args.file, answers, accept_defaults=args.accept_defaults, memory=memory)
    except KeywordError as e:
        print(f"Role keywords: {e}", file=sys.stderr)
        return 1

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
