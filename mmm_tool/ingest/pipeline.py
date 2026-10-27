"""Phase A: ingest (steps 1-6).

Turns a CSV or Excel export into a table where every column has a confirmed
role. Steps run in order; the run stops at the first step with a pending
blocking decision and reports what it needs. Re-run with the answers to go on.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .dates import parse_dates
from .granularity import detect_granularity
from .kpi import set_kpi_type
from .load import load_file
from .memory import RoleMemory
from .pairing import Channel, pair_channels
from .roles import map_roles
from .types import Answers, StepResult, Verdict


@dataclass
class PhaseAResult:
    status: str  # "complete", "needs_input" or "failed"
    steps: list[StepResult]
    table: pd.DataFrame | None = None
    roles: dict[str, str] = field(default_factory=dict)
    channels: list[Channel] = field(default_factory=list)
    kpi: dict[str, Any] = field(default_factory=dict)
    granularity: dict[str, Any] = field(default_factory=dict)
    decision_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def warnings(self) -> list[dict[str, Any]]:
        """Run-level warnings, shown again beside the numbers they affect."""
        return [{"step": s.step, "message": w} for s in self.steps for w in s.warnings]

    @property
    def pending(self) -> list[dict[str, Any]]:
        return [{**asdict(d), "pending": True} for s in self.steps for d in s.decisions if d.pending]

    def coord_to_columns(self) -> dict[str, Any]:
        """Column names for Meridian's CoordToColumns plus the *_to_channel maps
        its DataFrame loader needs. Media lists are aligned by channel."""
        if self.status != "complete":
            raise ValueError("Phase A is not complete.")
        by_role = lambda r: [c for c, v in self.roles.items() if v == r]  # noqa: E731
        one = lambda r: (by_role(r) or [None])[0]  # noqa: E731
        media = [c for c in self.channels if c.kind == "media"]
        rf = [c for c in self.channels if c.kind == "rf"]
        regional = self.granularity.get("geo_mode") == "regional"
        mapping: dict[str, Any] = {
            "time": one("date"),
            "geo": one("geo") if regional else None,
            "population": one("population") if regional else None,
            "kpi": one("kpi"),
            "revenue_per_kpi": self.kpi.get("revenue_per_kpi_column"),
            "controls": by_role("control"),
            "media": [c.exposure for c in media],
            "media_spend": [c.spend for c in media],
            "reach": [c.reach for c in rf],
            "frequency": [c.frequency for c in rf],
            "rf_spend": [c.spend for c in rf],
            "non_media_treatments": by_role("non_media_treatment"),
            "organic_media": by_role("organic_media"),
        }
        channel_maps = {
            "media_to_channel": {c.exposure: c.name for c in media},
            "media_spend_to_channel": {c.spend: c.name for c in media},
            "reach_to_channel": {c.reach: c.name for c in rf},
            "frequency_to_channel": {c.frequency: c.name for c in rf},
            "rf_spend_to_channel": {c.spend: c.name for c in rf},
        }
        return {
            "coord_to_columns": {k: v for k, v in mapping.items() if v not in (None, [])},
            **{k: v for k, v in channel_maps.items() if v},
            "kpi_type": self.kpi.get("kpi_type"),
            "revenue_per_kpi_value": self.kpi.get("revenue_per_kpi_value"),
        }

    def to_meridian(self):
        """Build meridian.data.load.CoordToColumns (needs meridian installed)."""
        from meridian.data.load import CoordToColumns  # type: ignore

        return CoordToColumns(**self.coord_to_columns()["coord_to_columns"])

    def to_dict(self) -> dict[str, Any]:
        out = {
            "status": self.status,
            "steps": [s.to_dict() for s in self.steps],
            "pending": self.pending,
            "warnings": self.warnings,
            "decision_log": self.decision_log,
        }
        if self.status == "complete":
            out.update(
                roles=self.roles,
                channels=[asdict(c) for c in self.channels],
                kpi=self.kpi,
                granularity=self.granularity,
                meridian=self.coord_to_columns(),
                rows=len(self.table),
            )
        return out


def run_phase_a(path: str | Path, answers: dict[str, Any] | Answers | None = None, accept_defaults: bool = False,
                memory: RoleMemory | None = None) -> PhaseAResult:
    """Run steps 1-6. With a RoleMemory, step 3 suggests roles the user
    confirmed before and stores the roles confirmed in this run."""
    ans = answers if isinstance(answers, Answers) else Answers(answers, accept_defaults)
    steps: list[StepResult] = []

    def stop(step: StepResult) -> PhaseAResult:
        steps.append(step)
        status = "failed" if step.verdict == Verdict.FAIL else "needs_input"
        return PhaseAResult(status, steps, decision_log=ans.log)

    table, step = load_file(path, ans)
    if step.blocks or table is None:
        return stop(step)
    steps.append(step)

    table, date_col, step = parse_dates(table, ans, lines=steps[0].details.get("data_lines"))
    if step.blocks or table is None:
        return stop(step)
    steps.append(step)

    roles, table, step = map_roles(table, date_col, ans, memory, Path(path).name)
    if step.blocks or roles is None:
        return stop(step)
    steps.append(step)
    if roles.get(date_col) != "date":
        # The user moved the date role; parse that column instead.
        new_date = next(c for c, r in roles.items() if r == "date")
        ans.values["date_column"] = new_date
        return run_phase_a(path, ans, memory=memory)

    channels, roles, step = pair_channels(roles, ans, table)
    if step.blocks:
        return stop(step)
    steps.append(step)

    kpi, step, table = set_kpi_type(table, roles, ans)
    if step.blocks:
        return stop(step)
    steps.append(step)

    gran, step = detect_granularity(table, roles, ans)
    if step.blocks:
        return stop(step)
    steps.append(step)

    # The revenue total behind a derived revenue per KPI stays (set aside) so
    # Phase B can clean it and rebuild the ratio from cleaned totals.
    table = table.drop(columns=[c for c, r in roles.items() if r == "excluded" and c != kpi.get("revenue_column")])
    return PhaseAResult("complete", steps, table, roles, channels, kpi, gran, ans.log)
