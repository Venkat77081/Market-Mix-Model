"""Phase G: What-If (steps 43-47). See scenario.py."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def folder(run_dir: Path) -> Path:
    return Path(run_dir) / "fit" / "whatif"


def model_key(run_dir: Path) -> str:
    """Which saved model the cached results belong to (they are recomputed after a refit)."""
    p = Path(run_dir) / "fit" / "model.pkl"
    return hashlib.sha256(f"{p.stat().st_mtime_ns}:{p.stat().st_size}".encode()).hexdigest()[:12] if p.exists() else ""


def scenario_key(period: str, mult: dict[str, float]) -> str:
    body = json.dumps([period, sorted((c, round(float(v), 4)) for c, v in mult.items() if round(float(v), 4) != 1.0)])
    return hashlib.sha256(body.encode()).hexdigest()[:12]


def write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)
