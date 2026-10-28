"""Step 58: refit an approved model on new data, then compare its ROI with the approved version.

A refit is a new run, from Phase A, on the new export (more weeks, usually), reusing every
answer the approved run gave (and the same seed and sampler), so only what is new in the data
asks for a decision. Once it is fitted, each channel's ROI is compared with the approved
version's: a channel whose new mean lies outside the approved 90% range is flagged.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def start(source_dir: Path, new_dir: Path, new_file: Path, line: str, version: int,
          roles: dict[str, str] | None = None) -> dict[str, Any]:
    """Make the refit run's meta.json: the approved run's answers, seed and fit options.

    roles are the approved run's confirmed roles. They are kept for the columns still in the new
    file, so an accepted suggestion can't change under a refit; a new column is suggested and asked."""
    src = json.loads((Path(source_dir) / "meta.json").read_text(encoding="utf-8"))
    answers = dict(src.get("answers", {}))
    if roles:
        new_dir.mkdir(parents=True, exist_ok=True)
        cols = _columns(Path(new_file), answers)
        answers["roles"] = {c: r for c, r in roles.items() if c in cols}
    fd = Path(source_dir) / "fit"
    req = json.loads((fd / "request.json").read_text(encoding="utf-8")) if (fd / "request.json").exists() else {}
    opts = {k: v for k, v in (req.get("options") or {}).items() if k in ("sampler", "stability", "placebo", "seed")}
    meta = {"file": Path(new_file).name, "answers": answers, "seed": opts.get("seed") or src.get("seed"),
            "refit_of": {"line": line, "version": version, "run_id": Path(source_dir).name, "file": src.get("file")},
            "refit_options": opts}
    new_dir.mkdir(parents=True, exist_ok=True)
    if Path(new_file).parent != new_dir:
        shutil.copy(new_file, new_dir / Path(new_file).name)
    (new_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def _columns(path: Path, answers: dict) -> set[str]:
    from ..ingest.load import load_file
    from ..ingest.types import Answers

    df, _ = load_file(path, Answers(answers))
    return set(df.columns) if df is not None else set()


def compare(base: dict, new: dict) -> dict[str, Any]:
    """Per channel: the approved ROI (mean, 90% range) and the refit's; flagged when the refit's
    mean is outside the approved range. Channels only in one of them are listed as such."""
    rows, flagged = [], []
    for ch in sorted(set(base.get("roi") or {}) | set(new.get("roi") or {})):
        a, b = (base.get("roi") or {}).get(ch), (new.get("roi") or {}).get(ch)
        if a is None or b is None:
            rows.append({"channel": ch, "approved": a, "refit": b, "status": "new channel" if a is None else "dropped"})
            continue
        outside = not (a["lo"] <= b["mean"] <= a["hi"])
        rows.append({"channel": ch, "approved": a, "refit": b, "moved": b["mean"] - a["mean"],
                     "status": "beyond its 90% range" if outside else "inside its range"})
        if outside:
            flagged.append(ch)
    changed = [r["channel"] for r in rows if r["status"] in ("new channel", "dropped")]
    if flagged:
        summary = (f"{', '.join(flagged)} flagged: ROI moved beyond {'its' if len(flagged) == 1 else 'their'} "
                   "approved 90% range")
    elif rows:
        summary = "ROI moved inside its range on all channels"
    else:
        summary = "No ROI to compare"
    if changed:
        summary += f"; channels changed: {', '.join(changed)}"
    return {"rows": rows, "flagged": flagged, "summary": summary,
            "approved": {"version": base.get("version"), "data_to": base.get("data_to"), "weeks": base.get("weeks")},
            "refit": {"data_to": new.get("data_to"), "weeks": new.get("weeks")}}
