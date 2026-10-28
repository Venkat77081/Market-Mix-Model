"""Step 57: the model registry. Every version is kept; one per product line is approved.

registry.json sits next to the runs (runs/registry.json):

    {"lines": {"<product line>": {"versions": [
        {"version": 1, "run_id": "...", "status": "draft" | "approved" | "superseded" | "retired",
         "registered_at", "approved_at", "note", "data_from", "data_to", "weeks", "kpi",
         "channels", "gates": {"30": true, ...}, "status_fit": "accepted" | "rejected",
         "roi": {channel: {"mean", "lo", "hi"}}, "roi_unit", "seed", "data_version",
         "refit_of": {"line", "version", "run_id"} | null, "refit": {comparison} | null}]}}}

A version is registered from a finished fit (a rejected one stays a draft); approving an
accepted model makes it the line's approved version and the previous one superseded.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

LINE_OK = re.compile(r"^[^\x00-\x1f/\\]{1,60}$")


class RegistryError(ValueError):
    pass


def path(runs_dir: Path) -> Path:
    return Path(runs_dir) / "registry.json"


def load(runs_dir: Path) -> dict[str, Any]:
    p = path(runs_dir)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"lines": {}}


def save(runs_dir: Path, reg: dict[str, Any]) -> None:
    p = path(runs_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    tmp.replace(p)


def find_run(reg: dict, run_id: str) -> tuple[str, dict] | None:
    for line, x in reg["lines"].items():
        for v in x["versions"]:
            if v["run_id"] == run_id:
                return line, v
    return None


def approved(reg: dict, line: str) -> dict | None:
    return next((v for v in reg["lines"].get(line, {}).get("versions", []) if v["status"] == "approved"), None)


def snapshot(run_dir: Path, period: dict | None = None) -> dict[str, Any]:
    """What a version records about its model: from the fit result and, for an accepted model,
    the Model Summary's ROI table (all weeks). period ({data_from, data_to, weeks}, from the run's
    Phase D table) fills in the dates a rejected model has no summary for."""
    fd = Path(run_dir) / "fit"
    if not (fd / "result.json").exists():
        raise RegistryError("This run has no finished fit to register: fit it in Phase E first.")
    res = json.loads((fd / "result.json").read_text(encoding="utf-8"))
    if res.get("status") not in ("accepted", "rejected"):
        raise RegistryError(f"The fit {res.get('status') or 'did not finish'}; only a finished fit can be registered.")
    summary = json.loads((fd / "summary.json").read_text(encoding="utf-8")) if (fd / "summary.json").exists() else {}
    v = summary.get("version") or {}
    roi = {}
    for r in ((summary.get("roi") or {}).get("all") or {}).get("rows", []):
        if r["channel"] != "all paid":
            roi[r["channel"]] = {k: float(r["roi"][k]) for k in ("mean", "lo", "hi")}
    if not roi:                                    # a rejected model has no summary: the step 32 fit's ROI
        for c in res.get("channels", []):
            if c.get("paid"):
                roi[c["channel"]] = {"mean": c["roi_posterior_mean"], "lo": c["roi_posterior_ci_lo"], "hi": c["roi_posterior_ci_hi"]}
    model = res.get("model") or {}
    period = period or {}
    return {"status_fit": res["status"], "gates": {str(g["step"]): g["passed"] for g in res.get("gates", [])},
            "roi": roi, "channels": sorted(roi), "roi_unit": v.get("roi_unit"), "kpi": v.get("kpi"),
            "data_from": v.get("first_week") or period.get("data_from"), "data_to": v.get("last_week") or period.get("data_to"),
            "weeks": v.get("weeks") or model.get("weeks") or period.get("weeks"),
            "seed": res.get("seed") or model.get("seed"), "data_version": v.get("data_version") or model.get("data_version"),
            "not_converged": bool(res.get("not_converged"))}


def register(runs_dir: Path, run_dir: Path, line: str, note: str = "", refit_of: dict | None = None,
             period: dict | None = None) -> dict:
    line = (line or "").strip()
    if not LINE_OK.match(line):
        raise RegistryError("Give the product line a name (up to 60 characters, no slashes).")
    reg = load(runs_dir)
    run_id = Path(run_dir).name
    if find_run(reg, run_id):
        raise RegistryError("This run is already registered.")
    snap = snapshot(run_dir, period)
    versions = reg["lines"].setdefault(line, {"versions": []})["versions"]
    entry = {"version": 1 + max((v["version"] for v in versions), default=0), "run_id": run_id, "status": "draft",
             "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "approved_at": None, "note": note[:500],
             "refit_of": refit_of, "refit": None, **snap}
    if refit_of:
        base = next((v for v in reg["lines"].get(refit_of["line"], {}).get("versions", [])
                     if v["version"] == refit_of["version"]), None)
        if base:
            from .refit import compare
            entry["refit"] = compare(base, entry)
    versions.append(entry)
    save(runs_dir, reg)
    return entry


def set_status(runs_dir: Path, line: str, version: int, action: str, stale: bool = False) -> dict:
    reg = load(runs_dir)
    versions = reg["lines"].get(line, {}).get("versions", [])
    v = next((x for x in versions if x["version"] == version), None)
    if v is None:
        raise RegistryError(f"No version {version} of '{line}'.")
    if action == "approve":
        if v["status"] == "retired":
            raise RegistryError("A retired version can't be approved again; register the run's model as a new version.")
        if v["status_fit"] != "accepted":
            raise RegistryError("Only an accepted model can be approved; this one failed a gate in Phase E.")
        if v.get("not_converged"):
            raise RegistryError("This model did not converge (step 30), so it can't be approved.")
        if stale:
            raise RegistryError("The run's data or settings changed after this fit; refit before approving.")
        for x in versions:
            if x["status"] == "approved":
                x["status"] = "superseded"
        v["status"], v["approved_at"] = "approved", time.strftime("%Y-%m-%dT%H:%M:%S")
    elif action == "retire":
        v["status"] = "retired"
    else:
        raise RegistryError(f"Unknown action '{action}'.")
    save(runs_dir, reg)
    return v


def rows(reg: dict) -> list[dict]:
    """One row per product line, as the registry table shows it."""
    out = []
    for line, x in sorted(reg["lines"].items()):
        vs = x["versions"]
        ap = next((v for v in vs if v["status"] == "approved"), None)
        latest = max(vs, key=lambda v: v["version"])
        last_refit = next((v for v in sorted(vs, key=lambda v: -v["version"]) if v.get("refit")), None)
        out.append({"line": line, "approved": ap, "latest": latest, "versions": vs,
                    "last_refit": last_refit["refit"]["summary"] if last_refit else None,
                    "last_refit_version": last_refit["version"] if last_refit else None})
    return out
