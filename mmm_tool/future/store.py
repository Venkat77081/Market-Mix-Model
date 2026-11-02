"""A project's future plans, kept with the project (runs/projects/<project>/planning/):

  period.json    the future period being planned: dates, template, cost changes, forecast, the
                 model (run and model key) it was set up on
  plans.json     plans A, B, C… for that period: spend per channel and week, predictions per week
                 and in total, where each came from; read-only once saved
  tracking.json  the tracked plan (and earlier ones, with the week each switch took effect), the
                 frozen weekly targets, tracking settings, confirmed column renames, events,
                 the actuals' weeks
  log.json       the planning decision log: every choice, who made it and when

No Meridian here: plans arrive already evaluated (engine.py).
"""
from __future__ import annotations

import getpass
import json
import string
import time
from pathlib import Path

STATUSES = ("draft", "tracked", "archived")


class PlanStoreError(ValueError):
    pass


def who() -> str:
    try:
        return getpass.getuser() or "you"
    except Exception:  # noqa: BLE001 - no user name on this machine
        return "you"


def folder(project_dir: Path) -> Path:
    return Path(project_dir) / "planning"


def _read(p: Path, default):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else default


def _write(p: Path, data) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")
    tmp.replace(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# The planning decision log --------------------------------------------------------------------------

def log(project_dir: Path, what: str, details: dict | None = None, by: str | None = None) -> dict:
    p = folder(project_dir) / "log.json"
    data = _read(p, {"entries": []})
    entry = {"at": now(), "by": by or who(), "what": what, "details": details or {}}
    data["entries"].append(entry)
    _write(p, data)
    return entry


def entries(project_dir: Path) -> list[dict]:
    return _read(folder(project_dir) / "log.json", {"entries": []})["entries"]


# The future period ----------------------------------------------------------------------------------

def period(project_dir: Path) -> dict | None:
    return _read(folder(project_dir) / "period.json", None)


SNAPSHOT_KEYS = ("resolved", "cost", "rpk", "forecast")


def setup_hash(setup: dict) -> str:
    """What a plan's predictions depend on besides the model and the forecast: the weeks, the
    template and lead-in, the cost changes and the money value per KPI."""
    import hashlib

    body = json.dumps([setup.get("resolved"), setup.get("cost") or {}, setup.get("rpk") or None], sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()[:12]


def save_period(project_dir: Path, setup: dict) -> dict:
    """Save the future period. Setting it up again with the same dates keeps its plans (and the
    tracked one) under the same id; what earlier setups were is kept, so a plan made on one is
    still tracked on its own weeks, cost changes and forecast."""
    old = period(project_dir) or {}
    pid = f"{setup['plan'][0]}_{len(setup['plan'])}w"
    snaps = dict(old.get("snapshots") or {}) if old.get("id") == pid else {}
    h = setup_hash(setup)
    snaps[f"{h}:{(setup.get('forecast') or {}).get('hash')}"] = {k: setup.get(k) for k in SNAPSHOT_KEYS}
    setup = {**setup, "id": pid, "setup_hash": h, "snapshots": snaps, "set_at": now(), "set_by": who()}
    _write(folder(project_dir) / "period.json", setup)
    return setup


def snapshot(setup: dict, plan: dict | None) -> dict:
    """The setup a plan was made on (the current one when it isn't known)."""
    if plan:
        snap = (setup.get("snapshots") or {}).get(f"{plan.get('setup_hash')}:{plan.get('forecast_hash')}")
        if snap:
            return {**setup, **snap}
    return setup


# Plans ------------------------------------------------------------------------------------------------

def plans(project_dir: Path, period_id: str | None = None) -> list[dict]:
    allp = _read(folder(project_dir) / "plans.json", {"plans": []})["plans"]
    return [p for p in allp if period_id is None or p["period_id"] == period_id]


def _save_plans(project_dir: Path, items: list[dict]) -> None:
    _write(folder(project_dir) / "plans.json", {"plans": items})


def _next_id(existing: list[str]) -> str:
    letters = string.ascii_uppercase
    n = 0
    while True:
        pid, k = "", n
        while True:
            pid = letters[k % 26] + pid
            k = k // 26 - 1
            if k < 0:
                break
        if pid not in existing:
            return pid
        n += 1


def record(evaluation: dict, name: str, source: dict, setup: dict, run_id: str, model_key: str) -> dict:
    """A plan record from engine.evaluate's result (spend, predictions per week and in total)."""
    pw, weeks = evaluation["per_week"], evaluation["weeks"]
    rows = []
    for i, w in enumerate(weeks):
        g = lambda key, sub=None: (pw[key][sub][i] if sub else pw[key][i]) if pw.get(key) is not None else None  # noqa: E731
        rows.append({"week": w, "spend": {c: v[i] for c, v in evaluation["spend"].items()},
                     "reference": g("reference"), "media": g("media", "mean"), "media_lo": g("media", "lo"), "media_hi": g("media", "hi"),
                     "forecast": g("forecast"), "forecast_lower": g("forecast_lower"), "forecast_upper": g("forecast_upper"),
                     "total": g("total", "mean"), "total_lo": g("total", "lo"), "total_hi": g("total", "hi")})
    f = setup.get("forecast") or {}
    return {"name": name, "source": source, "by": who(), "at": now(), "run_id": run_id, "model_key": model_key,
            "period_id": setup["id"], "setup_hash": setup.get("setup_hash"), "plan_weeks": weeks, "template_weeks": setup["template"],
            "cost": setup.get("cost") or {}, "forecast_source": f.get("source"), "forecast_hash": f.get("hash"),
            "multipliers": evaluation.get("multipliers"), "weeks": rows, "totals": evaluation["totals"],
            "by_channel": evaluation.get("by_channel"), "status": "draft"}


def add(project_dir: Path, plan: dict) -> dict:
    items = plans(project_dir)
    plan = {**plan, "id": _next_id([p["id"] for p in items if p["period_id"] == plan["period_id"]])}
    items.append(plan)
    _save_plans(project_dir, items)
    log(project_dir, "plan saved", {"plan": plan["id"], "name": plan["name"], "source": plan["source"],
                                    "spend": plan["totals"]["spend"], "total": (plan["totals"].get("total") or {}).get("mean"),
                                    "model": plan["model_key"]})
    return plan


def get(project_dir: Path, period_id: str, plan_id: str) -> dict:
    p = next((x for x in plans(project_dir, period_id) if x["id"] == plan_id), None)
    if p is None:
        raise PlanStoreError(f"No plan {plan_id}.")
    return p


def update(project_dir: Path, period_id: str, plan_id: str, name: str | None = None, status: str | None = None) -> dict:
    items = plans(project_dir)
    p = next((x for x in items if x["period_id"] == period_id and x["id"] == plan_id), None)
    if p is None:
        raise PlanStoreError(f"No plan {plan_id}.")
    if name is not None:
        name = name.strip()
        if not 1 <= len(name) <= 80:
            raise PlanStoreError("A plan's name is 1 to 80 characters.")
        log(project_dir, "plan renamed", {"plan": plan_id, "from": p["name"], "to": name})
        p["name"] = name
    if status is not None:
        if status not in ("draft", "archived"):
            raise PlanStoreError("A plan can be archived or restored; choose it for tracking to track it.")
        if p["status"] == "tracked":
            raise PlanStoreError("The tracked plan can't be archived; choose another plan for tracking first.")
        log(project_dir, "plan archived" if status == "archived" else "plan restored", {"plan": plan_id})
        p["status"] = status
    _save_plans(project_dir, items)
    return p


def duplicate(project_dir: Path, period_id: str, plan_id: str) -> dict:
    src = get(project_dir, period_id, plan_id)
    copy = {k: v for k, v in src.items() if k != "id"}
    copy.update(name=f"Copy of {src['name']}"[:80], status="draft", at=now(), by=who(),
                source={**src["source"], "duplicate_of": plan_id})
    return add(project_dir, copy)


def stale(plan: dict, setup: dict | None, current_model_key: str | None) -> list[str]:
    """Why a plan's predictions may be out of date: a newer model, or a changed forecast."""
    why = []
    if current_model_key and plan.get("model_key") != current_model_key:
        why.append("model")
    if setup and (setup.get("forecast") or {}).get("hash") != plan.get("forecast_hash"):
        why.append("forecast")
    if setup and plan.get("setup_hash") and setup.get("setup_hash") and plan["setup_hash"] != setup["setup_hash"]:
        why.append("setup")
    return why


def reference(project_dir: Path, setup: dict) -> dict | None:
    """Plan A of the current setup: the latest reference (or uploaded) plan made on it."""
    refs = [p for p in plans(project_dir, setup["id"]) if p["source"].get("kind") in ("reference", "upload")
            and "recomputed_from" not in p["source"]]
    cur = [p for p in refs if not stale(p, setup, None)]
    return (cur or refs or [None])[-1]


# Tracking ---------------------------------------------------------------------------------------------

TRACKING_PER_PERIOD = ("history", "targets", "actuals", "actual_weeks", "all_weeks", "actuals_file",
                       "pending_key", "pending_payload", "pending_upload")


def tracking(project_dir: Path, period_id: str | None = None) -> dict:
    """Tracking state. Given the current period's id, the state of an earlier period (its targets,
    actuals and history) is put aside under "earlier", so a new period starts clean; settings,
    confirmed column renames and events stay with the project."""
    from .period import settings

    t = _read(folder(project_dir) / "tracking.json", {})
    if period_id and t.get("period_id") not in (None, period_id):
        t.setdefault("earlier", []).append({"period_id": t["period_id"], **{k: t.pop(k) for k in TRACKING_PER_PERIOD if k in t}})
    if period_id:
        t["period_id"] = period_id
    t.setdefault("history", [])
    t.setdefault("targets", {})
    t.setdefault("settings", dict(settings()["tracking"]))
    for k, v in settings()["tracking"].items():
        t["settings"].setdefault(k, v)
    t.setdefault("renames", {})
    t.setdefault("events", [])
    t.setdefault("actual_weeks", [])
    return t


def save_tracking(project_dir: Path, t: dict) -> None:
    _write(folder(project_dir) / "tracking.json", t)


def tracked(project_dir: Path, period_id: str) -> dict | None:
    return next((p for p in plans(project_dir, period_id) if p["status"] == "tracked"), None)


def choose(project_dir: Path, period_id: str, plan_id: str, reason: str = "") -> dict:
    """Choose a plan for tracking: its weekly predictions are frozen as the targets. Switching from
    another plan takes effect from the next week without actuals; weeks already tracked keep theirs."""
    items = plans(project_dir)
    new = next((x for x in items if x["period_id"] == period_id and x["id"] == plan_id), None)
    if new is None:
        raise PlanStoreError(f"No plan {plan_id}.")
    if new["status"] == "archived":
        raise PlanStoreError("Restore the plan before tracking it.")
    old = next((x for x in items if x["period_id"] == period_id and x["status"] == "tracked"), None)
    if old is not None and old["id"] == plan_id:
        return {"plan": new, "from_week": None, "switched_from": None}
    t = tracking(project_dir, period_id)
    done = set(t["actual_weeks"])
    weeks = new["plan_weeks"]
    from_week = next((w for w in weeks if w not in done), None)
    if from_week is None:
        raise PlanStoreError("Every week of the plan already has actuals; there is no week left to track it on.")
    if old is not None and not reason.strip():
        raise PlanStoreError("Say why tracking switches plans; the reason goes into the decision log.")
    for row in new["weeks"]:                        # freeze the targets from from_week on
        if from_week is not None and row["week"] >= from_week:
            t["targets"][row["week"]] = {**row, "plan_id": plan_id}
    t["history"].append({"plan_id": plan_id, "from_week": from_week, "at": now(), "by": who(),
                         "reason": reason.strip(), "switched_from": old["id"] if old else None})
    if old is not None:
        old["status"] = "draft"
    new["status"] = "tracked"
    _save_plans(project_dir, items)
    save_tracking(project_dir, t)
    tot = new["totals"]
    log(project_dir, "plan chosen for tracking" if old is None else "tracking switched",
        {"plan": plan_id, "from_plan": old["id"] if old else None, "from_week": from_week, "reason": reason.strip(),
         "spend": tot["spend"], "total": (tot.get("total") or {}).get("mean"), "media": tot["media"]["mean"],
         "model": new["model_key"]})
    return {"plan": new, "from_week": from_week, "switched_from": old["id"] if old else None}
