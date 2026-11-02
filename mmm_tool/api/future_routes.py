"""API for future planning (Phases G and H in Future mode) and the Tracking screen.

Meridian runs only in background jobs (python -m mmm_tool.future); these routes read the run's
model table (pandas), save the project's plans, and start or read those jobs.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

router = APIRouter()


def A():
    from . import app as a                      # the app module: RUNS_DIR is read at call time (tests patch it)

    return a


# Context: the run, its model table and the project ---------------------------------------------------------

def _ctx(run_id: str) -> dict[str, Any]:
    from .. import projects, whatif
    from ..fit import job
    from ..plan.optimize import display_name

    a = A()
    d = a._run_dir(run_id)
    a._accepted_model(d)
    slug = projects.project_of(d)
    if not slug:
        raise HTTPException(409, "Future plans are kept with a project: move this run into a project first (Projects).")
    run = a._pipeline(d)
    if run.d is None or run.d.status != "complete":
        raise HTTPException(409, "Phase D isn't complete for this run's current answers.")
    cols = run.d.coord_to_columns()["coord_to_columns"]
    date, kpi_col, geo = cols["time"], cols["kpi"], cols.get("geo")
    t = run.d.table.copy()
    t[date] = pd.to_datetime(t[date])
    chans = [c for c in run.d.channels if c.kind == "media"] + [c for c in run.d.channels if c.kind == "rf"]
    weekly = t.groupby(date)
    weeks = [str(w.date()) for w in sorted(t[date].unique())]
    kpi = {str(k.date()): float(v) for k, v in weekly[kpi_col].sum().items()}
    spend = {c.name: {str(k.date()): float(v) for k, v in weekly[c.spend].sum().items()} for c in chans}
    res = (job.status(d) or {}).get("result") or {}
    s31 = next((s["details"] for s in res.get("steps", []) if s.get("step") == 31), None)
    geos = sorted(str(g) for g in t[geo].unique()) if geo else ["national_geo"]
    rpk_col = cols.get("revenue_per_kpi")
    rpk = {}
    if rpk_col and rpk_col in t.columns:
        rpk = {str(g): float(x[rpk_col].tail(13).mean()) for g, x in (t.groupby(geo) if geo else [("national_geo", t)])}
    return {"d": d, "run": run, "slug": slug, "pdir": projects.folder(a.RUNS_DIR, slug), "weeks": weeks, "kpi": kpi,
            "spend": spend, "channels": [c.name for c in chans], "chan_objs": chans,
            "display": {c.name: display_name(c.name) for c in chans},
            "spend_only": {c.name: bool(c.kind == "media" and (c.exposure_is_spend or c.spend_sources)) for c in chans},
            "lag": int(run.d.config.get("max_lag", 0)), "step31": s31, "model_key": whatif.model_key(d),
            "geos": geos, "rpk": rpk, "kpi_name": run.d.kpi.get("kpi"), "date_col": date, "kpi_col": kpi_col, "geo_col": geo}


def _resolve(c: dict, setup: dict) -> dict:
    from ..future import period

    try:
        return period.resolve(setup, c["weeks"], c["lag"])
    except period.PeriodError as e:
        raise HTTPException(400, str(e)) from e


# Background jobs ------------------------------------------------------------------------------------------

def _job(d: Path, payload: dict) -> tuple[str, dict | None]:
    """Start (or find) a future job: (key, its result if already computed on this model)."""
    from .. import whatif
    from ..future.__main__ import folder

    mk = whatif.model_key(d)
    key = hashlib.sha256(json.dumps([payload, mk], sort_keys=True, default=str).encode()).hexdigest()[:12]
    fd = folder(d)
    done = _result(d, key)
    if done is not None and not done.get("error"):           # an error is never cached: asking again retries
        return key, done
    whatif.write(fd / f"request_{key}.json", payload)
    a, tag = A(), f"{d.name}|future_{key}"
    proc = a._whatif_jobs.get(tag)
    if proc is None or proc.poll() is not None:
        root = str(Path(__file__).resolve().parents[2])
        env = {**os.environ, "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", "")}
        log = open(fd / f"log_{key}.txt", "w", encoding="utf-8")
        a._whatif_jobs[tag] = subprocess.Popen([sys.executable, "-m", "mmm_tool.future", "--run-dir", str(d), "--key", key],
                                               stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
    return key, None


def _result(d: Path, key: str) -> dict | None:
    from .. import whatif
    from ..future.__main__ import folder

    p = folder(d) / f"result_{key}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if data.get("model_key") == whatif.model_key(d) else None


def _running(d: Path, key: str) -> bool:
    proc = A()._whatif_jobs.get(f"{d.name}|future_{key}")
    return proc is not None and proc.poll() is None


def _payload(setup: dict, kind: str, **kw) -> dict:
    return {"kind": kind, "resolved": setup["resolved"], "forecast": setup.get("forecast"), "cost": setup.get("cost") or {},
            "rpk": setup.get("rpk") or None, **kw}


def _settle_pending(c: dict, setup: dict) -> dict:
    """Plans waiting for their job (plan A, a recompute): saved once the job is done."""
    from ..future import store

    left = []
    for p in setup.get("pending_plans") or []:
        res = _result(c["d"], p["key"])
        if res is None and not _running(c["d"], p["key"]) and p.get("tries", 0) < 1:
            _job(c["d"], p["payload"])                 # restart a job lost to a server restart
            p["tries"] = p.get("tries", 0) + 1
        if res is None and not _running(c["d"], p["key"]) and p.get("tries", 0) >= 1 and p.get("restarted"):
            left.append({**p, "error": "The calculation stopped before it finished; set the period up again to retry."})
        elif res is None:
            left.append({**p, "restarted": p.get("tries", 0) >= 1})
        elif res.get("error"):
            left.append({**p, "error": res["error"]})
        else:
            store.add(c["pdir"], store.record(res, p["name"], p["source"], setup, c["d"].name, c["model_key"]))
    if left != (setup.get("pending_plans") or []):
        setup["pending_plans"] = left
        from ..future.store import _write, folder

        _write(folder(c["pdir"]) / "period.json", setup)
    return setup


def _plans_view(c: dict, setup: dict | None) -> list[dict]:
    from ..future import store

    if not setup:
        return []
    out = []
    a_plan = store.reference(c["pdir"], setup)
    for p in store.plans(c["pdir"], setup["id"]):
        t = p["totals"]
        out.append({k: p[k] for k in ("id", "name", "source", "by", "at", "status", "run_id", "model_key", "forecast_source", "multipliers")}
                   | {"spend": t["spend"], "total": t.get("total"), "media": t["media"], "change": t["change"], "revenue": t.get("revenue"),
                      "vs_a": {"outcome": (t["media"]["mean"] - a_plan["totals"]["media"]["mean"]) if a_plan else None,
                               "spend": (t["spend"] - a_plan["totals"]["spend"]) if a_plan else None},
                      "is_reference": bool(a_plan and p["id"] == a_plan["id"]),
                      "stale": store.stale(p, setup, c["model_key"] if p["run_id"] == c["d"].name else None)
                      + (["model"] if p["run_id"] != setup.get("run_id") else [])})
    return out


# Future period ------------------------------------------------------------------------------------------

class SetupIn(BaseModel):
    start: str
    weeks: int
    template: dict[str, Any] = {"kind": "last_year"}
    cost: dict[str, float] = {}
    forecast: dict[str, Any] = {"source": "builtin"}
    rpk: dict[str, float] | None = None
    upload: dict[str, list[float]] | None = None          # planned spend per channel and week ("upload" template)


@router.get("/api/runs/{run_id}/future")
def future_screen(run_id: str) -> dict:
    """Everything the Future setup and the plans need."""
    from ..future import forecast as fc
    from ..future import period, store
    from ..whatif.scenario import settings as wsettings

    c = _ctx(run_id)
    setup = store.period(c["pdir"])
    if setup:
        setup = _settle_pending(c, setup)
    curves = None
    if setup and setup.get("curves_key") and setup.get("run_id") == c["d"].name:
        curves = _result(c["d"], setup["curves_key"]) or ({"building": True} if _running(c["d"], setup["curves_key"]) else None)
        if curves is None:
            key, done = _job(c["d"], _payload(setup, "curves", grid=_grid()))
            curves = done or {"building": True}
    tracked = store.tracked(c["pdir"], setup["id"]) if setup else None
    return A()._clean({
        "defaults": period.defaults(c["weeks"]), "data": {"first": c["weeks"][0], "last": c["weeks"][-1], "weeks": len(c["weeks"])},
        "lag": c["lag"], "channels": c["channels"], "display": c["display"], "spend_only": c["spend_only"],
        "geos": c["geos"], "rpk": c["rpk"], "kpi": c["kpi_name"], "growth": fc.default_growth(c["weeks"], c["kpi"]),
        "step31": {"naive_mape": (c["step31"] or {}).get("naive_mape"),
                   "holdout_mape": (((c["step31"] or {}).get("accuracy") or {}).get("Test") or {}).get("MAPE")},
        "settings": period.settings(), "whatif": wsettings(), "project": c["slug"],
        "period": setup, "same_run": bool(setup and setup.get("run_id") == c["d"].name), "curves": curves,
        "plans": _plans_view(c, setup), "tracked": tracked and tracked["id"], "log": store.entries(c["pdir"])[-30:]})


def _grid() -> list[float]:
    from ..whatif.scenario import grid

    return grid()


@router.post("/api/runs/{run_id}/future/setup")
def future_setup(run_id: str, body: SetupIn) -> dict:
    """Set up the future period: dates, template, cost changes, forecast. Plan A (the template at
    100%) is made as the reference; the quick curves for the sliders start computing."""
    from ..future import forecast as fc
    from ..future import store

    c = _ctx(run_id)
    s = body.model_dump()
    R = _resolve(c, s)
    for ch, v in (s["cost"] or {}).items():
        if ch not in c["channels"]:
            raise HTTPException(400, f"'{ch}' isn't a channel of this model.")
        if not -90 <= float(v) <= 500:
            raise HTTPException(400, f"'{ch}': a cost change is between −90% and +500%.")
    cost = {ch: float(v) / 100 for ch, v in (s["cost"] or {}).items() if float(v)}
    try:
        f = fc.make(s["forecast"], R["plan"], R["template"], c["weeks"], c["kpi"], c["step31"])
    except fc.ForecastError as e:
        raise HTTPException(400, str(e)) from e
    tmpl_spend = {ch: [c["spend"][ch][w] for w in R["template"]] for ch in c["channels"]}
    spec = {k: v for k, v in s["forecast"].items() if k != "csv"}
    setup = {"run_id": c["d"].name, "model_key": c["model_key"], "start": R["plan"][0], "weeks": len(R["plan"]),
             "plan": R["plan"], "template": R["template"], "template_kind": (s["template"] or {}).get("kind", "last_year"),
             "resolved": R, "cost": cost, "cost_pct": s["cost"], "forecast": f, "forecast_spec": spec,
             "rpk": s.get("rpk"), "template_spend": tmpl_spend, "warnings": R["warnings"]}
    if setup["template_kind"] == "upload":
        if not s.get("upload"):
            raise HTTPException(400, "Upload the planned spend file (week × channel).")
        setup["upload"] = s["upload"]
    setup = store.save_period(c["pdir"], setup)
    store.log(c["pdir"], "future period set", {"plan": f"{R['plan'][0]} → {R['plan'][-1]} ({len(R['plan'])} weeks)",
                                                "template": f"{setup['template_kind']}: {R['template'][0]} → {R['template'][-1]}",
                                                "cost_change": s["cost"], "forecast": f.get("note"), "run": c["d"].name, "model": c["model_key"]})
    ref_payload = _payload(setup, "evaluate", **({"spend": setup["upload"]} if setup.get("upload") else {}))
    have = store.reference(c["pdir"], setup)
    same_a = have and have["run_id"] == c["d"].name and not store.stale(have, setup, c["model_key"]) \
        and not setup.get("upload") and have["source"].get("kind") == "reference"
    key_a, _ = _job(c["d"], ref_payload) if not same_a else (None, None)
    setup["pending_plans"] = [] if same_a else [{"key": key_a, "payload": ref_payload,
                               "name": "Uploaded plan" if setup.get("upload") else {
                                   "last_year": "Same as last year", "recent": "Same as the latest weeks"}.get(
                                   setup["template_kind"], f"Same as from {R['template'][0]}"),
                               "source": {"kind": "upload" if setup.get("upload") else "reference"}}]
    setup["curves_key"], _ = _job(c["d"], _payload(setup, "curves", grid=_grid()))
    from ..future.store import _write, folder

    _write(folder(c["pdir"]) / "period.json", setup)
    return future_screen(run_id)


class EvalIn(BaseModel):
    multipliers: dict[str, float] | None = None
    spend: dict[str, list[float]] | None = None


def _setup_here(c: dict) -> dict:
    from ..future import store

    setup = store.period(c["pdir"])
    if not setup:
        raise HTTPException(409, "Set up the future period first.")
    if setup.get("run_id") != c["d"].name:
        raise HTTPException(409, "This project's future period was set up on another run's model; open that run, or set it up again here.")
    return setup


@router.post("/api/runs/{run_id}/future/evaluate")
def future_evaluate(run_id: str, body: EvalIn) -> dict:
    """A plan (slider multipliers, or spend per week) on every posterior draw, in the background."""
    c = _ctx(run_id)
    setup = _setup_here(c)
    mult = {k: round(float(v), 4) for k, v in (body.multipliers or {}).items() if k in c["channels"]}
    key, done = _job(c["d"], _payload(setup, "evaluate", mult=mult or None, spend=body.spend))
    return A()._clean({"key": key, "result": done} if done else {"key": key, "building": True})


@router.get("/api/runs/{run_id}/future/result/{key}")
def future_result(run_id: str, key: str) -> dict:
    a = A()
    d = a._run_dir(run_id)
    if not all(ch in "0123456789abcdef" for ch in key) or len(key) != 12:
        raise HTTPException(404, "No such result.")
    done = _result(d, key)
    if done is not None:
        return a._clean({"key": key, "result": done})
    if _running(d, key):
        return {"key": key, "building": True}
    raise HTTPException(404, "No result for this plan: run it again.")


class SavePlanIn(BaseModel):
    key: str
    name: str
    source: dict[str, Any] = {"kind": "what-if"}


@router.post("/api/runs/{run_id}/future/plans")
def save_plan(run_id: str, body: SavePlanIn) -> dict:
    """Save an evaluated plan (from the sliders or the optimizer) as plan B, C…"""
    from ..future import store

    c = _ctx(run_id)
    setup = _setup_here(c)
    res = _result(c["d"], body.key)
    if res is None or res.get("error"):
        raise HTTPException(409, "Work the plan out first (its result isn't ready).")
    evaluation = res.get("evaluation", res)                    # an optimizer result carries its evaluation
    source = dict(body.source)
    if res.get("kind") == "optimize":
        source = {"kind": "optimizer", "settings": {k: res["plan"].get(k) for k in ("budget_type", "budget", "objective")},
                  "limits": [{"channel": r["channel"], "lowest": r["lowest"], "highest": r["highest"]} for r in res["plan"]["channels"]]}
    name = (body.name or "").strip() or ("Optimized" if source["kind"] == "optimizer" else "What-if plan")
    plan = store.add(c["pdir"], store.record(evaluation, name[:80], source, setup, c["d"].name, c["model_key"]))
    return A()._clean({"plan": plan["id"], "plans": _plans_view(c, setup)})


class PlanActionIn(BaseModel):
    name: str | None = None
    reason: str = ""


@router.post("/api/runs/{run_id}/future/plans/{plan_id}/{action}")
def plan_action(run_id: str, plan_id: str, action: str, body: PlanActionIn) -> dict:
    """rename · duplicate · archive · restore · track (choose for tracking) · recompute (same spend, current model)."""
    from ..future import store

    c = _ctx(run_id)
    setup = store.period(c["pdir"])
    if not setup:
        raise HTTPException(409, "Set up the future period first.")
    try:
        if action == "rename":
            store.update(c["pdir"], setup["id"], plan_id, name=body.name or "")
        elif action in ("archive", "restore"):
            store.update(c["pdir"], setup["id"], plan_id, status="archived" if action == "archive" else "draft")
        elif action == "duplicate":
            store.duplicate(c["pdir"], setup["id"], plan_id)
        elif action == "track":
            store.choose(c["pdir"], setup["id"], plan_id, body.reason)
        elif action == "recompute":
            setup = _setup_here(c)
            p = store.get(c["pdir"], setup["id"], plan_id)
            spend = {ch: [row["spend"].get(ch, 0.0) for row in p["weeks"]] for ch in c["channels"]}
            payload = _payload(setup, "evaluate", spend=spend)
            key, _ = _job(c["d"], payload)
            setup.setdefault("pending_plans", []).append({"key": key, "payload": payload, "name": f"{p['name']} (recomputed)"[:80],
                                                          "source": {**p["source"], "recomputed_from": plan_id}})
            from ..future.store import _write, folder

            _write(folder(c["pdir"]) / "period.json", setup)
            store.log(c["pdir"], "plan recomputed on the current model", {"plan": plan_id, "model": c["model_key"]})
        else:
            raise HTTPException(404, "Unknown action.")
    except store.PlanStoreError as e:
        raise HTTPException(400, str(e)) from e
    return future_screen(run_id)


# Plan, Optimize in Future mode: the same plan checks, on the future period -------------------------------------

def future_plan_inputs(run_id: str, inp: dict) -> dict:
    """The Plan screen's inputs for the future period: one period, the template's spend as history,
    and the highest past spend of any window as long, for the "outside what the model has seen" warning."""
    from ..future import store

    c = _ctx(run_id)
    setup = _setup_here(c)
    plan = setup["plan"]
    P = len(plan)
    up = setup.get("upload") or {}                                # an uploaded plan is where the optimizer starts
    hist = {ch: float(sum(up.get(ch) or setup["template_spend"][ch])) for ch in c["channels"]}
    peaks = {}
    for ch in c["channels"]:
        s = pd.Series([c["spend"][ch][w] for w in c["weeks"]])
        peaks[ch] = float(s.rolling(P).sum().max()) if len(s) >= P else float(s.sum())
    return {**inp, "periods": {"future": (plan[0], plan[-1])}, "hist": {"future": hist}, "peaks": {"future": peaks},
            "future": {"setup_id": setup["id"], "plan": plan, "upload": bool(setup.get("upload")), "forecast": setup.get("forecast"), "tracked": (store.tracked(c["pdir"], setup["id"]) or {}).get("id")}}


def future_optimize(run_id: str, checked: dict) -> dict:
    c = _ctx(run_id)
    setup = _setup_here(c)
    key, done = _job(c["d"], _payload(setup, "optimize", checked=checked, **({"spend": setup["upload"]} if setup.get("upload") else {})))
    return A()._clean({"key": key, "result": done, "future": True} if done else {"key": key, "building": True, "future": True})


# Tracking -----------------------------------------------------------------------------------------------

def _tracking_ctx(run_id: str) -> tuple[dict, dict, dict | None]:
    """(context of the run the tracked plan was made on, the period, the tracked plan)."""
    from ..future import store

    c = _ctx(run_id)
    setup = store.period(c["pdir"])
    if not setup:
        return c, None, None
    tp = store.tracked(c["pdir"], setup["id"])
    if tp and tp["run_id"] != c["d"].name:
        try:
            other = _ctx(tp["run_id"])
            if other["slug"] == c["slug"]:              # a run moved to another project isn't this project's
                c = other
        except HTTPException:
            pass
    return c, setup, tp


@router.get("/api/runs/{run_id}/tracking")
def tracking_screen(run_id: str) -> dict:
    from ..future import store, tracking

    c, setup, tp = _tracking_ctx(run_id)
    if setup is None or tp is None:
        return A()._clean({"tracked": None, "period": setup, "project": c["slug"]})
    t = store.tracking(c["pdir"], setup and setup["id"])
    result = None
    if t.get("pending_key"):
        res = _result(c["d"], t["pending_key"])
        if res is None and not _running(c["d"], t["pending_key"]) and t.get("pending_payload"):
            _job(c["d"], t["pending_payload"])
        if res is not None and not res.get("error"):
            act = t.get("actuals") or {}
            holdout = (((c["step31"] or {}).get("accuracy") or {}).get("Test") or {}).get("MAPE")
            result = tracking.compare(act, t["targets"], res, t["settings"], holdout, tp.get("by_channel"))
        elif res is not None:
            result = {"error": res["error"]}
        else:
            result = {"building": True}
    return A()._clean({"tracked": {k: tp[k] for k in ("id", "name", "plan_weeks", "model_key", "run_id", "totals", "by_channel")},
                       "targets": t["targets"], "history": t["history"], "settings": t["settings"], "events": t["events"],
                       "renames": t["renames"], "actual_weeks": t["actual_weeks"], "result": result, "period": setup,
                       "display": c["display"], "channels": c["channels"], "kpi": c["kpi_name"], "project": c["slug"],
                       "log": store.entries(c["pdir"])[-30:], "pending_upload": t.get("pending_upload")})


def _training_columns(c: dict) -> list[str]:
    return list(c["run"].a.table.columns) if c["run"].a is not None and c["run"].a.table is not None else []


def _process_actuals(c: dict, setup: dict, tp: dict, path: Path, renames: dict[str, str | None]) -> dict:
    """Read the uploaded actuals like the training file, keep the plan weeks, and start the
    'predicted at actual spend' job."""
    from .. import settings as rs
    from ..future import store, tracking
    from ..ingest.load import load_file
    from ..ingest.types import Answers
    from ..pipeline import run_pipeline

    a = A()
    meta = a._meta(c["d"])
    df, _ = load_file(path, Answers(meta["answers"]))
    if df is None:
        raise HTTPException(400, "The actuals file couldn't be read.")
    df = df.rename(columns={n: o for n, o in renames.items() if o})
    keep = [col for col in df.columns if col in _training_columns(c)]
    tmp = path.with_name(path.stem + "_matched.csv")
    df[keep].to_csv(tmp, index=False)
    roles = {k: v for k, v in (c["run"].a.roles or {}).items() if k in keep}
    answers = {**meta["answers"], "roles": roles}
    r = run_pipeline(tmp, answers, accept_defaults=True, settings=rs.for_run(c["d"], meta)[0])
    if r.b is None or r.b.status != "complete":
        raise HTTPException(400, f"The actuals couldn't be read like the training file: {r.last.steps[-1].summary}")
    act = tracking.actuals_from_table(r.b.table, c["date_col"], c["geo_col"], c["kpi_col"], c["chan_objs"])
    if act["missing_channels"]:
        raise HTTPException(400, f"The actuals have no spend for {', '.join(act['missing_channels'])}.")
    on = tracking.on_plan(act, tp["plan_weeks"])
    if not on["weeks"]:
        raise HTTPException(400, f"None of the uploaded weeks ({act['weeks'][0]} → {act['weeks'][-1]}) are in the plan "
                                 f"({tp['plan_weeks'][0]} → {tp['plan_weeks'][-1]}).")
    t = store.tracking(c["pdir"], setup and setup["id"])
    on = tracking.merge(t.get("actuals"), on, tp["plan_weeks"])     # earlier uploads keep their weeks
    spend = tracking.spend_for_model(on, t["targets"], tp["plan_weeks"], c["channels"])
    units = {ch: [on["units"][ch][on["weeks"].index(w)] if w in on["weeks"] else None for w in tp["plan_weeks"]]
             for ch in on["units"]}
    units = {ch: v for ch, v in units.items() if any(x is not None for x in v)} or None
    payload = _payload(store.snapshot(setup, tp), "evaluate", spend=spend, units=units)
    key, _ = _job(c["d"], payload)
    t.update(actuals=on, actual_weeks=on["weeks"], all_weeks=sorted(set(t.get("all_weeks") or []) | set(act["weeks"])), pending_key=key, pending_payload=payload,
             actuals_file=path.name, actuals_files=list(dict.fromkeys((t.get("actuals_files") or ([t["actuals_file"]] if t.get("actuals_file") else []))
                                                                   + [path.name])),
             renames_used={**(t.get("renames_used") or {}), path.name: {n: o for n, o in renames.items() if o}}, pending_upload=None)
    store.save_tracking(c["pdir"], t)
    store.log(c["pdir"], "actuals uploaded", {"file": path.name, "plan_weeks": len(on["weeks"]),
                                              "all_weeks": len(act["weeks"]), "renames": {n: o for n, o in renames.items() if o}})
    return {"weeks": on["weeks"], "key": key}


@router.post("/api/runs/{run_id}/tracking/upload")
async def tracking_upload(run_id: str, file: UploadFile = File(...)) -> dict:
    """Actuals in the training file's format. Asks only about columns it can't match."""
    from ..future import store, tracking
    from ..ingest.load import load_file
    from ..ingest.types import Answers

    c, setup, tp = _tracking_ctx(run_id)
    if tp is None:
        raise HTTPException(409, "Choose a plan for tracking first (What-If or Plan, Optimize, in Future mode).")
    name = A()._upload_name(file.filename)
    folder = store.folder(c["pdir"]) / "actuals"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(await file.read())
    df, _ = load_file(path, Answers(A()._meta(c["d"])["answers"]))
    if df is None:
        raise HTTPException(400, "The file couldn't be read.")
    t = store.tracking(c["pdir"], setup and setup["id"])
    m = tracking.match_columns(list(df.columns), _training_columns(c), t["renames"], float(t["settings"]["similarity"]))
    if m["offered"]:
        t["pending_upload"] = {"file": name, "match": m}
        store.save_tracking(c["pdir"], t)
        return A()._clean({"needs_confirm": True, "file": name, "match": m})
    out = _process_actuals(c, setup, tp, path, m["confirmed"])
    return A()._clean({"processed": True, "match": m, **out})


class ConfirmIn(BaseModel):
    file: str
    renames: dict[str, str | None]            # uploaded column -> training column, or None to leave it out


@router.post("/api/runs/{run_id}/tracking/confirm")
def tracking_confirm(run_id: str, body: ConfirmIn) -> dict:
    """Confirm the offered renames once (remembered for later uploads), then read the actuals."""
    from ..future import store

    c, setup, tp = _tracking_ctx(run_id)
    if tp is None:
        raise HTTPException(409, "Choose a plan for tracking first.")
    path = store.folder(c["pdir"]) / "actuals" / Path(body.file).name
    if not path.exists():
        raise HTTPException(404, "Upload the actuals again.")
    t = store.tracking(c["pdir"], setup and setup["id"])
    train = set(_training_columns(c))
    for new, old in body.renames.items():
        if old is not None and old not in train:
            raise HTTPException(400, f"'{old}' isn't a column of the training file.")
        if old:
            t["renames"][new] = old
    store.save_tracking(c["pdir"], t)
    store.log(c["pdir"], "columns matched", {"renames": {n: o for n, o in body.renames.items() if o},
                                             "left_out": [n for n, o in body.renames.items() if not o]})
    renames = {**t["renames"], **{n: o for n, o in body.renames.items()}}
    out = _process_actuals(c, setup, tp, path, {n: o for n, o in renames.items() if o})
    return A()._clean({"processed": True, **out})


class TrackSettingsIn(BaseModel):
    settings: dict[str, float]


@router.put("/api/runs/{run_id}/tracking/settings")
def tracking_settings(run_id: str, body: TrackSettingsIn) -> dict:
    from ..future import store

    c, setup, tp = _tracking_ctx(run_id)
    t = store.tracking(c["pdir"], setup and setup["id"])
    bad = [k for k in body.settings if k not in t["settings"]]
    if bad:
        raise HTTPException(400, f"Unknown tracking settings: {bad}.")
    if any(v < 0 for v in body.settings.values()):
        raise HTTPException(400, "Thresholds can't be negative.")
    before = dict(t["settings"])
    t["settings"].update({k: float(v) for k, v in body.settings.items()})
    store.save_tracking(c["pdir"], t)
    store.log(c["pdir"], "tracking settings changed", {k: {"from": before[k], "to": t["settings"][k]} for k in body.settings if before[k] != t["settings"][k]})
    return tracking_screen(run_id)


class EventIn(BaseModel):
    date: str
    note: str
    end: str | None = None


@router.post("/api/runs/{run_id}/tracking/events")
def tracking_event(run_id: str, body: EventIn) -> dict:
    """A dated note (a competitor's price cut, an outage): kept, logged, and offered as a calendar flag at the next refit."""
    from ..future import store

    c, setup, tp = _tracking_ctx(run_id)
    try:
        start, end = pd.Timestamp(body.date), pd.Timestamp(body.end or body.date)
    except ValueError as e:
        raise HTTPException(400, "Give the event a date (YYYY-MM-DD).") from e
    if not body.note.strip() or end < start:
        raise HTTPException(400, "An event needs a note, and an end on or after its start.")
    t = store.tracking(c["pdir"], setup and setup["id"])
    ev = {"date": str(start.date()), "end": str(end.date()), "note": body.note.strip()[:200], "by": store.who(), "at": store.now()}
    t["events"].append(ev)
    store.save_tracking(c["pdir"], t)
    store.log(c["pdir"], "event added", ev)
    return tracking_screen(run_id)


@router.get("/api/runs/{run_id}/tracking.csv")
def tracking_csv(run_id: str) -> Response:
    data = tracking_screen(run_id)
    res = data.get("result") or {}
    rows = res.get("rows") or []
    out = [{"week": r["week"], "plan": r["plan_id"], "planned_spend": r["planned_spend"], "actual_spend": r["actual_spend"],
            "plan_predicted": r.get("plan_predicted"), "predicted_at_actual_spend": r.get("predicted"), "range_low": r.get("lo"),
            "range_high": r.get("hi"), "actual": r["actual"], "gap_pct": r.get("gap_pct"), "execution_gap": r.get("execution_gap"),
            "model_gap": r.get("model_gap"), "status": r["status"],
            **{f"spend_{ch}_planned": v["planned"] for ch, v in r["channels"].items()},
            **{f"spend_{ch}_actual": v["actual"] for ch, v in r["channels"].items()}} for r in rows]
    return Response(pd.DataFrame(out).to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="tracking.csv"'})


@router.post("/api/runs/{run_id}/tracking/refit")
def tracking_refit(run_id: str) -> dict:
    """A refit with the new weeks: the training file plus the actuals' weeks (renamed to the
    training columns), a new run in the project with the same answers, roles, seed and sampler; the
    events become calendar flags it offers in step 21."""
    from .. import projects
    from ..export import refit as rf
    from ..export import registry as reg
    from ..future import store
    from ..ingest.load import load_file
    from ..ingest.types import Answers

    c, setup, tp = _tracking_ctx(run_id)
    t = store.tracking(c["pdir"], setup and setup["id"])
    if not t.get("actuals_file"):
        raise HTTPException(409, "Upload actuals first.")
    a = A()
    meta = a._meta(c["d"])
    old, _ = load_file(c["d"] / meta["file"], Answers(meta["answers"]))
    parts = []
    for fname in t.get("actuals_files") or [t["actuals_file"]]:          # every upload, oldest first (later ones win)
        f = store.folder(c["pdir"]) / "actuals" / fname
        if not f.is_file():
            continue
        part, _ = load_file(f, Answers(meta["answers"]))
        if part is None:
            continue
        used = (t.get("renames_used") or {}).get(fname) or {n: o for n, o in t["renames"].items() if o}
        part = part.rename(columns=used)
        parts.append(part[[col for col in old.columns if col in part.columns]])
    if not parts:
        raise HTTPException(409, "The uploaded actuals files are missing; upload them again.")
    new = pd.concat(parts, ignore_index=True)
    date = next((col for col, r in (c["run"].a.roles or {}).items() if r == "date"), old.columns[0])
    both = pd.concat([old, new], ignore_index=True)
    both["_d"] = pd.to_datetime(both[date], errors="coerce")
    keys = ["_d"] + ([c["geo_col"]] if c["geo_col"] and c["geo_col"] in both.columns else [])
    both = both.drop_duplicates(keys, keep="last").sort_values(keys).drop(columns="_d")
    d = projects.runs_folder(a.RUNS_DIR, c["slug"]) / __import__("uuid").uuid4().hex[:12]
    d.mkdir(parents=True)
    name = f"{Path(meta['file']).stem}_to_{str(pd.to_datetime(new[date], errors='coerce').max().date())}.csv"
    both.to_csv(d / name, index=False)
    hit = reg.find_run(reg.load(a._reg_dir(c["d"])), c["d"].name)
    roles = c["run"].a.roles if c["run"].a is not None else None
    rf.start(c["d"], d, d / name, hit[0] if hit else None, hit[1]["version"] if hit else None, roles=roles)
    m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if not hit:
        m.pop("refit_of", None)
        m["refit_from"] = c["d"].name
    events = [{"name": f"event_{e['date'].replace('-', '_')}", "label": e["note"], "ranges": [{"start": e["date"], "end": e["end"]}]}
              for e in t["events"]]
    if events:
        m["answers"]["custom_events"] = list(m["answers"].get("custom_events") or []) + events
    (d / "meta.json").write_text(json.dumps(m), encoding="utf-8")
    store.log(c["pdir"], "refit started from tracking", {"run": d.name, "file": name, "events": len(events)})
    return a._evaluate(d)
