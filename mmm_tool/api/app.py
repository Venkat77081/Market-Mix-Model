"""HTTP API and web UI for the pipeline.

Each run is a folder under the runs directory holding the uploaded file and
the answers given so far. Phase A is fast, so every call re-runs it from the
file with the current answers and returns the full result.

    uvicorn mmm_tool.api.app:app --reload     # or: mmm-ui
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from .. import samples
from ..fit import job
from ..fit.context import settings as fit_settings
from ..fit.model import meridian_installed
from ..pipeline import run_pipeline
from ..ingest.keywords import KeywordError, load_keywords
from ..ingest.memory import RoleMemory
from ..ingest.load import EXCEL_EXT, TEXT_EXT

RUNS_DIR = Path(os.environ.get("MMM_RUNS_DIR", "runs"))
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
RUN_ID = re.compile(r"^[0-9a-f]{12}$")

app = FastAPI(title="Meridian MMM Tool")


class AnswersIn(BaseModel):
    answers: dict[str, Any]


class FitIn(BaseModel):
    sampler: str = "quick"                      # a preset from fit.yaml, or "custom"
    custom: dict[str, Any] | None = None        # n_chains, n_adapt, n_burnin, n_keep for "custom"
    seed: int | None = None                     # defaults to the run's own seed
    stability: bool = False
    placebo: bool = False


def _clean(x: Any) -> Any:
    """Make a result JSON-safe: NaN becomes null, numpy scalars become numbers."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return str(x)


def _run_dir(run_id: str) -> Path:
    if not RUN_ID.match(run_id):
        raise HTTPException(404, "Run not found.")
    d = RUNS_DIR / run_id
    if not (d / "meta.json").exists():
        raise HTTPException(404, "Run not found.")
    return d


def _meta(d: Path) -> dict:
    return json.loads((d / "meta.json").read_text())


_META_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _update_meta(d: Path, change) -> dict:
    """Read, change and write a run's meta.json under the run's lock, atomically, so a slow request
    never writes back answers a newer one replaced, and a reader never sees half a file."""
    with _LOCKS_GUARD:
        lock = _META_LOCKS.setdefault(str(d), threading.Lock())
    with lock:
        meta = _meta(d)
        change(meta)
        tmp = d / "meta.json.tmp"
        tmp.write_text(json.dumps(meta))
        tmp.replace(d / "meta.json")
        return meta


def _pipeline(d: Path, meta: dict | None = None):
    """Phases A-D for a run, read with the run's own keyword packs and calendar."""
    from .. import settings as rs

    meta = meta if meta is not None else _meta(d)
    return run_pipeline(d / meta["file"], meta["answers"], memory=_memory(), settings=rs.for_run(d, meta)[0])


def _memory() -> RoleMemory:
    """Remembered column roles, shared by every run (MMM_ROLE_MEMORY, or next to the runs)."""
    return RoleMemory(os.environ.get("MMM_ROLE_MEMORY") or RUNS_DIR / "role_memory.json")


def _preview(table) -> dict:
    t = table.head(8)
    return {
        "columns": list(t.columns),
        "rows": [[None if v is None else str(v)[:10] if hasattr(v, "date") else v for v in r] for r in t.itertuples(index=False)],
    }


def _evaluate(d: Path) -> dict:
    """Run Phase A, then Phase B once A is complete, with the run's answers."""
    from .. import settings as rs

    meta = _meta(d)
    try:
        run = _pipeline(d, meta)
    except KeywordError as e:
        raise HTTPException(400, f"Your role keywords file has a problem: {e}") from e
    except rs.SettingsError as e:
        raise HTTPException(400, f"This run's reading settings have a problem: {e}") from e
    out_a = run.a.to_dict()
    if run.a.status == "complete":
        out_a["table_preview"] = _preview(run.a.table)
        run.a.table.to_csv(d / "ingested.csv", index=False)
    out_b = None
    if run.b is not None:
        out_b = run.b.to_dict()
        if run.b.status == "complete":
            out_b["table_preview"] = _preview(run.b.table)
            run.b.table.to_csv(d / "cleaned.csv", index=False)
    out_c = None
    if run.c is not None:
        out_c = run.c.to_dict()
        if run.c.status == "complete":
            out_c["table_preview"] = _preview(run.c.table)
            run.c.table.to_csv(d / "profiled.csv", index=False)
    out_d = None
    if run.d is not None:
        out_d = run.d.to_dict()
        if run.d.status == "complete":
            out_d["table_preview"] = _preview(run.d.table)
            run.d.table.to_csv(d / "model_data.csv", index=False)
            (d / "model_config.json").write_text(json.dumps(_clean(run.d.config), indent=2))
    fit = job.status(d)
    if fit is not None:
        built_from = (fit.get("result") or {}).get("inputs") or fit.get("inputs")
        fit["stale"] = bool(built_from and run.d is not None and run.d.status == "complete"
                            and built_from != job.inputs_hash(run.d))
    final = run.last
    used = meta["answers"]

    def status(m: dict) -> None:               # only if the answers are still the ones this run used
        if m.get("answers") == used:
            m.update(phase=run.phase, status=final.status, updated=time.time(), decisions=len(final.pending))
    _update_meta(d, status)
    key = _answer_key(d)
    if key and fit and fit.get("result") and run.d is not None and run.d.status == "complete":
        from ..fit import truth

        fit["truth"] = truth.compare(fit["result"], run.d.channels, key, run.d.kpi)   # display only
    return {"run_id": d.name, "file": meta["file"], "answers": meta["answers"], "phase": run.phase,
            "result": _clean(out_a), "result_b": _clean(out_b), "result_c": _clean(out_c), "result_d": _clean(out_d),
            "fit": _clean(fit), "meridian_installed": meridian_installed(),
            "answer_key": {"loaded": bool(key), "file": (key or {}).get("_file")},
            "refit_of": meta.get("refit_of"), "refit_options": meta.get("refit_options"),
            "settings": _settings_view(d, meta)}


def _settings_view(d: Path, meta: dict) -> dict:
    from .. import settings as rs

    s, source = rs.for_run(d, meta)
    return {**s, "source": source, "label": rs.describe(s)}


class SettingsIn(BaseModel):
    packs: list[str] = []
    calendar: str = "none"


@app.get("/api/settings")
def get_settings() -> dict:
    """Every keyword pack and calendar preset, and the settings a new run starts with."""
    from .. import settings as rs

    new = rs.for_new_run(RUNS_DIR)
    return {**rs.catalog(), "default": {**new, "label": rs.describe(new), "remembered": rs.remembered(RUNS_DIR) is not None},
            "sample": {**samples.SETTINGS, "label": rs.describe(samples.SETTINGS)}}


@app.put("/api/settings/default")
def set_default_settings(body: SettingsIn) -> dict:
    """"Use these for my new runs"."""
    from .. import settings as rs

    try:
        s = rs.remember(RUNS_DIR, body.model_dump())
    except rs.SettingsError as e:
        raise HTTPException(400, str(e)) from e
    return {**s, "label": rs.describe(s), "remembered": True}


@app.put("/api/runs/{run_id}/settings")
def set_run_settings(run_id: str, body: SettingsIn) -> dict:
    """Change how a run's file is read (the chip): it is re-read from Phase A. Answers stay, except
    the step 21 calendar pick, so the new calendar applies."""
    from .. import settings as rs

    d = _run_dir(run_id)
    try:
        s = rs.clean(body.model_dump())
    except rs.SettingsError as e:
        raise HTTPException(400, str(e)) from e

    def change(m: dict) -> None:
        if m.get("settings") != s:
            m["answers"] = {k: v for k, v in m.get("answers", {}).items() if k != "calendar_preset"}
        m["settings"] = s
    _update_meta(d, change)
    return _evaluate(d)


ANSWER_KEY = "answer_key.json"
# What the tool writes into a run's folder: an upload must never take one of these names.
RUN_FILES = {"meta.json", "ingested.csv", "cleaned.csv", "profiled.csv", "model_data.csv", "model_config.json", ANSWER_KEY, "fit"}


def _upload_name(filename: str | None) -> str:
    name = Path(filename or "upload.csv").name or "upload.csv"
    return f"upload_{name}" if name.lower() in RUN_FILES else name


def _answer_key(d: Path) -> dict | None:
    """A test file's known truth, if one was loaded (step 32). Never passed to the model."""
    p = d / ANSWER_KEY
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _new_seed() -> int:
    """A run's seed: random, made once, shown on the Phase E screen and saved with every fit."""
    import secrets

    return secrets.randbelow(2**31 - 1) + 1


def _run_seed(d: Path) -> int:
    def seed(m: dict) -> None:
        if not m.get("seed"):
            m["seed"] = _new_seed()
    return int(_update_meta(d, seed)["seed"])


def _new_run(filename: str, settings: dict | None = None) -> Path:
    """A run folder; its settings are the ones given, else the remembered default, else the machine's."""
    from .. import settings as rs

    chosen = rs.clean(settings) if settings is not None else rs.for_new_run(RUNS_DIR)
    d = RUNS_DIR / uuid.uuid4().hex[:12]
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"file": filename, "answers": {}, "seed": _new_seed(), "settings": chosen}))
    return d


def _form_settings(packs: str | None, calendar: str | None) -> dict | None:
    """The upload form's settings: packs as a comma list ("" for none); both absent means the default."""
    from .. import settings as rs

    if packs is None and calendar is None:
        return None
    try:
        return rs.clean({"packs": packs or "", "calendar": calendar or "none"})
    except rs.SettingsError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/", response_class=HTMLResponse)
def landing() -> FileResponse:
    return FileResponse(WEB_DIR / "landing.html")


@app.get("/app", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/runs")
def list_runs(limit: int = 10) -> list[dict]:
    """Most recent runs first, with the status saved at their last evaluation."""
    if not RUNS_DIR.exists():
        return []
    runs = []
    for d in RUNS_DIR.iterdir():
        if RUN_ID.match(d.name) and (d / "meta.json").exists():
            meta = _meta(d)
            runs.append({"run_id": d.name, "file": meta["file"], "phase": meta.get("phase", "A"), "status": meta.get("status"),
                         "decisions": meta.get("decisions", 0), "updated": meta.get("updated") or d.stat().st_mtime})
    return sorted(runs, key=lambda r: r["updated"], reverse=True)[:limit]


@app.get("/api/samples")
def list_samples() -> list[dict]:
    return [{"name": k, "description": v[0]} for k, v in samples.SAMPLES.items()]


@app.post("/api/runs")
async def create_run(file: UploadFile = File(...), packs: str | None = Form(None), calendar: str | None = Form(None),
                     remember: bool = Form(False)) -> dict:
    from .. import settings as rs

    chosen = _form_settings(packs, calendar)
    name = _upload_name(file.filename)
    if Path(name).suffix.lower() not in EXCEL_EXT | TEXT_EXT:
        raise HTTPException(400, "Upload a CSV, TSV, TXT or Excel file.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is over 50 MB.")
    if chosen is not None and remember:
        rs.remember(RUNS_DIR, chosen)
    d = _new_run(name, chosen)
    (d / name).write_bytes(data)
    return _evaluate(d)


@app.post("/api/runs/sample/{name}")
def create_sample_run(name: str) -> dict:
    if name not in samples.SAMPLES:
        raise HTTPException(404, "Unknown sample.")
    tmp = _new_run("", samples.SETTINGS)
    path = samples.SAMPLES[name][1](tmp)
    _update_meta(tmp, lambda m: m.update(file=path.name))
    return _evaluate(tmp)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return _evaluate(_run_dir(run_id))


@app.put("/api/runs/{run_id}/answers")
def set_answers(run_id: str, body: AnswersIn) -> dict:
    """Replace the run's answers and re-run Phase A."""
    d = _run_dir(run_id)
    _update_meta(d, lambda m: m.update(answers=body.answers))
    return _evaluate(d)


@app.get("/api/runs/{run_id}/table.csv")
def download_table(run_id: str) -> Response:
    """The table after Phase A (columns with confirmed roles)."""
    d = _run_dir(run_id)
    if _evaluate(d)["result"]["status"] != "complete":
        raise HTTPException(409, "Phase A is not complete yet.")
    stem = Path(_meta(d)["file"]).stem
    return FileResponse(d / "ingested.csv", media_type="text/csv", filename=f"{stem}_phase_a.csv")


@app.get("/api/runs/{run_id}/cleaned.csv")
def download_cleaned(run_id: str) -> Response:
    """The table after Phase B: one complete row per week, ready for Meridian."""
    d = _run_dir(run_id)
    b = _evaluate(d)["result_b"]
    if not b or b["status"] != "complete":
        raise HTTPException(409, "Phase B is not complete yet.")
    stem = Path(_meta(d)["file"]).stem
    return FileResponse(d / "cleaned.csv", media_type="text/csv", filename=f"{stem}_phase_b.csv")


@app.get("/api/runs/{run_id}/profiled.csv")
def download_profiled(run_id: str) -> Response:
    """The table after Phase C: rare channels merged, moving-together channels combined, mediators dropped."""
    d = _run_dir(run_id)
    c = _evaluate(d)["result_c"]
    if not c or c["status"] != "complete":
        raise HTTPException(409, "Phase C is not complete yet.")
    stem = Path(_meta(d)["file"]).stem
    return FileResponse(d / "profiled.csv", media_type="text/csv", filename=f"{stem}_phase_c.csv")


def _phase_d_file(run_id: str, name: str, media_type: str, suffix: str) -> Response:
    d = _run_dir(run_id)
    res = _evaluate(d)["result_d"]
    if not res or res["status"] != "complete":
        raise HTTPException(409, "Phase D is not complete yet.")
    stem = Path(_meta(d)["file"]).stem
    return FileResponse(d / name, media_type=media_type, filename=f"{stem}_{suffix}")


@app.get("/api/runs/{run_id}/model_data.csv")
def download_model_data(run_id: str) -> Response:
    """The table Meridian fits: Phase C's table with calendar flags added."""
    return _phase_d_file(run_id, "model_data.csv", "text/csv", "model_data.csv")


@app.get("/api/runs/{run_id}/model_config.json")
def download_model_config(run_id: str) -> Response:
    """Phase D's model configuration: carryover, knots, priors, calibration, holdout."""
    return _phase_d_file(run_id, "model_config.json", "application/json", "model_config.json")


_previews: dict[str, dict] = {}


def _phase_d(d: Path):
    """The run's pipeline result, which must have reached a complete Phase D."""
    meta = _meta(d)
    run = _pipeline(d, meta)
    if run.d is None or run.d.status != "complete":
        raise HTTPException(409, "Phase D is not complete yet.")
    return run


@app.get("/api/runs/{run_id}/baseline-preview")
def baseline_preview(run_id: str) -> dict:
    """Step 24's preview: the baseline for every number of knots, computed once.
    The slider then redraws in the browser without calling the server."""
    from ..configure import baseline

    if not baseline.meridian_installed():
        raise HTTPException(503, "The knots preview uses Meridian's knot weights: pip install -r requirements-model.txt.")
    from ..channel_stats import ChannelStatsError

    run = _phase_d(_run_dir(run_id))
    outliers = [o["date"] for s in run.b.steps if s.step == 11 for o in s.details.get("outliers", [])]
    t = run.d.table
    chans = sorted((c.name, c.spend, tuple(c.sources())) for c in run.d.channels)
    key = hashlib.sha256(t.to_csv(index=False).encode() + json.dumps(
        [run.d.config.get("max_lag"), sorted(run.d.roles.items()), chans, outliers], default=str).encode()).hexdigest()[:16]
    if key not in _previews:
        try:
            _previews[key] = _clean({**baseline.preview(run.d, outliers), "key": key})
        except ChannelStatsError as e:
            raise HTTPException(500, f"Step 24 can't check burst channels: {e}") from e
    return _previews[key]


@app.get("/api/runs/{run_id}/knots-suggestion")
def knots_suggestion(run_id: str) -> dict:
    """Meridian's automatic knot selection, shown on the slider and never applied."""
    from ..configure import baseline

    if not baseline.meridian_installed():
        raise HTTPException(503, "Suggestions use Meridian: pip install -r requirements-model.txt.")
    run = _phase_d(_run_dir(run_id))
    return _clean(baseline.suggest(run.d))


@app.get("/api/fit-settings")
def get_fit_settings() -> dict:
    """Sampler presets and gate thresholds (fit.yaml), for the Phase E screen."""
    return _clean(fit_settings())


@app.post("/api/runs/{run_id}/fit")
def start_fit(run_id: str, body: FitIn) -> dict:
    """Start Phase E as a background job. Poll GET .../fit for progress."""
    from ..fit.context import check_custom

    d = _run_dir(run_id)
    if not meridian_installed():
        raise HTTPException(503, "Phase E needs google-meridian: pip install -r requirements-model.txt, then restart.")
    options = body.model_dump()
    if body.sampler == "custom":
        errors, _ = check_custom(body.custom or {})
        if errors:
            raise HTTPException(400, "Custom sampler: " + "; ".join(errors) + ".")
        options["sampler"] = {"label": "Custom", **{k: int(body.custom[k]) for k in ("n_chains", "n_adapt", "n_burnin", "n_keep")}}
    elif body.sampler not in fit_settings()["samplers"]:
        raise HTTPException(400, f"Unknown sampler '{body.sampler}'.")
    run_seed = _run_seed(d)
    options["seed_source"] = "user" if body.seed is not None and body.seed != run_seed else "default"
    options["seed"] = int(body.seed) if body.seed is not None else run_seed
    options.pop("custom", None)
    res = _evaluate(d)["result_d"]
    if not res or res["status"] != "complete":
        raise HTTPException(409, "Phase D is not complete yet.")
    return _start_job(d, options)


def _start_job(d: Path, options: dict) -> dict:
    memory = os.environ.get("MMM_ROLE_MEMORY") or str(RUNS_DIR / "role_memory.json")
    try:
        return _clean(job.start(d, options, memory=memory))
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e


@app.post("/api/runs/{run_id}/fit/continue")
def continue_fit(run_id: str) -> dict:
    """Step 30 failed: carry on with steps 31+ anyway, from the saved holdout fit. Every later
    result is badged 'did not converge'."""
    d = _run_dir(run_id)
    st = job.status(d) or {}
    if not (st.get("result") or {}).get("waiting"):
        raise HTTPException(409, "Nothing is waiting: continue is only offered when step 30 fails.")
    req = json.loads((job.fit_dir(d) / "request.json").read_text(encoding="utf-8"))
    return _start_job(d, {**req["options"], "continue_unconverged": True})


_calibration: subprocess.Popen | None = None


@app.get("/api/runs/{run_id}/fit-estimate")
def fit_estimate(run_id: str) -> dict:
    """What the Phase E form needs to estimate run time as the user types: a and b for this
    machine, and this run's size. Starts the one-off calibration if there's nothing to go on."""
    global _calibration
    from ..fit import estimate

    d = _run_dir(run_id)
    meta = _meta(d)
    run = _pipeline(d, meta)
    if run.d is None or run.d.status != "complete":
        raise HTTPException(409, "Phase D is not complete yet.")
    geo = run.d.coord_to_columns()["coord_to_columns"].get("geo")
    weeks = int(run.d.table[run.d.coord_to_columns()["coord_to_columns"]["time"]].nunique())
    geos = int(run.d.table[geo].nunique()) if geo else 1
    channels = len(run.d.channels) + sum(1 for r in run.d.roles.values() if r == "organic_media")
    model = estimate.model()
    calibrating = _calibration is not None and _calibration.poll() is None
    if model["a"] is None and not calibrating and meridian_installed():
        root = str(Path(__file__).resolve().parents[2])
        env = {**os.environ, "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", "")}
        _calibration = subprocess.Popen([sys.executable, "-m", "mmm_tool.fit", "--calibrate"], env=env,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        calibrating = True
    return _clean({**model, "calibrating": calibrating, "n_weeks": weeks, "n_geos": geos, "n_channels": channels,
                   "seed": _run_seed(d), "presets": fit_settings()["samplers"]})


@app.post("/api/runs/{run_id}/answer-key")
async def load_answer_key(run_id: str, file: UploadFile = File(...)) -> dict:
    """Step 32's optional comparison with known truth (test files only). Refused if it is for
    another KPI column. Stored with the run for display; never used by the model or a verdict."""
    from ..fit import truth

    d = _run_dir(run_id)
    try:
        key = json.loads((await file.read()).decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise HTTPException(400, f"The answer key isn't valid JSON: {e}") from e
    run = _pipeline(d)
    if run.a is None or run.a.status != "complete":
        raise HTTPException(409, "Finish Phase A first: the answer key is checked against the KPI column.")
    try:
        truth.validate(key, run.a.kpi["kpi"])
    except truth.TruthError as e:
        raise HTTPException(400, str(e)) from e
    key["_file"] = file.filename
    (d / ANSWER_KEY).write_text(json.dumps(key), encoding="utf-8")
    return _evaluate(d)


@app.delete("/api/runs/{run_id}/answer-key")
def remove_answer_key(run_id: str) -> dict:
    d = _run_dir(run_id)
    (d / ANSWER_KEY).unlink(missing_ok=True)
    return _evaluate(d)


@app.get("/api/runs/{run_id}/fit")
def fit_status(run_id: str) -> dict:
    """The fit's state and progress; the result once it has finished."""
    st = job.status(_run_dir(run_id))
    if st is None:
        raise HTTPException(404, "No fit has been started for this run.")
    return _clean(st)


@app.delete("/api/runs/{run_id}/fit")
def cancel_fit(run_id: str) -> dict:
    return _clean(job.cancel(_run_dir(run_id)) or {})


@app.get("/api/runs/{run_id}/model.pkl")
def download_model(run_id: str) -> Response:
    """The accepted model from step 36 (load it with meridian's load_mmm)."""
    d = _run_dir(run_id)
    path = job.fit_dir(d) / "model.pkl"
    if not path.exists():
        raise HTTPException(404, "No saved model: the fit hasn't finished, or a gate failed.")
    stem = Path(_meta(d)["file"]).stem
    return FileResponse(path, media_type="application/octet-stream", filename=f"{stem}_model.pkl")


_summary_jobs: dict[str, subprocess.Popen] = {}


@app.get("/api/runs/{run_id}/summary")
def get_summary(run_id: str) -> dict:
    """The Model Summary (Phase F), stored when the model was accepted."""
    d = _run_dir(run_id)
    path = job.fit_dir(d) / "summary.json"
    if not path.exists():
        proc = _summary_jobs.get(run_id)
        building = proc is not None and proc.poll() is None
        raise HTTPException(404, "The Model Summary is being built." if building else
                            "No Model Summary: Phase F opens once a model is accepted in Phase E.")
    st = job.status(d) or {}
    out = json.loads(path.read_text(encoding="utf-8"))
    built_from = (st.get("result") or {}).get("inputs")
    if built_from:
        run = _pipeline(d)
        out["stale"] = bool(run.d is not None and run.d.status == "complete" and job.inputs_hash(run.d) != built_from)
    out["downloads"] = {"meridian_html": (job.fit_dir(d) / "model_results_summary.html").exists(),
                        "model": (job.fit_dir(d) / "model.pkl").exists()}
    key = _answer_key(d)
    if key and st.get("result"):
        from ..fit import truth

        run = _pipeline(d)
        if run.d is not None and run.d.status == "complete":
            out["truth_marginal_roi"] = truth.compare(st["result"], run.d.channels, key, run.d.kpi)["true_marginal_roi"]
    return _clean(out)


@app.post("/api/runs/{run_id}/summary")
def rebuild_summary(run_id: str) -> dict:
    """Rebuild the Model Summary from the saved model, in the background (for models fitted before Phase F)."""
    d = _run_dir(run_id)
    if not (job.fit_dir(d) / "model.pkl").exists():
        raise HTTPException(409, "No accepted model to summarise: fit one in Phase E first.")
    proc = _summary_jobs.get(run_id)
    if proc is None or proc.poll() is not None:
        root = str(Path(__file__).resolve().parents[2])
        env = {**os.environ, "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", "")}
        log = open(job.fit_dir(d) / "summary_log.txt", "w", encoding="utf-8")
        _summary_jobs[run_id] = subprocess.Popen([sys.executable, "-m", "mmm_tool.summary", "--run-dir", str(d)],
                                                 stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
    return {"building": True}


# ---------- Phase G: What-If (steps 43-47) ----------
# Curves (step 44) and exact results (step 45) are computed from the saved model in a
# background process, so Meridian never loads in the server; results are cached by model.

class ExactIn(BaseModel):
    period: str = "all"
    multipliers: dict[str, float]


class SavedIn(BaseModel):
    scenarios: list[dict[str, Any]]


_whatif_jobs: dict[str, subprocess.Popen] = {}


def _accepted_model(d: Path) -> dict:
    st = job.status(d) or {}
    res = st.get("result") or {}
    if res.get("status") != "accepted" or not (job.fit_dir(d) / "model.pkl").exists():
        raise HTTPException(409, "What-If needs an accepted model: fit and accept one in Phase E first.")
    return st


def _whatif_job(d: Path, name: str, args: list[str]) -> bool:
    """Start a What-If job unless one with this name is running; True while it runs."""
    tag = f"{d.name}|{name}"
    proc = _whatif_jobs.get(tag)
    if proc is not None and proc.poll() is None:
        return True
    from .. import whatif

    folder = whatif.folder(d)
    folder.mkdir(parents=True, exist_ok=True)
    root = str(Path(__file__).resolve().parents[2])
    env = {**os.environ, "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", "")}
    log = open(folder / f"log_{name}.txt", "w", encoding="utf-8")
    _whatif_jobs[tag] = subprocess.Popen([sys.executable, "-m", "mmm_tool.whatif", "--run-dir", str(d), *args],
                                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
    return True


def _cached(path: Path, key: str) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if data.get("model_key") == key else None


@app.get("/api/runs/{run_id}/whatif")
def whatif(run_id: str, period: str = "all") -> dict:
    """Everything the What-If screen needs: settings, model facts, the period's curves (or that
    they are being computed) and the saved scenarios."""
    from .. import whatif as wi
    from ..whatif.scenario import settings as wsettings

    d = _run_dir(run_id)
    st = _accepted_model(d)
    fd, key = job.fit_dir(d), wi.model_key(d)
    summary = json.loads((fd / "summary.json").read_text(encoding="utf-8")) if (fd / "summary.json").exists() else {}
    version = summary.get("version") or {}
    meta = json.loads((fd / "model_meta.json").read_text(encoding="utf-8")) if (fd / "model_meta.json").exists() else {}
    periods = ["all"] + (["last_52"] if int(version.get("weeks") or meta.get("weeks") or 0) >= 104 else [])
    if period not in periods:
        raise HTTPException(400, f"Unknown period '{period}'; use {', '.join(periods)}.")
    curves = _cached(wi.folder(d) / f"curves_{period}.json", key)
    building = False
    if curves is None:
        building = _whatif_job(d, f"curves_{period}", ["--curves", period])
    saved = _cached(wi.folder(d) / "saved.json", key) or {"scenarios": []}
    out = {
        "settings": wsettings(), "periods": periods, "period": period, "model_key": key,
        "weeks_all": int(version.get("weeks") or meta.get("weeks") or 0) or None,
        "kpi": version.get("kpi") or (st.get("result") or {}).get("model", {}).get("kpi"), "kpi_type": version.get("kpi_type"),
        "money": version.get("money"), "currency": version.get("currency"),
        "no_saturation": sorted(c for c, v in ((meta.get("config") or {}).get("saturation") or {}).items() if v == "none"),
        "curves": None if curves and curves.get("error") else curves, "curves_error": (curves or {}).get("error"),
        "building": building and curves is None, "saved": saved["scenarios"],
    }
    built_from = (st.get("result") or {}).get("inputs")
    if built_from:
        run = _pipeline(d)
        out["stale"] = bool(run.d is not None and run.d.status == "complete" and job.inputs_hash(run.d) != built_from)
    return _clean(out)


@app.post("/api/runs/{run_id}/whatif/exact")
def whatif_exact(run_id: str, body: ExactIn) -> dict:
    """Step 45: the scenario on every posterior draw of the saved model (a few seconds, in the background)."""
    from .. import whatif as wi

    d = _run_dir(run_id)
    _accepted_model(d)
    if any(not (v >= 0) for v in body.multipliers.values()):
        raise HTTPException(400, "Spend can't go below zero.")
    k, key = wi.scenario_key(body.period, body.multipliers), wi.model_key(d)
    done = _cached(wi.folder(d) / f"exact_{k}.json", key)
    if done is not None:
        return _clean({"key": k, "result": done})
    wi.write(wi.folder(d) / f"request_{k}.json", {"period": body.period, "multipliers": body.multipliers})
    _whatif_job(d, f"exact_{k}", ["--exact", k])
    return {"key": k, "building": True}


@app.get("/api/runs/{run_id}/whatif/exact/{key}")
def whatif_exact_result(run_id: str, key: str) -> dict:
    from .. import whatif as wi

    d = _run_dir(run_id)
    done = _cached(wi.folder(d) / f"exact_{key}.json", wi.model_key(d))
    if done is not None:
        return _clean({"key": key, "result": done})
    proc = _whatif_jobs.get(f"{d.name}|exact_{key}")
    if proc is None or proc.poll() is not None:
        raise HTTPException(404, "No exact result for this scenario: run it again.")
    return {"key": key, "building": True}


@app.put("/api/runs/{run_id}/whatif/saved")
def whatif_saved(run_id: str, body: SavedIn) -> dict:
    """Step 47: the saved scenarios (at most saved_max), kept with the run and the model they came from."""
    from .. import whatif as wi
    from ..whatif.scenario import settings as wsettings

    d = _run_dir(run_id)
    _accepted_model(d)
    most = int(wsettings()["saved_max"])
    if len(body.scenarios) > most:
        raise HTTPException(400, f"At most {most} saved scenarios; remove one first.")
    for sc in body.scenarios:
        if not isinstance(sc.get("name"), str) or not isinstance(sc.get("multipliers"), dict):
            raise HTTPException(400, "Each scenario needs a name and its multipliers.")
    wi.write(wi.folder(d) / "saved.json", {"scenarios": body.scenarios, "model_key": wi.model_key(d)})
    return {"saved": body.scenarios}


# ---------- Phase H: Plan, Optimize (steps 48-54) ----------

class PlanIn(BaseModel):
    plan: dict[str, Any]


def _plan_inputs(d: Path) -> dict[str, Any]:
    """The model's paid channels in Meridian's order, the periods, and each channel's historical
    spend per period, from the run's model table (no Meridian needed)."""
    st = _accepted_model(d)
    fd = job.fit_dir(d)
    summary = json.loads((fd / "summary.json").read_text(encoding="utf-8")) if (fd / "summary.json").exists() else {}
    version = summary.get("version") or {}
    run = _pipeline(d)
    if run.d is None or run.d.status != "complete":
        raise HTTPException(409, "Phase D isn't complete for this run's current answers.")
    cols = run.d.coord_to_columns()["coord_to_columns"]
    date = cols["time"]
    chans = [c for c in run.d.channels if c.kind == "media"] + [c for c in run.d.channels if c.kind == "rf"]
    names = [c.name for c in chans]
    paid = summary.get("paid")
    built_from = (st.get("result") or {}).get("inputs")
    stale = bool(built_from and job.inputs_hash(run.d) != built_from)
    if paid and sorted(paid) != sorted(names):
        raise HTTPException(409, "The channels changed after this model was fitted (steps 15-17); refit in Phase E first.")
    weeks = sorted(pd.to_datetime(run.d.table[date].unique()))
    wk = [str(w.date()) for w in weeks]
    periods = {"all": (wk[0], wk[-1])}
    if len(wk) >= 104:
        periods["last_52"] = (wk[-52], wk[-1])
    hist = {}
    for p_, (a, b) in periods.items():
        t = run.d.table[(pd.to_datetime(run.d.table[date]) >= a) & (pd.to_datetime(run.d.table[date]) <= b)]
        hist[p_] = {c.name: float(t[c.spend].sum()) for c in chans}
    meta = json.loads((fd / "model_meta.json").read_text(encoding="utf-8")) if (fd / "model_meta.json").exists() else {}
    money = version.get("money")
    if money is None:
        money = bool(run.d.kpi.get("kpi_type") == "revenue" or run.d.kpi.get("revenue_per_kpi_column") or run.d.kpi.get("revenue_per_kpi_value"))
    return {"channels": names, "periods": periods, "hist": hist, "money": bool(money), "kpi": version.get("kpi") or run.d.kpi.get("kpi"),
            "kpi_type": version.get("kpi_type") or run.d.kpi.get("kpi_type"), "currency": version.get("currency") or run.d.kpi.get("currency"),
            "no_saturation": sorted(c for c, v in ((meta.get("config") or {}).get("saturation") or {}).items() if v == "none"),
            "stale": stale, "run": run}


def _plan_check(inp: dict, plan: dict) -> dict:
    from ..plan.optimize import PlanError, check

    period = plan.get("period", "all")
    if period not in inp["hist"]:
        raise HTTPException(400, f"Unknown period '{period}'; use {', '.join(inp['hist'])}.")
    try:
        return check(plan, inp["hist"][period], inp["money"], inp["periods"])
    except PlanError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/runs/{run_id}/plan")
def plan_screen(run_id: str) -> dict:
    """What the Plan and Optimize screens need, and the last optimization."""
    from .. import whatif as wi
    from ..plan.optimize import defaults, settings as psettings

    d = _run_dir(run_id)
    inp = _plan_inputs(d)
    pd_ = job.fit_dir(d) / "plan"
    last = None
    if (pd_ / "last.json").exists():
        k = json.loads((pd_ / "last.json").read_text(encoding="utf-8")).get("key")
        last = _cached(pd_ / f"result_{k}.json", wi.model_key(d)) if k else None
        if last is None and k and (pd_ / f"request_{k}.json").exists():
            proc = _whatif_jobs.get(f"{d.name}|plan_{k}")
            if proc is not None and proc.poll() is None:
                last = {"key": k, "building": True}
    out = {k: v for k, v in inp.items() if k != "run"}
    out.update(settings=psettings(), defaults=defaults(inp["channels"], inp["money"]), last=last)
    key = _answer_key(d)
    st = job.status(d) or {}
    if key and st.get("result"):
        from ..fit import truth

        out["truth_marginal_roi"] = truth.compare(st["result"], inp["run"].d.channels, key, inp["run"].d.kpi)["true_marginal_roi"]
    return _clean(out)


@app.post("/api/runs/{run_id}/plan/check")
def plan_check(run_id: str, body: PlanIn) -> dict:
    """Step 53: the plan as BudgetOptimizer.optimize() arguments, or every problem with it
    (a plan being edited is often incomplete, so problems are an answer, not an error)."""
    try:
        return _clean({"ok": True, **_plan_check(_plan_inputs(_run_dir(run_id)), body.plan)})
    except HTTPException as e:
        if e.status_code != 400:
            raise
        return {"ok": False, "problems": e.detail}


@app.post("/api/runs/{run_id}/plan/optimize")
def plan_optimize(run_id: str, body: PlanIn) -> dict:
    """Step 54: run the optimizer in the background (about a minute). Cached by plan and model."""
    import hashlib as _h

    from .. import whatif as wi

    d = _run_dir(run_id)
    checked = _plan_check(_plan_inputs(d), body.plan)
    k = _h.sha256(json.dumps(checked["args"], sort_keys=True, default=str).encode()).hexdigest()[:12]
    pd_ = job.fit_dir(d) / "plan"
    wi.write(pd_ / "last.json", {"key": k})
    done = _cached(pd_ / f"result_{k}.json", wi.model_key(d))
    if done is not None:
        return _clean({"key": k, "result": done})
    wi.write(pd_ / f"request_{k}.json", checked)
    tag = f"{d.name}|plan_{k}"
    proc = _whatif_jobs.get(tag)
    if proc is None or proc.poll() is not None:
        root = str(Path(__file__).resolve().parents[2])
        env = {**os.environ, "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", "")}
        log = open(pd_ / f"log_{k}.txt", "w", encoding="utf-8")
        _whatif_jobs[tag] = subprocess.Popen([sys.executable, "-m", "mmm_tool.plan", "--run-dir", str(d), "--key", k],
                                             stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
    return {"key": k, "building": True}


@app.get("/api/runs/{run_id}/plan/result/{key}")
def plan_result(run_id: str, key: str) -> dict:
    from .. import whatif as wi

    d = _run_dir(run_id)
    done = _cached(job.fit_dir(d) / "plan" / f"result_{key}.json", wi.model_key(d))
    if done is not None:
        return _clean({"key": key, "result": done})
    proc = _whatif_jobs.get(f"{d.name}|plan_{key}")
    if proc is None or proc.poll() is not None:
        raise HTTPException(404, "No optimization for this plan: run it again.")
    return {"key": key, "building": True}


@app.get("/api/runs/{run_id}/plan/report/{key}")
def plan_report(run_id: str, key: str) -> Response:
    """Meridian's own optimization report for a finished optimization."""
    d = _run_dir(run_id)
    if not re.fullmatch(r"[0-9a-f]{12}", key):
        raise HTTPException(404, "No such report.")
    path = job.fit_dir(d) / "plan" / f"report_{key}" / "optimization_summary.html"
    if not path.exists():
        raise HTTPException(404, "No Meridian report for this optimization.")
    return FileResponse(path, media_type="text/html", filename=f"{Path(_meta(d)['file']).stem}_optimization.html")


# ---------- Phase I: Export, refit (steps 55-58) ----------

class RegisterIn(BaseModel):
    line: str
    note: str = ""


class VersionIn(BaseModel):
    line: str
    version: int


def _run_log(d: Path):
    meta = _meta(d)
    run = _pipeline(d, meta)
    return run, run.last.decision_log if run.last else []


def _stale(d: Path, run) -> bool:
    st = job.status(d) or {}
    built_from = (st.get("result") or {}).get("inputs")
    return bool(built_from and run.d is not None and run.d.status == "complete" and job.inputs_hash(run.d) != built_from)


def _period(run) -> dict | None:
    """The weeks the model table covers (its time column, as Meridian reads it)."""
    if run.d is None or run.d.status != "complete":
        return None
    t = pd.to_datetime(run.d.table[run.d.coord_to_columns()["coord_to_columns"]["time"]])
    return {"data_from": str(t.min())[:10], "data_to": str(t.max())[:10], "weeks": int(t.nunique())}


def _registry_entry(d: Path) -> dict | None:
    from ..export import registry as reg

    hit = reg.find_run(reg.load(RUNS_DIR), d.name)
    return {"line": hit[0], **hit[1]} if hit else None


@app.get("/api/runs/{run_id}/export")
def export_screen(run_id: str) -> dict:
    """Phase I: what can be downloaded, the registry, and (for a refit) its comparison."""
    from ..export import refit as rf
    from ..export import registry as reg
    from ..export import report as rpt

    d = _run_dir(run_id)
    meta = _meta(d)
    fd = job.fit_dir(d)
    run, log = _run_log(d)
    st = job.status(d) or {}
    res = st.get("result") or {}
    mine = _registry_entry(d)
    R = reg.load(RUNS_DIR)
    comparison = mine.get("refit") if mine else None
    base = None
    if meta.get("refit_of"):
        ro = meta["refit_of"]
        base = next((v for v in R["lines"].get(ro["line"], {}).get("versions", []) if v["version"] == ro["version"]), None)
        if comparison is None and base and res.get("status") in ("accepted", "rejected"):
            try:
                comparison = rf.compare(base, reg.snapshot(d, _period(run)))            # before it is registered
            except reg.RegistryError:
                comparison = None
    g = rpt.gather(d, log, run.d.kpi if run.d else (run.a.kpi if run.a and run.a.status == "complete" else {}), mine)
    return _clean({
        "fit_status": res.get("status"), "stale": _stale(d, run), "not_converged": bool(res.get("not_converged")),
        "parts": {"model": (fd / "model.pkl").exists(), "summary": (fd / "summary.json").exists(),
                  "meridian_html": (fd / "model_results_summary.html").exists(), "whatif": (fd / "whatif" / "saved.json").exists(),
                  "plan": (fd / "plan" / "last.json").exists(), "answer_key": (d / ANSWER_KEY).exists()},
        "csv": sorted(rpt.csv_tables(g)), "decisions": len(log) + len(res.get("decision_log", [])),
        "registry": reg.rows(R), "this": mine, "refit_of": meta.get("refit_of"), "base": base, "comparison": comparison,
        "seed": res.get("seed"), "file": meta["file"],
    })


@app.get("/api/runs/{run_id}/bundle.zip")
def download_bundle(run_id: str) -> Response:
    """Step 55: one zip that reproduces the model and every decision."""
    from ..export import bundle

    d = _run_dir(run_id)
    run, log = _run_log(d)
    kpi = run.d.kpi if run.d else {}
    data = bundle.build(d, log, kpi, _registry_entry(d), roles=run.a.roles if run.a and run.a.status == "complete" else None)
    name = f"{Path(_meta(d)['file']).stem}_{d.name}_bundle.zip"
    return Response(data, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/runs/{run_id}/report.html")
def download_report(run_id: str) -> Response:
    """Step 56: the run as one self-contained HTML report."""
    from ..export import report as rpt

    d = _run_dir(run_id)
    run, log = _run_log(d)
    res = (job.status(d) or {}).get("result") or {}
    g = rpt.gather(d, log + res.get("decision_log", []), run.d.kpi if run.d else {}, _registry_entry(d))
    return HTMLResponse(rpt.html_report(g))


@app.get("/api/runs/{run_id}/exports/{name}")
def download_csv(run_id: str, name: str) -> Response:
    from ..export import report as rpt

    d = _run_dir(run_id)
    run, log = _run_log(d)
    res = (job.status(d) or {}).get("result") or {}
    tables = rpt.csv_tables(rpt.gather(d, log + res.get("decision_log", []), run.d.kpi if run.d else {}, _registry_entry(d)))
    if name not in tables:
        raise HTTPException(404, f"No table '{name}' for this run; there are {', '.join(sorted(tables)) or 'none yet'}.")
    return Response(tables[name], media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/registry")
def get_registry() -> dict:
    from ..export import registry as reg

    return _clean({"lines": reg.rows(reg.load(RUNS_DIR))})


@app.post("/api/runs/{run_id}/register")
def register_run(run_id: str, body: RegisterIn) -> dict:
    """Step 57: add this run's model to a product line as a new (draft) version."""
    from ..export import registry as reg

    d = _run_dir(run_id)
    try:
        entry = reg.register(RUNS_DIR, d, body.line, body.note, _meta(d).get("refit_of"), _period(_run_log(d)[0]))
    except reg.RegistryError as e:
        raise HTTPException(400, str(e)) from e
    return _clean({"entry": entry, "registry": reg.rows(reg.load(RUNS_DIR))})


@app.post("/api/registry/{action}")
def registry_action(action: str, body: VersionIn) -> dict:
    """Approve a version (an accepted, converged, up-to-date model) or retire one."""
    from ..export import registry as reg

    if action not in ("approve", "retire"):
        raise HTTPException(404, "Unknown registry action.")
    R = reg.load(RUNS_DIR)
    v = next((x for x in R["lines"].get(body.line, {}).get("versions", []) if x["version"] == body.version), None)
    stale = False
    if v is not None and action == "approve" and (RUNS_DIR / v["run_id"]).exists():
        d = RUNS_DIR / v["run_id"]
        stale = _stale(d, _run_log(d)[0])
    try:
        entry = reg.set_status(RUNS_DIR, body.line, body.version, action, stale=stale)
    except reg.RegistryError as e:
        raise HTTPException(400, str(e)) from e
    return _clean({"entry": entry, "registry": reg.rows(reg.load(RUNS_DIR))})


@app.post("/api/refit")
async def start_refit(line: str = Form(...), version: int = Form(...), file: UploadFile = File(...)) -> dict:
    """Step 58: a new run on the new export, from Phase A, with the approved run's answers, seed and sampler."""
    from ..export import refit as rf
    from ..export import registry as reg

    R = reg.load(RUNS_DIR)
    v = next((x for x in R["lines"].get(line, {}).get("versions", []) if x["version"] == version), None)
    if v is None or not (RUNS_DIR / v["run_id"]).exists():
        raise HTTPException(404, f"No version {version} of '{line}' with its run on this machine.")
    if v["status"] != "approved":
        raise HTTPException(400, "Refit from the approved version of a product line.")
    name = _upload_name(file.filename)
    if Path(name).suffix.lower() not in EXCEL_EXT | TEXT_EXT:
        raise HTTPException(400, "Upload a CSV, TSV, TXT or Excel file.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is over 50 MB.")
    d = RUNS_DIR / uuid.uuid4().hex[:12]
    d.mkdir(parents=True)
    (d / name).write_bytes(data)
    src_run = _run_log(RUNS_DIR / v["run_id"])[0]
    roles = src_run.a.roles if src_run.a and src_run.a.status == "complete" else None
    rf.start(RUNS_DIR / v["run_id"], d, d / name, line, version, roles=roles)
    return _evaluate(d)


@app.get("/api/runs/{run_id}/model_results_summary.html")
def download_meridian_summary(run_id: str) -> Response:
    """Meridian's own HTML results summary, written next to the accepted model."""
    d = _run_dir(run_id)
    path = job.fit_dir(d) / "model_results_summary.html"
    if not path.exists():
        raise HTTPException(404, "No Meridian summary: it is written when a model is accepted.")
    return FileResponse(path, media_type="text/html", filename=f"{Path(_meta(d)['file']).stem}_meridian_summary.html")


@app.get("/api/role-memory")
def get_role_memory() -> dict:
    kw = load_keywords()
    return {"columns": _memory().to_list(), "keyword_files": kw.sources}


@app.delete("/api/role-memory")
def forget_roles(column: str | None = None) -> dict:
    """Forget one remembered column (?column=name) or all of them."""
    return {"forgotten": _memory().forget(column)}


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str) -> dict:
    shutil.rmtree(_run_dir(run_id))
    return {"deleted": run_id}


def main() -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(prog="mmm-ui", description="Start the MMM tool web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    print(f"MMM tool on http://{args.host}:{args.port}  (runs saved in {RUNS_DIR.resolve()})")
    uvicorn.run(app, host=args.host, port=args.port)
