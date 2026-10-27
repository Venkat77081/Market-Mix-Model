"""HTTP API and web UI for the pipeline.

Each run is a folder under the runs directory holding the uploaded file and
the answers given so far. Phase A is fast, so every call re-runs it from the
file with the current answers and returns the full result.

    uvicorn mmm_tool.api.app:app --reload     # or: mmm-ui
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
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
    sampler: str = "quick"
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
    meta = _meta(d)
    try:
        run = run_pipeline(d / meta["file"], meta["answers"], memory=_memory())
    except KeywordError as e:
        raise HTTPException(400, f"Your role keywords file has a problem: {e}") from e
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
    meta.update(phase=run.phase, status=final.status, updated=time.time(), decisions=len(final.pending))
    (d / "meta.json").write_text(json.dumps(meta))
    return {"run_id": d.name, "file": meta["file"], "answers": meta["answers"], "phase": run.phase,
            "result": _clean(out_a), "result_b": _clean(out_b), "result_c": _clean(out_c), "result_d": _clean(out_d),
            "fit": _clean(fit), "meridian_installed": meridian_installed()}


def _new_run(filename: str) -> Path:
    d = RUNS_DIR / uuid.uuid4().hex[:12]
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"file": filename, "answers": {}}))
    return d


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
async def create_run(file: UploadFile = File(...)) -> dict:
    name = Path(file.filename or "upload.csv").name
    if Path(name).suffix.lower() not in EXCEL_EXT | TEXT_EXT:
        raise HTTPException(400, "Upload a CSV, TSV, TXT or Excel file.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is over 50 MB.")
    d = _new_run(name)
    (d / name).write_bytes(data)
    return _evaluate(d)


@app.post("/api/runs/sample/{name}")
def create_sample_run(name: str) -> dict:
    if name not in samples.SAMPLES:
        raise HTTPException(404, "Unknown sample.")
    tmp = _new_run("")
    path = samples.SAMPLES[name][1](tmp)
    (tmp / "meta.json").write_text(json.dumps({"file": path.name, "answers": {}}))
    return _evaluate(tmp)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return _evaluate(_run_dir(run_id))


@app.put("/api/runs/{run_id}/answers")
def set_answers(run_id: str, body: AnswersIn) -> dict:
    """Replace the run's answers and re-run Phase A."""
    d = _run_dir(run_id)
    meta = _meta(d)
    meta["answers"] = body.answers
    (d / "meta.json").write_text(json.dumps(meta))
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


@app.get("/api/fit-settings")
def get_fit_settings() -> dict:
    """Sampler presets and gate thresholds (fit.yaml), for the Phase E screen."""
    return _clean(fit_settings())


@app.post("/api/runs/{run_id}/fit")
def start_fit(run_id: str, body: FitIn) -> dict:
    """Start Phase E as a background job. Poll GET .../fit for progress."""
    d = _run_dir(run_id)
    if not meridian_installed():
        raise HTTPException(503, "Phase E needs google-meridian: pip install -r requirements-model.txt, then restart.")
    if body.sampler not in fit_settings()["samplers"]:
        raise HTTPException(400, f"Unknown sampler '{body.sampler}'.")
    res = _evaluate(d)["result_d"]
    if not res or res["status"] != "complete":
        raise HTTPException(409, "Phase D is not complete yet.")
    memory = os.environ.get("MMM_ROLE_MEMORY") or str(RUNS_DIR / "role_memory.json")
    try:
        return _clean(job.start(d, body.model_dump(), memory=memory))
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e


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
