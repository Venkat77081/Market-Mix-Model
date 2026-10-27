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
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from .. import samples
from ..ingest import run_phase_a
from ..ingest.load import EXCEL_EXT, TEXT_EXT

RUNS_DIR = Path(os.environ.get("MMM_RUNS_DIR", "runs"))
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
RUN_ID = re.compile(r"^[0-9a-f]{12}$")

app = FastAPI(title="Meridian MMM Tool")


class AnswersIn(BaseModel):
    answers: dict[str, Any]


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


def _evaluate(d: Path) -> dict:
    meta = _meta(d)
    result = run_phase_a(d / meta["file"], meta["answers"])
    out = result.to_dict()
    if result.status == "complete":
        t = result.table.head(8)
        out["table_preview"] = {
            "columns": list(t.columns),
            "rows": [[None if v is None else str(v)[:10] if hasattr(v, "date") else v for v in r] for r in t.itertuples(index=False)],
        }
        result.table.to_csv(d / "ingested.csv", index=False)
    return {"run_id": d.name, "file": meta["file"], "answers": meta["answers"], "result": _clean(out)}


def _new_run(filename: str) -> Path:
    d = RUNS_DIR / uuid.uuid4().hex[:12]
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"file": filename, "answers": {}}))
    return d


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


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
    d = _run_dir(run_id)
    if _evaluate(d)["result"]["status"] != "complete":
        raise HTTPException(409, "Phase A is not complete yet.")
    stem = Path(_meta(d)["file"]).stem
    return FileResponse(d / "ingested.csv", media_type="text/csv", filename=f"{stem}_phase_a.csv")


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
