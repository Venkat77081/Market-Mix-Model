"""Run Phase E as a background job for a run folder.

The web API starts `python -m mmm_tool.fit --run-dir <run>` as a separate
process, so a fit that takes minutes never blocks a request. The job writes
into <run>/fit/:

  request.json   what was asked (sampler, optional checks)
  status.json    state, current step, progress and a heartbeat
  result.json    the Phase E result, once finished
  model.pkl      the final model (step 36), with model_meta.json
  log.txt        the job's output

Liveness comes from the heartbeat (works the same on Windows, where probing a
process id isn't safe) and, while the API that started it is running, from
the process handle.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

HEARTBEAT_SECONDS = 15
STALE_SECONDS = 180
ACTIVE = ("queued", "running")
_procs: dict[str, subprocess.Popen] = {}


def fit_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "fit"


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write(path: Path, data: dict) -> None:
    """Write via a temporary file so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
    os.replace(tmp, path)


def inputs_hash(d) -> str:
    """What the fit was built from: the model table and the Phase D config."""
    h = hashlib.sha256(d.table.to_csv(index=False).encode())
    h.update(json.dumps(d.config, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def status(run_dir: Path) -> dict | None:
    """The job's state with its result, or None if no fit was ever started."""
    fd = fit_dir(run_dir)
    st = _read(fd / "status.json")
    if st is None:
        return None
    if st.get("state") in ACTIVE:
        proc = _procs.get(str(run_dir))
        ended = proc is not None and proc.poll() is not None
        silent = time.time() - float(st.get("heartbeat", 0)) > STALE_SECONDS
        if ended or silent:
            st.update(state="failed", message="The fit stopped unexpectedly; see fit/log.txt in the run folder.")
            _write(fd / "status.json", st)
    res = _read(fd / "result.json")
    if res is not None and st.get("state") not in ACTIVE:
        st["result"] = res
    st["summary"] = (fd / "summary.json").exists()           # Phase F's stored numbers
    return st


def start(run_dir: Path, options: dict[str, Any], memory: str | None = None) -> dict:
    run_dir = Path(run_dir)
    current = status(run_dir)
    if current and current.get("state") in ACTIVE:
        raise RuntimeError("A fit is already running for this run.")
    fd = fit_dir(run_dir)
    fd.mkdir(parents=True, exist_ok=True)
    for name in ("result.json", "status.json"):
        (fd / name).unlink(missing_ok=True)
    now = time.time()
    _write(fd / "request.json", {"options": options, "memory": memory, "requested": now})
    _write(fd / "status.json", {"state": "queued", "step": None, "message": "Starting", "fraction": 0.0,
                                "started": now, "heartbeat": now, "options": options})
    log = open(fd / "log.txt", "w", encoding="utf-8")
    kwargs: dict[str, Any] = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    root = str(Path(__file__).resolve().parents[2])  # so the job imports this same copy of mmm_tool
    env = {**os.environ, "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", "")}
    _procs[str(run_dir)] = subprocess.Popen([sys.executable, "-m", "mmm_tool.fit", "--run-dir", str(run_dir)], env=env, **kwargs)
    return status(run_dir) or {}


def cancel(run_dir: Path) -> dict | None:
    proc = _procs.pop(str(run_dir), None)
    if proc is not None and proc.poll() is None:
        proc.terminate()
    st = _read(fit_dir(run_dir) / "status.json")
    if st and st.get("state") in ACTIVE:
        st.update(state="cancelled", message="Cancelled.")
        _write(fit_dir(run_dir) / "status.json", st)
    return st


def run_job(run_dir: Path) -> int:
    """The job itself (runs in the child process)."""
    from ..ingest.memory import RoleMemory
    from ..pipeline import run_pipeline
    from .pipeline import run_phase_e

    run_dir = Path(run_dir)
    fd = fit_dir(run_dir)
    req = _read(fd / "request.json") or {}
    st = _read(fd / "status.json") or {"started": time.time()}
    lock = threading.Lock()

    def save(**kw) -> None:
        with lock:
            st.update(kw, heartbeat=time.time())
            _write(fd / "status.json", st)

    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(HEARTBEAT_SECONDS):
            save()

    threading.Thread(target=beat, daemon=True).start()
    try:
        save(state="running", message="Preparing the data (phases A-D)")
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        memory = RoleMemory(req["memory"]) if req.get("memory") else None
        from ..settings import for_run

        run = run_pipeline(run_dir / meta["file"], meta["answers"], memory=memory, settings=for_run(run_dir, meta)[0])
        if run.d is None or run.d.status != "complete":
            save(state="failed", message="Phase D isn't complete; answer the open questions first.")
            return 1
        save(inputs=inputs_hash(run.d))
        result = run_phase_e(run.d, req.get("options"), out_dir=fd,
                             progress=lambda step, msg, frac: save(step=step, message=msg, fraction=round(frac, 3)))
        out = result.to_dict()
        out["inputs"] = st["inputs"]
        _write(fd / "result.json", _jsonable(out))
        save(state="done", status=result.status, fraction=1.0,
             message={"accepted": "Accepted: every gate passed.", "rejected": "Rejected: a gate failed.",
                      "failed": "Stopped by an error."}[result.status], finished=time.time())
        return 0
    except Exception as e:  # anything outside the steps (bad run folder, Meridian missing)
        save(state="failed", message=f"{type(e).__name__}: {e}")
        raise
    finally:
        stop.set()


def _jsonable(x: Any) -> Any:
    import math

    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x if x is None or isinstance(x, (str, int, float, bool)) else str(x)
