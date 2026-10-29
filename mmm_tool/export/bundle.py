"""Step 55: the model bundle. One zip that reproduces the model and every decision:

  input/           the export as uploaded
  answers.json     every answer given (the run's answers; defaults are re-derived)
  decisions.json   the full decision log, phases A-E, who decided what (also decisions.csv)
  data/            the table after each phase and the model configuration (model_config.json)
  model/           model.pkl (Meridian's save_mmm) and model_meta.json
  validation/      Phase E's result (every gate) and the fit request (sampler, seed)
  summary/         the Model Summary (summary.json) and Meridian's HTML report
  whatif/, plan/   saved What-If scenarios and the last optimization
  reports/         the HTML report and CSV tables (step 56)
  MANIFEST.json    versions (tool, Python, Meridian), seed, and a SHA-256 for every file
  REPRODUCE.md     how to rebuild and refit it

verify() reruns phases A-D from the bundle's own input and answers and checks the model
table and configuration come out identical; refitting with the bundle's seed and sampler
then gives the same posterior (Phase E is deterministic for a seed).
"""
from __future__ import annotations

import hashlib
import io
import json
import platform
import re
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

from . import report as rpt

DATA_FILES = ("ingested.csv", "cleaned.csv", "profiled.csv", "model_data.csv", "model_config.json")


def tool_version() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    out: dict[str, Any] = {"package": "mmm_tool"}
    try:                                           # tomllib is 3.11+; the version line is simple enough to read
        text = (root / "pyproject.toml").read_text(encoding="utf-8")
        out["version"] = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)
    except Exception:  # noqa: BLE001 - versions are information
        pass
    try:
        out["commit"] = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                                       timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        out["commit"] = None
    return out


def _sampler_arg(sampler: Any) -> str:
    if isinstance(sampler, dict):
        return "'" + json.dumps({k: sampler[k] for k in ("n_chains", "n_adapt", "n_burnin", "n_keep") if k in sampler}) + "'"
    return str(sampler or "quick")


def reproduce_md(meta: dict, request: dict, manifest: dict) -> str:
    opts = request.get("options") or {}
    extra = (" --stability" if opts.get("stability") else "") + (" --placebo" if opts.get("placebo") else "")
    return f"""# Reproduce this model

Made with mmm_tool {manifest['tool'].get('version', '')} (commit {manifest['tool'].get('commit') or 'unknown'}),
Python {manifest['python']}, Meridian {manifest.get('meridian') or 'unknown'}.

1. Check the bundle rebuilds phases A-D exactly (same model table and configuration; no fit, seconds):

       python -m mmm_tool.export --verify this_bundle.zip

2. Refit it (Phase E) with the same answers, seed and sampler; the same versions give the same
   posterior:

       unzip this_bundle.zip -d bundle
       MMM_ROLES_FILE=bundle/config/roles.yaml MMM_CALENDARS=bundle/config/calendars.yaml \\
       python -m mmm_tool.fit "bundle/input/{meta.get('file', '')}" --answers bundle/answers.json \\
           --packs "{','.join(f'bundle/config/packs/{n}.yaml' for n in manifest.get('packs') or [])}" --calendar {manifest.get('calendar') or 'none'} \\
           --seed {opts.get('seed', manifest.get('seed'))} --sampler {_sampler_arg(opts.get('sampler'))}{extra} --out refit/

3. Or load the saved model directly: `meridian.model.model.load_mmm("bundle/model/model.pkl")`
   (needs the same Meridian version).

Every decision, and who made it, is in decisions.json / decisions.csv.
"""


def _settings_files(run_settings: dict) -> dict[str, bytes]:
    """The keyword and calendar settings the run was read with, so a rebuild elsewhere reads it the same way."""
    from ..configure.context import user_calendars_file
    from ..ingest.keywords import _pack_path, user_file

    out: dict[str, bytes] = {}
    uf = user_file()
    if uf.is_file():
        out["config/roles.yaml"] = uf.read_bytes()
    packs = list(run_settings["packs"])
    for n in packs:
        out[f"config/packs/{Path(n).stem}.yaml"] = _pack_path(n).read_bytes()
    out["config/packs.json"] = json.dumps([Path(n).stem for n in packs]).encode()
    out["config/settings.json"] = json.dumps({"packs": [Path(n).stem for n in packs], "calendar": run_settings["calendar"]}).encode()
    cf = user_calendars_file()
    if cf.is_file():
        out["config/calendars.yaml"] = cf.read_bytes()
    return out


def build(run_dir: Path, decision_log: list[dict], kpi: dict, registry_entry: dict | None = None,
          roles: dict[str, str] | None = None) -> bytes:
    run_dir = Path(run_dir)
    fd = run_dir / "fit"
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    request = json.loads((fd / "request.json").read_text(encoding="utf-8")) if (fd / "request.json").exists() else {}
    model_meta = json.loads((fd / "model_meta.json").read_text(encoding="utf-8")) if (fd / "model_meta.json").exists() else {}
    files: dict[str, bytes] = {}

    def add_file(arc: str, p: Path) -> None:
        if p.exists() and p.is_file():
            files[arc] = p.read_bytes()

    add_file(f"input/{meta['file']}", run_dir / meta["file"])
    # The confirmed roles are written out in full: accepted suggestions depend on keywords and
    # remembered roles, which another machine may not have.
    answers = {**meta.get("answers", {}), **({"roles": roles} if roles else {})}
    files["answers.json"] = json.dumps(answers, indent=1).encode()
    from ..settings import for_run

    files.update(_settings_files(for_run(run_dir, meta)[0]))
    full_log = list(decision_log) + [{**e, "phase": "E"} for e in (json.loads((fd / "result.json").read_text(encoding="utf-8")).get("decision_log", [])
                                                                if (fd / "result.json").exists() else [])]
    files["decisions.json"] = json.dumps(full_log, indent=1, default=str).encode()
    for name in DATA_FILES:
        add_file(f"data/{name}", run_dir / name)
    add_file("model/model.pkl", fd / "model.pkl")
    add_file("model/model_meta.json", fd / "model_meta.json")
    add_file("validation/result.json", fd / "result.json")
    add_file("validation/request.json", fd / "request.json")
    add_file("summary/summary.json", fd / "summary.json")
    add_file("summary/model_results_summary.html", fd / "model_results_summary.html")
    add_file("whatif/saved.json", fd / "whatif" / "saved.json")
    last = fd / "plan" / "last.json"
    if last.exists():
        k = json.loads(last.read_text(encoding="utf-8")).get("key")
        add_file(f"plan/result_{k}.json", fd / "plan" / f"result_{k}.json")
        add_file("plan/optimization_summary.html", fd / "plan" / f"report_{k}" / "optimization_summary.html")
    add_file("test/answer_key.json", run_dir / "answer_key.json")
    g = rpt.gather(run_dir, full_log, kpi, registry_entry)
    files["reports/report.html"] = rpt.html_report(g).encode()
    for name, text in rpt.csv_tables(g).items():
        files[f"reports/{name}"] = text.encode()
    files.setdefault("decisions.csv", files.get("reports/decisions.csv", b""))
    manifest = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_id": run_dir.name, "tool": tool_version(),
                "packs": json.loads(files["config/packs.json"]),
                "calendar": json.loads(files["config/settings.json"])["calendar"],
                "python": platform.python_version(), "meridian": model_meta.get("meridian"), "seed": (request.get("options") or {}).get("seed"),
                "sampler": (request.get("options") or {}).get("sampler"), "registry": registry_entry and
                {k: registry_entry[k] for k in ("line", "version", "status") if k in registry_entry},
                "files": [{"path": k, "bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()} for k, v in sorted(files.items())]}
    files["REPRODUCE.md"] = reproduce_md(meta, request, manifest).encode()
    files["MANIFEST.json"] = json.dumps(manifest, indent=1, default=str).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for k, v in sorted(files.items()):
            z.writestr(k, v)
    return buf.getvalue()


def verify(bundle: Path | bytes, work: Path) -> dict[str, Any]:
    """Rerun phases A-D from the bundle's input and answers; compare with its data files and checksums."""
    import pandas as pd

    from ..pipeline import run_pipeline

    z = zipfile.ZipFile(io.BytesIO(bundle) if isinstance(bundle, (bytes, bytearray)) else bundle)
    manifest = json.loads(z.read("MANIFEST.json"))
    bad_sums = [f["path"] for f in manifest["files"] if hashlib.sha256(z.read(f["path"])).hexdigest() != f["sha256"]]
    inp = next(n for n in z.namelist() if n.startswith("input/"))
    work.mkdir(parents=True, exist_ok=True)
    src = work / Path(inp).name
    src.write_bytes(z.read(inp))
    answers = json.loads(z.read("answers.json"))
    names = set(z.namelist())
    env = {}
    for arc, var in (("config/roles.yaml", "MMM_ROLES_FILE"), ("config/calendars.yaml", "MMM_CALENDARS")):
        if arc in names:
            (work / Path(arc).name).write_bytes(z.read(arc))
            env[var] = str(work / Path(arc).name)
        else:
            env[var] = str(work / f"none_{Path(arc).name}")           # none in the bundle: none now either
    packs = json.loads(z.read("config/packs.json")) if "config/packs.json" in names else []
    bad = [n for n in packs if not (isinstance(n, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", n))]
    if bad:                                    # a pack name becomes a file name: never a path
        raise ValueError(f"Bundle lists keyword pack name(s) that aren't plain names: {bad}")
    for n in packs:
        (work / "packs").mkdir(exist_ok=True)
        (work / "packs" / f"{n}.yaml").write_bytes(z.read(f"config/packs/{n}.yaml"))
    env["MMM_KEYWORD_PACKS"] = ",".join(str(work / "packs" / f"{n}.yaml") for n in packs)
    import os

    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    run_settings = None
    if "config/settings.json" in names:              # the run's own packs (copied above) and calendar
        rs = json.loads(z.read("config/settings.json"))
        if sorted(rs.get("packs") or []) != sorted(packs):
            raise ValueError("The bundle's settings name keyword packs it doesn't contain.")
        run_settings = {"packs": [str(work / "packs" / f"{n}.yaml") for n in rs.get("packs") or []],
                        "calendar": rs.get("calendar") or "none"}
    try:
        run = run_pipeline(src, answers, settings=run_settings)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    checks = {"checksums": not bad_sums, "phase_d_complete": run.d is not None and run.d.status == "complete"}
    if checks["phase_d_complete"]:
        if "data/model_data.csv" in z.namelist():
            mine = run.d.table.to_csv(index=False)
            stored = pd.read_csv(io.BytesIO(z.read("data/model_data.csv")))
            checks["model_table"] = pd.read_csv(io.StringIO(mine)).equals(stored)       # same values, read the same way
        if "data/model_config.json" in z.namelist():
            from ..api.app import _clean

            checks["model_config"] = json.loads(json.dumps(_clean(run.d.config), default=str)) == json.loads(z.read("data/model_config.json"))
    return {"ok": all(checks.values()), "checks": checks, "bad_checksums": bad_sums, "files": len(manifest["files"]),
            "seed": manifest.get("seed"), "sampler": manifest.get("sampler")}
