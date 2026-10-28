"""Phase I: Export, refit (steps 55-58).

The bundle, its verify(), the registry and the refit comparison need no Meridian: a fit is
stood in for by the files Phase E writes (status.json, result.json).
"""
from __future__ import annotations

import io
import json
import time
import zipfile

import pytest

from mmm_tool.export import refit as rf
from mmm_tool.export import registry as reg
from mmm_tool.export.bundle import verify
from mmm_tool.fit import job

ROI = {"tv": (1.0, 0.8, 1.2), "search": (2.0, 1.5, 2.5)}


def fake_fit(run_dir, status="accepted", roi=ROI, not_converged=False):
    fd = job.fit_dir(run_dir)
    job._write(fd / "status.json", {"state": "done", "status": status, "heartbeat": time.time()})
    job._write(fd / "result.json", {"status": status, "steps": [], "not_converged": not_converged, "seed": 7,
                                    "gates": [{"step": 30, "passed": True}, {"step": 31, "passed": status == "accepted"}],
                                    "channels": [{"channel": c, "paid": True, "roi_posterior_mean": m, "roi_posterior_ci_lo": lo,
                                                  "roi_posterior_ci_hi": hi} for c, (m, lo, hi) in roi.items()]})
    (fd / "model.pkl").write_bytes(b"not a model")


# Registry (step 57) --------------------------------------------------------------------------------

def test_versions_approve_supersede_retire(tmp_path):
    runs = tmp_path / "runs"
    a, b = runs / "aaa", runs / "bbb"
    for d in (a, b):
        fake_fit(d)
    v1 = reg.register(runs, a, "Line one", "first", period={"data_from": "2023-01-02", "data_to": "2024-12-30", "weeks": 105})
    assert (v1["version"], v1["status"], v1["weeks"], v1["roi"]["tv"]["mean"]) == (1, "draft", 105, 1.0)
    reg.set_status(runs, "Line one", 1, "approve")
    reg.register(runs, b, "Line one")
    reg.set_status(runs, "Line one", 2, "approve")
    vs = {v["version"]: v["status"] for v in reg.load(runs)["lines"]["Line one"]["versions"]}
    assert vs == {1: "superseded", 2: "approved"} and reg.approved(reg.load(runs), "Line one")["run_id"] == "bbb"
    reg.set_status(runs, "Line one", 1, "retire")
    assert reg.rows(reg.load(runs))[0]["latest"]["version"] == 2
    with pytest.raises(reg.RegistryError, match="already registered"):
        reg.register(runs, a, "Line one")


@pytest.mark.parametrize("kw, stale, msg", [
    ({"status": "rejected"}, False, "Only an accepted model"),
    ({"not_converged": True}, False, "did not converge"),
    ({}, True, "refit before approving"),
])
def test_what_cannot_be_approved(tmp_path, kw, stale, msg):
    runs = tmp_path / "runs"
    fake_fit(runs / "r1", **kw)
    reg.register(runs, runs / "r1", "L")
    with pytest.raises(reg.RegistryError, match=msg):
        reg.set_status(runs, "L", 1, "approve", stale=stale)


def test_register_needs_a_finished_fit_and_a_line_name(tmp_path):
    runs = tmp_path / "runs"
    (runs / "nofit").mkdir(parents=True)
    with pytest.raises(reg.RegistryError, match="no finished fit"):
        reg.register(runs, runs / "nofit", "L")
    fake_fit(runs / "r1")
    for bad in ("", "a/b", "x" * 61):
        with pytest.raises(reg.RegistryError, match="product line"):
            reg.register(runs, runs / "r1", bad)


# Refit comparison (step 58) ---------------------------------------------------------------------------

def test_compare_flags_a_mean_outside_the_approved_range():
    base = {"roi": {c: dict(zip(("mean", "lo", "hi"), v)) for c, v in ROI.items()}, "version": 1}
    new = {"roi": {"tv": {"mean": 1.1, "lo": 0.9, "hi": 1.3}, "search": {"mean": 2.9, "lo": 2.4, "hi": 3.3}}}
    c = rf.compare(base, new)
    assert c["flagged"] == ["search"] and c["summary"] == "search flagged: ROI moved beyond its approved 90% range"
    assert {r["channel"]: r["status"] for r in c["rows"]} == {"search": "beyond its 90% range", "tv": "inside its range"}
    assert rf.compare(base, base)["summary"] == "ROI moved inside its range on all channels"


def test_compare_lists_new_and_dropped_channels():
    base = {"roi": {"tv": {"mean": 1, "lo": 0.5, "hi": 1.5}, "radio": {"mean": 1, "lo": 0.5, "hi": 1.5}}}
    new = {"roi": {"tv": {"mean": 1, "lo": 0.5, "hi": 1.5}, "social": {"mean": 3, "lo": 2, "hi": 4}}}
    c = rf.compare(base, new)
    assert {r["channel"]: r["status"] for r in c["rows"]} == {"radio": "dropped", "social": "new channel", "tv": "inside its range"}
    assert c["flagged"] == [] and c["summary"].endswith("channels changed: radio, social")


def test_refit_start_keeps_answers_seed_and_roles_still_in_the_file(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "meta.json").write_text(json.dumps({"file": "old.csv", "answers": {"x": 1, "roles": {}}, "seed": 11}))
    job._write(job.fit_dir(src) / "request.json", {"options": {"sampler": "full", "seed": 11, "placebo": True, "seed_source": "user"}})
    new = tmp_path / "new.csv"
    new.write_text("week,spend_a,sales\n2024-01-01,1,2\n")
    meta = rf.start(src, tmp_path / "run2", new, "L", 3, roles={"week": "date", "spend_a": "media_spend", "sales": "kpi", "gone": "control"})
    assert meta["answers"]["x"] == 1 and meta["answers"]["roles"] == {"week": "date", "spend_a": "media_spend", "sales": "kpi"}
    assert meta["refit_options"] == {"sampler": "full", "seed": 11, "placebo": True} and meta["seed"] == 11
    assert meta["refit_of"] == {"line": "L", "version": 3, "run_id": "src", "file": "old.csv"}
    assert (tmp_path / "run2" / "new.csv").exists()


# API, bundle (step 55) and reports (step 56) ------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from mmm_tool.api import app as app_mod

    monkeypatch.setattr(app_mod, "RUNS_DIR", tmp_path / "runs")
    c = TestClient(app_mod.app)
    c.runs = tmp_path / "runs"
    return c


def _run(client, fit="accepted"):
    run_id = client.post("/api/runs/sample/ins_clean").json()["run_id"]
    client.put(f"/api/runs/{run_id}/answers", json={"answers": {"roles": {}}})
    if fit:
        fake_fit(client.runs / run_id, fit)
    return run_id


def test_bundle_contents_and_verify(client, tmp_path, monkeypatch):
    run_id = _run(client)
    r = client.get(f"/api/runs/{run_id}/bundle.zip")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(z.namelist())
    for need in ("MANIFEST.json", "REPRODUCE.md", "answers.json", "decisions.json", "decisions.csv", "input/ins_clean.csv",
                 "data/model_data.csv", "data/model_config.json", "model/model.pkl", "validation/result.json", "reports/report.html"):
        assert need in names, need
    manifest = json.loads(z.read("MANIFEST.json"))
    assert {f["path"] for f in manifest["files"]} == names - {"MANIFEST.json", "REPRODUCE.md"}
    roles = json.loads(z.read("answers.json"))["roles"]
    assert roles and set(roles.values()) >= {"date", "kpi"}                # confirmed roles frozen in full
    assert "--verify" in z.read("REPRODUCE.md").decode()
    monkeypatch.setenv("MMM_KEYWORD_PACKS", "")                            # settings here differ: the bundle's own are used
    res = verify(r.content, tmp_path / "v")
    assert res["ok"], res
    assert res["checks"] == {"checksums": True, "phase_d_complete": True, "model_table": True, "model_config": True}


def test_verify_catches_a_changed_file(client, tmp_path):
    run_id = _run(client)
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/runs/{run_id}/bundle.zip").content))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as out:
        for n in z.namelist():
            out.writestr(n, b"tampered" if n == "decisions.json" else z.read(n))
    res = verify(buf.getvalue(), tmp_path / "v")
    assert res["ok"] is False and res["bad_checksums"] == ["decisions.json"]


def test_reports_and_csv(client):
    run_id = _run(client)
    html = client.get(f"/api/runs/{run_id}/report.html")
    assert html.status_code == 200 and "<html" in html.text.lower()
    scr = client.get(f"/api/runs/{run_id}/export").json()
    assert "decisions.csv" in scr["csv"] and scr["fit_status"] == "accepted" and scr["parts"]["model"]
    text = client.get(f"/api/runs/{run_id}/exports/decisions.csv").text
    assert text.startswith("step,decision,answer,source") and "accepted as suggested" in text     # not a bare {}
    assert client.get(f"/api/runs/{run_id}/exports/nonesuch.csv").status_code == 404


def test_register_approve_and_refit_through_the_api(client):
    run_id = _run(client)
    e = client.post(f"/api/runs/{run_id}/register", json={"line": "Brand", "note": "v1"}).json()["entry"]
    assert e["version"] == 1 and e["weeks"] and e["data_to"]                  # the period comes from Phase D's table
    new = (client.runs / run_id / "ins_clean.csv").read_bytes()
    files = {"file": ("newer.csv", new, "text/csv")}
    assert client.post("/api/refit", data={"line": "Brand", "version": 1}, files=files).status_code == 400   # not approved yet
    assert client.post("/api/registry/approve", json={"line": "Brand", "version": 1}).status_code == 200
    r = client.post("/api/refit", data={"line": "Brand", "version": 1}, files=files).json()
    assert r["refit_of"]["run_id"] == run_id and r["answers"]["roles"]          # roles frozen from the approved run
    assert r["phase"] == "D" and r["result_d"]["status"] == "complete"         # same data: nothing new to decide
    fake_fit(client.runs / r["run_id"], roi={"tv": (5.0, 4.0, 6.0)})
    scr = client.get(f"/api/runs/{r['run_id']}/export").json()
    assert scr["comparison"]["refit"]["weeks"] == e["weeks"]
    e2 = client.post(f"/api/runs/{r['run_id']}/register", json={"line": "Brand"}).json()["entry"]
    assert e2["version"] == 2 and e2["refit_of"]["version"] == 1 and e2["refit"]["summary"]
    rows = client.get("/api/registry").json()["lines"]
    assert rows[0]["approved"]["version"] == 1 and rows[0]["last_refit_version"] == 2


def test_a_rejected_model_registers_as_a_draft_only(client):
    run_id = _run(client, fit="rejected")
    assert client.post(f"/api/runs/{run_id}/register", json={"line": "B"}).json()["entry"]["status"] == "draft"
    bad = client.post("/api/registry/approve", json={"line": "B", "version": 1})
    assert bad.status_code == 400 and "accepted" in bad.json()["detail"]
