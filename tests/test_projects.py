"""Projects: a folder per business with its settings, runs and registry; older runs move in."""
from __future__ import annotations

import json

import pytest
from helpers import project_in, run_path
from test_generic import shop
from test_phase_i import fake_fit

from mmm_tool import projects

INDIA = {"packs": ["india", "insurance"], "calendar": "india_insurance"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from mmm_tool.api import app as app_mod

    monkeypatch.setattr(app_mod, "RUNS_DIR", tmp_path / "runs")
    c = TestClient(app_mod.app)
    c.runs, c.app_mod = tmp_path / "runs", app_mod
    return c


def _new(client, name, **kw):
    r = client.post("/api/projects", json={"name": name, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def _upload(client, slug):
    data = shop(money_sign="").to_csv(index=False).encode()
    return client.post("/api/runs", files={"file": ("shop.csv", data, "text/csv")}, data={"project": slug}).json()


def _legacy_run(client, tmp_path) -> str:
    """A run made before projects: directly in the runs folder, with no settings recorded."""
    slug = project_in("Scratch")
    r = _upload(client, slug)
    src = run_path(r["run_id"])
    dest = client.runs / r["run_id"]
    src.rename(dest)
    meta = json.loads((dest / "meta.json").read_text())
    meta.pop("settings")
    (dest / "meta.json").write_text(json.dumps(meta))
    return r["run_id"]


def test_create_list_edit(client):
    p = _new(client, "Acme Insurance – India", description="Motor and health", **INDIA)
    assert p["slug"] == "acme-insurance-india" and p["settings"] == INDIA
    folder = client.runs / "projects" / p["slug"]
    assert (folder / "project.json").is_file() and (folder / "runs").is_dir()
    assert client.post("/api/projects", json={"name": "acme insurance – india"}).status_code == 400   # same name
    assert client.post("/api/projects", json={"name": "  "}).status_code == 400
    assert client.post("/api/projects", json={"name": "X", "packs": ["nonesuch"]}).status_code == 400
    assert _new(client, "Acme Insurance  India")["slug"] == "acme-insurance-india-2"            # same slug, new folder
    e = client.put(f"/api/projects/{p['slug']}", json={"name": "Acme India", "description": "", "packs": [], "calendar": "none"}).json()
    assert e["name"] == "Acme India" and e["settings"] == {"packs": [], "calendar": "none"} and e["slug"] == p["slug"]
    names = [x["name"] for x in client.get("/api/projects").json()["projects"]]
    assert sorted(names) == sorted(["Acme Insurance  India", "Acme India"])
    assert client.get("/api/projects/nonesuch").status_code == 404


def test_a_new_run_lives_in_its_project_and_starts_with_its_settings(client):
    p = _new(client, "India shop", **INDIA)
    r = _upload(client, p["slug"])
    d = run_path(r["run_id"])
    assert d.parent == client.runs / "projects" / p["slug"] / "runs"
    assert r["project"] == {"slug": p["slug"], "name": "India shop"} and r["settings"]["packs"] == INDIA["packs"]
    client.put(f"/api/projects/{p['slug']}", json={"name": "India shop", "packs": [], "calendar": "none"})
    assert client.get(f"/api/runs/{r['run_id']}").json()["settings"]["packs"] == INDIA["packs"]       # keeps its own
    client.put(f"/api/runs/{r['run_id']}/answers", json={"answers": {"roles": {}}})
    page = client.get(f"/api/projects/{p['slug']}").json()
    row = page["runs"][0]
    assert row["run_id"] == r["run_id"] and row["period"]["weeks"] == 156 and row["project"]["slug"] == p["slug"]
    assert client.post("/api/runs/sample/ins_clean", params={"project": "nonesuch"}).status_code == 404
    recent = client.get("/api/runs").json()
    assert recent[0]["project"]["name"] == "India shop"


def test_each_project_has_its_own_registry(client):
    a, b = _new(client, "A")["slug"], _new(client, "B")["slug"]
    ra, rb = _upload(client, a)["run_id"], _upload(client, b)["run_id"]
    for rid in (ra, rb):
        client.put(f"/api/runs/{rid}/answers", json={"answers": {"roles": {}}})
        fake_fit(run_path(rid))
        assert client.post(f"/api/runs/{rid}/register", json={"line": "Orders"}).json()["entry"]["version"] == 1   # v1 in each
    assert (client.runs / "projects" / a / "registry.json").is_file() and not (client.runs / "registry.json").exists()
    assert client.post("/api/registry/approve", json={"line": "Orders", "version": 1, "project": a}).status_code == 200
    pa, pb = client.get(f"/api/projects/{a}").json(), client.get(f"/api/projects/{b}").json()
    assert pa["registry"][0]["approved"]["run_id"] == ra and pb["registry"][0]["approved"] is None
    listed = {p["slug"]: p for p in client.get("/api/projects").json()["projects"]}
    assert listed[a]["approved"] == [{"line": "Orders", "version": 1}] and listed[b]["versions"] == 1


def test_refit_stays_in_the_project_and_needs_one(client):
    a = _new(client, "A")["slug"]
    rid = _upload(client, a)["run_id"]
    client.put(f"/api/runs/{rid}/answers", json={"answers": {"roles": {}}})
    fake_fit(run_path(rid))
    client.post(f"/api/runs/{rid}/register", json={"line": "Orders"})
    client.post("/api/registry/approve", json={"line": "Orders", "version": 1, "project": a})
    files = {"file": ("newer.csv", (run_path(rid) / "shop.csv").read_bytes(), "text/csv")}
    assert client.post("/api/refit", data={"line": "Orders", "version": 1}, files=files).status_code == 400
    r = client.post("/api/refit", data={"line": "Orders", "version": 1, "project": a}, files=files).json()
    assert run_path(r["run_id"]).parent.parent.name == a and r["refit_of"]["run_id"] == rid


def test_moving_older_runs_in(client, tmp_path):
    rid = _legacy_run(client, tmp_path)
    home = client.get("/api/projects").json()
    assert [r["run_id"] for r in home["unfiled"]] == [rid]
    assert client.get(f"/api/runs/{rid}").json()["project"] is None
    india = _new(client, "India", **INDIA)["slug"]
    m = client.post(f"/api/runs/{rid}/move", json={"project": india}).json()
    assert m["moved"] == [rid] and run_path(rid).parent.parent.name == india
    r = client.get(f"/api/runs/{rid}").json()
    assert r["project"]["slug"] == india and r["settings"]["source"] == "project" and r["settings"]["packs"] == INDIA["packs"]
    assert client.get("/api/projects").json()["unfiled"] == []
    assert client.post(f"/api/runs/{rid}/move", json={"project": "nonesuch"}).status_code == 400


def test_a_registered_run_moves_with_its_whole_product_line(client, tmp_path):
    from mmm_tool.export import registry as reg

    v1, v2 = _legacy_run(client, tmp_path), _legacy_run(client, tmp_path)
    for rid in (v1, v2):
        fake_fit(run_path(rid))
        reg.register(client.runs, run_path(rid), "Online shop")
    other = _legacy_run(client, tmp_path)
    shop_p = _new(client, "Shop")["slug"]
    m = client.post(f"/api/runs/{v2}/move", json={"project": shop_p}).json()
    assert sorted(m["moved"]) == sorted([v1, v2])                          # v1 came with it
    R = reg.load(client.runs / "projects" / shop_p)
    assert [v["version"] for v in R["lines"]["Online shop"]["versions"]] == [1, 2]
    assert "Online shop" not in reg.load(client.runs)["lines"]
    assert run_path(other).parent == client.runs                           # the unregistered one stayed
    # the project already has that line: refused, nothing moved
    v3 = _legacy_run(client, tmp_path)
    fake_fit(run_path(v3))
    reg.register(client.runs, run_path(v3), "Online shop")
    bad = client.post(f"/api/runs/{v3}/move", json={"project": shop_p})
    assert bad.status_code == 400 and "already has a product line" in bad.json()["detail"]
    assert run_path(v3).parent == client.runs


def test_a_run_with_a_fit_in_progress_isnt_moved(client, tmp_path):
    from mmm_tool.fit import job

    rid = _legacy_run(client, tmp_path)
    job._write(job.fit_dir(run_path(rid)) / "status.json", {"state": "running", "heartbeat": __import__("time").time()})
    dest = _new(client, "P")["slug"]
    bad = client.post(f"/api/runs/{rid}/move", json={"project": dest})
    assert bad.status_code == 400 and "in progress" in bad.json()["detail"] and run_path(rid).parent == client.runs


def test_slugs_never_escape_the_projects_folder(tmp_path):
    with pytest.raises(projects.ProjectError):
        projects.folder(tmp_path, "../x")
    assert projects.find_run(tmp_path, "../../etc") is None
