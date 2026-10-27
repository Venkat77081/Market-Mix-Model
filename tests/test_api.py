"""API tests: the UI's calls, end to end through Phase A."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mmm_tool.api import app as app_module


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "RUNS_DIR", tmp_path / "runs")
    return TestClient(app_module.app)


def test_pages_and_samples(client):
    assert "What caused sales" in client.get("/").text
    assert "Upload a spend-and-sales export" in client.get("/app").text
    names = [s["name"] for s in client.get("/api/samples").json()]
    assert "ins_clean" in names and "ins_monthly" in names


def test_sample_run_stops_at_roles_then_completes(client):
    r = client.post("/api/runs/sample/ins_clean").json()
    run_id = r["run_id"]
    assert r["result"]["status"] == "needs_input"
    assert r["result"]["pending"][0]["id"] == "roles"
    roles = r["result"]["steps"][2]["details"]["roles"]

    r = client.put(f"/api/runs/{run_id}/answers", json={"answers": {"roles": roles}}).json()
    assert r["result"]["status"] == "complete"
    assert r["result"]["meridian"]["coord_to_columns"]["kpi"] == "new_policies"
    assert r["result"]["table_preview"]["rows"]

    assert client.get(f"/api/runs/{run_id}").json()["answers"] == {"roles": roles}
    csv = client.get(f"/api/runs/{run_id}/table.csv")
    assert csv.status_code == 200 and csv.text.startswith("week_start,")


def test_upload_and_date_decision(client):
    from mmm_tool import samples

    path = samples.write_ambiguous_monthly(app_module.RUNS_DIR.parent)
    with open(path, "rb") as f:
        r = client.post("/api/runs", files={"file": ("monthly.csv", f, "text/csv")}).json()
    assert r["result"]["pending"][0]["id"] == "date_order"
    r = client.put(f"/api/runs/{r['run_id']}/answers", json={"answers": {"date_order": "dayfirst"}}).json()
    assert r["result"]["pending"][0]["id"] == "roles"


def test_rejects_bad_input(client):
    assert client.post("/api/runs", files={"file": ("x.json", b"{}", "application/json")}).status_code == 400
    assert client.get("/api/runs/../../etc").status_code == 404
    assert client.get("/api/runs/0123456789ab").status_code == 404
    r = client.post("/api/runs/sample/ins_clean").json()
    assert client.get(f"/api/runs/{r['run_id']}/table.csv").status_code == 409


def test_recent_runs_list(client):
    assert client.get("/api/runs").json() == []
    a = client.post("/api/runs/sample/ins_clean").json()
    b = client.post("/api/runs/sample/ins_monthly").json()
    runs = client.get("/api/runs").json()
    assert [r["run_id"] for r in runs] == [b["run_id"], a["run_id"]]
    assert runs[0]["status"] == "needs_input" and runs[0]["decisions"] == 1


def test_phase_b_through_the_api(client):
    r = client.post("/api/runs/sample/ins_messy").json()
    run_id = r["run_id"]
    assert r["phase"] == "A" and r["result_b"] is None
    r = client.put(f"/api/runs/{run_id}/answers", json={"answers": {"roles": {}}}).json()
    assert r["phase"] == "B" and r["result_b"]["pending"][0]["id"] == "kpi_gaps"
    assert client.get(f"/api/runs/{run_id}/cleaned.csv").status_code == 409

    neg = {"roles": {}, "kpi_gaps": "interpolate"}
    r = client.put(f"/api/runs/{run_id}/answers", json={"answers": neg}).json()
    cells = r["result_b"]["pending"][0]["context"]["cells"]
    r = client.put(f"/api/runs/{run_id}/answers",
                   json={"answers": {**neg, "negative_values": {c["key"]: "zero" for c in cells}}}).json()
    assert r["result_b"]["status"] == "complete"
    assert r["result_b"]["table_preview"]["rows"]
    csv = client.get(f"/api/runs/{run_id}/cleaned.csv")
    assert csv.status_code == 200 and len(csv.text.strip().splitlines()) == 157
    runs = client.get("/api/runs").json()
    assert runs[0]["phase"] == "B" and runs[0]["status"] == "complete"


def test_decisions_say_whether_they_are_pending(client):
    r = client.post("/api/runs/sample/ins_clean").json()
    roles = r["result"]["steps"][2]["decisions"][0]
    assert roles["id"] == "roles" and roles["pending"] is True
