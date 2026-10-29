"""How a run's file is read: keyword packs and calendar preset chosen per run (upload panel,
the run's chip, --packs / --calendar), remembered for new runs, kept by bundles and refits."""
from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile

import pytest
from test_generic import shop

from mmm_tool import settings as rs
from mmm_tool.pipeline import run_pipeline

INDIA = {"packs": ["india", "insurance"], "calendar": "india_insurance"}
GENERIC = {"packs": [], "calendar": "none"}


def write(tmp_path, df, name="shop.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def preset(r) -> str:
    return next(s for s in r.d.steps if s.step == 21).details["preset_name"]


def test_clean_settings():
    assert rs.clean({"packs": "india, insurance", "calendar": "us_retail"}) == {"packs": ["india", "insurance"], "calendar": "us_retail"}
    assert rs.clean({}) == GENERIC
    with pytest.raises(rs.SettingsError, match="Unknown keyword pack"):
        rs.clean({"packs": ["nonesuch"]})
    with pytest.raises(rs.SettingsError, match="Unknown calendar preset"):
        rs.clean({"calendar": "nonesuch"})
    assert rs.describe(INDIA) == "India · Insurance · India insurance calendar" and rs.describe(GENERIC) == "Generic"


def test_a_runs_settings_win_over_the_machine_wide_ones(tmp_path):
    p = write(tmp_path, shop(money_sign=""))                 # no currency in names or cells
    generic = run_pipeline(p, {}, accept_defaults=True, settings=GENERIC)   # the tests' environment turns India on
    india = run_pipeline(p, {}, accept_defaults=True, settings=INDIA)
    us = run_pipeline(p, {}, accept_defaults=True, settings={"packs": [], "calendar": "us_retail"})
    print(f"\ngeneric {generic.a.kpi['currency']['code']} / {preset(generic)}; india {india.a.kpi['currency']['code']} / "
          f"{preset(india)}; us {preset(us)}")
    assert generic.a.kpi["currency"]["code"] is None and preset(generic) == "none"
    assert india.a.kpi["currency"]["code"] == "INR" and preset(india) == "india_insurance"
    assert preset(us) == "us_retail"
    assert rs.active() is None                               # nothing leaks out of a run


def test_for_run_order(tmp_path):
    runs = tmp_path / "runs"
    (runs / "r1").mkdir(parents=True)
    assert rs.for_run(runs / "r1", {"settings": INDIA}) == (INDIA, "run")
    assert rs.for_run(runs / "r1", {})[1] == "machine"
    rs.remember(runs, {"packs": ["india"], "calendar": "none"})
    assert rs.for_run(runs / "r1", {}) == ({"packs": ["india"], "calendar": "none"}, "default")   # an older run
    assert rs.for_new_run(runs) == {"packs": ["india"], "calendar": "none"}


# API --------------------------------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from mmm_tool.api import app as app_mod

    monkeypatch.setattr(app_mod, "RUNS_DIR", tmp_path / "runs")
    c = TestClient(app_mod.app)
    c.runs = tmp_path / "runs"
    return c


def _upload(client, **form):
    data = shop(money_sign="").to_csv(index=False).encode()
    return client.post("/api/runs", files={"file": ("shop.csv", data, "text/csv")}, data=form)


def test_upload_with_settings_and_remember(client):
    cat = client.get("/api/settings").json()
    assert {p["name"] for p in cat["packs"]} >= {"india", "insurance"} and all(p["describe"] for p in cat["packs"])
    assert [c["name"] for c in cat["calendars"]][0] == "none" and cat["sample"]["label"].startswith("India")
    r = _upload(client, packs="", calendar="us_retail").json()
    assert r["settings"]["packs"] == [] and r["settings"]["calendar"] == "us_retail" and r["settings"]["source"] == "run"
    assert r["settings"]["label"] == "US retail calendar"
    r2 = _upload(client, packs="india,insurance", calendar="india_insurance", remember="true").json()
    assert r2["settings"]["label"] == "India · Insurance · India insurance calendar"
    assert client.get("/api/settings").json()["default"]["remembered"]
    r3 = _upload(client).json()                                 # no choice: the remembered default
    assert r3["settings"]["packs"] == ["india", "insurance"]
    assert _upload(client, packs="nonesuch").status_code == 400


def test_the_chip_rereads_a_run_and_drops_the_old_calendar_pick(client):
    r = _upload(client, packs="india,insurance", calendar="india_insurance").json()
    run_id = r["run_id"]
    client.put(f"/api/runs/{run_id}/answers", json={"answers": {"roles": {}, "calendar_preset": "india_insurance", "x": 1}})
    d = client.put(f"/api/runs/{run_id}/settings", json={"packs": [], "calendar": "us_retail"}).json()
    assert d["settings"]["label"] == "US retail calendar" and d["answers"] == {"roles": {}, "x": 1}
    s5 = next(s for s in d["result"]["steps"] if s["step"] == 5)
    assert (s5["details"].get("currency") or {}).get("code") is None             # no longer INR
    assert client.put(f"/api/runs/{run_id}/settings", json={"packs": ["x"]}).status_code == 400


def test_samples_use_their_own_settings_and_old_runs_the_default(client):
    from mmm_tool import samples

    r = client.post("/api/runs/sample/ins_clean").json()
    assert r["settings"]["packs"] == samples.SETTINGS["packs"] and r["settings"]["calendar"] == samples.SETTINGS["calendar"]
    meta_p = client.runs / r["run_id"] / "meta.json"
    meta = json.loads(meta_p.read_text())
    del meta["settings"]                                        # a run made before runs kept settings
    meta_p.write_text(json.dumps(meta))
    rs.remember(client.runs, GENERIC)
    again = client.get(f"/api/runs/{r['run_id']}").json()
    assert again["settings"]["source"] == "default" and again["settings"]["label"] == "Generic"


def test_bundle_and_refit_keep_the_runs_settings(client, tmp_path, monkeypatch):
    from test_phase_i import fake_fit

    from mmm_tool.export.bundle import verify

    run_id = _upload(client, packs="india", calendar="us_retail").json()["run_id"]
    client.put(f"/api/runs/{run_id}/answers", json={"answers": {"roles": {}}})
    fake_fit(client.runs / run_id)
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/runs/{run_id}/bundle.zip").content))
    assert json.loads(z.read("config/settings.json")) == {"packs": ["india"], "calendar": "us_retail"}
    assert "--packs" in z.read("REPRODUCE.md").decode() and "--calendar us_retail" in z.read("REPRODUCE.md").decode()
    monkeypatch.setenv("MMM_KEYWORD_PACKS", "insurance")          # this machine reads differently
    res = verify(client.get(f"/api/runs/{run_id}/bundle.zip").content, tmp_path / "v")
    assert res["ok"], res
    monkeypatch.delenv("MMM_KEYWORD_PACKS")
    from mmm_tool.export import refit as rf

    new = tmp_path / "newer.csv"
    new.write_bytes((client.runs / run_id / "shop.csv").read_bytes())
    meta = rf.start(client.runs / run_id, client.runs / "refit1", new, "L", 1)
    assert meta["settings"] == {"packs": ["india"], "calendar": "us_retail"}


def test_command_line_packs_and_calendar(tmp_path):
    p = write(tmp_path, shop(money_sign=""))
    out = subprocess.run([sys.executable, "-m", "mmm_tool.ingest", str(p), "--accept-defaults", "--json", "--phase", "a",
                          "--packs", "india", "--calendar", "none"], capture_output=True, text=True, cwd=tmp_path,
                         env={**__import__("os").environ, "PYTHONPATH": str(__import__("pathlib").Path(__file__).parents[1])})
    assert out.returncode == 0, out.stderr
    kpi_step = next(s for s in json.loads(out.stdout)["steps"] if s["step"] == 5)
    assert kpi_step["details"]["currency"]["code"] == "INR"
    bad = subprocess.run([sys.executable, "-m", "mmm_tool.ingest", str(p), "--packs", "nonesuch"], capture_output=True, text=True,
                         cwd=tmp_path, env={**__import__("os").environ, "PYTHONPATH": str(__import__("pathlib").Path(__file__).parents[1])})
    assert bad.returncode == 1 and "Unknown keyword pack" in bad.stderr
