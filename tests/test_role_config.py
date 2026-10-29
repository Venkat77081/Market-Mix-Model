"""Role keywords from YAML, and remembered column roles."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import gen
from mmm_tool.api import app as app_module
from mmm_tool.ingest import run_phase_a
from mmm_tool.ingest.keywords import KeywordError, load_keywords
from mmm_tool.ingest.memory import RoleMemory
from mmm_tool.ingest.roles import channel_of, suggest_roles
from helpers import project_in, run_path  # noqa: F401

REPO = Path(__file__).parent.parent


@pytest.fixture()
def user_roles(tmp_path, monkeypatch):
    """Write a user roles.yaml and point MMM_ROLES_FILE at it."""
    path = tmp_path / "roles.yaml"
    monkeypatch.setenv("MMM_ROLES_FILE", str(path))

    def write(text: str) -> Path:
        path.write_text(text)
        st = path.stat()
        os.utime(path, (st.st_atime, st.st_mtime + 1))  # make sure the cache sees a change
        return path

    return write


def frame(**cols) -> pd.DataFrame:
    n = 30
    base = {"week_start": pd.date_range("2024-01-01", periods=n, freq="W-MON"), "new_policies": range(n)}
    return pd.DataFrame({**base, **{k: [v] * n if not isinstance(v, list) else v for k, v in cols.items()}})


def roles_of(df: pd.DataFrame) -> dict[str, str]:
    return {s.column: s.role for s in suggest_roles(df, "week_start")}


# Keywords ----------------------------------------------------------------

def test_builtin_keywords_load_and_project_roles_yaml_is_valid():
    kw = load_keywords(REPO / "roles.yaml")
    assert any(r.origin == "your" for r in kw.rules)
    assert kw.aliases["policybazaar"] == "aggregator"
    assert len(load_keywords(REPO / "missing.yaml").rules) > 10


def test_user_words_add_a_role_and_channel_words(user_roles):
    rng = np.random.default_rng(1)
    df = frame(tv_kharcha=list(rng.gamma(4, 1e5, 30)), search_kharcha=list(rng.gamma(4, 1e5, 30)))
    assert roles_of(df)["tv_kharcha"] == "needs_role"  # unknown word before: no silent control

    user_roles("rules:\n  - role: media_spend\n    words: [kharcha]\nchannel_words_to_strip: [kharcha]\n")
    s = {x.column: x for x in suggest_roles(df, "week_start")}
    assert s["tv_kharcha"].role == "media_spend" and s["tv_kharcha"].source == "your words"
    assert s["tv_kharcha"].channel == "tv"


def test_user_rules_are_checked_before_builtin(user_roles):
    df = frame(ooh_spend=1.0)
    assert roles_of(df)["ooh_spend"] == "media_spend"
    user_roles("rules:\n  - role: excluded\n    words: [ooh]\n    reason: we don't model outdoor\n")
    assert roles_of(df)["ooh_spend"] == "excluded"


def test_channel_alias_from_user_file(user_roles):
    assert channel_of("policybazaar_spend") == "policybazaar"
    user_roles("channel_aliases:\n  policybazaar: aggregator\n")
    assert channel_of("policybazaar_spend") == "aggregator"


def test_also_needs_and_unless():
    df = frame(branded_search_queries=1.0, brand_search_spend=1.0, brand_awareness_index=1.0)
    r = roles_of(df)
    assert r["branded_search_queries"] == "excluded"
    assert r["brand_search_spend"] == "media_spend"
    assert r["brand_awareness_index"] == "control"


def test_whole_word_matching():
    r = roles_of(frame(popular_flag=[0, 1] * 15, adult_pop=1.0))
    assert r["popular_flag"] == "control"  # "pop" must not match "popular"


def test_bad_user_file_is_reported(user_roles):
    user_roles("rules:\n  - role: spendy\n    words: [x]\n")
    with pytest.raises(KeywordError, match="unknown role 'spendy'"):
        load_keywords()
    user_roles("rules: [oops")
    with pytest.raises(KeywordError, match="not valid YAML"):
        load_keywords()


def test_kpi_type_words_come_from_yaml(user_roles, tmp_path):
    from mmm_tool.ingest.kpi import kpi_type_guess

    assert kpi_type_guess("new_policies") == "non_revenue"
    assert kpi_type_guess("gross_written_premium") == "revenue"
    assert kpi_type_guess("bima_count") == "non_revenue"
    assert kpi_type_guess("bima_kul") == "non_revenue"  # unknown defaults to count
    user_roles("kpi_money_words: [kul]\n")
    assert kpi_type_guess("bima_kul") == "revenue"


# Memory ------------------------------------------------------------------

@pytest.fixture()
def clean_csv(tmp_path):
    return gen.write_clean(tmp_path)


def test_confirmed_roles_are_suggested_next_time(clean_csv, tmp_path):
    mem = RoleMemory(tmp_path / "mem.json")
    r1 = run_phase_a(clean_csv, {"roles": {"vehicle_registrations": "excluded"}}, accept_defaults=True, memory=mem)
    assert r1.status == "complete"
    assert mem.recall("vehicle_registrations")["role"] == "excluded"

    # A later file names the same column differently and must still confirm roles.
    df = pd.read_csv(clean_csv).rename(columns={"vehicle_registrations": "Vehicle Registrations"})
    later = tmp_path / "later.csv"
    df.to_csv(later, index=False)
    r2 = run_phase_a(later, memory=RoleMemory(tmp_path / "mem.json"))
    assert r2.status == "needs_input" and r2.pending[0]["id"] == "roles"
    s = {x["column"]: x for x in r2.steps[2].details["suggestions"]}
    assert s["Vehicle Registrations"]["role"] == "excluded"
    assert s["Vehicle Registrations"]["source"] == "memory"
    assert s["Vehicle Registrations"]["confidence"] == "high"
    assert "clean" in s["Vehicle Registrations"]["reason"]
    assert r2.steps[2].details["remembered"] == 20  # every column but the date, which step 2 finds


def test_defaults_are_not_remembered(clean_csv, tmp_path):
    mem = RoleMemory(tmp_path / "mem.json")
    run_phase_a(clean_csv, accept_defaults=True, memory=mem)
    assert mem.entries == {} and not (tmp_path / "mem.json").exists()


def test_invalid_roles_are_not_remembered(clean_csv, tmp_path):
    mem = RoleMemory(tmp_path / "mem.json")
    run_phase_a(clean_csv, {"roles": {"new_policies": "control"}}, memory=mem)  # no KPI left
    assert mem.entries == {}


def test_forget(tmp_path):
    mem = RoleMemory(tmp_path / "mem.json")
    mem.remember({"tv_spend": "media_spend", "xyz": "control"}, "a.csv")
    assert mem.forget("TV Spend") == 1
    assert RoleMemory(tmp_path / "mem.json").recall("tv_spend") is None
    assert mem.forget() == 1 and RoleMemory(tmp_path / "mem.json").entries == {}


def test_damaged_memory_file_is_ignored(tmp_path):
    (tmp_path / "mem.json").write_text("{not json")
    assert RoleMemory(tmp_path / "mem.json").entries == {}


def test_api_remembers_and_forgets(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "RUNS_DIR", tmp_path / "runs")
    c = TestClient(app_module.app)
    r = c.post("/api/runs/sample/ins_clean", params={"project": project_in()}).json()
    roles = {**r["result"]["steps"][2]["details"]["roles"], "vehicle_registrations": "excluded"}
    c.put(f"/api/runs/{r['run_id']}/answers", json={"answers": {"roles": roles}})
    cols = {e["key"]: e["role"] for e in c.get("/api/role-memory").json()["columns"]}
    assert cols["vehicle registrations"] == "excluded"

    r2 = c.post("/api/runs/sample/ins_clean", params={"project": project_in()}).json()
    sugg = {s["column"]: s for s in r2["result"]["steps"][2]["details"]["suggestions"]}
    assert sugg["vehicle_registrations"]["role"] == "excluded"

    assert c.delete("/api/role-memory").json()["forgotten"] == 21
    assert c.get("/api/role-memory").json()["columns"] == []
