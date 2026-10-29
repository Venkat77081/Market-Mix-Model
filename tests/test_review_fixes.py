"""Fixes from the review of phases A-H: decimal commas, every listed currency, uploads named like
the tool's own files, unknown control directions, concurrent answers, generic answer keys and
spend columns that share a channel. No keyword packs or calendar here: nothing India-specific."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from test_generic import shop

from mmm_tool import currency
from mmm_tool.fit import truth
from mmm_tool.ingest.pairing import Channel, propose_pairs
from mmm_tool.ingest.roles import channel_of
from mmm_tool.ingest.values import decimal_mark, to_number
from mmm_tool.pipeline import run_pipeline
from helpers import project_in, run_path  # noqa: F401


@pytest.fixture(autouse=True)
def generic(monkeypatch, tmp_path):
    monkeypatch.setenv("MMM_KEYWORD_PACKS", "")
    monkeypatch.setenv("MMM_CALENDARS", str(tmp_path / "no_calendars.yaml"))


# Numbers --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("column, cell, expected", [
    (["12,5", "3,25"], "12,5", 12.5),
    (["1.234,56"], "1.234,56", 1234.56),
    (["€ 1.200", "€ 3.450,00"], "€ 1.200", 1200),
    (["1 200,50"], "1 200,50", 1200.5),
    (["1.234.567"], "1.234.567", 1234567),
    (["1,20,000", "4,500"], "1,20,000", 120000),            # Indian grouping stays thousands
    (["1,234", "0.75"], "1,234", 1234),
    (["0.75"], "0.75", 0.75),
    (["1'200"], "1'200", 1200),
    (["¥1200"], "¥1200", 1200),
    (["R$ 1.200,00"], "R$ 1.200,00", 1200),
    (["A$1,200"], "A$1,200", 1200),
    (["AED 500"], "AED 500", 500),
    (["(120)"], "(120)", -120),
])
def test_cells_read_with_the_columns_decimal_mark(column, cell, expected):
    mark = decimal_mark(column)
    got = to_number(cell, {}, mark)
    print(f"\n{cell!r:14} decimal '{mark}' -> {got}")
    assert got == pytest.approx(expected)


def _european(x: float) -> str:
    return f"€ {x:,.2f}".replace(",", " ").replace(".", ",").replace(" ", ".")      # € 1.234,56


def test_a_semicolon_file_with_decimal_commas_reads_its_real_values(tmp_path):
    df = shop(money_sign="€")
    money = [c for c in df.columns if "€" in c and "spend" in c]
    raw = df.copy()
    for c in money:
        raw[c] = raw[c].map(_european)
    p = tmp_path / "eu.csv"
    raw.to_csv(p, sep=";", index=False)
    r = run_pipeline(p, {}, accept_defaults=True)
    assert r.b is not None and r.b.status == "complete", r.last.steps[-1].summary
    for c in money:
        assert r.b.table[c].sum() == pytest.approx(df[c].sum(), rel=1e-9), c       # not off by 10-1000x
    assert r.a.kpi["currency"]["code"] == "EUR"
    s7 = next(s for s in r.b.steps if s.step == 7)
    assert {x["column"]: x["decimal_mark"] for x in s7.details["columns"]}[money[0]] == ","


@pytest.mark.parametrize("sign, code", [("¥", "JPY"), ("R$", "BRL"), ("A$", "AUD"), ("C$", "CAD"), ("S$", "SGD"),
                                        ("AED", "AED"), ("₹", "INR"), ("£", "GBP")])
def test_every_listed_currency_names_and_cells(tmp_path, sign, code):
    df = shop(money_sign=sign)
    money = [c for c in df.columns if sign in c and "spend" in c]
    raw = df.copy()
    for c in money:
        raw[c] = raw[c].map(lambda x: f"{sign}{x:,.2f}")
    p = tmp_path / "shop.csv"
    raw.to_csv(p, index=False)
    r = run_pipeline(p, {}, accept_defaults=True)
    assert r.b is not None and r.b.status == "complete", r.last.steps[-1].summary
    names = sorted(c.name for c in r.a.channels)
    search = next(c for c in r.a.channels if c.name == "search")
    print(f"\n{sign}: currency {r.a.kpi['currency']['code']}, channels {names}, search exposure {search.exposure}")
    assert r.a.kpi["currency"]["code"] == code
    assert names == ["search", "social", "tv"] and search.exposure == "search_clicks"     # spend pairs with clicks
    for c in money:
        assert r.b.table[c].sum() == pytest.approx(df[c].sum(), rel=1e-9)


def test_currency_codes_never_stay_in_a_channel_name():
    for col in ("TV Spend (₹)", "tv_spend_jpy", "Meta Spend (R$)", "TV Spend (A$)", "TV Spend (¥)", "Radio Spend US$", "tv_spend_aed"):
        assert channel_of(col) in ("tv", "meta", "radio"), col
    assert currency.detect(pd.DataFrame(), ["TV Spend (A$)", "Meta Spend (A$)"]) == "AUD"
    assert currency.detect(pd.DataFrame(), ["TV Spend (¥)"]) == "JPY"


# Pairing --------------------------------------------------------------------------------------------

def test_two_spend_columns_of_one_channel_never_share_an_exposure():
    pairs, _ = propose_pairs({"TV Spend": "media_spend", "TV Cost": "media_spend", "TV GRPs": "media"})
    exposures = [p["exposure"] for p in pairs.values() if p["exposure"]]
    assert exposures == ["TV GRPs"] and len(pairs) == 2


# Answer keys ------------------------------------------------------------------------------------------

def test_a_generic_answer_key_compares_kpi_per_spend():
    channels = [Channel("tv", "media", "tv_spend")]
    res = {"steps": [{"step": 32, "details": {"roi_in_money": False}}],
           "channels": [{"channel": "tv", "paid": True, "share_of_kpi": 0.2, "roi_posterior_mean": 0.004,
                         "roi_posterior_ci_lo": 0.003, "roi_posterior_ci_hi": 0.005}]}
    key = {"kpi_column": "orders", "kpi_per_spend_unit": 1000,
           "channels": [{"source_columns": ["tv_spend"], "true_kpi_per_spend": 4.2}]}
    t = truth.compare(res, channels, key, {"kpi": "orders", "currency": "USD"})
    r = t["rows"][0]
    print(f"\n{t['label']}: model {r['model']}, true {r['true']}")
    assert t["label"] == "orders per $1.0K" and t["per"] == 1000
    assert r["model"] == pytest.approx(4.0) and r["true"] == 4.2 and r["inside"]


# API --------------------------------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from mmm_tool.api import app as app_mod

    monkeypatch.setattr(app_mod, "RUNS_DIR", tmp_path / "runs")
    c = TestClient(app_mod.app)
    c.runs, c.app_mod = tmp_path / "runs", app_mod
    return c


def test_an_upload_named_like_a_tool_file_is_kept(client, tmp_path):
    src = shop().to_csv(index=False).encode()
    r = client.post("/api/runs", files={"file": ("cleaned.csv", src, "text/csv")}, data={"project": project_in()}).json()
    assert r["file"] == "upload_cleaned.csv"
    d = run_path(r["run_id"])
    client.put(f"/api/runs/{r['run_id']}/answers", json={"answers": {"roles": {}}})
    assert (d / "upload_cleaned.csv").read_bytes() == src and (d / "cleaned.csv").exists()   # both, neither replaced


@pytest.mark.parametrize("bad", [{"price_index": "up"}, {"nonesuch": "raises"}, "lowers"])
def test_an_unknown_direction_asks_again_instead_of_crashing(client, bad):
    run_id = client.post("/api/runs", files={"file": ("shop.csv", shop().to_csv(index=False).encode(), "text/csv")}, data={"project": project_in()}).json()["run_id"]
    r = client.put(f"/api/runs/{run_id}/answers", json={"answers": {"roles": {}, "control_direction": bad}})
    assert r.status_code == 200
    s21 = next(s for s in r.json()["result_d"]["steps"] if s["step"] == 21)
    assert s21["verdict"] == "needs_input" and ("either, raises or lowers" in s21["summary"])
    assert client.get(f"/api/runs/{run_id}").status_code == 200


def test_a_slow_request_never_writes_back_older_answers(client, monkeypatch):
    run_id = client.post("/api/runs", files={"file": ("shop.csv", shop().to_csv(index=False).encode(), "text/csv")}, data={"project": project_in()}).json()["run_id"]
    d = run_path(run_id)
    real, once = client.app_mod.run_pipeline, []

    def slow(path, answers, **kw):            # an answer arrives while this request is running
        if not once:
            once.append(1)
            client.app_mod._update_meta(d, lambda m: m.update(answers={"roles": {}, "newer": True}))
        return real(path, answers, **kw)
    monkeypatch.setattr(client.app_mod, "run_pipeline", slow)
    client.get(f"/api/runs/{run_id}")
    assert json.loads((d / "meta.json").read_text())["answers"] == {"roles": {}, "newer": True}
    assert not (d / "meta.json.tmp").exists()
