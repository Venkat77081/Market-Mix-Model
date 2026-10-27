"""Check Phase A+B against tests/data/phase_b_tests: expected_phase_b.json and truth/.

Each file is answered the way the key says a user should answer; everything
else takes the tool's default. Prints check-by-check results per file.
"""
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmm_tool.pipeline import run_pipeline  # noqa: E402

# PHASE_B_DIR points the checks at another set, e.g. one made with
# gen_phase_b.py --seed N (problems land in different weeks).
B = Path(os.environ.get("PHASE_B_DIR", Path(__file__).parent / "data" / "phase_b_tests"))
K = json.loads((B / "expected_phase_b.json").read_text())


def truth(name):
    return pd.read_csv(B / "truth" / name.replace(".csv", "_truth.csv"), parse_dates=["week_start"])


def run(name, answers):
    return run_pipeline(B / name, answers, accept_defaults=True)


def bstep(r, n):
    return next((s for s in (r.b.steps if r.b else []) if s.step == n), None)


def all_decisions(r):
    out = {}
    for ph in (r.a, r.b):
        if ph:
            for s in ph.steps:
                for d in s.decisions:
                    out[d.id] = d
    return out


def table(r):
    """Phase B table keyed by week, with the premium total rebuilt if step 5 turned it into a per-policy value."""
    if not r.b or r.b.table is None:
        return None
    t = r.b.table.copy()
    date = next(c for c, v in r.b.roles.items() if v == "date")
    t = t.rename(columns={date: "week_start"}).set_index("week_start")
    rpk = r.b.kpi.get("revenue_per_kpi_column")
    kpi = r.b.kpi.get("kpi")
    if rpk == "revenue_per_kpi" and kpi in t.columns:
        t["new_premium_inr"] = t["revenue_per_kpi"] * t[kpi]
    return t


def compare(t, tr, skip_cells=(), cols=None, rtol=1e-6):
    """Columns in both, every week in truth. Returns (matching cols, mismatches dict, missing cols)."""
    tr = tr.set_index("week_start")
    cols = cols or [c for c in tr.columns if c in t.columns]
    missing = [c for c in tr.columns if c not in t.columns]
    bad = {}
    if not tr.index.isin(t.index).all() or len(t) != len(tr):
        bad["_rows"] = f"tool {len(t)} rows vs truth {len(tr)}; weeks missing: {[str(x.date()) for x in tr.index.difference(t.index)][:5]}"
        common = tr.index.intersection(t.index)
    else:
        common = tr.index
    for c in cols:
        a, b = t.loc[common, c].astype(float), tr.loc[common, c].astype(float)
        mask = ~a.index.isin(pd.to_datetime(list(skip_cells))) if skip_cells else np.ones(len(a), bool)
        close = np.isclose(a[mask], b[mask], rtol=rtol, equal_nan=True)
        if not close.all():
            wrong = a[mask][~close]
            bad[c] = f"{len(wrong)} week(s) differ, e.g. {wrong.index[0].date()}: tool {wrong.iloc[0]:,.4g} vs truth {b[mask][~close].iloc[0]:,.4g}"
    return cols, bad, missing


def report(name, checks):
    failed = [c for c in checks if not c[1]]
    print(f"\n== {name}: {'PASS' if not failed else 'FAIL'}")
    for desc, ok, actual in checks:
        print(f"   [{'ok' if ok else 'XX'}] {desc}" + ("" if ok else f"   actual: {actual}"))
    return not failed


ANSWERS = {
    "B03_split_rows_aggregation.csv": {"roles": {"active_agents": "control"}, "duplicate_dates": "merge",
                                       "aggregation_rules": {"active_agents": "mean"}},
    "B02_exact_duplicate_rows.csv": {"duplicate_dates": "merge"},
    "B07_missing_weeks.csv": {"kpi_gaps": "interpolate", "media_gaps": "interpolate"},
    "B09_kpi_gaps.csv": {"kpi_gaps": "interpolate"},
    "B13_end_to_end.csv": {"duplicate_dates": "merge", "kpi_gaps": "interpolate", "media_gaps": "interpolate"},
}

EXP = sorted(k for k in K if k.endswith(".csv"))


def check(name):
    e, tr = K[name], truth(name)
    answers = dict(ANSWERS.get(name, {}))
    checks = []
    add = lambda d, ok, a="": checks.append((d, bool(ok), a))  # noqa: E731

    if name in ("B11_impossible_values.csv", "B13_end_to_end.csv"):  # answer the negatives with the corrected values
        r0 = run(name, {})
        t0 = tr.set_index("week_start")
        for s in (r0.b.steps if r0.b else []):
            for d in s.decisions:
                if d.id in ("negative_values", "kpi_zero_weeks", "control_jumps"):
                    answers[d.id] = {c["key"]: float(t0.loc[c["date"], c["column"]]) for c in d.context["cells"]}
        if name.startswith("B13"):
            back = {v: k for k, v in e["column_names"].items()}
            r0 = run(name, answers)
            for s in (r0.b.steps if r0.b else []):
                for d in s.decisions:
                    if d.id == "negative_values":
                        answers[d.id] = {c["key"]: float(t0.loc[c["date"], back.get(c["column"], c["column"])]) for c in d.context["cells"]}
        if r0.b and r0.b.status != "complete":  # negatives block: the first run stops before 12's other cells
            r1 = run(name, answers)
            for s in (r1.b.steps if r1.b else []):
                for d in s.decisions:
                    if d.id in ("kpi_zero_weeks", "control_jumps"):
                        answers[d.id] = {c["key"]: float(t0.loc[c["date"], c["column"]]) for c in d.context["cells"]}
    r = run(name, answers)
    if r.a.status != "complete" and r.a.steps[-1].step == 3 and "Choose a role" in r.a.steps[-1].summary:
        stuck = [c for c, v in r.a.steps[-1].details["roles"].items() if v == "needs_role"]
        add("no unexpected role prompt at step 3", False, f"asks for a role: {stuck} " + "; ".join(
            x["reason"][:90] for x in r.a.steps[-1].details["suggestions"] if x["column"] in stuck))
        answers["roles"] = {**answers.get("roles", {}), **{c: "revenue_for_kpi" for c in stuck}}
        r = run(name, answers)
    t = table(r)
    dec = all_decisions(r)
    add("Phase B completes with the key's answers", r.b is not None and r.b.status == "complete",
        f"phase {r.phase if r.b is None else 'B'}: {(r.b or r.a).status} at step {(r.b or r.a).steps[-1].step}: {(r.b or r.a).steps[-1].summary[:110]}")

    if name == "B01_text_and_number_formats.csv" and t is not None:
        nulls = [(d, c) for c in tr.columns[1:] for d in tr.loc[tr[c].isna(), "week_start"]]
        cols, bad, missing = compare(t, tr, skip_cells=[d for d, _ in nulls])
        add(f"{len(cols)} columns equal truth on every cell the truth has a value for", not bad, bad)
        add("cells missing after step 7 are not filled with 0 later (always-on channel, index)",
            all(t.loc[d, c] != 0 for d, c in nulls if c in t.columns), {f"{c} {d.date()}": t.loc[d, c] for d, c in nulls if c in t.columns})
        add("social_spend: 5 text nulls read as missing in step 7",
            next(c for c in bstep(r, 7).details["columns"] if c["column"] == "social_spend")["text_nulls"] + next(c for c in bstep(r, 7).details["columns"] if c["column"] == "social_spend")["blank"] == 5,
            next(c for c in bstep(r, 7).details["columns"] if c["column"] == "social_spend"))
        idx = next((c for c in bstep(r, 7).details["columns"] if c["column"] == "competitor_spend_index"), None)
        raw = pd.read_csv(B / name, dtype=str, keep_default_na=False).set_index("week_start")
        dash_week = raw.index[raw["competitor_spend_index"].str.strip() == "-"]
        add("'-' in an index is missing, not 0 (step 7)", idx is not None and idx["text_nulls"] + idx["blank"] >= 1,
            f"competitor_spend_index '-' on {list(dash_week)} read as {r.a.table.set_index(r.a.table.columns[0]).loc[pd.Timestamp(dash_week[0]), 'competitor_spend_index'] if len(dash_week) else '?'}")
        if missing:
            add(f"columns not in the final table: {missing}", True)
    elif name in ("B02_exact_duplicate_rows.csv", "B03_split_rows_aggregation.csv", "B05_sunday_week_starts.csv",
                  "B06_week_start_changes.csv", "B10_kpi_outliers.csv", "B11_impossible_values.csv", "B12_unit_change.csv") and t is not None:
        cols, bad, missing = compare(t, tr)
        add(f"{len(cols)} columns equal truth over {len(tr)} weeks", not bad, bad)
    if name == "B02_exact_duplicate_rows.csv":
        add("asks before merging", "duplicate_dates" in dec)
    if name == "B03_split_rows_aggregation.csv":
        s8 = bstep(r, 8)
        rules = {x["column"]: x.get("applied") for x in (s8.details.get("rules", []) if s8 else [])}
        add("new_vehicle_registrations kept, not summed", rules.get("new_vehicle_registrations") != "sum", rules.get("new_vehicle_registrations"))
        add("asked how to combine active_agents", s8 and "active_agents" in (dec.get("aggregation_rules").context.get("unclassified", []) if "aggregation_rules" in dec else []),
            dec.get("aggregation_rules") and dec["aggregation_rules"].context)
    if name == "B04_daily_to_weekly.csv" and t is not None:
        s9 = bstep(r, 9)
        add("104 weeks (edge partial weeks dropped by default)", len(t) == 104, len(t))
        add("all 7 incomplete weeks flagged (2 edge + 5 inner)",
            len(s9.details.get("partial_weeks_dropped", [])) + len(s9.details.get("partial_weeks_kept", {})) + len(s9.details.get("short_weeks", [])) == 7,
            {k: s9.details.get(k) for k in ("partial_weeks_dropped", "partial_weeks_kept", "short_weeks")})
        add("inner incomplete weeks offered scale/interpolate/keep", any(i in ("short_weeks",) or "inner" in i or "incomplete" in i for i in dec), list(dec))
        tri = tr.set_index("week_start")
        raw = pd.read_csv(B / name, parse_dates=["date"])
        raw["wk"] = raw["date"] - pd.to_timedelta(raw["date"].dt.dayofweek, unit="D")
        days = raw.groupby("wk").size()
        full = days[days == 7].index.intersection(t.index)
        add("weeks complete in the file equal truth", np.allclose(t.loc[full, "policies"], tri.loc[full, "policies"]), f"{len(full)} weeks")
        inner = days[(days < 7)].index[1:-1]
        err = (t.loc[inner, "policies"] / tri.loc[inner, "policies"] - 1).abs()
        add("inner weeks with a missing day within 5% of truth (scaled)", (err < 0.05).all(), {str(k.date()): f"{v:.1%} low" for k, v in err.items()})
    if name == "B05_sunday_week_starts.csv" and t is not None:
        add("156 weeks, values unchanged", len(t) == 156)
    if name == "B06_week_start_changes.csv" and t is not None:
        s10 = bstep(r, 10)
        add("0 missing weeks reported", s10 and not s10.details.get("missing_weeks"), s10 and s10.details.get("missing_weeks"))
        add("156 continuous weeks", len(t) == 156 and (pd.Series(t.index).diff().dropna() == pd.Timedelta(days=7)).all(), len(t))
    if name in ("B07_missing_weeks.csv", "B09_kpi_gaps.csv") and t is not None:
        exp = e["expected_interpolated_kpi"]
        got = {d: float(t.loc[pd.Timestamp(d), "new_policies"]) if pd.Timestamp(d) in t.index else None for d in exp}
        add("interpolated KPI = key's linear targets", all(v is not None and abs(v - exp[d]) < 0.11 for d, v in got.items()),
            {d: (got[d], exp[d]) for d in exp})
        add("kpi_imputed flag column", "kpi_imputed" in t.columns)
        if name == "B07_missing_weeks.csv":
            add("156 rows", len(t) == 156, len(t))
            add("spend in missing weeks asked, not set to 0", "media_gaps" in dec and dec["media_gaps"].default != "zero",
                dec.get("media_gaps") and f"asked: {dec['media_gaps'].question[:60]}…, default '{dec['media_gaps'].default}'")
            mw = pd.to_datetime(e["missing_weeks"])
            add("spend in missing weeks is not 0", (t.loc[mw, "social_spend"] > 0).all(), t.loc[mw, "social_spend"].tolist())
        else:
            add("4 leading weeks trimmed: 152 rows", len(t) == 152, len(t))
    if name == "B08_zero_vs_missing.csv" and t is not None:
        raw = pd.read_csv(B / name, parse_dates=["week_start"]).set_index("week_start")
        tri = tr.set_index("week_start")
        add("tv_spend zeros kept", (t["tv_spend"] == raw["tv_spend"].fillna(-1)).all() or np.allclose(t["tv_spend"], tri["tv_spend"]))
        add("radio blanks: asked whether 0", "media_gaps" in dec and "radio" in dec["media_gaps"].question, dec.get("media_gaps") and dec["media_gaps"].question[:80])
        exp = e["expected_interpolated_social"]
        got = {d: float(t.loc[pd.Timestamp(d), "social_spend"]) for d in exp}
        add("social blanks interpolated to the key's targets", all(abs(got[d] - exp[d]) < 1 for d in exp), {d: (got[d], exp[d]) for d in exp})
        add("ctv 0 before launch", np.allclose(t["ctv_spend"], tri["ctv_spend"]) if "ctv_spend" in t else False)
        add("print 0 after stop", np.allclose(t["print_spend"], tri["print_spend"]))
    if name == "B10_kpi_outliers.csv":
        s11 = bstep(r, 11)
        add("exactly the 2 keyed weeks flagged", s11 and sorted(o["date"] for o in s11.details.get("outliers", [])) == sorted(e["flagged"]),
            s11 and [o["date"] for o in s11.details.get("outliers", [])])
        add("event flag column added", t is not None and any(c.startswith("event_") or c == "event_flag" for c in t.columns))
    if name == "B11_impossible_values.csv":
        s12 = bstep(r, 12)
        found = []
        if s12:
            found = [(x["date"], x["column"]) for k in ("negative", "kpi_zero", "jumps") for x in s12.details.get(k, [])]
        keyed = [(d, v.split()[0]) for grp in e["findings"].values() for d, v in grp.items()]
        add("all 7 cells listed", sorted(set(found) & set(keyed)) == sorted(keyed), f"found {sorted(found)}; missing {sorted(set(keyed) - set(found))}")
        add("negatives block until answered", "negative_values" in dec)
    if name == "B12_unit_change.csv":
        s = [s for s in (r.b.steps if r.b else []) if s.step in (12, 13)]
        txt = " ".join(x.summary + " ".join(x.warnings) + " ".join(d.question for d in x.decisions) for x in s)
        add("level shift (~100,000x) detected and conversion offered", "100,000" in txt or "lakh" in txt.lower(), txt[:200])
    if name == "B13_end_to_end.csv" and t is not None:
        t = t.rename(columns={v: k for k, v in e["column_names"].items()})
        t = t.loc[:, ~t.columns.duplicated()]  # the kept premium total and the one rebuilt from revenue per KPI
        if "new_premium_inr" not in t.columns and "revenue_per_kpi" in t.columns:
            t["new_premium_inr"] = t["revenue_per_kpi"] * t["new_policies"]
        exp_skip = list(e["missing_weeks"]) + list(e["social_nulls"])
        cols, bad, missing = compare(t, tr.drop(columns=["event_flag"]), skip_cells=exp_skip)
        add(f"{len(cols)} columns equal truth (missing weeks and social nulls skipped)", not bad, bad)
        add("event flags on the 2 outlier weeks", bstep(r, 11) and sorted(o["date"] for o in bstep(r, 11).details.get("outliers", [])) == sorted(e["kpi_outliers"]),
            bstep(r, 11) and [o["date"] for o in bstep(r, 11).details.get("outliers", [])])
        add("156 rows", len(t) == 156, len(t))
    return r, checks


def main() -> int:
    names = [n for n in EXP if not sys.argv[1:] or n in sys.argv[1:]]
    results = {n: report(n, check(n)[1]) for n in names}
    print(f"\n{sum(results.values())} of {len(results)} files pass")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
