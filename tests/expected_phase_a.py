"""Checks of Phase A against tests/data/phase_a_tests/expected_phase_a.json.

Each file runs with every suggestion accepted (as a user clicking through);
check(name) turns that file's expected entries into (description, ok, actual)
checks. Run as a script to print a report for every file:

    python tests/expected_phase_a.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as a script from anywhere
from mmm_tool.ingest import Answers, run_phase_a  # noqa: E402
from mmm_tool.pipeline import run_pipeline  # noqa: E402

D = Path(__file__).parent / "data" / "phase_a_tests"
EXP = json.loads((D / "expected_phase_a.json").read_text())


def run(name, answers=None):
    return run_phase_a(D / name, Answers(answers or {}, accept_defaults=True))


def step(r, n):
    return next((s for s in r.steps if s.step == n), None)


def decisions(r):
    return {d.id: d for s in r.steps for d in s.decisions}


def suggestions(r):
    s3 = step(r, 3)
    return {x["column"]: x for x in (s3.details.get("suggestions", []) if s3 else [])}


def blocked(r):
    return r.status != "complete"


ROLE_WORDS = [("exclude", "excluded"), ("revenue (for", "revenue_for_kpi"), ("media_spend", "media_spend"),
              ("media (exposure)", "media"), ("control", "control"), ("kpi", "kpi"), ("date", "date")]


def role_of(text):
    return next(r for w, r in ROLE_WORDS if text.startswith(w))


def check(name):
    e = EXP[name]
    c = []  # (description, ok, actual)
    r = run(name)
    s1, s2, s3, s5, s6 = (step(r, n) for n in (1, 2, 3, 5, 6))
    dec = decisions(r)
    sug = suggestions(r)
    roles = {k: v["role"] for k, v in sug.items()}

    def add(desc, ok, actual=""):
        c.append((desc, bool(ok), actual))

    add(f"blocks={e['blocks']}", blocked(r) == e["blocks"], f"status={r.status} at step {r.steps[-1].step}")
    if "delimiter" in e:
        add("delimiter", s1 and s1.details.get("delimiter") == e["delimiter"], s1 and s1.details.get("delimiter"))
    if "encoding" in e:
        want = {"utf-8": "utf-8", "utf-8 with BOM": "utf-8-sig"}[e["encoding"]]
        add("encoding", s1 and s1.details.get("encoding") == want, s1 and s1.details.get("encoding"))
    if "header_row" in e:
        add("header_row", s1 and s1.details.get("header_row") == e["header_row"], s1 and s1.details.get("header_row"))
    if "preamble_rows" in e:
        add("preamble_rows", s1 and s1.details.get("preamble_rows") == e["preamble_rows"], s1 and s1.details.get("preamble_rows"))
    if "expected_rows" in e or ("rows" in e and isinstance(e["rows"], int)):
        n = e.get("expected_rows", e.get("rows"))
        got = s2.details.get("rows") if s2 and "rows" in s2.details else (s1.details.get("rows") if s1 else None)
        add(f"rows={n}", got == n, got)
    if "check" in e:
        add("first column is week_start", s1 and s1.details.get("columns", [""])[0] == "week_start", s1 and repr(s1.details.get("columns", [""])[0]))
    if "data_sheet" in e:
        add("data sheet", s1 and s1.details.get("sheet") == e["data_sheet"], s1 and s1.details.get("sheet"))
        add("sheet prompt", "sheet" in dec, list(dec))
    if "expected_unique_dates" in e:
        add("unique dates", s2 and s2.details.get("unique_dates") == e["expected_unique_dates"], s2 and s2.details.get("unique_dates"))
        add("first/last", s2 and (s2.details.get("first"), s2.details.get("last")) == (e["expected_first"], e["expected_last"]),
            s2 and (s2.details.get("first"), s2.details.get("last")))
    if "granularity" in e:
        add("granularity", s6 and s6.details.get("frequency") == e["granularity"], s6 and s6.details.get("frequency"))
    if "geo" in e and e["geo"] is None:
        add("no region", s6 and not s6.details.get("regions"), s6 and s6.details.get("regions"))
    if "kpi" in e:
        k = r.kpi.get("kpi") if r.status == "complete" else next((c for c, v in roles.items() if v == "kpi"), None)
        add("kpi", k == e["kpi"], k)
    if "kpi_type" in e:
        add("kpi_type", r.kpi.get("kpi_type") == e["kpi_type"], r.kpi.get("kpi_type"))
    if name == "A01_clean_weekly.csv":
        add("new_premium_inr = Revenue for KPI (high)", sug.get("new_premium_inr", {}).get("role") == "revenue_for_kpi"
            and sug["new_premium_inr"]["confidence"] == "high", sug.get("new_premium_inr", {}).get("role"))
        add("reason gives ₹ range", "9,310" in sug.get("new_premium_inr", {}).get("reason", "") and "11,096" in sug.get("new_premium_inr", {}).get("reason", ""),
            sug.get("new_premium_inr", {}).get("reason", "")[:60])
        add("step 5 offers count vs revenue", "kpi_choice" in dec and dec["kpi_choice"].options == ["count", "revenue"], list(dec))
        add("revenue per KPI is per unit", r.status == "complete" and r.kpi.get("revenue_per_kpi_column") == "revenue_per_kpi"
            and 9000 < r.table["revenue_per_kpi"].median() < 11500, r.kpi.get("revenue_per_kpi_column"))
    if "media_spend" in e:
        got = sorted(c for c, v in roles.items() if v == "media_spend")
        add("media_spend columns", got == sorted(e["media_spend"]), got)
    if "exposure_pairs" in e:
        got = {ch.spend: ch.exposure for ch in r.channels if not ch.exposure_is_spend}
        add("exposure pairs", got == e["exposure_pairs"], got)
    if "controls" in e:
        got = sorted(c for c, v in roles.items() if v == "control")
        add("controls", got == sorted(e["controls"]), got)
    if "suggest_exclude" in e:
        add("suggest exclude", all(roles.get(c) == "excluded" for c in e["suggest_exclude"]), [roles.get(c) for c in e["suggest_exclude"]])
    if name == "A06_ambiguous_dates_monthly.csv":
        d = dec.get("date_order")
        add("asks to confirm, suggests day-first", d is not None and d.default == "dayfirst", d and d.default)
    if "expected_roles" in e:
        wrong = {col: (roles.get(col), role_of(want)) for col, want in e["expected_roles"].items() if roles.get(col) != role_of(want)}
        add("roles (A07)", not wrong, wrong)
        add("asks about Fees and Payout", any("Fees" in str(d.question) or "Fees" in str(d.context) for d in dec.values() if d.id != "roles")
            or all(sug.get(x, {}).get("confidence") in ("low", "medium") and roles.get(x) == "media_spend" for x in ("PolicyBazaar Fees", "Affiliate Payout")),
            [(x, sug.get(x, {}).get("role"), sug.get(x, {}).get("confidence")) for x in ("PolicyBazaar Fees", "Affiliate Payout")])
    if "pairs" in e:
        ch = {c.spend: c for c in r.channels}
        add("tv_spend↔tv_grps", ch.get("tv_spend") and ch["tv_spend"].exposure == "tv_grps", ch.get("tv_spend") and ch["tv_spend"].exposure)
        add("Search Cost↔Search Clicks", ch.get("Search Cost") and ch["Search Cost"].exposure == "Search Clicks", ch.get("Search Cost") and ch["Search Cost"].exposure)
        add("asks FB spend↔Meta impressions", any("FB spend" in str(d.question) and "Meta" in str(d.question) for d in dec.values()),
            ch.get("FB spend") and ch["FB spend"].exposure)
        add("radio_spend spend-only", ch.get("radio_spend") and ch["radio_spend"].exposure_is_spend, ch.get("radio_spend") and ch["radio_spend"].exposure)
        rf = next((x for x in r.channels if x.kind == "rf"), None)
        add("yt reach & frequency", rf and (rf.spend, rf.reach, rf.frequency) == ("yt_spend", "yt_reach", "yt_frequency"), rf and (rf.spend, rf.reach, rf.frequency))
        got = sorted(c for c, v in r.roles.items() if v == "organic_media")
        add("organic media", got == sorted(e["organic_media"]), got)
    if "revenue_per_kpi" in e and e["revenue_per_kpi"] is None:
        add("no revenue-per-unit prompt", "revenue_per_kpi_value" not in dec, list(dec))
    if name == "A10_leads_no_revenue.csv":
        add("asks value per quote", "revenue_per_kpi_value" in dec, list(dec))
    if name == "A11_daily.csv":
        p = run_pipeline(D / name, {}, accept_defaults=True)
        n = len(p.b.table) if p.b and p.b.table is not None else None
        add("Phase B → 105 weeks", n == 105, n)
    if name == "A12_monthly.csv":
        add("36 dates 2023-01-01..2025-12-01", s2 and (s2.details.get("unique_dates"), s2.details.get("first"), s2.details.get("last")) == (36, "2023-01-01", "2025-12-01"),
            s2 and (s2.details.get("unique_dates"), s2.details.get("first"), s2.details.get("last")))
        add("step 6: monthly, 36 periods", s6 and s6.summary.lower().startswith("monthly, 36 periods"), s6 and s6.summary[:40])
        add("warns under 104 weeks", s6 and any("under 104 weeks" in w for w in s6.warnings), s6 and s6.warnings)
    if name == "A13_regions_long.csv":
        add("region column state", roles.get("state") == "geo", roles.get("state"))
        add("region values", s6 and sorted(s6.details.get("regions", [])) == sorted(e["geo_values"]), s6 and s6.details.get("regions"))
        add("population", roles.get("adult_population") == "population", roles.get("adult_population"))
        add("asks regional vs national", "geo_mode" in dec, list(dec))
        add("no reshape prompt", "wide_regions" not in dec, list(dec))
    if name == "A14_regions_wide.csv":
        add("reshape prompt (default reshape)", "wide_regions" in dec and dec["wide_regions"].default == "reshape", list(dec))
        add("468 rows", r.table is not None and len(r.table) == 468, None if r.table is None else len(r.table))
        add("channels tv_spend, search_spend", sorted(c.spend for c in r.channels) == ["search_spend", "tv_spend"], sorted(c.spend for c in r.channels))
        add("step 6: 3 regions", s6 and len(s6.details.get("regions", [])) == 3, s6 and s6.details.get("regions"))
        add("choice in decision log", any(x["decision"] == "wide_regions" for x in r.decision_log), [x["decision"] for x in r.decision_log])
    if "bad_rows" in e:
        rows = [u["line"] for u in s2.details.get("unreadable", [])] if s2 else []
        add("lists all 9 bad lines", rows == e["bad_rows"], rows)
        add("offers Excel serial / ISO week fixes", any("serial" in str(d.question).lower() or "iso week" in str(d.question).lower() for d in dec.values()),
            [d.question[:60] for d in dec.values()])
    if name == "A16_no_media.csv":
        add("stops: no media", r.steps[-1].step == 3 and "media" in r.steps[-1].summary, r.steps[-1].summary[:70])
    if name == "A17_empty.csv":
        add("stops: empty file", "empty" in r.steps[-1].summary.lower(), r.steps[-1].summary)
    if name == "A17_header_only.csv":
        add("stops: 0 data rows", "0 data rows" in r.steps[-1].summary or "no data rows" in r.steps[-1].summary.lower(), r.steps[-1].summary)
    if name == "A18_indian_number_formats.csv":
        p = run_pipeline(D / name, {}, accept_defaults=True)
        b12 = p.b and next((s for s in p.b.steps if s.step == 12), None)
        neg = b12.details.get("negative", []) if b12 else []
        add("numeric before roles", all(v not in ("excluded", "geo") for k, v in roles.items() if k != "week_start"),
            {k: v for k, v in roles.items() if v in ("excluded", "geo")})
        add("2 negatives flagged at step 12", len(neg) == 2, len(neg) if b12 else f"Phase B stopped: {p.status}")
    return e, r, c


def main() -> int:
    passed = 0
    for name in sorted(EXP):
        _, r, c = check(name)
        failed = [x for x in c if not x[1]]
        passed += not failed
        print(f"\n== {name}: {'PASS' if not failed else 'FAIL'}  (status {r.status}, stopped at step {r.steps[-1].step})")
        for desc, ok, actual in c:
            print(f"   [{'ok' if ok else 'XX'}] {desc}" + ("" if ok else f"   actual: {actual}"))
    print(f"\n{passed} of {len(EXP)} files pass")
    return 0 if passed == len(EXP) else 1


if __name__ == "__main__":
    sys.exit(main())
