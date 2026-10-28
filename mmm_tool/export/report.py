"""Step 56: reports. A self-contained HTML report of the run and CSV tables, built from what the
run stored (no Meridian): the decisions, the validation gates, the Model Summary, saved What-If
scenarios, the last optimization and, for a refit, its comparison with the approved version.
"""
from __future__ import annotations

import csv
import html
import io
import json
from pathlib import Path
from typing import Any

from .. import currency as money_fmt


def _read(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def gather(run_dir: Path, decision_log: list[dict], kpi: dict, registry_entry: dict | None = None) -> dict[str, Any]:
    """Everything the report and the CSVs are made of, from the run folder."""
    run_dir = Path(run_dir)
    fd = run_dir / "fit"
    meta = _read(run_dir / "meta.json") or {}
    plan_last = _read(fd / "plan" / "last.json")
    plan = _read(fd / "plan" / f"result_{plan_last['key']}.json") if plan_last else None
    return {"meta": meta, "kpi": kpi, "result": _read(fd / "result.json"), "summary": _read(fd / "summary.json"),
            "saved": (_read(fd / "whatif" / "saved.json") or {}).get("scenarios", []),
            "plan": plan if plan and not plan.get("error") else None, "decisions": decision_log,
            "registry": registry_entry}


def _answer_text(e: dict) -> str:
    if e.get("lines"):
        return "; ".join(e["lines"])
    a = e.get("answer")
    if a == {} and isinstance(e.get("default"), dict) and e["default"]:
        return "accepted as suggested: " + json.dumps(e["default"], default=str)      # {} confirms every suggestion
    return json.dumps(a, default=str) if isinstance(a, (dict, list)) else str(a)


def decisions_rows(log: list[dict]) -> list[dict]:
    src = {"user": "you", "tool": "found in the data"}
    out = []
    for e in log:
        if "text" in e:                                   # Phase E logs "<what> <value>" as one line
            what, _, val = e["text"].partition(" ")
            row = {"decision": what, "answer": val}
        else:
            row = {"decision": e.get("decision"), "answer": _answer_text(e)}
        out.append({"step": e.get("step"), **row, "source": src.get(e.get("source"), "default")})
    return out


def csv_tables(g: dict[str, Any]) -> dict[str, str]:
    """{file name: CSV text} for what the run has."""
    out = {}

    def table(rows: list[dict]) -> str:
        if not rows:
            return ""
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        return buf.getvalue()

    out["decisions.csv"] = table(decisions_rows(g["decisions"]))
    s = g["summary"]
    if s:
        rows = []
        for period, t in (s.get("roi") or {}).items():
            for r in (t or {}).get("rows", []):
                rows.append({"period": period, "channel": r["channel"], "spend": r["spend"],
                             **{f"{m}_{k}": (r.get(m) or {}).get(k) for m in ("roi", "mroi", "kpi_driven", "revenue_driven", "cost_per_kpi")
                                for k in ("mean", "lo", "hi")}, "share_of_kpi": r.get("share_of_kpi")})
        out["roi.csv"] = table(rows)
        parts = []
        for period, b in (s.get("breakdown") or {}).items():
            if not b or "total" not in b:                    # "weekly" is the series, below
                continue
            parts.append({"period": period, "part": "baseline", **b["baseline"]})
            parts += [{"period": period, "part": c, **v} for c, v in b["channels"].items()]
        out["breakdown.csv"] = table(parts)
        wk = (s.get("breakdown") or {}).get("weekly")
        if wk and wk.get("weeks"):
            out["weekly_breakdown.csv"] = table([{"week": w, "baseline": wk["baseline"][i], **{c: v[i] for c, v in wk["channels"].items()}}
                                                 for i, w in enumerate(wk["weeks"])])
        f = s.get("fit") or {}
        if f.get("weeks"):
            out["weekly_fit.csv"] = table([{"week": w, "actual": a, "predicted": e, "predicted_lo": lo, "predicted_hi": hi}
                                           for w, a, e, lo, hi in zip(f["weeks"], f["actual"], f["expected"], f["expected_lo"], f["expected_hi"])])
        curves = []
        for c, x in (s.get("curves") or {}).items():
            curves += [{"channel": c, "spend_multiplier": m, "spend": sp, "kpi": k, "kpi_lo": lo, "kpi_hi": hi}
                       for m, sp, k, lo, hi in zip(x["multiplier"], x["spend"], x["kpi"], x["kpi_lo"], x["kpi_hi"])]
        out["response_curves.csv"] = table(curves)
    if g["plan"]:
        p = g["plan"]
        out["optimized_plan.csv"] = table([{"channel": c, "current_spend": p["before"]["channels"][c]["spend"],
                                            "optimized_spend": p["after"]["channels"][c]["spend"],
                                            "optimized_outcome": p["after"]["channels"][c]["outcome"]["mean"],
                                            "optimized_roi": p["after"]["channels"][c]["roi"]["mean"],
                                            "optimized_mroi": p["after"]["channels"][c]["mroi"]["mean"]} for c in p["channels"]])
    return {k: v for k, v in out.items() if v}


CSS = """
:root { --bg: #f6f7f9; --surface: #fff; --ink: #18202a; --muted: #5b6675; --line: #dde2e8; --pass: #1f7a4d; --warn: #9a6200; --fail: #b3261e; }
@media (prefers-color-scheme: dark) { :root { --bg: #0e141a; --surface: #161e27; --ink: #e8edf2; --muted: #9aa6b4; --line: #2a3542; --pass: #5cc48d; --warn: #e0a641; --fail: #ef7b72; } }
body { background: var(--bg); color: var(--ink); font: 14px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; margin: 0; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 26px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 28px 0 8px; } p { margin: 6px 0; }
.muted { color: var(--muted); } .card { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; } th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 500; } td.n { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.pass { color: var(--pass); } .warn { color: var(--warn); } .fail { color: var(--fail); } .skipped { color: var(--muted); }
@media print { body { background: #fff; color: #000; } .card { border: 0; padding: 0; } }
"""


def html_report(g: dict[str, Any]) -> str:
    e = html.escape
    meta, res, s, kpi = g["meta"], g["result"] or {}, g["summary"] or {}, g["kpi"] or {}
    cur = money_fmt.of(kpi)
    m = lambda x: "–" if x is None else money_fmt.money_short(float(x), cur)  # noqa: E731
    n = lambda x, d=0: "–" if x is None else f"{float(x):,.{d}f}"  # noqa: E731
    v = s.get("version") or {}
    kname = v.get("kpi") or kpi.get("kpi") or "KPI"
    parts = [f"<h1>Marketing mix model: {e(kname)}</h1>",
             f"<p class='muted'>{e(meta.get('file', ''))} · data {e(str(v.get('first_week', '–')))} to {e(str(v.get('last_week', '–')))}"
             f" · {e(str(v.get('weeks', '–')))} weeks · model {e(str(v.get('data_version', '–')))} · seed {e(str(res.get('seed', '–')))}"
             f" · Meridian {e(str(v.get('meridian', '–')))}</p>"]
    reg = g.get("registry")
    if reg:
        parts.append(f"<p>Registry: <b>{e(reg['line'])} v{reg['version']}</b> ({e(reg['status'])})</p>")
    status = res.get("status", "not fitted")
    parts.append(f"<p>Phase E: <b class='{'pass' if status == 'accepted' else 'fail'}'>{e(status)}</b>"
                 + (" · <b class='fail'>did not converge</b>" if res.get("not_converged") else "") + "</p>")
    # Validation
    if res.get("steps"):
        rows = "".join(f"<tr><td>{st['step']}</td><td>{e(st['name'])}</td><td class='{e(st['verdict'])}'>{e(st['verdict'])}</td>"
                       f"<td>{e(st['summary'])}</td></tr>" for st in res["steps"])
        parts.append(f"<h2>Validation (steps 28–36)</h2><div class='card'><table><tr><th>Step</th><th></th><th>Verdict</th><th>Result</th></tr>{rows}</table></div>")
    # ROI
    roi = ((s.get("roi") or {}).get("all") or {}).get("rows", [])
    if roi:
        rows = "".join(f"<tr><td>{e(r['channel'])}</td><td class='n'>{m(r['spend'])}</td><td class='n'>{n(r['roi']['mean'], 2)}</td>"
                       f"<td class='n'>{n(r['roi']['lo'], 2)}–{n(r['roi']['hi'], 2)}</td><td class='n'>{n((r.get('mroi') or {}).get('mean'), 2)}</td>"
                       f"<td class='n'>{n(r['kpi_driven']['mean'])}</td><td class='n'>{m((r.get('cost_per_kpi') or {}).get('mean'))}</td></tr>" for r in roi)
        parts.append(f"<h2>Return on spend (all weeks, {e(str(v.get('roi_unit', '')))})</h2><div class='card'><table><tr><th>Channel</th><th>Spend</th><th>ROI</th>"
                     f"<th>90% range</th><th>Next-unit ROI</th><th>{e(kname)} driven</th><th>Cost per KPI</th></tr>{rows}</table></div>")
    b = (s.get("breakdown") or {}).get("all")
    if b:
        rows = [("baseline", b["baseline"])] + sorted(b["channels"].items(), key=lambda kv: -kv[1]["mean"])
        parts.append("<h2>Sales breakdown (all weeks)</h2><div class='card'><table><tr><th>Part</th><th>" + e(kname)
                     + "</th><th>90% range</th><th>Share</th></tr>" + "".join(
                         f"<tr><td>{e(k)}</td><td class='n'>{n(x['mean'])}</td><td class='n'>{n(x['lo'])}–{n(x['hi'])}</td>"
                         f"<td class='n'>{x['mean'] / b['total']:.1%}</td></tr>" for k, x in rows) + "</table></div>")
    # What-If and plan
    if g["saved"]:
        rows = "".join(f"<tr><td>{e(x['name'])}</td><td>{e(x['period'])}</td><td class='n'>{m(x['quick']['spend'])}</td>"
                       f"<td class='n'>{n(x['quick']['kpi'])}</td><td>{e(', '.join(f'{c} {(k - 1) * 100:+.0f}%' for c, k in x['multipliers'].items() if abs(k - 1) > 1e-4))}</td></tr>"
                       for x in g["saved"])
        parts.append(f"<h2>Saved What-If scenarios</h2><div class='card'><table><tr><th>Name</th><th>Period</th><th>Spend</th><th>{e(kname)}</th><th>Changes</th></tr>{rows}</table></div>")
    if g["plan"]:
        p = g["plan"]
        rows = "".join(f"<tr><td>{e(c)}</td><td class='n'>{m(p['before']['channels'][c]['spend'])}</td><td class='n'>{m(p['after']['channels'][c]['spend'])}</td>"
                       f"<td class='n'>{n(p['after']['channels'][c]['roi']['mean'], 2)}</td></tr>" for c in p["channels"])
        d = p["totals"]["delta_kpi"]
        parts.append(f"<h2>Last optimized plan</h2><div class='card'><p>{e(p['plan']['budget_type'])} budget, {e(p['plan']['period'])}: "
                     f"{e(kname)} from paid media {d['mean']:+,.0f} (90%: {d['lo']:+,.0f} to {d['hi']:+,.0f}).</p>"
                     f"<table><tr><th>Channel</th><th>Current</th><th>Optimized</th><th>ROI</th></tr>{rows}</table></div>")
    if reg and reg.get("refit"):
        r = reg["refit"]
        rows = "".join(f"<tr><td>{e(x['channel'])}</td><td class='n'>{_rng(x.get('approved'))}</td><td class='n'>{_rng(x.get('refit'))}</td>"
                       f"<td class='{'fail' if 'beyond' in x['status'] else ''}'>{e(x['status'])}</td></tr>" for x in r["rows"])
        parts.append(f"<h2>Refit against v{r['approved']['version']}</h2><div class='card'><p>{e(r['summary'])}.</p><table><tr><th>Channel</th>"
                     f"<th>Approved ROI (90%)</th><th>Refit ROI (90%)</th><th></th></tr>{rows}</table></div>")
    # Decisions
    rows = "".join(f"<tr><td>{x['step']}</td><td>{e(str(x['decision']))}</td><td>{e(str(x['answer']))[:400]}</td><td>{e(x['source'])}</td></tr>"
                   for x in decisions_rows(g["decisions"]))
    parts.append(f"<h2>Every decision</h2><div class='card'><table><tr><th>Step</th><th>Decision</th><th>Answer</th><th>By</th></tr>{rows}</table></div>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>MMM report: {e(kname)}</title><style>{CSS}</style></head><body><main>{''.join(parts)}</main></body></html>")


def _rng(x: dict | None) -> str:
    return "–" if not x else f"{x['mean']:.2f} ({x['lo']:.2f}–{x['hi']:.2f})"
