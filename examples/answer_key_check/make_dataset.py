"""A synthetic MMM export with a known answer, for checking the tool end to end.

    python examples/answer_key_check/make_dataset.py [--seed 11] [--out DIR]

writes, into DIR (default: this folder):

  home_goods_mmm.csv     the export to upload: 3 regions x 156 weeks, KPI 'Orders', a revenue
                         column, 4 paid channels, 3 controls, a population column, and a few
                         planted data problems for Phases A and B to find
  truth_for_tool.json    the answer key to load at step 32 (the tool's own format)
  truth_details.json     everything else that is known: each channel's carryover and diminishing
                         returns, the controls' effects, the planted problems, weekly contributions

How the KPI is made (additive, in orders per region-week, like Meridian's model):

  orders = baseline (trend x seasonality) + holiday bump + price effect + competitor effect
           + sum over channels of  beta_c x scale_region x hill(adstock(units_c) / median, ec_c, slope_c)
           + noise

adstock is geometric (decay alpha_c, 8 weeks, weights normalised to 1, as Meridian's), hill is
x^s / (x^s + ec^s) on units relative to the region's median adstocked units. Because the media
terms add up, each channel's true contribution is its own term exactly; its true ROI is that
contribution valued at each week's order value, over its spend. The true marginal ROI is the
extra revenue from 1% more spend in every week, over that 1%.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WEEKS = 156
START = "2023-01-02"                       # a Monday
REGIONS = {"North": 4_200_000, "South": 3_100_000, "West": 2_400_000}
LAG = 8
WHATIF = (0.0, 0.5, 0.8, 1.2, 1.5)         # spend factors with a known exact What-If answer
# channel: (spend column, units column or None, cost per unit, decay, beta, ec, slope)
CHANNELS = {
    "tv":      ("TV Spend ($)", "TV GRPs", 900.0, 0.65, 900.0, 1.0, 1.5),
    "search":  ("Search Spend ($)", "Search Clicks", 1.10, 0.15, 700.0, 0.8, 1.0),
    "social":  ("Social Spend ($)", "Social Impressions", 0.006, 0.35, 400.0, 1.2, 1.2),
    "display": ("Display Spend ($)", None, 1.0, 0.20, 80.0, 1.0, 1.0),
}


def adstock(x: np.ndarray, alpha: float, lag: int = LAG) -> np.ndarray:
    w = alpha ** np.arange(lag + 1)
    w = w / w.sum()
    out = np.zeros_like(x, dtype=float)
    for k, wk in enumerate(w):
        out[k:] += wk * x[: len(x) - k]
    return out


def hill(x: np.ndarray, ec: float, slope: float) -> np.ndarray:
    return x ** slope / (x ** slope + ec ** slope)


def flights(rng, n: int, on_share: float) -> np.ndarray:
    """On/off in runs of 3 to 6 weeks, about on_share of weeks on."""
    on, t, out = rng.random() < on_share, 0, np.zeros(n, bool)
    while t < n:
        k = int(rng.integers(3, 7))
        out[t:t + k] = on
        t += k
        on = rng.random() < on_share
    return out


def holiday_weeks(dates: pd.DatetimeIndex) -> np.ndarray:
    """Black Friday week and the two weeks before Christmas."""
    flag = np.zeros(len(dates), int)
    for y in sorted(set(dates.year)):
        nov = pd.date_range(f"{y}-11-01", f"{y}-11-30")
        bf = nov[nov.weekday == 3][3] + pd.Timedelta(days=1)                 # the day after the 4th Thursday
        for d in (bf, pd.Timestamp(f"{y}-12-11"), pd.Timestamp(f"{y}-12-18")):
            i = np.where((dates <= d) & (d < dates + pd.Timedelta(days=7)))[0]
            flag[i] = 1
    return flag


def media_effect(units: dict[str, np.ndarray], scale: float, med: dict[str, float], mult=1.0) -> dict[str, np.ndarray]:
    """Each channel's orders per week; mult is one factor for all channels or {channel: factor}."""
    out = {}
    for c, (_, _, _, alpha, beta, ec, slope) in CHANNELS.items():
        m = mult.get(c, 1.0) if isinstance(mult, dict) else mult
        a = adstock(units[c] * m, alpha)
        out[c] = beta * scale * hill(a / med[c], ec, slope)
    return out


def make(seed: int = 11) -> tuple[pd.DataFrame, dict, dict]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, periods=WEEKS, freq="W-MON")
    t = np.arange(WEEKS)
    pop_total = sum(REGIONS.values())
    hol = holiday_weeks(dates)
    promo = (rng.random(WEEKS) < 0.15).astype(int)
    season = 1 + 0.10 * np.sin(2 * np.pi * (dates.isocalendar().week.values.astype(float) - 10) / 52)
    # National spend, then split across regions by population with a little noise
    tv_on = flights(rng, WEEKS, 0.55)
    nat = {"tv": tv_on * rng.gamma(9, 1e4, WEEKS),
           "search": 4e4 * season * rng.lognormal(0, 0.25, WEEKS),
           "social": flights(rng, WEEKS, 0.85) * rng.gamma(6, 5e3, WEEKS),
           "display": rng.gamma(5, 4e3, WEEKS)}
    rows, contrib, contrib_up, spend_tot = [], {c: 0.0 for c in CHANNELS}, {c: 0.0 for c in CHANNELS}, {c: 0.0 for c in CHANNELS}
    rev_contrib, rev_contrib_up = {c: 0.0 for c in CHANNELS}, {c: 0.0 for c in CHANNELS}
    weekly = {c: np.zeros(WEEKS) for c in CHANNELS}
    whatif = {c: {f: {"all": [0.0, 0.0], "last_52": [0.0, 0.0]} for f in WHATIF} for c in CHANNELS}   # [orders, revenue]
    totals = {"orders": 0.0, "revenue": 0.0, "baseline": 0.0, "holiday": 0.0, "price": 0.0, "competitor": 0.0, "noise": 0.0}
    _STATE.clear()
    for g, pop in REGIONS.items():
        scale = pop / 3_000_000
        share = pop / pop_total
        spend = {c: np.round(nat[c] * share * rng.lognormal(0, 0.05, WEEKS), 2) for c in CHANNELS}
        units = {}
        for c, (_, ucol, cpu, *_r) in CHANNELS.items():
            units[c] = spend[c] / cpu * (rng.lognormal(0, 0.03, WEEKS) if ucol else 1.0)
            units[c] = np.round(units[c], 1 if c == "tv" else 0)
        med = {c: float(np.median(adstock(units[c], CHANNELS[c][3])[adstock(units[c], CHANNELS[c][3]) > 0])) for c in CHANNELS}
        eff = media_effect(units, scale, med)
        eff_up = media_effect(units, scale, med, 1.01)
        base = 2600 * scale * (1 + 0.15 * t / WEEKS) * season
        price = np.clip(100 + np.cumsum(rng.normal(0, 0.5, WEEKS)), 94, 106).round(1)
        e_hol, e_price, e_comp = 0.2 * base * hol, -35 * scale * (price - 100), -250 * scale * promo
        noise = rng.normal(0, 0.025, WEEKS) * base
        orders = base + e_hol + e_price + e_comp + sum(eff.values()) + noise
        orders = np.round(orders)
        aov = np.round(58 + rng.normal(0, 2, WEEKS), 2)
        revenue = np.round(orders * aov, 2)
        _STATE.append({"region": g, "pop": pop, "units": units, "spend": spend, "scale": scale, "med": med, "aov": aov, "eff": eff})
        last = slice(WEEKS - 52, WEEKS)
        for c in CHANNELS:                                # one channel's spend changed in every week
            for f in WHATIF:
                d_eff = media_effect(units, scale, med, {c: f})[c] - eff[c]
                for per, sl in (("all", slice(None)), ("last_52", last)):
                    # over a period: spend changed in its weeks only (carryover from earlier weeks unchanged)
                    if per == "last_52":
                        u = {**units, c: np.where(np.arange(WEEKS) >= WEEKS - 52, units[c] * f, units[c])}
                        d_eff = media_effect(u, scale, med)[c] - eff[c]
                    whatif[c][f][per][0] += float(d_eff[sl].sum())
                    whatif[c][f][per][1] += float((d_eff * aov)[sl].sum())
        for c in CHANNELS:
            contrib[c] += eff[c].sum()
            rev_contrib[c] += (eff[c] * aov).sum()
            rev_contrib_up[c] += (eff_up[c] * aov).sum()
            contrib_up[c] += eff_up[c].sum()
            spend_tot[c] += spend[c].sum()
            weekly[c] += eff[c]
        totals["orders"] += orders.sum()
        totals["revenue"] += revenue.sum()
        totals["baseline"] += base.sum()
        totals["holiday"] += e_hol.sum()
        totals["price"] += e_price.sum()
        totals["competitor"] += e_comp.sum()
        totals["noise"] += noise.sum()
        df = pd.DataFrame({"Week": dates.strftime("%Y-%m-%d"), "Region": g, "Orders": orders.astype(int), "Revenue ($)": revenue})
        for c, (scol, ucol, *_r) in CHANNELS.items():
            df[scol] = spend[c]
            if ucol:
                df[ucol] = units[c]
        df["Price Index"] = price
        df["Competitor Promo"] = promo
        df["Holiday Week"] = hol
        df["Population"] = pop
        rows.append(df)
    data = pd.concat(rows, ignore_index=True)
    # The tool's answer key
    key = {"kpi_column": "Orders", "assumes_decisions": "every paid channel kept as its own channel; Revenue ($) as the revenue column",
           "kpi_per_spend_unit": 1000,
           "channels": [{"source_columns": [CHANNELS[c][0]],
                         "true_roi": round(rev_contrib[c] / spend_tot[c], 3),
                         "true_share_of_kpi_pct": round(100 * contrib[c] / totals["orders"], 2),
                         "true_kpi_per_spend": round(1000 * contrib[c] / spend_tot[c], 3),
                         "true_marginal_roi": round((rev_contrib_up[c] - rev_contrib[c]) / (0.01 * spend_tot[c]), 3)}
                        for c in CHANNELS],
           "true_non_media_share_pct": round(100 * (1 - sum(contrib.values()) / totals["orders"]), 2)}
    details = {"seed": seed, "weeks": WEEKS, "start": START, "regions": REGIONS, "max_lag": LAG,
               "channels": {c: {"spend_column": s, "units_column": u, "cost_per_unit": cpu, "adstock_decay": a,
                                "beta_orders_per_week_per_3m_people": b, "half_saturation_vs_median": ec, "hill_slope": sl,
                                "total_spend": round(spend_tot[c], 2), "contribution_orders": round(contrib[c], 1),
                                "contribution_revenue": round(rev_contrib[c], 2),
                                "weeks_with_spend": int((nat[c] > 0).sum())}
                            for c, (s, u, cpu, a, b, ec, sl) in CHANNELS.items()},
               "controls": {"Price Index": "-35 orders per point above 100, per 3M people",
                            "Competitor Promo": "-250 orders in a promo week, per 3M people",
                            "Holiday Week": "+20% of baseline (Black Friday week, the 2 weeks before Christmas)"},
               "totals": {k: round(v, 1) for k, v in totals.items()},
               "what_if": {c: {f"{int(round(f * 100))}%": {per: {"orders_change": round(v[0], 1), "revenue_change": round(v[1], 2)}
                                                              for per, v in pp.items()} for f, pp in ff.items()}
                           for c, ff in whatif.items()},
               "weekly_contribution_orders": {c: [round(x, 1) for x in v] for c, v in weekly.items()}}
    data, planted = plant_problems(data, rng)
    details["planted_problems"] = planted
    return data, key, details


_STATE: list[dict] = []          # each region's units and true response, kept by make() for true_change()


def true_change(multipliers: dict[str, float], period: str = "all", seed: int = 11) -> dict[str, float]:
    """The true change in orders and revenue if each channel's spend is multiplied as given in every
    week of the period ("all" or "last_52"), against the data as it is: an answer key for any What-If
    scenario or optimized plan."""
    if not _STATE:
        make(seed)
    first = 0 if period == "all" else WEEKS - 52
    d_orders = d_rev = 0.0
    for r in _STATE:
        units = {c: np.where(np.arange(WEEKS) >= first, r["units"][c] * multipliers.get(c, 1.0), r["units"][c]) for c in CHANNELS}
        eff = media_effect(units, r["scale"], r["med"])
        diff = sum(eff[c] - r["eff"][c] for c in CHANNELS)[first:]
        d_orders += float(diff.sum())
        d_rev += float((diff * r["aov"][first:]).sum())
    return {"orders_change": round(d_orders, 1), "revenue_change": round(d_rev, 2)}


FUTURE_WEEKS = 13
FUTURE_TV = 0.5                  # in the actuals, TV ran at half of the same weeks last year


def season_of(dates: pd.DatetimeIndex) -> np.ndarray:
    return 1 + 0.10 * np.sin(2 * np.pi * (dates.isocalendar().week.values.astype(float) - 10) / 52)


def future_actuals(seed: int = 11) -> tuple[pd.DataFrame, dict]:
    """13 weeks after the data, made by the same process (carryover from the data's last weeks
    included): spend is the same weeks last year in every region, except TV at FUTURE_TV of it.
    No holidays fall in these weeks; prices stay at 100 and there is no competitor promotion.
    The truth: what each week's orders would have been with TV as planned (the execution gap)."""
    if not _STATE:
        make(seed)
    rng = np.random.default_rng(seed + 1000)
    dates = pd.date_range(START, periods=WEEKS + FUTURE_WEEKS, freq="W-MON")[WEEKS:]
    t = np.arange(WEEKS, WEEKS + FUTURE_WEEKS)
    season = season_of(dates)
    frames, exec_gap, true_media_plan, true_media_actual = [], np.zeros(FUTURE_WEEKS), np.zeros(FUTURE_WEEKS), np.zeros(FUTURE_WEEKS)
    for r in _STATE:
        last_year = slice(WEEKS - 52, WEEKS - 52 + FUTURE_WEEKS)
        plan_spend = {c: r["spend"][c][last_year].copy() for c in CHANNELS}
        act_spend = {c: np.round(plan_spend[c] * (FUTURE_TV if c == "tv" else 1.0), 2) for c in CHANNELS}

        def units_of(sp):
            out = {}
            for c, (_, ucol, cpu, *_r) in CHANNELS.items():
                u = sp[c] / cpu * (np.exp(rng.normal(0, 0.03, FUTURE_WEEKS)) if ucol else 1.0)
                out[c] = np.round(u, 1 if c == "tv" else 0)
            return out
        u_act = units_of(act_spend)
        u_plan = {c: (u_act[c] / FUTURE_TV if c == "tv" else u_act[c]) for c in CHANNELS}
        full = lambda u: {c: np.concatenate([r["units"][c], u[c]]) for c in CHANNELS}  # noqa: E731
        e_act = media_effect(full(u_act), r["scale"], r["med"])
        e_plan = media_effect(full(u_plan), r["scale"], r["med"])
        m_act = sum(e_act[c][WEEKS:] for c in CHANNELS)
        m_plan = sum(e_plan[c][WEEKS:] for c in CHANNELS)
        base = 2600 * r["scale"] * (1 + 0.15 * t / WEEKS) * season
        noise = rng.normal(0, 0.025, FUTURE_WEEKS) * base
        orders = np.round(base + m_act + noise)
        aov = np.round(58 + rng.normal(0, 2, FUTURE_WEEKS), 2)
        exec_gap += m_act - m_plan
        true_media_plan += m_plan
        true_media_actual += m_act
        df = pd.DataFrame({"Week": dates.strftime("%Y-%m-%d"), "Region": r["region"], "Orders": orders.astype(int),
                           "Revenue ($)": np.round(orders * aov, 2)})
        for c, (scol, ucol, *_r) in CHANNELS.items():
            df[scol] = act_spend[c]
            if ucol:
                df[ucol] = u_act[c]
        df["Search Spend ($)"] = df["Search Spend ($)"].map(lambda v: f"${v:,.2f}")    # exported like the training file
        df["Price Index"], df["Competitor Promo"], df["Holiday Week"], df["Population"] = 100.0, 0, 0, r["pop"]
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    truth = {"weeks": [str(d.date()) for d in dates], "spend_rule": f"same weeks last year; TV at {FUTURE_TV:.0%}",
             "true_execution_gap_orders": [round(float(x), 1) for x in exec_gap],
             "true_execution_gap_total": round(float(exec_gap.sum()), 1),
             "true_media_orders_as_planned": round(float(true_media_plan.sum()), 1),
             "true_media_orders_as_run": round(float(true_media_actual.sum()), 1),
             "actual_orders_total": int(data["Orders"].sum())}
    return data, truth


def plant_problems(df: pd.DataFrame, rng) -> tuple[pd.DataFrame, list[dict]]:
    """Problems Phases A and B should find and fix, none of which changes the true answer."""
    df = df.copy()
    planted = []
    # 1. Search spend exported as text with a currency sign and thousands separators.
    df["Search Spend ($)"] = df["Search Spend ($)"].map(lambda v: f"${v:,.2f}")
    planted.append({"what": "Search Spend ($) is text like '$12,345.67'", "where": "every row", "expect": "Phase B step 7 reads it as numbers (a currency sign and separators removed)"})
    # 2. One exact duplicate row.
    i = int(df.index[(df["Region"] == "South")][40])
    df = pd.concat([df.iloc[: i + 1], df.iloc[[i]], df.iloc[i + 1:]], ignore_index=True)
    planted.append({"what": "one exact duplicate row", "where": f"South, week {df.loc[i, 'Week']}", "expect": "Phase B step 8 drops it"})
    # 3. One blank control value.
    j = int(df.index[(df["Region"] == "West")][100])
    planted.append({"what": "one blank Price Index", "where": f"West, week {df.loc[j, 'Week']} (true value {df.loc[j, 'Price Index']})",
                    "expect": "Phase B fills or flags it (a missing control value)"})
    df.loc[j, "Price Index"] = np.nan
    # 4. TV off for whole flights: real zeros in a media column.
    planted.append({"what": "TV off in runs of 3 to 6 weeks (real zeros)", "where": "about 45% of weeks, all regions",
                    "expect": "Phase B step 10 counts them as real zeros in a media column, not missing data; TV is flighted, not always-on"})
    return df, planted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=str(Path(__file__).parent))
    ap.add_argument("--check-plan", metavar="JSON", help='spend multipliers, e.g. \'{"search": 1.35, "display": 0.7}\': '
                    "prints the true change in orders and revenue (all weeks; add --period last_52)")
    ap.add_argument("--period", default="all", choices=["all", "last_52"])
    a = ap.parse_args()
    if a.check_plan:
        print(json.dumps(true_change(json.loads(a.check_plan), a.period, a.seed)))
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data, key, details = make(a.seed)
    data.to_csv(out / "home_goods_mmm.csv", index=False)
    (out / "truth_for_tool.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    fut, fut_truth = future_actuals(a.seed)
    details["future_actuals"] = fut_truth
    fut.to_csv(out / "home_goods_actuals_next13.csv", index=False)
    (out / "truth_details.json").write_text(json.dumps(details, indent=1), encoding="utf-8")
    print(json.dumps(key, indent=1))
    print({c: v["weeks_with_spend"] for c, v in details["channels"].items()}, details["totals"])


if __name__ == "__main__":
    main()
