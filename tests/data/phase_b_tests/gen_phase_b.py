"""Phase B test files for the MMM tool (steps 7-13: clean and align to weeks).

Every file has a clean version in truth/<name>_truth.csv: the tool's cleaned output
should match it (exactly for counts and sums, within the stated tolerance for
interpolated values). expected_phase_b.json lists every injected problem.

Usage:  python3 gen_phase_b.py                 # the shared set (seed 11)
        python3 gen_phase_b.py --seed 99 --out hidden_b   # a hidden set: same problems, new positions
"""
import argparse, json, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--out", default="phase_b_tests")
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
OUT = args.out.rstrip("/") + "/"
os.makedirs(OUT + "truth", exist_ok=True)

base = pd.read_csv("insurance_mmm_sample.csv", parse_dates=["week_start"])
N = len(base)
SPEND = [c for c in base.columns if c.endswith("_spend")]
FLAGS = ["tax_season", "fy_end", "festive_flag", "regulatory_event"]
INDEX = ["avg_premium_index", "competitor_spend_index", "brand_search_index"]
ADDITIVE = ["new_policies", "new_premium_inr", "tv_grps"] + SPEND
iso = lambda s: pd.to_datetime(s).dt.strftime("%Y-%m-%d")
wk = lambda i: base.week_start[int(i)].strftime("%Y-%m-%d")
truth_base = base.assign(week_start=iso(base.week_start))
key = {"_about": {"seed": args.seed, "base": "insurance_mmm_sample.csv, 156 weeks, Monday starts 2023-01-02 to 2025-12-22",
                  "truth": "truth/<file>_truth.csv is the correctly cleaned weekly table"}}


def save(name, df, truth, **entry):
    df.to_csv(OUT + name, index=False)
    truth.to_csv(OUT + "truth/" + name.replace(".csv", "_truth.csv"), index=False)
    key[name] = entry


def interior(k, lo=6, hi=N - 6, avoid=()):
    pool = np.setdiff1d(np.arange(lo, hi), np.array(list(avoid), dtype=int))
    return sorted(int(x) for x in rng.choice(pool, k, replace=False))


def linear_targets(series, idx):
    """value a tool should produce by linear interpolation between neighbouring known weeks"""
    s = pd.Series(series, dtype=float).copy()
    s.iloc[idx] = np.nan
    s = s.interpolate(limit_area="inside")
    return {wk(i): round(float(s.iloc[i]), 1) for i in idx}


def inr(v):
    s = str(int(abs(v)))
    if len(s) > 3:
        head, tail, parts = s[:-3], s[-3:], []
        while len(head) > 2:
            parts.insert(0, head[-2:]); head = head[:-2]
        s = ",".join(([head] if head else []) + parts + [tail])
    return ("-" if v < 0 else "") + s


# ---------------------------------------------------------------- B01 step 7
b = truth_base.astype(object).copy()
t = truth_base.copy()
b.loc[base.tv_spend == 0, "tv_spend"] = "-"                         # accounting dash = zero
b["search_spend"] = [inr(v) for v in base.search_spend]              # 12,13,000
b["aggregator_spend"] = ["₹" + inr(v) for v in base.aggregator_spend]
lakh10 = (base.ooh_spend / 1e4).round().astype(int)                  # 12.6 L
b["ooh_spend"] = [f"{x / 10:.1f} L" if x else "0" for x in lakh10]
t["ooh_spend"] = lakh10 * 10_000
cr100 = (base.new_premium_inr / 1e5).round().astype(int)             # 3.88 Cr
b["new_premium_inr"] = [f"{x / 100:.2f} Cr" for x in cr100]
t["new_premium_inr"] = cr100 * 100_000
b["print_spend"] = [f"{v:.3e}" for v in base.print_spend]            # 9.940e+05
t["print_spend"] = [float(f"{v:.3e}") for v in base.print_spend]
pol_fmt = rng.choice(["plain", "dotzero", "comma_space"], N, p=[.6, .2, .2])
b["new_policies"] = [str(v) if f == "plain" else f"{v}.0" if f == "dotzero" else f" {inr(v)} "
                     for v, f in zip(base.new_policies, pol_fmt)]
b["tax_season"] = np.where(base.tax_season == 1, "Yes", "No")
b["festive_flag"] = np.where(base.festive_flag == 1, "TRUE", "FALSE")
b["fy_end"] = np.where(base.fy_end == 1, "Y", "N")
b["regulatory_event"] = base.regulatory_event.astype(float).astype(str)
null_tokens = ["N/A", "NULL", "#N/A", "nan", " "]
soc_null = interior(5)
for i, tok in zip(soc_null, null_tokens):
    b.loc[i, "social_spend"] = tok
t["social_spend"] = t["social_spend"].astype(float)
t.loc[soc_null, "social_spend"] = np.nan
comp_dash = interior(1, avoid=soc_null)
b.loc[comp_dash, "competitor_spend_index"] = "-"
t.loc[comp_dash, "competitor_spend_index"] = np.nan
save("B01_text_and_number_formats.csv", b, t,
     steps="7",
     conversions={"tv_spend": f"'-' -> 0 in {int((base.tv_spend == 0).sum())} cells (accounting dash in a money column)",
                  "search_spend": "Indian grouping 12,13,000", "aggregator_spend": "₹ prefix + Indian grouping",
                  "ooh_spend": "'12.6 L' = 12.6 lakh = 1,260,000; '0' stays 0",
                  "new_premium_inr": "'3.88 Cr' = 38,800,000", "print_spend": "scientific 9.940e+05",
                  "new_policies": "'3836', '3836.0' and ' 3,836 ' all -> 3836",
                  "tax_season": "Yes/No -> 1/0", "festive_flag": "TRUE/FALSE -> 1/0", "fy_end": "Y/N -> 1/0",
                  "regulatory_event": "'1.0'/'0.0' -> 1/0"},
     missing_after_step_7={"social_spend": {wk(i): tok for i, tok in zip(soc_null, null_tokens)},
                           "competitor_spend_index": {wk(comp_dash[0]): "'-' is missing here, not zero: not a money column"}},
     trap="'-' means 0 only in money columns; in an index it means not available",
     blocks=False)

# ---------------------------------------------------------------- B02 step 8
dup = interior(12)
d = pd.concat([truth_base, truth_base.iloc[dup]]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
save("B02_exact_duplicate_rows.csv", d, truth_base,
     steps="8", rows_in=len(d), rows_out=N, exact_duplicate_weeks=[wk(i) for i in dup],
     expected="drop the 12 exact copies (never sum them: that doubles spend and KPI); sort the shuffled rows by date",
     blocks=False)

# ---------------------------------------------------------------- B03 step 8
d = truth_base.copy()
d["active_agents"] = (1200 + 2 * np.arange(N) + rng.normal(0, 15, N)).round().astype(int)
t = d.copy().astype({"active_agents": float})
flagged = np.where((base.tax_season == 1) | (base.festive_flag == 1))[0]
pick = sorted(set(int(x) for x in rng.choice(flagged[(flagged > 5) & (flagged < N - 6)], 3, replace=False))
              | set(interior(7, avoid=flagged)))
rows = []
for i in range(N):
    r = d.iloc[i].to_dict()
    if i not in pick:
        rows.append(r); continue
    a, c = dict(r), dict(r)
    for col in ADDITIVE:
        part = int(round(r[col] * rng.uniform(.3, .7)))
        a[col], c[col] = part, int(r[col]) - part
    for col in FLAGS:
        c[col] = 0                                   # second source does not carry the flag
    delta = int(rng.choice([2, 4, 6]))
    c["active_agents"] = int(r["active_agents"]) + delta   # two systems counted at different times
    t.loc[i, "active_agents"] = r["active_agents"] + delta / 2
    rows += [a, c]
d = pd.DataFrame(rows)
save("B03_split_rows_aggregation.csv", d, t,
     steps="8", rows_in=len(d), rows_out=N, split_weeks=[wk(i) for i in pick],
     rules={"sum (values differ, money or count)": ADDITIVE,
            "keep the value (identical in both rows; summing doubles it)": INDEX + ["new_vehicle_registrations"],
            "max (flags)": FLAGS,
            "ask the user (values differ, role unknown)": ["active_agents -> correct answer: mean"]},
     traps=["new_vehicle_registrations is a count by name but must not be summed: both rows repeat the same market figure",
            "a flag summed or averaged gives 0.5 or 2; it must be the max"],
     prompts=["aggregation rule for active_agents"], blocks=True)

# ---------------------------------------------------------------- B04 step 9
days = pd.date_range("2024-01-03", "2025-12-27", freq="D")          # Wednesday .. Saturday
nd = len(days)
promo = np.zeros(nd, int)
for s in rng.choice(np.arange(0, nd - 10), 14, replace=False):
    promo[s:s + rng.integers(3, 9)] = 1
daily = pd.DataFrame({"date": days.strftime("%Y-%m-%d"),
                      "policies": (rng.poisson(540, nd) + np.where(days.dayofweek < 5, 60, -110) + 80 * promo).astype(int),
                      "tv_spend": np.where((days.dayofyear // 14) % 2 == 0, rng.integers(5, 10, nd) * 100_000, 0),
                      "search_spend": rng.integers(150, 220, nd) * 1_000,
                      "avg_premium_index": (1 + 0.00008 * np.arange(nd) + rng.normal(0, .004, nd)).round(4),
                      "promo_flag": promo})
monday = (days - pd.to_timedelta(days.dayofweek, unit="D")).strftime("%Y-%m-%d")
weekly = daily.assign(week_start=monday).groupby("week_start").agg(
    policies=("policies", "sum"), tv_spend=("tv_spend", "sum"), search_spend=("search_spend", "sum"),
    avg_premium_index=("avg_premium_index", "mean"), promo_flag=("promo_flag", "max"),
    days_in_week=("date", "count")).reset_index()
wk_list = weekly.week_start.tolist()
inner_weeks = rng.choice(np.arange(2, len(wk_list) - 2), 5, replace=False)
drop_days = []
for w in sorted(inner_weeks):
    cand = np.where(monday == wk_list[w])[0]
    drop_days.append(int(rng.choice(cand)))
file_daily = daily.drop(index=drop_days).reset_index(drop=True)
weekly["avg_premium_index"] = weekly.avg_premium_index.round(4)
save("B04_daily_to_weekly.csv", file_daily, weekly,
     steps="9", daily_rows=len(file_daily), weeks_out=len(weekly), week_start="Monday",
     aggregation={"sum": ["policies", "tv_spend", "search_spend"], "mean": ["avg_premium_index"], "max": ["promo_flag"]},
     partial_weeks={"first": f"{wk_list[0]} (Wed-Sun, 5 days)", "last": f"{wk_list[-1]} (Mon-Sat, 6 days)",
                    "missing_day_inside": {wk_list[w]: daily.date[d] for w, d in zip(sorted(inner_weeks), drop_days)}},
     expected=("7 incomplete weeks flagged, never silently summed as if complete. Edge weeks: drop (default) or keep flagged. "
               "Inner weeks with one missing day: scale sums by 7/6 or interpolate the day, and add a flag column. "
               "The truth file holds the full-week totals from complete daily data; a 7/6-scaled estimate lands within about 5%."),
     prompts=["edge partial weeks: drop or keep", "inner incomplete weeks: scale, interpolate, or keep"], blocks=False)

# ---------------------------------------------------------------- B05 step 9
sun = truth_base.copy()
sun["week_start"] = iso(base.week_start - pd.Timedelta(days=1))
save("B05_sunday_week_starts.csv", sun, sun,
     steps="9, 6", expected=("weekly with Sunday starts. Already weekly, so no resampling: values must be unchanged and 156 weeks kept. "
                             "Relabelling to the following Monday is acceptable if the tool standardises on Monday; "
                             "binning into Monday-start weeks is not (it relabels every week to the previous Monday)."),
     blocks=False)

# ---------------------------------------------------------------- B06 step 9
mix = truth_base.copy()
mix.loc[:51, "week_start"] = iso(base.week_start[:52] - pd.Timedelta(days=1))
save("B06_week_start_changes.csv", mix, truth_base,
     steps="9, 10", change_point=f"rows 1-52 are Sundays ({mix.week_start[0]} .. {mix.week_start[51]}); from {mix.week_start[52]} Mondays",
     gap_at_change="8 days between the last Sunday and the first Monday: not a missing week",
     expected="detect the weekday change, shift the Sunday labels forward one day, end with 156 continuous Monday weeks: 0 missing, 0 duplicates",
     traps=["reporting a missing week at the change point",
            "binning by Monday-start weeks: all of 2023 moves one week earlier and a fake empty week appears at 2023-12-25"],
     blocks=False)

# ---------------------------------------------------------------- B07 step 10
run_start = interior(1, lo=30, hi=60)[0]
run = [run_start, run_start + 1, run_start + 2]
singles = interior(3, lo=70, hi=N - 6)
while min(np.diff(singles)) < 3:
    singles = interior(3, lo=70, hi=N - 6)
gone = run + singles
save("B07_missing_weeks.csv", truth_base.drop(index=gone).reset_index(drop=True), truth_base,
     steps="10", rows_in=N - 6, rows_out=N, missing_weeks=[wk(i) for i in gone],
     runs={"3-week run": [wk(i) for i in run], "single weeks": [wk(i) for i in singles]},
     expected=("insert the 6 weeks; KPI and controls interpolated with an 'imputed' flag; spend in missing weeks is UNKNOWN, "
               "not zero: ask (interpolate always-on channels, 0 only if the user confirms the channel was off). "
               "Interpolated KPI must equal expected_interpolated_kpi (linear between neighbours); the truth file holds the real values for reference only (noise puts them 5-20% away)."),
     expected_interpolated_kpi=linear_targets(base.new_policies, gone),
     trap="filling missing-week spend with 0 invents a dark period that the model will read as a media effect",
     prompts=["KPI gaps: interpolate + flag, or trim", "spend in missing weeks"], blocks=True)

# ---------------------------------------------------------------- B08 step 10
d = truth_base.copy().astype(object)
t = truth_base.copy()
launch, stop = 78, 118                                                # ctv starts / print stops
ctv = np.where(np.arange(N) >= launch, (rng.integers(6, 12, N) * 100_000), 0)
d["ctv_spend"] = np.where(np.arange(N) >= launch, ctv, None)
t["ctv_spend"] = ctv
d.loc[stop:, "print_spend"] = None
t.loc[stop:, "print_spend"] = 0
d.loc[base.radio_spend == 0, "radio_spend"] = None                  # export writes blank for zero
soc_blank = interior(4)
d.loc[soc_blank, "social_spend"] = None
save("B08_zero_vs_missing.csv", d, t,
     steps="10",
     categories={"tv_spend": f"{int((base.tv_spend == 0).sum())} literal zeros: genuine zero, leave unchanged",
                 "radio_spend": (f"{int((base.radio_spend == 0).sum())} blanks, no literal zeros, blanks fall in runs between bursts: "
                                 "probably zero. Suggest 0 and ask."),
                 "social_spend": f"4 blanks in an always-on channel ({', '.join(wk(i) for i in soc_blank)}): missing. Interpolate + flag; targets in expected_interpolated_social.",
                 "ctv_spend": f"blank before first value ({wk(launch)}): channel not started. Fill 0.",
                 "print_spend": f"blank after {wk(stop - 1)}: channel stopped. Fill 0, keep literal zeros before."},
     expected_interpolated_social=linear_targets(base.social_spend, soc_blank),
     trap="one rule for every blank (all zero or all missing) gets at least three of these five columns wrong",
     prompts=["radio blanks = zero?"], blocks=False)

# ---------------------------------------------------------------- B09 step 10
d = truth_base.copy().astype({"new_policies": float})
lead = [0, 1, 2, 3]
sc = interior(3, lo=20, hi=100)
while min(np.diff(sc)) < 3:
    sc = interior(3, lo=20, hi=100)
pair_start = interior(1, lo=110, hi=N - 8, avoid=sc)[0]
inner = sc + [pair_start, pair_start + 1]
d.loc[lead + inner, "new_policies"] = np.nan
save("B09_kpi_gaps.csv", d, truth_base.iloc[4:].reset_index(drop=True),
     steps="10", leading_gap=[wk(i) for i in lead], inner_gaps=[wk(i) for i in sorted(inner)],
     expected=("trim the 4 leading weeks (nothing before them to interpolate from): 152 rows remain. "
               "Interpolate the 5 inner weeks (pass test: expected_interpolated_kpi; truth values are reference only) and add a kpi_imputed flag column. "
               "Never fill KPI gaps with 0."),
     expected_interpolated_kpi=linear_targets(base.new_policies, sorted(inner)),
     prompts=["trim leading KPI gap", "interpolate inner KPI gaps"], blocks=True)

# ---------------------------------------------------------------- B10 step 11
d = truth_base.copy()
o_low, o_high, near = interior(3)
d.loc[o_low, "new_policies"] = int(base.new_policies[o_low] * 0.2)
d.loc[o_high, "new_policies"] = int(base.new_policies[o_high] * 1.8)
d.loc[near, "new_policies"] = int(base.new_policies[near] * 1.2)
k = d.new_policies.astype(float)
z = 0.6745 * (k - k.median()) / (k - k.median()).abs().median()
truth = d.copy()
truth["event_flag"] = 0
truth.loc[[o_low, o_high], "event_flag"] = 1
save("B10_kpi_outliers.csv", d, truth,
     steps="11", flagged={wk(o_low): f"0.2x (outage), robust z {z[o_low]:.1f}", wk(o_high): f"1.8x (spike), robust z {z[o_high]:.1f}"},
     not_flagged={wk(near): f"1.2x, robust z {z[near]:.1f} (below 3.5)"},
     max_other_abs_z=round(float(z.drop([o_low, o_high]).abs().max()), 2),
     expected="flag exactly 2 weeks; default action adds an event_flag column. No row is deleted and no KPI value is changed.",
     blocks=False)

# ---------------------------------------------------------------- B11 step 12
d = truth_base.copy()
neg_aff = interior(2)
neg_agg = interior(1, avoid=neg_aff)
for i in neg_aff:
    d.loc[i, "affiliate_spend"] = -int(rng.integers(20, 60)) * 1_000
d.loc[neg_agg, "aggregator_spend"] = -int(rng.integers(50, 90)) * 1_000
zero_kpi = interior(2, avoid=neg_aff + neg_agg)
d.loc[zero_kpi, "new_policies"] = 0
jump_idx = interior(1, avoid=neg_aff + neg_agg + zero_kpi)[0]
d.loc[jump_idx, "avg_premium_index"] = round(base.avg_premium_index[jump_idx] * 10, 3)
tv_on = np.where(base.tv_spend > 0)[0]
tv_bad = int(rng.choice(tv_on[(tv_on > 6) & (tv_on < N - 6) & ~np.isin(tv_on, neg_aff + neg_agg + zero_kpi + [jump_idx])]))
d.loc[tv_bad, "tv_spend"] = int(base.tv_spend[tv_bad]) * 100
save("B11_impossible_values.csv", d, truth_base,
     steps="12",
     findings={"negative spend (blocks)": {**{wk(i): f"affiliate_spend {d.affiliate_spend[i]}" for i in neg_aff},
                                           wk(neg_agg[0]): f"aggregator_spend {d.aggregator_spend[neg_agg[0]]}"},
               "KPI zero in a trading week": {wk(i): f"new_policies 0 while premium is {base.new_premium_inr[i]:,} (inconsistent)" for i in zero_kpi},
               "10x jump": {wk(jump_idx): f"avg_premium_index {d.avg_premium_index[jump_idx]} (neighbours about {base.avg_premium_index[jump_idx]})"},
               "100x jump": {wk(tv_bad): f"tv_spend {d.tv_spend[tv_bad]:,} (true {base.tv_spend[tv_bad]:,}: units slip)"}},
     expected="list all 7 cells with their values; negatives block until the user corrects, zeroes or confirms them; truth file has the corrected values",
     blocks=True)

# ---------------------------------------------------------------- B12 step 13
d = truth_base.copy().astype({"search_spend": float})
t = truth_base.copy()
lakhs = (base.search_spend[:52] / 1e3).round().astype(int)          # 2 decimals of a lakh
d.loc[:51, "search_spend"] = lakhs / 100
t.loc[:51, "search_spend"] = lakhs * 1_000
ratio = base.search_spend[52:].median() / (lakhs / 100).median()
save("B12_unit_change.csv", d, t,
     steps="12, 13", change=f"search_spend is in lakhs for {wk(0)} .. {wk(51)}, rupees from {wk(52)}",
     level_ratio=f"median after / median before = {ratio:,.0f} (about 1e5)",
     expected="detect the level shift and its ratio close to a power of 10; offer to multiply the 2023 rows by 100,000; do not treat it as a real spend change",
     inflation="no detection expected; step 13 offers an optional adjustment to a chosen base year",
     prompts=["convert lakhs rows to rupees"], blocks=False)

# ---------------------------------------------------------------- B13 all steps
names = {"week_start": "Week Commencing", "new_policies": "Policies Issued", "new_premium_inr": "NB Premium (INR)",
         "tv_spend": "TV Spend", "search_spend": "Google Search Cost", "social_spend": "Meta Spend",
         "aggregator_spend": "PolicyBazaar Fees", "affiliate_spend": "Affiliate Payout", "festive_flag": "Diwali/Festive"}
cols = ["week_start", "new_policies", "new_premium_inr", "tv_spend", "search_spend", "social_spend",
        "aggregator_spend", "affiliate_spend", "avg_premium_index", "tax_season", "festive_flag"]
truth = truth_base[cols].copy()
d = truth.copy().astype(object)
miss = interior(3)
outl = interior(2, avoid=miss)
neg = interior(1, avoid=miss + outl)
nulls = interior(3, avoid=miss + outl + neg)
dups = interior(5, avoid=miss)
splits = interior(4, avoid=miss + outl + neg + nulls + dups)
for i in outl:
    d.loc[i, "new_policies"] = int(base.new_policies[i] * 0.2)
    truth.loc[i, "new_policies"] = int(base.new_policies[i] * 0.2)   # outliers are flagged, not changed
truth["event_flag"] = 0
truth.loc[outl, "event_flag"] = 1
d.loc[neg[0], "affiliate_spend"] = -35_000
d["date_obj"] = base.week_start
rows = []
for i in range(N):
    if i in miss:
        continue
    r = d.iloc[i].to_dict()
    if i in splits:
        a, c = dict(r), dict(r)
        for col in ["new_policies", "new_premium_inr", "tv_spend", "search_spend", "social_spend", "aggregator_spend", "affiliate_spend"]:
            part = int(round(int(r[col]) * .4))
            a[col], c[col] = part, int(r[col]) - part
        c["festive_flag"] = 0
        rows += [a, c]
    else:
        rows.append(r)
    if i in dups:
        rows.append(dict(rows[-1]))
d = pd.DataFrame(rows)
for i, tok in zip(nulls, ["N/A", "NULL", "#N/A"]):
    d.loc[d.date_obj == base.week_start[i], "social_spend"] = tok
d["search_spend"] = [inr(v) if str(v).lstrip("-").isdigit() else v for v in d.search_spend]
d["new_premium_inr"] = ["₹" + inr(v) for v in d.new_premium_inr]
d["week_start"] = [x.strftime("%d/%m/%Y") if j % 2 else x.strftime("%Y-%m-%d") for j, x in enumerate(d.date_obj)]
d = d.drop(columns="date_obj").rename(columns=names).sample(frac=1, random_state=args.seed + 1)
truth.loc[nulls, "social_spend"] = np.nan
with open(OUT + "B13_end_to_end.csv", "w") as f:
    f.write("Acme General Insurance - weekly marketing export\nGenerated 2026-01-05\n\n")
    d.to_csv(f, index=False)
truth.to_csv(OUT + "truth/B13_end_to_end_truth.csv", index=False)
key["B13_end_to_end.csv"] = dict(
    steps="1-13", preamble_lines=3, rows_in=len(d), column_names=names,
    dates="alternating ISO and dd/mm/yyyy; rows shuffled",
    exact_duplicates=[wk(i) for i in dups], split_weeks=[wk(i) for i in splits],
    missing_weeks=[wk(i) for i in miss], kpi_outliers=[wk(i) for i in outl],
    negative_spend={wk(neg[0]): "affiliate_spend -35,000"}, social_nulls=[wk(i) for i in nulls],
    formats={"Google Search Cost": "Indian grouping", "NB Premium (INR)": "₹ + Indian grouping"},
    expected=("after Phases A and B: 156 weekly rows matching the truth file, with social nulls and missing weeks "
              "interpolated linearly (truth values for those cells are reference only), event_flag on the 2 outlier weeks, and the negative resolved by the user"),
    blocks=True)

json.dump(key, open(OUT + "expected_phase_b.json", "w"), indent=2, ensure_ascii=False, default=str)
print(f"seed {args.seed}: wrote {len(key) - 1} files to {OUT}")
