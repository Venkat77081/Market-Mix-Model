"""Phase A test files for the MMM tool. Each file targets specific steps 1-6.
Base data: insurance_mmm_sample.csv (156 weeks, 2023-01-02 .. 2025-12-22)."""
import json, numpy as np, pandas as pd
rng = np.random.default_rng(7)
OUT = "phase_a_tests/"
base = pd.read_csv("insurance_mmm_sample.csv", parse_dates=["week_start"])
key = {}

def save(name, df, **kw):
    df.to_csv(OUT + name, index=False, **kw)

# A01 clean baseline -------------------------------------------------------
save("A01_clean_weekly.csv", base.assign(week_start=base.week_start.dt.strftime("%Y-%m-%d")))
key["A01_clean_weekly.csv"] = dict(steps="1-6 baseline", delimiter=",", encoding="utf-8", header_row=1,
    date_format="%Y-%m-%d", granularity="weekly", geo=None, kpi="new_policies", kpi_type="non_revenue",
    revenue_per_kpi="derive: new_premium_inr / new_policies (or use new_premium_inr as a revenue KPI)",
    media_spend=[c for c in base.columns if c.endswith("_spend")],
    exposure_pairs={"tv_spend": "tv_grps"},
    controls=["avg_premium_index","competitor_spend_index","new_vehicle_registrations","tax_season","fy_end","festive_flag","regulatory_event"],
    suggest_exclude=["brand_search_index"], prompts=["KPI choice: new_policies vs new_premium_inr"], blocks=False)

# A02 semicolon + UTF-8 BOM -------------------------------------------------
a = base.copy(); a["week_start"] = a.week_start.dt.strftime("%Y-%m-%d")
a.to_csv(OUT + "A02_semicolon_bom.csv", index=False, sep=";", encoding="utf-8-sig")
key["A02_semicolon_bom.csv"] = dict(steps="1", delimiter=";", encoding="utf-8 with BOM",
    header_row=1, check="first column must be named 'week_start', not '\\ufeffweek_start'", blocks=False)

# A03 report preamble + footer total ----------------------------------------
pre = ("Acme General Insurance - Marketing Performance Export\n"
       "Generated: 2026-01-05 09:14 IST\nCurrency: INR\n\n")
body = a.to_csv(index=False)
tot = ["TOTAL"] + [str(a[c].sum()) if np.issubdtype(a[c].dtype, np.number) else "" for c in a.columns[1:]]
with open(OUT + "A03_preamble_footer.csv", "w") as f:
    f.write(pre + body + ",".join(tot) + "\n")
key["A03_preamble_footer.csv"] = dict(steps="1, 2", header_row=5, preamble_rows=4,
    footer="last row 'TOTAL' is a summary, must be dropped (not an unparseable date)",
    expected_rows=156, blocks=False)

# A04 Excel with several sheets, header not on row 1 ------------------------
with pd.ExcelWriter(OUT + "A04_multisheet.xlsx") as xw:
    pd.DataFrame({"Notes": ["Data owner: Marketing Analytics", "Spend in INR, weekly, Monday start"]}).to_excel(xw, "Notes", index=False)
    a.to_excel(xw, "Weekly_Data", index=False, startrow=2)
    a.groupby(a.week_start.str[:4])[["new_policies", "tv_spend"]].sum().to_excel(xw, "Yearly_Pivot")
key["A04_multisheet.xlsx"] = dict(steps="1", sheets=["Notes","Weekly_Data","Yearly_Pivot"],
    data_sheet="Weekly_Data", header_row=3, prompts=["which sheet (suggest Weekly_Data: most rows, has a date column)"], blocks=False)

# A05 mixed date formats per row ---------------------------------------------
m = base.copy()
fmts = ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%y", "%d.%m.%Y"]
m["week_start"] = [d.strftime(fmts[i % 4]) for i, d in enumerate(m.week_start)]
save("A05_mixed_dates.csv", m)
key["A05_mixed_dates.csv"] = dict(steps="2", formats_present=fmts,
    rule="slash dates are day-first: proven by rows with day > 12 (e.g. 16/01/2023)",
    expected_first="2023-01-02", expected_last="2025-12-22", expected_unique_dates=156,
    duplicates_created_if_parsed_month_first="must be 0 after correct parsing", blocks=False)

# A06 every slash date ambiguous (monthly, 1st of month) ----------------------
mo = pd.date_range("2023-01-01", periods=36, freq="MS")
amb = pd.DataFrame({"month": mo.strftime("%d/%m/%Y"),
    "policies": rng.integers(14000, 19000, 36), "tv_spend": rng.integers(0, 3, 36) * 9_000_000,
    "search_spend": rng.integers(4_000_000, 6_000_000, 36)})
save("A06_ambiguous_dates_monthly.csv", amb)
key["A06_ambiguous_dates_monthly.csv"] = dict(steps="2, 6",
    trap="day-first gives monthly 2023-01..2025-12; month-first gives 2023-01-01..01-12 daily-ish then invalid (13th month) for... no: all days are 01, so month-first reads 01/02/2023 as Jan 2",
    expected="prefer day-first: it gives an evenly spaced monthly series; month-first gives an irregular series. Ask the user to confirm",
    granularity="monthly", phase_c_note="36 rows -> would be refused at step 14", prompts=["confirm day-first"], blocks=False)

# A07 messy, real-world column names -----------------------------------------
n = base.rename(columns={
    "week_start": "Week Commencing", "new_policies": "Policies Issued (New)", "new_premium_inr": "New Business GWP (INR)",
    "tv_spend": "TV Spend (INR)", "tv_grps": "TV GRPs", "radio_spend": "Radio - Cost", "print_spend": "Print Ads ₹",
    "ooh_spend": "OOH/Hoardings spend", "search_spend": "Google Search Cost", "social_spend": "Meta Spend",
    "youtube_ott_spend": "YT+OTT Investment", "aggregator_spend": "PolicyBazaar Fees", "affiliate_spend": "Affiliate Payout",
    "brand_search_index": "Brand Search (GTrends)", "avg_premium_index": "Avg Premium Idx",
    "competitor_spend_index": "Cmpttr SOV", "new_vehicle_registrations": "Vahan Registrations",
    "tax_season": "Tax Season?", "fy_end": "FY End Flag", "festive_flag": "Diwali/Festive", "regulatory_event": "IRDAI change"})
n["Week Commencing"] = base.week_start.dt.strftime("%Y-%m-%d")
n.insert(0, "Unnamed: 0", range(len(n)))
n["Notes"] = ["" if rng.random() > .05 else "campaign launch" for _ in range(len(n))]
save("A07_messy_column_names.csv", n)
key["A07_messy_column_names.csv"] = dict(steps="3",
    expected_roles={"Unnamed: 0": "exclude (row index)", "Week Commencing": "date", "Policies Issued (New)": "kpi",
        "New Business GWP (INR)": "revenue (for revenue_per_kpi)", "TV Spend (INR)": "media_spend", "TV GRPs": "media (exposure)",
        "Radio - Cost": "media_spend", "Print Ads ₹": "media_spend", "OOH/Hoardings spend": "media_spend",
        "Google Search Cost": "media_spend", "Meta Spend": "media_spend", "YT+OTT Investment": "media_spend",
        "PolicyBazaar Fees": "media_spend (needs confirmation: 'fees' is not an obvious spend word)",
        "Affiliate Payout": "media_spend (needs confirmation)", "Brand Search (GTrends)": "exclude (suggested: caused by media)",
        "Avg Premium Idx": "control", "Cmpttr SOV": "control", "Vahan Registrations": "control",
        "Tax Season?": "control (flag)", "FY End Flag": "control (flag)", "Diwali/Festive": "control (flag)",
        "IRDAI change": "control (flag)", "Notes": "exclude (free text)"}, blocks=False)

# A08 spend/exposure pairing incl. reach & frequency and organic ------------
p = pd.DataFrame({"week": base.week_start.dt.strftime("%Y-%m-%d"), "policies": base.new_policies,
    "tv_spend": base.tv_spend, "tv_grps": base.tv_grps,
    "Search Cost": base.search_spend, "Search Clicks": (base.search_spend / 18 * rng.normal(1, .05, 156)).round(),
    "FB spend": base.social_spend, "Meta impressions": (base.social_spend / 0.12 * rng.normal(1, .05, 156)).round(),
    "radio_spend": base.radio_spend,
    "yt_spend": base.youtube_ott_spend, "yt_reach": (base.youtube_ott_spend / 4.5).round(), "yt_frequency": np.where(base.youtube_ott_spend > 0, rng.uniform(2, 4, 156).round(2), 0),
    "email_sends": rng.integers(200_000, 600_000, 156), "whatsapp_msgs": rng.integers(50_000, 200_000, 156)})
save("A08_spend_exposure_pairs.csv", p)
key["A08_spend_exposure_pairs.csv"] = dict(steps="3, 4",
    pairs={"tv_spend": "tv_grps (name match)", "Search Cost": "Search Clicks (name match)",
           "FB spend": "Meta impressions (names differ: tool should ask, not auto-pair)", "radio_spend": "none -> spend used as exposure"},
    reach_frequency={"spend": "yt_spend", "reach": "yt_reach", "frequency": "yt_frequency"},
    organic_media=["email_sends", "whatsapp_msgs"], prompts=["confirm FB spend <-> Meta impressions"], blocks=False)

# A09 revenue KPI ----------------------------------------------------------
r = base.drop(columns=["new_policies"]).assign(week_start=base.week_start.dt.strftime("%Y-%m-%d"))
save("A09_revenue_kpi.csv", r)
key["A09_revenue_kpi.csv"] = dict(steps="5", kpi="new_premium_inr", kpi_type="revenue", revenue_per_kpi=None,
    note="ROI reported directly in INR; no revenue-per-unit prompt", blocks=False)

# A10 non-revenue KPI, no money column -------------------------------------
l = pd.DataFrame({"week_start": base.week_start.dt.strftime("%Y-%m-%d"),
    "quote_requests": (base.new_policies * rng.uniform(4.5, 5.5, 156)).round().astype(int),
    "tv_spend": base.tv_spend, "search_spend": base.search_spend, "social_spend": base.social_spend,
    "aggregator_spend": base.aggregator_spend, "tax_season": base.tax_season})
save("A10_leads_no_revenue.csv", l)
key["A10_leads_no_revenue.csv"] = dict(steps="5", kpi="quote_requests", kpi_type="non_revenue",
    prompts=["value per quote request in INR, or continue with ROI in quotes per INR"], blocks=False)

# A11 daily data -------------------------------------------------------------
days = pd.date_range("2024-01-01", "2025-12-31", freq="D")
d = pd.DataFrame({"date": days.strftime("%Y-%m-%d"),
    "policies": rng.poisson(550, len(days)) + np.where(days.dayofweek < 5, 60, -120),
    "tv_spend": np.where((days.dayofyear // 14) % 2 == 0, rng.integers(500_000, 900_000, len(days)), 0),
    "search_spend": rng.integers(150_000, 220_000, len(days)), "social_spend": rng.integers(80_000, 120_000, len(days))})
save("A11_daily.csv", d)
key["A11_daily.csv"] = dict(steps="6", granularity="daily", rows=len(d),
    expected="detect daily; Phase B step 9 resamples to 105 weeks (spend and KPI summed)", blocks=False)

# A12 monthly data -------------------------------------------------------------
mm = pd.DataFrame({"month": pd.date_range("2023-01-01", periods=36, freq="MS").strftime("%Y-%m"),
    "policies": rng.integers(15000, 19000, 36), "tv_spend": rng.integers(0, 3, 36) * 9_000_000,
    "search_spend": rng.integers(4_000_000, 6_000_000, 36)})
save("A12_monthly.csv", mm)
key["A12_monthly.csv"] = dict(steps="2, 6", date_format="%Y-%m", granularity="monthly", rows=36,
    expected="detect monthly; warn that monthly data cannot be split into weeks and 36 rows will be refused at step 14", blocks=False)

# A13 regions, long format -------------------------------------------------------
states = {"Tamil Nadu": 5.6e7, "Karnataka": 4.9e7, "Maharashtra": 9.1e7, "Delhi": 1.5e7, "Gujarat": 5.0e7}
rows = []
for s, pop in states.items():
    w = pop / sum(states.values())
    for _, b in base.iterrows():
        rows.append({"week_start": b.week_start.strftime("%Y-%m-%d"), "state": s, "adult_population": int(pop),
            "new_policies": int(b.new_policies * w * rng.normal(1, .06)),
            "tv_spend": int(b.tv_spend * w), "search_spend": int(b.search_spend * w * rng.normal(1, .1)),
            "social_spend": int(b.social_spend * w * rng.normal(1, .1)), "tax_season": b.tax_season})
save("A13_regions_long.csv", pd.DataFrame(rows))
key["A13_regions_long.csv"] = dict(steps="3, 6", geo="state", geo_values=list(states), population="adult_population",
    rows=780, expected="5 regions x 156 weeks; population role suggested from a constant-per-region column",
    prompts=["model regions together (recommended) or aggregate to national"], blocks=False)

# A14 regions, wide format --------------------------------------------------------
wide = pd.DataFrame({"week_start": base.week_start.dt.strftime("%Y-%m-%d")})
for s, share in [("TN", .30), ("KA", .25), ("MH", .45)]:
    # each region gets its own spend pattern, so the three columns are not copies of each other
    tv_on = rng.random(156) < .55
    wide[f"policies_{s}"] = (base.new_policies * share * rng.normal(1, .06, 156)).round().astype(int)
    wide[f"tv_spend_{s}"] = np.where(tv_on, base.tv_spend.mean() * share * rng.lognormal(0, .35, 156), 0).round(-3).astype(int)
    wide[f"search_spend_{s}"] = (base.search_spend * share * rng.lognormal(0, .2, 156)).round(-3).astype(int)
save("A14_regions_wide.csv", wide)
key["A14_regions_wide.csv"] = dict(steps="3, 6",
    expected=("detect the repeated suffixes _TN/_KA/_MH on the same stems (policies, tv_spend, search_spend); "
              "offer to reshape to long: 468 rows = 3 regions x 156 weeks, geo=state, KPI=policies, channels tv and search; "
              "step 6 must then report 3 regions, not one national series"),
    wrong_outcomes=["policies_TN as KPI with policies_KA/MH excluded",
                    "6 channels like tv_spend_KA (another region's spend explaining TN's policies)",
                    "'one national series' at step 6"],
    alternative="sum the regions into one national series (policies, tv_spend, search_spend), if the user declines the reshape",
    prompts=["reshape wide regions to long"], blocks=False)

# A15 unparseable dates --------------------------------------------------------------
bd = a.copy()
bad_idx = rng.choice(156, 9, replace=False)
bd.loc[bad_idx, "week_start"] = ["TBD", "Week 12", "", "2024-13-01", "31/02/2024", "#VALUE!", "n/a", "2024/W15", "45292"]
save("A15_bad_dates.csv", bd)
key["A15_bad_dates.csv"] = dict(steps="2", bad_rows=sorted((bad_idx + 2).tolist()),
    bad_values=["TBD","Week 12","(blank)","2024-13-01","31/02/2024","#VALUE!","n/a","2024/W15","45292 (Excel serial = 2024-01-01)"],
    expected="list all 9 rows with their values; offer to read 45292 as an Excel serial and 2024/W15 as ISO week; block until the rest are fixed or dropped",
    blocks=True)

# A16 no media columns ------------------------------------------------------------------
save("A16_no_media.csv", base[["week_start", "new_policies", "avg_premium_index", "tax_season"]]
     .assign(week_start=base.week_start.dt.strftime("%Y-%m-%d")))
key["A16_no_media.csv"] = dict(steps="3", expected="block: no column can be a media channel", blocks=True)

# A17 empty and header-only files -----------------------------------------------------------
open(OUT + "A17_empty.csv", "w").close()
open(OUT + "A17_header_only.csv", "w").write(",".join(base.columns) + "\n")
key["A17_empty.csv"] = dict(steps="1", expected="block: file is empty", blocks=True)
key["A17_header_only.csv"] = dict(steps="1", expected="block: header found but 0 data rows", blocks=True)

# A18 Indian number formats and units -----------------------------------------------------
def inr(v):
    s = f"{int(v):d}"
    if len(s) <= 3: return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2: parts.insert(0, head[-2:]); head = head[:-2]
    if head: parts.insert(0, head)
    return ",".join(parts + [tail])
f = pd.DataFrame({"week_start": base.week_start.dt.strftime("%Y-%m-%d"),
    "new_policies": [inr(v) for v in base.new_policies],
    "tv_spend": [f"₹{v/1e7:.2f} Cr" if v else "-" for v in base.tv_spend],
    "search_spend": [f"{v/1e5:.1f} L" for v in base.search_spend],
    "social_spend": [inr(v) for v in base.social_spend],
    "discount_pct": [f"{x}%" for x in rng.integers(0, 15, 156)],
    "affiliate_spend": [f"({inr(v)})" if i in (20, 77) else inr(v) for i, v in enumerate(base.affiliate_spend)]})
save("A18_indian_number_formats.csv", f)
key["A18_indian_number_formats.csv"] = dict(steps="3 (with Phase B step 7)",
    formats=["1,20,000 lakh grouping", "₹4.50 Cr", "12.6 L", "'-' for zero", "12%", "(1,234) negative"],
    expected="columns recognised as numeric before role suggestion; tv_spend '-' -> 0; affiliate rows 22 and 79 negative -> flagged at step 12",
    blocks=False)

json.dump(key, open(OUT + "expected_phase_a.json", "w"), indent=2, ensure_ascii=False)
print("\n".join(sorted(key)))
