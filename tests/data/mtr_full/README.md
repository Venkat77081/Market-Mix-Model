# MTR_FULL: test the MMM tool from Phase A to Tracking

A synthetic motor insurer: 157 weeks of history (2023-01-02 to 2025-12-29), a planning period (2026-01-05 to 2026-03-30), a forecast for that period, and 13 weeks of actuals to track. Built from a known truth, so every step has a correct answer. Every number here is generated from `expected/` by `write_readme_mtr.py`.

**New compared with INS_FULL and HLT_FULL:** an Excel file with a notes sheet and title rows; exposure columns (GRPs, impressions,
clicks) whose cost per unit rose until Dec-2024; premium reported in ₹ lakh; a channel that ended (Print); a preset that doesn't apply
(tax season); future planning in Phases G and H; saved plans; Choose for tracking; and a Tracking run with an execution gap (Search
under-spent for 3 weeks) and an outside shock (a dealer supply shortage in March 2026 that's in no column).

## Files

| File | Upload at | What it is |
|---|---|---|
| `raw/MTR_FULL_raw.xlsx` | Phase A | Main export: sheet "Weekly", 2 title rows, header on row 3 |
| `raw/lift_tests.csv` | Step 26, run 2 | Search lift test |
| `raw/forecast_2026Q1.csv` | Future setup (Phase G/H) | The forecasting tool's output for the planning period: forecast and lower/upper |
| `raw/MTR_actuals_2026Q1.csv` | Tracking | 13 weeks of actuals, same columns as the main export, no title rows |
| `expected/expected_steps_1_27.json` | | Every expected value for steps 1–27 and the Tracking input |
| `expected/expected_weekly_clean.csv`, `model_input.csv` | | Tables after Phase B and after Phase D |
| `expected/reference_fits/*.json`, `tracking_model.csv` | | Reference Meridian results, plans and Tracking predictions |
| `expected/truth_targets.json`, `tracking_truth.csv`, `truth_for_tool.json` | | The truth for every later screen |
| `truth/` | | Reality: weekly values, contributions, parameters |

## Decision script

| Step | Prompt | Answer |
|---|---|---|
| 1 | Sheet / header | Sheet "Weekly", header on row 3 |
| 2 | Date format | Day-first (dd/mm/yyyy) |
| 3 | Roles | As in the step 3 table. **Branded Search Vol as Control on purpose** (step 18 must catch it). Comments → Excluded |
| 4 | Pairs | TV Spend ↔ TV GRPs, YouTube Spend ↔ YouTube Impr., Search Spend ↔ Search Clicks |
| 5 | KPI | New Policies; value per policy = Premium (₹ lakh) × 1,00,000 ÷ New Policies |
| 8 | Duplicates | Drop the 3 exact copies |
| 10 | Missing week, Print blanks | Interpolate everything in the missing week; Print blank after it ended → 0 |
| 11 | Outlier | Keep the fleet-deal week, add event_flag |
| 15–17 | Activity, variation, together | Keep all 6 channels |
| 18 | Marketing-caused | Exclude Branded Search Vol |
| 21 | Calendar | Use the file's Festive Season; **don't add tax season or the Diwali preset**; add FY end |
| 22–27 | Model setup | 8 weeks, geometric, 12 knots, no priors, holdout 13 weeks; lift test in run 2 only |
| 29 | Sampler | 4 chains, 500 adapt, 500 burn-in, 1,000 kept, seed 42 |
| 57 | Approve | Run 2 |
| G/H future | Period | Future · 05-Jan-2026 → 30-Mar-2026; pattern = same weeks last year; media cost change 0%; forecast = upload `raw/forecast_2026Q1.csv` |
| G future | Plan B | Sliders TV 80%, Search 120%, others 100%; save as plan B |
| H future | Plan C | Fixed budget = plan A spend, every channel −30% / +50% (Print has no spend in these weeks); save as plan C |
| Plans | Track | **Choose plan B for tracking** |
| Tracking | Actuals | Upload `raw/MTR_actuals_2026Q1.csv` |

## Phase A: steps 1–6

**Step 1:** Excel with 2 sheets; data on 'Weekly', 2 title rows, header on row 3: **159 rows × 18 columns**.

**Step 2:** 159 rows, **156 unique dates**, 2023-01-02 to 2025-12-29. Day-first proven by 97 rows (day > 12); 56 ambiguous; 6 with day = month.

**Step 3 · Roles**

| Column | Expected role |
|---|---|
| Week (dd/mm/yyyy) | Date |
| New Policies | KPI (a count) |
| Premium (₹ lakh) | Revenue for KPI (in ₹ lakh: × 1,00,000) |
| TV Spend | Paid media spend |
| TV GRPs | Exposure for TV |
| YouTube Spend | Paid media spend |
| YouTube Impr. | Exposure for YouTube |
| Search Spend | Paid media spend |
| Search Clicks | Exposure for Search |
| Meta Spend | Paid media spend (text with Indian commas) |
| PolicyBazaar Commission | Paid media spend ('Commission' is a spend word) |
| Print Spend | Paid media spend (blank after it ended) |
| Branded Search Vol | Control (DECISION: set as Control on purpose, so step 18 can catch it) |
| Car Sales (SIAM) | Control |
| Fuel Price Idx | Control |
| Competitor Discount | Control (event flag) |
| Festive Season | Control (season flag) |
| Comments | Excluded (free text) |

**Step 4 · Pairs:** TV Spend ↔ TV GRPs, YouTube Spend ↔ YouTube Impr., Search Spend ↔ Search Clicks; Meta, PolicyBazaar, Print spend-only. Media costs rose until dec-2024 (tv +8%/yr, youtube +5%, search +6%), so exposure is not proportional to spend.

**Step 5:** KPI New Policies (non-revenue); value per policy = Premium × 1,00,000 ÷ policies, median **₹14,890**. **Trap:** premium is in lakh: without × 1,00,000 the value per policy is about ₹0.14 and every ROI is 1,00,000× too small.

**Step 6:** weekly, Monday, one national series.

## Phase B: steps 7–13

| Step | Expected |
|---|---|
| 7 | Meta Spend: text with Indian grouping (e.g. 7,99,000): 159 cells |
| 7 | Print Spend: blank cells: 79 cells |
| 7 | Premium (₹ lakh): decimals: 159 cells |
| 8 | 159 → **156** rows; exact duplicates 2023-05-22, 2024-05-13, 2025-07-21; no split weeks, no revisions |
| 9 | Already weekly; no resampling |
| 10 | Missing week **2024-09-16**: policies 2,854, Search ₹1,446,000, Meta ₹885,500, TV 0 (off on both sides). Print ended after 2024-06-24: 79 weeks → 0. Real zero weeks: TV 68, YouTube 92, Print 48 (total 208) |
| 11 | Median 3,081, MAD 130; exactly **one flagged: 2024-07-15** (z 9.4), the fleet deal; next |z| 3.34 |
| 12 | Nothing to flag |
| 13 | No adjustment; premium in lakh is converted when the value per policy is calculated |

Every non-interpolated cell of `expected/expected_weekly_clean.csv` matches the truth exactly.

## Phase C: steps 14–20

**Step 14:** 157 weeks, 6 channels, 6 controls: 4.24 per parameter (3 per channel) → thin; 3.65 (4 per channel) → thin.

| Channel | Weeks with spend | Share of spend | Spend variation | Badge | Festive corr. |
|---|---|---|---|---|---|
| TV | 88 (56.1%) | 32.9% | 0.999 |  | 0.13 |
| YouTube | 64 (40.8%) | 6.6% | 1.324 |  | 0.16 |
| Search | 157 (100.0%) | 22.3% | 0.118 | hard to measure | −0.24 |
| Meta | 157 (100.0%) | 13.7% | 0.356 |  | 0.00 |
| PolicyBazaar | 157 (100.0%) | 22.6% | 0.180 | hard to measure | −0.01 |
| Print | 30 (19.1%) | 1.9% | 2.110 |  | −0.06 |

Step 15: all active (Print passes (19% of weeks, 1.9% of spend) although it stopped in Jun-2024: keep it). Step 17: largest correlation 0.41 (TV and YouTube), VIF ≤ 1.29: pass. Step 19: nothing flagged. Step 20: total spend ₹111.39 Cr.

**Step 18**

| Control | Level rule (≥ 0.6) | Change rule (≥ 0.4) |
|---|---|---|
| branded_search | 0.71 → **flag** | 0.68 → **flag** |
| car_sales | 0.43 → clear | 0.18 → clear |
| fuel_index | 0.48 → clear | 0.17 → clear |
| competitor_discount | 0.18 → clear | 0.19 → clear |

Only Branded Search is flagged by either rule: exclude it.

## Phase D: steps 21–27

**Step 21:** preset tax season (41 weeks) → **don't add**; file's Festive Season (16 weeks) → use; FY end (8 weeks) → add. Controls: car_sales, fuel_index, competitor_discount, festive, fy_end, event_flag.

**Steps 22–27:** 8 weeks; geometric; 12 knots at weeks 0, 14, 28, 42, 56, 70, 85, 99, 113, 127, 141, 156; no priors; run 2 lift test Search 2025-06-02 → 2025-07-27, ROI 2.74 (2.34–3.14; true 2.739); holdout 2025-10-06 → 2025-12-29 (144 training weeks).

## Phase E: steps 28–36

**Step 28:** actual 484,692 policies; prior 90% 486,466 – 761,627 → **Pass with warning**; run 2 516,456 – 772,916 → **Pass with warning**.

**Step 30:** largest R-hat 1.0071 (holdout), 1.0029 (run 1 final), 1.0107 (run 2 final): pass, no retry.

**Step 31**

| | Default knots | Knots inside training weeks |
|---|---|---|
| Train MAPE | 1.9% | 1.9% |
| **Test MAPE** | **28.1%** | **1.3%** |
| Test bias | +28.1% | +0.2% |
| Naive: same week last year | 6.9% | 6.9% |

Expected verdict: **FAIL** with default knots, **Pass** with knots inside the training weeks.

**Step 32–33 · run 1 (final fit)**

| Channel | ROI (90%) | True ROI | Truth in range? | Share | True share | Narrower by |
|---|---|---|---|---|---|---|
| TV | 1.26 (0.93–1.63) | 1.31 | ✓ | 6.4% | 6.6% | 87% |
| YouTube | 1.46 (0.65–2.55) | 1.60 | ✓ | 1.5% | 1.6% | 62% |
| Search | 3.36 (1.04–5.78) | 2.80 | ✓ | 11.6% | 9.6% | 9% |
| Meta | 1.19 (0.29–2.63) | 1.41 | ✓ | 2.5% | 3.0% | 53% |
| PolicyBazaar | 1.37 (0.28–3.20) | 3.00 | ✓ | 4.8% | 10.5% | 45% |
| Print | 0.90 (0.25–1.85) | 0.60 | ✓ | 0.3% | 0.2% | 69% |
| **Non-media** | | | | **73.0%** | **68.5%** | |

**Run 2 (Search lift test) · approve this one at step 57**

| Channel | ROI (90%) | True ROI | Truth in range? | Share | True share | Narrower by |
|---|---|---|---|---|---|---|
| TV | 1.28 (0.95–1.66) | 1.31 | ✓ | 6.5% | 6.6% | 87% |
| YouTube | 1.43 (0.65–2.60) | 1.60 | ✓ | 1.4% | 1.6% | 61% |
| Search | 2.86 (2.45–3.33) | 2.80 | ✓ | 9.8% | 9.6% | 59% |
| Meta | 1.22 (0.29–2.67) | 1.41 | ✓ | 2.6% | 3.0% | 52% |
| PolicyBazaar | 1.40 (0.29–3.33) | 3.00 | ✓ | 4.9% | 10.5% | 43% |
| Print | 0.89 (0.25–1.85) | 0.60 | ✓ | 0.3% | 0.2% | 69% |
| **Non-media** | | | | **74.4%** | **68.5%** | |

Tolerances against these reference values: TV ROI ±15%, others ±40%, non-media ±4 points, same ✓/✗ pattern. Expected: Search moves from 3.36 to 2.86 (truth 2.80); PolicyBazaar has no lift test and stays as run 1 left it.

## Phase F: Model Summary (run 2)

| Channel | Next-₹ ROI | True next-₹ ROI | Carryover alpha (90%) | True alpha |
|---|---|---|---|---|
| TV | 0.87 | 0.83 | 0.60 (0.50–0.69) | 0.60 |
| YouTube | 0.94 | 0.99 | 0.38 (0.09–0.68) | 0.40 |
| Search | 1.70 | 1.40 | 0.17 (0.01–0.41) | 0.10 |
| Meta | 0.58 | 0.72 | 0.47 (0.08–0.87) | 0.30 |
| PolicyBazaar | 0.74 | 1.45 | 0.34 (0.03–0.82) | 0.10 |
| Print | 0.57 | 0.39 | 0.45 (0.07–0.90) | 0.45 |

## Phase G: What-if, Past mode (all 157 weeks, run 2)

| Scenario | Spend change | True premium change | Model (90%) | Share of draws better |
|---|---|---|---|---|
| tv +20% | +7.34 Cr | +5.74 Cr | +6.07 Cr (+4.35 to +7.88) | 100% |
| search -30% | −7.43 Cr | −12.28 Cr | −14.38 Cr (−17.64 to −10.51) | 0% |
| agg +50% | +12.61 Cr | +14.66 Cr | +7.70 Cr (+1.08 to +20.09) | 100% |
| meta off | −15.29 Cr | −21.53 Cr | −18.54 Cr (−40.72 to −4.42) | 0% |
| all -10% | −11.14 Cr | −12.72 Cr | −11.45 Cr (−14.75 to −8.97) | 0% |
| all +10% | +11.14 Cr | +11.63 Cr | +10.63 Cr (+8.27 to +13.74) | 100% |

The exact result (step 45) should land within about ±10% of the model column.

## Phase H: Optimize, Past mode (last 52 weeks, fixed budget, −30% / +50%)

| Channel | True best | Run 2 |
|---|---|---|
| TV | ×0.72 | ×0.99 |
| YouTube | ×0.94 | ×1.11 |
| Search | ×1.35 | ×1.50 |
| Meta | ×0.70 | ×0.70 |
| PolicyBazaar | ×1.23 | ×0.71 |
| Budget | ₹37.08 Cr | ₹37.09 Cr |

True gain ₹1.46 Cr; run 2 expects ₹3.0 Cr. Print had no spend in these weeks, so it's left out.

## Future mode (Phases G and H): 05-Jan-2026 → 30-Mar-2026

Setup: pattern from 2025-01-06 → 2025-03-31 (same ISO weeks); media cost change 0% (costs frozen since Jan-2025); forecast file total **41,610 policies** (the forecast assumes last year's spend pattern, so plan A's total equals it exactly).

| Plan | How made | Spend | Change vs A (true) | Change vs A (model, 90%) | Total expected (model) |
|---|---|---|---|---|---|
| A | auto: 100% everywhere | ₹8.75 Cr | +0 policies | +0 (+0 to +0) | 41,610 |
| B | What-if sliders: TV 80%, Search 120% | ₹8.68 Cr | +30 policies | +119 (−47 to +273) | 41,729 |
| C | Optimizer: fixed budget, −30% / +50% | ₹8.75 Cr | −161 policies | +460 (−205 to +949) | 42,070 |

Plan C multipliers (run 2): TV ×0.95, YouTube ×1.32, Search ×1.50, Meta ×0.70, PolicyBazaar ×0.70. True best re-split: TV ×0.73, YouTube ×1.21, Search ×1.24, Meta ×0.70, PolicyBazaar ×1.19, true gain ₹0.27 Cr. Checks: plan A change = 0 exactly; plan A total = forecast total; the optimizer's plan C spend = plan A spend within 0.1%.

**Why plan B is tracked, not C:** the model expects plan C to add +460 policies, but on the truth it changes them by -161 because it cuts PolicyBazaar, which run 2 still under-measures. The model's own range for C (-205 to +949) includes a loss, so a careful user would hold off. Tracking plan B keeps the Tracking expectations independent of the model.

**Choose for tracking: plan B.** Expected: only one plan tracked for this period; a decision-log entry with plan, model version and totals; plan B's week-by-week predictions frozen.

## Tracking (actuals = `raw/MTR_actuals_2026Q1.csv`, plan B tracked)

**Input:** 13 rows, 2026-01-05 → 2026-03-30; same mapping as the main file (no prompts); Print blank → 0. Spend vs plan overall **-2.8%**.

| Week | Search vs plan | Forecast | Predicted (plan) | Predicted at actual spend: true / model | Actual | Gap vs prediction | Status (truth) |
|---|---|---|---|---|---|---|---|
| 2026-01-05 | -0.1% | 3,301 | 3,308 | 3,299 / 3,307 | 3,249 | -1.5% | inside |
| 2026-01-12 | +0.7% | 3,175 | 3,180 | 3,173 / 3,181 | 3,187 | +0.4% | inside |
| 2026-01-19 | -1.1% | 3,283 | 3,285 | 3,276 / 3,282 | 3,221 | -1.7% | inside |
| 2026-01-26 | -40.8% | 3,071 | 3,077 | 3,006 / 3,008 | 2,966 | -1.3% | inside |
| 2026-02-02 | -39.6% | 3,157 | 3,171 | 3,088 / 3,085 | 3,025 | -2.1% | inside |
| 2026-02-09 | -39.3% | 3,050 | 3,071 | 2,989 / 2,981 | 3,003 | +0.5% | inside |
| 2026-02-16 | -0.3% | 3,108 | 3,114 | 3,103 / 3,104 | 3,085 | -0.6% | inside |
| 2026-02-23 | -0.3% | 3,289 | 3,286 | 3,282 / 3,282 | 3,293 | +0.3% | inside |
| 2026-03-02 | -0.6% | 3,285 | 3,283 | 3,276 / 3,279 | 2,905 | -11.3% | below |
| 2026-03-09 | -0.1% | 3,121 | 3,127 | 3,117 / 3,125 | 2,812 | -9.8% | below |
| 2026-03-16 | -1.3% | 3,375 | 3,388 | 3,380 / 3,386 | 3,014 | -10.8% | below |
| 2026-03-23 | -1.8% | 3,186 | 3,206 | 3,199 / 3,204 | 2,888 | -9.7% | below |
| 2026-03-30 | -0.0% | 3,209 | 3,234 | 3,227 / 3,233 | 2,928 | -9.3% | below |

**Totals:** actual 39,576; predicted at actual spend 41,415 (true) / 41,457 (model); execution gap −225 (true) / −272 (model); model/outside gap −1,839 (true).

**Expected alerts** (default rules: spend ±10% for 2+ weeks; outside the range 3+ weeks):
- **Amber, from 2026-02-02:** Search spend about 40% under plan for 3 weeks (2026-01-26 → 02-09). Execution gap about −70 policies a week; policies stay inside the range because Search drives only a small part of each week.
- **Red, from 2026-03-16:** actual below the range for the 3rd week running (since 2026-03-02), about 10% under prediction **while spend matched the plan**, so the wording must be 'spend matched the plan, so the gap is not explained by media'. The cause (a dealer supply shortage) is in no column: the user adds it with **Add an event**.
- Tiles at week 13: 8 of 13 weeks inside the range, 5 below.

The expected range here is the forecast file's lower/upper (±4%) shifted by the media change; a tool that also adds the model's media range gets a slightly wider band, which changes no status above.

## Phase I: refit with the tracked weeks

**Start a refit** → Phases A–E rerun on 157 + 13 = **170 weeks** (`expected/model_input_refit.csv`), every earlier answer reused, and the 'dealer supply shortage' event offered as a control flag for 2026-03-02 → 2026-03-30. Without that flag the refit treats the shock as a baseline drop.

## What this file should reveal about the tool

| Step | What to watch |
|---|---|
| 1 | Finds the data sheet and the header on row 3 |
| 3–5 | Exposure columns paired with spend; premium in lakh multiplied by 1,00,000 |
| 10 | Print's blanks after June 2024 are "ended" (0), not missing |
| 21 | A preset that doesn't apply (tax season) can be declined |
| Future setup | Forecast file loaded; plan A change exactly 0; plan A total = forecast |
| Plans | B from the what-if, C from the optimizer, both in one Saved plans table |
| Tracking | Execution gap (Search) separated from the outside gap (shock); amber then red alert with the right wording |
| Refit | Event added in Tracking appears as a control flag |
| G/H/Tracking | Meridian ignores `media_spend` in `new_data` for outcomes: scenarios must scale the exposure columns (GRPs, impressions, clicks), not only spend. A tool that scales only spend shows no change for TV, YouTube and Search |
| H future | Plan C cuts PolicyBazaar to ×0.70 because run 2 still under-measures it (no lift test); the true best raises it to ×1.19. A lift test on PolicyBazaar is the fix |

## Rebuilding

```
python3 gen_mtr.py          # truth and raw files (seed 77)
python3 check_mtr.py        # steps 1-27 and the Tracking input, checked against the truth
python3 truth_mtr.py        # truth for Phases F-H, future plans and Tracking
python3 ref_mtr.py          # 4 reference Meridian runs, plans and Tracking predictions (about 10 minutes on 2 cores)
python3 truth_mtr.py        # again: adds the true effect of the model's plan C
python3 write_readme_mtr.py # this file
```
