# Answer-key check: is the tool working?

A made-up but realistic export where **the right answer is known**, and what each screen should show
on it. Run it after any change to the tool, or on a new machine, and compare.

| File | What it is |
|---|---|
| `home_goods_mmm.csv` | the file to upload: an online home-goods shop, 3 regions × 156 weeks (2023-01-02 → 2025-12-22) |
| `truth_for_tool.json` | the answer key, loaded at step 32 (the tool's own format) |
| `home_goods_actuals_next13.csv` | 13 weeks after the data (2025-12-29 → 2026-03-23), for the Tracking screen |
| `truth_details.json` | everything else that is known: each channel's carryover and saturation, What-If answers, the Tracking answer |
| `make_dataset.py` | makes all of the above (`--seed` for a fresh set); `--check-plan` gives the true result of any spend split |

## The truth in one table

Orders are made additively, as Meridian models them: baseline (trend + seasonality) + holiday weeks
+ price + competitor promotions + four paid channels, each with carryover (geometric, 8 weeks) and
diminishing returns (Hill curve) + 2.5% noise. Revenue = orders × an order value of about $58.

| Channel | Spend (3 years) | Weeks with spend | True ROI ($ per $) | True next-$ ROI | True share of orders | True carryover (week 1) |
|---|---|---|---|---|---|---|
| TV | $7.72M | 84 (flighted) | **1.11** | 0.78 | 7.9% | 0.65 |
| Search | $6.43M | 156 | **1.76** | 0.77 | 10.4% | 0.15 |
| Social | $4.39M | 142 (a few pauses) | **1.11** | 0.70 | 4.5% | 0.35 |
| Display | $3.00M | 156 | **0.39** | 0.19 | 1.1% | 0.20 |
| Non-media (baseline, controls) | | | | | 76.1% | |

Controls: Price Index (−35 orders per point above 100, per 3M people), Competitor Promo (−250 orders
in a promo week, per 3M people), Holiday Week (+20%: Black Friday week and the two weeks before
Christmas). Population: North 4.2M, South 3.1M, West 2.4M.

## Planted problems

| Problem | Where | The tool should |
|---|---|---|
| Search Spend ($) is text: `$12,345.67` | every row | read it as numbers (step 7) |
| One exact duplicate row | South, week 2023-10-09 | drop it (step 8), and not let it hide the Region column (step 2) |
| One blank Price Index | West, week 2024-12-02 (true value 105.9) | label and fill it (step 10) |
| TV off in runs of 3–6 weeks | ~45% of weeks | keep them as real zeros (step 10); TV is flighted, so no "always-on" question |
| Social paused for 3–5 weeks a few times | 14 of 156 weeks | ask about the pauses (step 10); the right answer is **pause** |

## Step by step

"Seen" is what the tool showed on 2026-11-03 with the Quick sampler. Fitted numbers move a little
from run to run (the sampler is random); the checks in **bold** are the ones that matter.

### Before you start
1. Projects → **New project** (any name, "Read as" Generic) → **New run** → upload `home_goods_mmm.csv`.

### Phase A · Ingest (steps 1–6)
| Step | Expected | Seen |
|---|---|---|
| 1 Load file | 469 rows × 15 columns | ✓ |
| 2 Parse dates | **156 dates × 3 values of 'Region'**, 2023-01-02 → 2025-12-22, no question | ✓ |
| 3 Map column roles | KPI **Orders**; Revenue ($) as revenue for the KPI; 4 spend columns; TV GRPs, Search Clicks, Social Impressions as exposure; **Price Index, Competitor Promo, Holiday Week as controls**; Population; Region as geo. Click **Confirm** | ✓ (the only question in A–D) |
| 4 Pair spend with exposure | 4 channels: 3 paired, **display spend-only** | ✓ |
| 5 KPI type | a count; revenue per order = Revenue ÷ Orders each week; USD | ✓ |
| 6 Granularity | weekly, 156 weeks, **3 regions, modelled regionally**, weeks start Monday | ✓ |

### Phase B · Clean (steps 7–13)
| Step | Expected | Seen |
|---|---|---|
| 7 Coerce types | **469 formatted numbers read (e.g. '$15,928.84' in Search Spend ($))** | ✓ |
| 8 Duplicates | **1 exact copy row dropped; 469 → 468** | ✓ |
| 9 Resample | already weekly, Monday | ✓ |
| 10 Gaps | warn: **1 empty cell labelled and filled; 516 real zeros in media columns**; asks about social's 42 zero weeks in 9 runs: answer **pause** (the default) | ✓ |
| 11 Outliers | none | ✓ |
| 12 Impossible values | none | ✓ |
| 13 Money | 5 money columns, no inflation adjustment; revenue per order median **$58** | ✓ |

### Phase C · Profile (steps 14–20)
| Step | Expected | Seen |
|---|---|---|
| 14 Sufficiency | 156 weeks × 3 regions, adequate (468 rows, ~15.6 per parameter) | ✓ |
| 15 Activity | all 4 channels active | ✓ |
| 16 Spend variation | warn: **search hard to measure** (CV 0.28). True: search spend only follows the season | ✓ |
| 17–19 | pass: no channels moving together, no control caused by media, no spend timed with season flags | ✓ |
| 20 Profile | $21.5M total; largest TV (36%) | ✓ |

### Phase D · Configure (steps 21–27)
Keep every default. The truth matches the defaults: carryover up to 8 weeks (step 22), geometric
(step 23), diminishing returns for all four (step 23). Step 25 warns that search is on the default
prior; step 27 holds back the last 13 weeks (2025-09-29 → 2025-12-22).

### Phase E · Fit and validate (steps 28–36)
1. In step 32's card, **Load answer key** → `truth_for_tool.json`.
2. **Fit** with the Quick sampler (about 3 minutes; Full takes longer and moves less between runs).

| Step | Expected | Seen |
|---|---|---|
| 28 Prior check | pass | covers the actual 1,865,813 orders |
| 30 Convergence | **every R-hat below 1.1** | largest 1.06 |
| 31 Holdout | **test MAPE well under the same-week-last-year forecast's** | 1.5% vs 12.9% (R² 0.92) |
| 32 Plausibility + truth | **every true ROI inside its 90% range** | **4 of 4** ✓ — table below |
| 33 Learned from data | every channel narrower than its prior | ✓ |
| 36 Final fit | saved | ✓ |

| Channel | True ROI | Model (holdout fit) | 90% range | Inside |
|---|---|---|---|---|
| TV | 1.11 | 1.10 | 1.06–1.15 | ✓ |
| Search | 1.76 | 1.87 | 1.36–2.62 | ✓ |
| Social | 1.11 | 1.08 | 0.90–1.27 | ✓ |
| Display | 0.39 | 0.85 | 0.24–1.78 | ✓ (wide: the smallest channel) |
| Non-media share | 76% | 74% | | |

The "Mean absolute ROI error" line (32% here) is the average *relative* error, and display's 0.85
vs 0.39 makes most of it.

### Phase F · Model Summary (final fit, all 156 weeks)
| Channel | ROI (90%) | true | Next-$ ROI | true | Week-1 carryover | true | Share | true |
|---|---|---|---|---|---|---|---|---|
| TV | 1.10 (1.05–1.15) | 1.11 | 0.80 | 0.78 | 0.60 | 0.65 | 7.8% | 7.9% |
| Search | 1.77 (1.34–2.42) | 1.76 | 1.03 | 0.77 | 0.28 | 0.15 | 10.5% | 10.4% |
| Social | 0.98 (0.83–1.14) | 1.11 | 0.49 | 0.70 | 0.16 | 0.35 | 4.0% | 4.5% |
| Display | 0.94 (0.30–1.83) | 0.39 | 0.44 | 0.19 | 0.70 (0.32–0.94) | 0.20 | 2.6% | 1.1% |

**Expected:** TV and search ROIs within ~10% of the truth; the ranking display lowest; display's
ranges wide. Next-$ ROI and carryover are harder to recover than ROI: search's next-$ ROI is high
(the model's search curve saturates less than the truth's; step 16 warned about search).

### Phase G · What-If (step 45, "Work out exactly")
Exact answers are in `truth_details.json` → `what_if`, or from
`python make_dataset.py --check-plan '{"search": 1.2}'` (add `--period last_52`).

| Scenario | True change in orders | Seen (90% range) | Inside |
|---|---|---|---|
| TV −50%, last 52 weeks | **−23,549** | −23,064 (−24,528 to −21,663) | ✓ |
| Display to 0, all weeks | **−20,133** | −48,586 (−94,948 to −15,527) | ✓ (wide) |
| Search +20%, all weeks | **+15,513** | +21,258 (+17,481 to +25,411) | ✗ expected: search's response is over-estimated (step 16) |

### Phase H · Plan, Optimize (defaults: all weeks, fixed budget $21.54M, ±30%)
| | Seen |
|---|---|
| Budget kept | ✓ $21.54M |
| Split | display −30%, social −30%, search +35%, TV unchanged |
| Predicted gain | +13,765 orders (+4,018 to +23,279), 99% of draws better |
| **True gain of that split** (`--check-plan` with the multipliers shown) | **+2,968 orders, +$172k**: really better, but less than predicted |
| True best split under the same limits | display −30%, TV +8%, search +6%, social −2%: +8,307 orders, +$481k |

**Expected:** the optimizer cuts display (right: lowest true next-$ ROI) and its split truly beats
the current one. It also cuts social and over-predicts the gain, because it under-rates social and
over-rates search (Phase F).

### Future planning and Tracking
1. What-If → **Future** → keep the defaults (2025-12-29, 13 weeks, same weeks last year, built-in
   forecast) → **Set up**. Plan A appears in seconds.
2. Plan A → **Choose for tracking**.
3. **Tracking** (T) → **Upload actuals** → `home_goods_actuals_next13.csv`. In these weeks spend is
   the same weeks last year, **except TV at half**.

| Check | Expected | Seen |
|---|---|---|
| Built-in forecast | growth from the data ~+5.8% (true trend +5%/year); range ±12.9% (step 31) | ✓ |
| Upload | 13 weeks read, no column questions | ✓ |
| Spend vs plan | **TV −50%** (furthest channel); total −17% | ✓ |
| **Execution gap** | **true −4,762 orders** | **−4,795** ✓ |
| Alerts | amber **execution** naming TV; amber **spend off plan: TV** | ✓ both |
| Weeks inside the range | most or all | 13 of 13; gap −1.0% |

### Phase I · Export, refit
| Check | Expected | Seen |
|---|---|---|
| Model bundle | includes the model, answer key (`test/`) and future plans (`future/`) | ✓ 35 files |
| `python -m mmm_tool.export --verify <bundle.zip>` | no bad checksums; model table and config identical | ✓ |
| Register → Approve | v1 approved | ✓ |
| Tracking → **Start a refit** | a new run with **169 weeks** (156 + 13), reaches Phase D with no questions | ✓ |

## Making a fresh set

`python examples/answer_key_check/make_dataset.py --seed 12 --out /some/folder` makes a new file
with the same structure (different spend, noise and true numbers); its `truth_for_tool.json` and
`truth_details.json` hold that set's answers. `tests/test_answer_key_dataset.py` checks that the
committed files are what the generator writes and that Phases A–D read them as above.
