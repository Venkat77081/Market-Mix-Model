# Meridian MMM Tool

Built so far: **Phase A, Ingest (steps 1–6)**, **Phase B, Clean (steps 7–13)**, **Phase C,
Profile (steps 14–20)**, **Phase D, Configure (steps 21–27)**, **Phase E, Fit and validate
(steps 28–36)** and the **Model Summary screen (Phase F, steps 37–42)** from the roadmap, a web UI for all of them, and a clickable design mock of the later screens in `mock/` (sample data only,
not connected).

## Web UI

```bash
pip install -r requirements.txt
python -m mmm_tool.api     # then open http://127.0.0.1:8000
```

Or install it as a package, which also adds the `mmm-ui` and `mmm-ingest` commands:
`pip install -e ".[ui]"` then `mmm-ui`.

`/` is the landing page (what the tool does, the phases, the validation gates, and your recent runs);
`/app` is the Phase A screen. Upload a CSV or Excel export (or pick a sample), answer the highlighted decisions, and download the
ingested table when all six steps pass. Runs and answers are saved under `./runs/` (set `MMM_RUNS_DIR`
to change it), so a page reload keeps your place. The API is under `/api/` (docs at `/docs`).

## Command line and tests

```bash
pip install -r requirements-dev.txt
pytest                                      # Phase A and API tests on synthetic insurance exports
python -m mmm_tool.ingest data.xlsx         # stops at the first decision prompt
python -m mmm_tool.ingest data.xlsx --answers answers.json
python -m mmm_tool.ingest data.xlsx --accept-defaults --json
```

| Step | Module | Blocks the run |
|---|---|---|
| 1 Load file | `ingest/load.py` | no (sheet prompt uses a default) |
| 2 Parse dates | `ingest/dates.py` | when every date is ambiguous, or some can't be read |
| 3 Map column roles | `ingest/roles.py` | until roles are confirmed and date, KPI and 1+ spend are set |
| 4 Pair spend with exposure | `ingest/pairing.py` | no |
| 5 Set KPI type | `ingest/kpi.py` | no |
| 6 Detect granularity | `ingest/granularity.py` | only for a regional model with no population column |

Answers are keyed by decision id (`sheet`, `date_order`, `unparseable_dates`, `roles`, `pairs`,
`unpaired:<col>`, `kpi_type`, `revenue_per_kpi_value`, `geo_mode`). `roles` takes a partial
`{column: role}` override. `PhaseAResult.coord_to_columns()` gives the field lists for Meridian's
`CoordToColumns` plus the `*_to_channel` maps; `to_meridian()` builds the object when Meridian is installed.

## How column roles are suggested (step 3)

Each column's suggested role comes from, in order:

1. **Roles you confirmed before.** Every time you confirm step 3, the roles are saved (UI:
   `runs/role_memory.json`; command line: `.mmm/role_memory.json`; or set `MMM_ROLE_MEMORY`).
   A later file with the same column name ("TV Spend" matches "tv_spend") gets your earlier role,
   marked *remembered*. Clear them with the **Forget remembered roles** button,
   `mmm-ingest --forget`, or by deleting the file.
2. **Your words** in `roles.yaml` at the project root (or `MMM_ROLES_FILE`), checked before the
   built-in words. Example: `kharcha` means spend, and `policybazaar` is the aggregator channel.
   Edits apply on the next run; no restart needed.
3. **Keyword packs** you turn on, checked after your words: words for one industry or country, in
   `mmm_tool/ingest/packs/`. Two come with the tool: `insurance` (policies, premium, GWP, vehicle
   registrations; a premium index lowers sales) and `india` (₹/INR/Rs, lakh and crore, festivals;
   suggests INR when the file shows no currency). Turn them on with `packs: [india, insurance]` in
   `roles.yaml` (this project's `roles.yaml` does), or `MMM_KEYWORD_PACKS=india,insurance`. A pack
   of your own is a `.yaml` file in the same format, named by its path.
4. **Built-in words** in `mmm_tool/ingest/role_keywords.yaml`, which documents every field. They hold
   only words that mean the same in any industry or country.
5. **The values**: text with a few repeating values is a region; a 0/1 column is a flag.
   Anything else becomes a low-confidence control.

You still confirm every role before the run continues.

**Currency (step 5).** The currency comes from the money columns' names and cells (`TV Spend ($)`,
`spend_eur`, `₹4,500`), or a pack's default, and you can change it (`currency`: `INR`, `USD`,
`EUR`, `GBP`, … or `none`). Every amount and the ROI unit use it, with its digit grouping: lakh and
crore for INR (₹12,34,567; ₹2.3 Cr), thousands and millions otherwise ($1,234,567; $2.3M). With no
currency, amounts are plain numbers and ROI is "return per unit of spend".

## Phase B: clean and align to weeks (steps 7–13)

Runs automatically once Phase A is complete, and produces one complete row per week (per region in a
regional model) with no empty cells, which is what Meridian reads. Download it from the UI, or
`python -m mmm_tool.ingest file.csv --out clean.csv` (add `--phase a` to stop after ingest).

| Step | What it does | You decide | Blocks |
|---|---|---|---|
| 7 Coerce types | `N/A`, `NULL`, blanks → missing; `-` → 0 in money and media columns, missing elsewhere; `1,20,000`, `₹4,500`, `(250)`, `12.6 L` → numbers; yes/no → 1/0 | nothing | no |
| 8 Collapse duplicate weeks | groups rows by date, or date + region when there is a region column (one row per region is normal); drops exact copies (equal after rounding to 6 significant digits), never adding them up; judges every other multi-row date against a normal week (below); combines split rows by each column's rule (below); adds regions up for a national model | what to do with a likely revision or an unclear week (`multi_row_weeks`: `later`, `sum` or `mean`); reach and any column that differs between split rows (`aggregation_rules`) | a likely revision or unclear week |
| 9 Resample to weeks | days → weeks; weekly dates moved to the nearest week start; partial edge weeks kept or dropped; weeks with days missing inside the period scaled, interpolated or kept, and flagged in `partial_week_flag` | `week_start`, `partial_weeks`, `short_weeks` | no |
| 10 Label every gap | adds absent weeks; labels empty cells not started, ended or missing; a zero stays zero. KPI gaps at the edges are trimmed; inside, interpolated + `kpi_imputed` flag. Media blanks are judged per column: a column that never records 0 and is often blank gets 0 suggested, other blanks are interpolated; spend in absent weeks is interpolated unless you say the channels were off | `kpi_edge_gaps`, `kpi_gaps`, `media_gaps` (one answer, or one per column plus `missing_weeks`) | KPI gaps inside; absent weeks |
| 11 Flag outliers | robust z-score on the KPI, weeks above 3.5 | event flag or keep, per week (`outliers`) | no |
| 12 Check impossible values | negative spend, KPI zero in a trading week, an index or price jumping 10x, a spend or exposure cell 10x the weeks around it | zero, keep, interpolate or a corrected number, per cell (`negative_values`, `kpi_zero_weeks`, `control_jumps`) | negative values |
| 13 Normalise money | a level change close to a power of 10 (lakhs, then rupees) converted; crore/lakh columns → ₹; mixed currency warning; optional inflation restatement; then revenue per KPI is calculated once (below) | `level_shifts`, `money_units`, `inflation` (`{"base_year": 2025, "annual_rate_pct": 5}`) | no |

**Step 8: split or revision.** Each date (+ region) with more than one row is compared with a
normal week: the median of each spend and KPI column over the 4 single-row weeks on each side, same
region, leaving other multi-row weeks out. The KPI decides, because it moves least from week to
week while spend runs in flights; total paid spend is used when the KPI has no normal level.

- The rows add up to about one normal week (within ±30%): a **split**, added up by the rules below.
  Identical rows that together make one normal week are an even split, not copies.
- Each row is about a full normal week: a **likely revision**. Asked: Keep the later row (default),
  Add up or Average, showing both rows and the normal week. It is never added up without your choice.
- Anything else, or a KPI split whose rows all carry the same full week of spend: asked the same way.

**Step 8: how split rows combine.** From the step 3 role and the values only; a dropdown only where
there is a real choice, and the rule shown is the rule applied.

| Column | Shown as | Rule |
|---|---|---|
| Paid spend, KPI, revenue column | fixed | Add up |
| Exposure: impressions, GRPs, clicks | fixed | Add up |
| Reach | dropdown, default Maximum | true combined reach is between the larger row and the sum (audiences overlap) |
| Frequency | fixed | average weighted by the channel's impressions (by reach without them) |
| 0/1 columns | fixed | Maximum |
| Any other column the same on every row of every split week | fixed, "Keep (same on every row)" | first row |
| Any other column that differs in at least one split week | dropdown, default Average, listing those weeks | Add up / Average / Maximum |

The only red is Add up on a column that is the same on every row of every split week (it would double).

**Revenue per KPI** is calculated once, after step 13, from the cleaned columns, and never merged,
interpolated or outlier-checked as a ratio. Step 5 records only where it comes from:

- the KPI is revenue: no revenue per KPI;
- a revenue column (premium beside policies): revenue ÷ KPI per date (+ region); weeks where the KPI
  is 0 use the whole file's total revenue ÷ total KPI and are listed in a warning;
- a value-per-unit column: turned into a revenue total (value × KPI) before rows are combined, then as above;
- a fixed ₹ value per unit you type: the same in every week (it is not restated for inflation).

The revenue total is then dropped: Meridian gets the KPI and revenue per KPI. When step 12 fills or
corrects a zero-KPI week, a zero revenue in that week is filled with it.

The unit words in step 13 are in `role_keywords.yaml` (million, thousand, billion) and the india pack (lakh, crore); add your own in `roles.yaml`. The thresholds Phase B uses (when blanks count as
zero spend, the 10x jump ratio, how close to a power of 10 a level shift must be) are in
`mmm_tool/clean/cleaning.yaml`.

## Answer keys

`tests/data/phase_a_tests/` holds 19 test exports and `expected_phase_a.json`, their answer key.
`python tests/expected_phase_a.py` prints a check-by-check report for every file; the same checks run
in `pytest` (`tests/test_expected_phase_a.py`).

`tests/data/phase_b_tests/` holds 13 Phase B exports, `expected_phase_b.json` and a cleaned `truth/`
file for each. `python tests/expected_phase_b.py` reports on them (`tests/test_expected_phase_b.py`
in `pytest`). To test on data the code has never seen, make a fresh set with
`python gen_phase_b.py --seed N --out DIR` (run from `tests/data/phase_b_tests/`) and point the checks
at it with `PHASE_B_DIR=DIR`.

## Phase C: profile and sufficiency gates (steps 14–20)

Runs once Phase B is complete. Decides whether the data can support an MMM and warns about each
channel before fitting. Channel findings become badges shown beside that channel's ROI later.

| Step | Check | Threshold | Action |
|---|---|---|---|
| 14 Data sufficiency | weeks of history; observations per parameter | under 104 weeks or under 3 per parameter: refused; 104–155 weeks thin; 3–6 per parameter needs strong priors | refuse, or warn |
| 15 Channel activity | weeks with spend; share of spend | under 10% of weeks or 1% of spend | merge into Other, drop, or keep (`rare_channels`) |
| 16 Spend variation | CV of weekly spend (median across regions in a regional model) | under 0.3 | "hard to measure" badge |
| 17 Channels moving together | correlation; VIF | above 0.7; above 10 | combine or keep, per pair (`together:a|b`) |
| 18 Marketing-caused variables | control vs spend 1–2 weeks earlier | above 0.6 | exclude or keep (`mediator:col`) |
| 19 Spend timed with seasons | spend vs 0/1 season flags | above 0.6 | warning and badge |
| 20 Profile report | spend share, KPI with spend overlay, activity heatmap | | report |

Parameters are counted as 3 per paid or unpaid media channel (effect, carryover, saturation), 1 per
control or lever, one baseline knot per quarter and one intercept per region; the counts and every
threshold are in `mmm_tool/profile/context.py`. Samples: `ins_short`, `ins_rare`, `ins_flat`,
`ins_collinear`, `ins_mediator`, `ins_season`.

## Phase D: configure the model (steps 21–27)

Runs once Phase C is complete and completes on its own: every step has a default. You add business
calendars and past lift tests, and can change any setting. The result is a JSON model configuration
(download it from the UI, or `--config file.json` on the command line) and the table Meridian fits.

| Step | Default | You can change | Meridian setting |
|---|---|---|---|
| 21 Calendar flags | no preset; the presets in `mmm_tool/configure/calendars.yaml` are India insurance (tax season, financial year-end, Diwali and festive weeks) and US retail (Black Friday and Cyber week, Christmas shopping) | the preset (`calendar_preset`), its events on or off (`calendar_events`), your own date ranges (`custom_events`); each control's effect on sales, Either / Raises / Lowers (`control_direction`) | added to `controls`; direction: `gamma_c` |
| 22 Carryover length | 8 weeks | 2–13 (`max_lag`) | `max_lag` |
| 23 Carryover shape | geometric; binomial suggested for TV and outdoor; fade speed Don't know; diminishing returns Yes | per channel: shape (`adstock_shape`), how fast it fades (`carryover_speed`), diminishing returns (`diminishing_returns`) | `adstock_decay_spec`, `alpha_m`, `saturation_spec` |
| 24 Baseline flexibility | one knot per quarter | `knots`, with a live preview (below); amber above weeks/8, red above weeks/6 | `knots` |
| 25 ROI priors | average return (ROI), Meridian's default prior | what you know (`prior_type`: `roi`, `mroi` or `contribution`) and a 90% range per channel (`roi_ranges`) | `media_prior_type`, `roi_m` / `mroi_m` / `contribution_m` (and `_rf`) |
| 26 Lift-test calibration | none | tests: channel, dates, ROI and 90% range (`lift_tests`); several per channel are combined | prior + `roi_calibration` |
| 27 Holdout | last 13 weeks | 8–26 (`holdout_weeks`) | `holdout` |

**Step 24 preview.** Before you choose, the step shows what each number of knots does. Knot
placement and weights come from Meridian's `knots.get_knot_info`. For every K from 1 to weeks/4, a
quick regression splits weekly sales into baseline (the knots), controls and media: knot levels
are not penalised (Meridian's knot prior is very wide), controls and media get a light ridge, media
effects are held at zero or above, and media and controls are centred, so the knots own the
average level. The chart draws the baseline, with the burst channels as strips underneath. The
slider redraws in the browser with no server call; nothing is saved until **Apply**, which
records the choice in the decision log and marks a Phase E fit out of date. Three cards:

1. *Media's share of the ups and downs*: the share of the fitted week-to-week movement that the
   preview gives media, for every K. A guide, with no colour.
2. *Burst credit kept*: burst channels (spend CV 0.5 or more in step 16, and at least 5% of paid
   spend) keep this share of their total contribution compared with a straight-line baseline
   (2 knots). Green at 90% or more, amber 75–90%, red below; each channel's own figure is
   listed. White when there is no burst channel.
3. *Trend left out*, with one knot: a linear trend in sales after controls (|t| above 2).

**Suggest** shows Meridian's automatic knot selection, with card 2's value for it; **Preview** loads
it (it can lie beyond the slider) and **Apply** saves it. The preview is a guide; Meridian
estimates the real baseline in step 29. Code: `configure/baseline.py`; property tests on synthetic
data with a known truth in `tests/test_step24.py` and `tests/gen.py`.

**Calendar presets (step 21).** A `calendars.yaml` at the project root (or `MMM_CALENDARS`) adds
presets of your own and sets the one step 21 suggests (`default_preset`); this project's file
suggests India insurance. Without it, step 21 starts with none.

**Prior choices (steps 21, 23, 25).** Every choice goes into the model configuration, the
decision log (for example `#23 search fades fast (you)`) and the export; changing one marks a
Phase E fit out of date. Priors are float64 `meridian.backend.tfd` distributions, with one value
per channel or control looked up by name in `input_data`'s order, never by the order on screen.

- *How fast does it fade?* (23): "share of the effect still working next week". Don't know keeps
  Meridian's Uniform(0, 1), i.e. Beta(1, 1); Fast, Medium and Slow are Beta(2, 8), Beta(4, 6) and
  Beta(7, 3) on `alpha_m` (mean 20%, 40%, 70%). For binomial decay the share is
  `(1 - 1/W) ** (1/alpha - 1)` with W = max_lag + 1 (Meridian's `_map_alpha_for_binomial_decay`),
  so the preset is converted: a Beta on alpha whose median gives the same share.
- *Diminishing returns* (23): No sets `saturation_spec={channel: "none"}` (a straight response
  curve). No is hinted only for a channel under 5% of paid spend whose highest week is under 2x its
  typical week, and choosing it warns that the optimizer will add budget up to the channel's limit.
- *I know about each channel's…* (25): average return (`roi`), return on the next rupee (`mroi`)
  or share of sales (`contribution`), with a 90% low-high range per channel. Returns are LogNormal
  with mu = ln sqrt(low x high) and sigma = ln(high/low) / (2 x 1.645); a share is a Beta whose
  5th and 95th percentiles are solved to match the range (and checked). Blank channels keep
  Meridian's default. Without a ₹ value per KPI (step 5), only share of sales is offered. The
  table shows the median and 90% range of the distribution actually built. Lift-test calibration
  (26) needs the ROI type.
- *Effect on sales* (21): Either / Raises / Lowers per control and calendar flag. A direction is a
  TruncatedNormal(0, Meridian's default `gamma_c` scale) cut at zero. Names suggest one (price,
  premium, competitor: lowers; season, festive, holiday flags: raises; words in `calendars.yaml`),
  never applied for you. Step 32 warns when a control's posterior piles up at zero, i.e. the data
  points the other way.

With Meridian installed (`pip install -r requirements-model.txt`, about 1.5 GB with TensorFlow),
`mmm_tool.configure.spec.build_input_data(result)` and `build_model_spec(result.config, input_data)`
give the `InputData` and `ModelSpec` to fit. Tested with google-meridian 2.1.0: the tests build a
`Meridian` model from them and sample its priors (they skip when Meridian isn't installed).

## Phase E: fit and validate (steps 28–36)

Needs google-meridian (`pip install -r requirements-model.txt`). Fitting takes minutes, so it runs
as a background job: press **Start fit** on the Phase E screen and leave the page if you like; it
polls for progress and shows the result when the job finishes. From the command line:

```bash
python -m mmm_tool.fit data.csv --answers answers.json --sampler quick --out fit/
python -m mmm_tool.fit data.csv --defaults --sampler full --stability --placebo
```

A model is **accepted** only if it passes every gate (30, 31 and 32). A failed gate doesn't stop
the other checks, so every reason shows; the final fit (36) runs only for an accepted model.

| Step | What it checks | Pass rule | Kind |
|---|---|---|---|
| 28 Prior check | `sample_prior`: does the prior imply believable sales? | prior 90% range of total KPI covers the actual total and is at most 10× as wide as it | warning |
| 29 Fit with holdout | `sample_posterior` with the step 27 weeks held back, with the run's seed | quick: 2 chains, 300/300/300; full: 4 chains, 500 adapt, 500 burn-in, 1,000 kept; or Custom | — |
| 30 Convergence | largest R-hat, ignoring NaN (parameters the model doesn't use) | all below 1.1; otherwise one automatic retry with twice the adapt and burn-in, then fail and wait | gate |
| 31 Holdout accuracy | test and train MAPE and R² (national) | test MAPE below 10% and below a same-week-last-year forecast | gate |
| 32 Plausibility | ROI per channel, baseline share, contributions | ROI 0.05–20 (when ROI is ₹ per ₹); baseline 20–95% of the KPI; baseline + contributions = the fit | gate |
| 33 Did the data teach anything? | each channel's ROI, posterior vs prior; the prior's 90% range is exact from its distribution (LogNormal by default: 0.28–5.37), from prior draws only without a closed form | 1 − posterior width ÷ prior width at least 30%, else badge **prior-driven** | badge |
| 34 Stability (optional) | refits without the first 26, last 26, and both | ROI CV below 0.3, else badge **unstable**; a stable channel that is also hard to measure or prior-driven is noted as stable only because it stays near the starting assumption | badge; Skipped when not run |
| 35 Placebo (optional) | an extra channel with the weekly amounts of the biggest-spend channel whose spend CV is 0.5 or more (else the most variable one), in random weeks; its ROI is shown beside that channel's | placebo mean ROI under half the real channel's, and the low end of its 90% range under 0.3 | warning; Skipped when not run |
| 36 Final fit and save | refit on every week (same sampler, seed and retry rule as step 30), `save_mmm` | saved as `model.pkl` with `model_meta.json`: data version, settings, seed, attempts, gate results, decision log | — |

Thresholds and the two sampler presets are in `mmm_tool/fit/fit.yaml`.

**Sampler, seed and run time (step 29).** *Custom* takes chains (at least 2), adapt, burn-in and
kept draws (each at least 100); fewer than 4 chains or 500 kept draws shows an amber hint. Every
run gets a random seed once, shown on the form (you can type your own, or `--seed` on the command
line); it goes to `sample_prior` and `sample_posterior` and into the result and decision log, and
the same data, settings and seed reproduce the fit exactly. The run-time estimate is
`a + b x weeks x regions x channels x draws per chain x ceil(chains / cores)`, fitted by least
squares on this machine's fit log (`~/.cache/mmm_tool/fit_times.jsonl`, or `MMM_FIT_LOG`): "about
4 min (from 12 past runs)". Before 3 runs it times two fits of the bundled sample once
(`python -m mmm_tool.fit --calibrate`) and says "(rough)". a, the start-up cost, is large.

**When step 30 fails.** One automatic retry doubles adapt and burn-in (same chains, kept draws and
seed); the screen shows "Retrying with longer warm-up (about N min)" with Cancel, and a table of
both attempts (settings, largest R-hat, worst parameter, time). If it still fails, steps 31 onward
wait: the step lists the unsettled parameters in plain words and what to try (more chains and kept
draws; fewer knots; combine channels; tighter priors). **Continue anyway** carries on from the
saved holdout fit without refitting, and every later screen and the export (summary, Meridian's
HTML report, `model_meta.json`) carries a **did not converge** badge. Step 36 uses the same rule;
there is never more than one automatic retry.


Step 35 differs from the roadmap's wording ("its ROI range includes zero") on purpose: Meridian's
ROI priors are LogNormal, so an ROI range can never include zero. Instead the placebo is compared
with the real channel it copies. The 0.3 floor applies when ROI is money per money; with ROI in KPI
units only the comparison with the real channel is used. Steps 24 and 35 measure channels as they
are after step 15 drops and step 17 combinations (spend CV and share of paid spend from the model
table), never from step 16's stored numbers; a combined channel gets its own step 16 check.

**Compare with known truth (test files only).** Synthetic test sets come with an answer key
(`truth_for_tool.json`). Load it in the Phase E side panel and step 32 adds True ROI, Truth in range?
and True share for each channel (matched by the exact set of raw columns the channel is made of,
never by name; a channel with no exact match shows "no truth for this channel"), a Non-media row,
and "N of M channels have the truth inside their 90% range". With ROI in KPI units it compares
KPI per 1,00,000 of spend. The Model Summary adds "True next-₹ ROI". The key is refused if it is for
another KPI column, stays with the run (`answer_key.json`), never reaches the model and never
changes a verdict. Real data has no truth, so none of this shows without a key.

In the UI the job writes to `runs/<id>/fit/` (`status.json`, `result.json`, `model.pkl`,
`log.txt`). If you change any answer after a fit, the fit is marked out of date until you refit.
Download the accepted model at `/api/runs/<id>/model.pkl` and load it with
`meridian.model.model.load_mmm` using the same Meridian version.

`tests/test_step24.py` also drives the step 24 panel in a real browser (it needs Node with Playwright, and skips otherwise).

`pytest` checks the gate rules and the job bookkeeping in seconds. The tests that really fit with
Meridian take several minutes, so they run only with `MMM_FIT_TESTS=1 pytest tests/test_phase_e.py tests/test_priors_sampler.py`.
`tests/test_priors_sampler.py` holds the property tests for the prior choices, the sampler, the seed,
the estimate and the retry.

## Phase F: Model Summary (steps 37–42)

Opens once Phase E accepts a model. Every number is computed once, right after the final fit, and
stored as `fit/summary.json` (`mmm_tool/summary/build.py`), so the screen opens instantly and draws
its own charts from the stored tables. Panels follow the roadmap's rule, trust before results: fit
health and badges sit above the ROI table. The header shows the model version (its data version)
and the data's last week.

| Panel | Shows | Meridian source |
|---|---|---|
| 37 Fit health | actual vs predicted KPI by week with a 90% band; train and test MAPE and R² from the holdout fit (step 31); highest R-hat of the final fit | `expected_vs_actual_data`, `get_rhat` |
| 40 Trust badges | per channel: calibrated or model only, hard to measure, prior-driven, unstable (and Phase C's), each with its reason | Phases C, D and E |
| 39 ROI table | ROI, next-₹ ROI, KPI driven, cost per KPI unit, each with a 90% range; all weeks or the last 52 | `summary_metrics(confidence_level=0.9)` |
| 38 Sales breakdown | baseline and each channel, in total and week by week; the parts add up to the predicted KPI | `summary_metrics`, `baseline_summary_metrics` |
| 41 Response curves | KPI driven against spend, current spend marked | `response_curves` |
| 42 Carryover | share of a week's effect left in each following week, with the half-life | `adstock_decay` |

Meridian's own HTML report (`output_model_results_summary`) is written next to the model and stays
available as a download. For a model accepted before Phase F existed, the screen offers to build the
summary from the saved model (`python -m mmm_tool.summary --run-dir runs/<id>`). If the data or
settings change after the fit, the screen says the numbers describe the earlier version.

`tests/test_phase_f.py` checks the stored summary (parts add up, curves match the table, carryover
fades) and the screen in a browser, using `tests/data/phase_f/summary.json`, a summary produced by
the real Phase E path. With `MMM_FIT_TESTS=1` it also fits a model and builds the summary.

## Phase G: What-If (steps 43–47)

Opens once a model is accepted (the **What-If (G)** button on the Model Summary, or the rail). A
scenario changes each paid channel's spend over a period, all weeks or the last 52, by a share of
what it was: weekly spend, and media units with it, scale by the same factor, as in Meridian's own
response curves (for a reach-and-frequency channel, reach scales and frequency stays).

| Step | What it shows |
|---|---|
| 43 Enter a scenario | a slider per channel, −100% to +200% of its spend in the period; **Move** an amount from one channel to another; **Reset** |
| 44 Quick result | KPI (and revenue, with a money value) from paid media, total spend, ROI and cost per KPI, against the current plan, per channel too. It comes from response curves worked out once per period from the saved model, so a slider redraws in the browser with no server call |
| 45 Exact result | the same scenario on every posterior draw of the saved model: the KPI and revenue with their 90% ranges, and the change, whose 90% range is the range of the draw-by-draw difference (it says when that range includes zero) |
| 46 Extrapolation warning | a channel outside 50–200% of its spend in the period: the model hasn't seen such spend, so the result is a guess; a channel with no diminishing returns (step 23) grows in a straight line |
| 47 Save and compare | up to 3 named scenarios, kept with the run, against the current plan, with the exact range when it was run; **Load** puts one back on the sliders |

The quick and exact results agree (Meridian's channel effects add up); `tests/test_phase_g.py`
checks that on a real fit with `MMM_FIT_TESTS=1`. The curves and exact results are computed in a
background process (`python -m mmm_tool.whatif`) and cached in `runs/<id>/fit/whatif/` for the saved
model, so a refit recomputes them. The slider range, the curve grid, the 50–200% limits and the
number of saved scenarios are in `mmm_tool/whatif/whatif.yaml`.
