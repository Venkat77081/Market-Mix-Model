# Meridian MMM Tool

Built so far: **Phase A, Ingest (steps 1–6)**, **Phase B, Clean (steps 7–13)**, **Phase C,
Profile (steps 14–20)**, **Phase D, Configure (steps 21–27)** and **Phase E, Fit and validate
(steps 28–36)** from the roadmap, a web UI for all five, and a clickable design mock of the later screens in `mock/` (sample data only,
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
3. **Built-in words** in `mmm_tool/ingest/role_keywords.yaml`, which documents every field.
4. **The values**: text with a few repeating values is a region; a 0/1 column is a flag.
   Anything else becomes a low-confidence control.

You still confirm every role before the run continues.

## Phase B: clean and align to weeks (steps 7–13)

Runs automatically once Phase A is complete, and produces one complete row per week (per region in a
regional model) with no empty cells, which is what Meridian reads. Download it from the UI, or
`python -m mmm_tool.ingest file.csv --out clean.csv` (add `--phase a` to stop after ingest).

| Step | What it does | You decide | Blocks |
|---|---|---|---|
| 7 Coerce types | `N/A`, `NULL`, blanks → missing; `-` → 0 in money and media columns, missing elsewhere; `1,20,000`, `₹4,500`, `(250)`, `12.6 L` → numbers; yes/no → 1/0 | nothing | no |
| 8 Collapse duplicate weeks | drops exact copy rows; merges split rows on the same date (and adds regions up for a national model): spend, exposure, KPI summed; indexes, prices averaged; flags take the maximum; a value repeated on every split row is kept once | the rule for a column it can't classify (`aggregation_rules`) | only if rows merge |
| 9 Resample to weeks | days → weeks; weekly dates moved to the nearest week start; partial edge weeks kept or dropped; weeks with days missing inside the period scaled, interpolated or kept, and flagged in `partial_week_flag` | `week_start`, `partial_weeks`, `short_weeks` | no |
| 10 Label every gap | adds absent weeks; labels empty cells not started, ended or missing; a zero stays zero. KPI gaps at the edges are trimmed; inside, interpolated + `kpi_imputed` flag. Media blanks are judged per column: a column that never records 0 and is often blank gets 0 suggested, other blanks are interpolated; spend in absent weeks is interpolated unless you say the channels were off | `kpi_edge_gaps`, `kpi_gaps`, `media_gaps` (one answer, or one per column plus `missing_weeks`) | KPI gaps inside; absent weeks |
| 11 Flag outliers | robust z-score on the KPI, weeks above 3.5 | event flag or keep, per week (`outliers`) | no |
| 12 Check impossible values | negative spend, KPI zero in a trading week, an index or price jumping 10x, a spend or exposure cell 10x the weeks around it | zero, keep, interpolate or a corrected number, per cell (`negative_values`, `kpi_zero_weeks`, `control_jumps`) | negative values |
| 13 Normalise money | a level change close to a power of 10 (lakhs, then rupees) converted; crore/lakh columns → ₹; mixed currency warning; optional inflation restatement; revenue per KPI rebuilt from the cleaned revenue total; a fixed value per KPI unit becomes a column | `level_shifts`, `money_units`, `inflation` (`{"base_year": 2025, "annual_rate_pct": 5}`) | no |

The words used to classify columns in step 8 and the unit words in step 13 are in
`role_keywords.yaml`; add your own in `roles.yaml`. The thresholds Phase B uses (when blanks count as
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
| 21 Calendar flags | India insurance preset in `mmm_tool/configure/calendars.yaml`: tax season, financial year-end, Diwali and festive weeks | events on or off (`calendar_events`), your own date ranges (`custom_events`) | added to `controls` |
| 22 Carryover length | 8 weeks | 2–13 (`max_lag`) | `max_lag` |
| 23 Carryover shape | geometric; binomial suggested for TV and outdoor | per channel (`adstock_shape`) | `adstock_decay_spec` |
| 24 Baseline flexibility | one knot per quarter | `knots`, with a live preview (below); amber above weeks/8, red above weeks/6 | `knots` |
| 25 ROI priors | Meridian's LogNormal(0.2, 0.9) | a 90% ROI range per channel (`roi_ranges`) | `roi_m` / `roi_rf` |
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
| 29 Fit with holdout | `sample_posterior` with the step 27 weeks held back | quick: 2 chains, 300/300/300; full: 4 chains, 500 adapt, 500 burn-in, 1,000 kept | — |
| 30 Convergence | R-hat for every parameter | all below 1.1; otherwise one rerun with twice the draws, then fail | gate |
| 31 Holdout accuracy | test and train MAPE and R² (national) | test MAPE below 10% and below a same-week-last-year forecast | gate |
| 32 Plausibility | ROI per channel, baseline share, contributions | ROI 0.05–20 (when ROI is ₹ per ₹); baseline 20–95% of the KPI; baseline + contributions = the fit | gate |
| 33 Did the data teach anything? | each channel's ROI, posterior vs prior | 90% range at least 30% narrower than the prior's, else badge **prior-driven** | badge |
| 34 Stability (optional) | refits without the first 26, last 26, and both | ROI CV below 0.3, else badge **unstable** | badge |
| 35 Placebo (optional) | an extra channel: a real channel's weekly amounts in random weeks | not credited (see below) | warning |
| 36 Final fit and save | refit on every week, `save_mmm` | saved as `model.pkl` with `model_meta.json`: data version, settings, gate results | — |

Thresholds and the two sampler presets are in `mmm_tool/fit/fit.yaml`.

Step 35 differs from the roadmap's wording ("its ROI range includes zero") on purpose: Meridian's
ROI priors are LogNormal, so an ROI range can never include zero. The placebo counts as credited
when the data pulls its ROI *up* from the prior and the low end of its range stays above 0.05.

In the UI the job writes to `runs/<id>/fit/` (`status.json`, `result.json`, `model.pkl`,
`log.txt`). If you change any answer after a fit, the fit is marked out of date until you refit.
Download the accepted model at `/api/runs/<id>/model.pkl` and load it with
`meridian.model.model.load_mmm` using the same Meridian version.

`tests/test_step24.py` also drives the step 24 panel in a real browser (it needs Node with Playwright, and skips otherwise).

`pytest` checks the gate rules and the job bookkeeping in seconds. The tests that really fit with
Meridian take several minutes, so they run only with `MMM_FIT_TESTS=1 pytest tests/test_phase_e.py`.
