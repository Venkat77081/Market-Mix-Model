# Meridian MMM Tool

Built so far: **Phase A, Ingest (steps 1–6)**, **Phase B, Clean (steps 7–13)** and **Phase C,
Profile (steps 14–20)** from the roadmap, a web UI for all three, and a clickable design mock of the later screens in `mock/` (sample data only,
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
| 7 Coerce types | `N/A`, `NULL`, `-`, blanks → missing; `1,20,000`, `₹4,500`, `(250)` → numbers; yes/no → 1/0 | nothing | no |
| 8 Collapse duplicate weeks | merges rows on the same date (and adds regions up for a national model): spend, exposure, KPI summed; indexes, prices averaged; flags take the maximum | the rule for a column it can't classify (`aggregation_rules`) | only if rows merge |
| 9 Resample to weeks | days → weeks; weeks aligned to one start day; partial edge weeks dropped | week start (`week_start`) | no |
| 10 Label every gap | adds absent weeks; labels empty cells not started, ended or missing; a zero stays zero | KPI gaps: interpolate + `kpi_imputed` flag, or trim (`kpi_gaps`); media holes: 0 or interpolate (`media_gaps`) | KPI gaps |
| 11 Flag outliers | robust z-score on the KPI, weeks above 3.5 | event flag or keep, per week (`outliers`) | no |
| 12 Check impossible values | negative spend, KPI zero in a trading week, an index or price jumping 10x | zero, keep, interpolate or a corrected number, per cell (`negative_values`, `kpi_zero_weeks`, `control_jumps`) | negative values |
| 13 Normalise money | crore/lakh columns → ₹; mixed currency warning; optional inflation restatement; a fixed value per KPI unit becomes a column | `money_units`, `inflation` (`{"base_year": 2025, "annual_rate_pct": 5}`) | no |

The words used to classify columns in step 8 and the unit words in step 13 are in
`role_keywords.yaml`; add your own in `roles.yaml`.

## Phase A answer key

`tests/data/phase_a_tests/` holds 19 test exports and `expected_phase_a.json`, their answer key.
`python tests/expected_phase_a.py` prints a check-by-check report for every file; the same checks run
in `pytest` (`tests/test_expected_phase_a.py`).

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
