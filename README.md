# Meridian MMM Tool

Built so far: **Phase A, Ingest (steps 1–6)** from the roadmap, a web UI for it, and a clickable
design mock of the later screens in `mock/` (sample data only, not connected).

## Web UI

```bash
pip install -e '.[ui]'
mmm-ui                     # then open http://127.0.0.1:8000
```

`/` is the landing page (what the tool does, the phases, the validation gates, and your recent runs);
`/app` is the Phase A screen. Upload a CSV or Excel export (or pick a sample), answer the highlighted decisions, and download the
ingested table when all six steps pass. Runs and answers are saved under `./runs/` (set `MMM_RUNS_DIR`
to change it), so a page reload keeps your place. The API is under `/api/` (docs at `/docs`).

## Command line and tests

```bash
pip install -e '.[dev]'
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
