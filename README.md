# Meridian MMM Tool

Built so far: **Phase A, Ingest (steps 1–6)** from the roadmap, plus a clickable UI mock in `mock/`.

```bash
pip install -e '.[dev]'
pytest                                      # 32 tests on synthetic insurance exports
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
