# Phase A test files

Each file targets specific Phase A steps of the MMM tool (1 load, 2 dates, 3 roles, 4 spend/exposure pairs, 5 KPI type, 6 granularity).
The full answer key is in `expected_phase_a.json`. Regenerate with `python3 gen_phase_a.py`.

| File | Steps | What it tests | Expected result | Blocks? |
|---|---|---|---|---|
| A01_clean_weekly.csv | 1–6 | Clean baseline, 156 weeks | Everything detected; asks policies vs premium as KPI | No |
| A02_semicolon_bom.csv | 1 | `;` delimiter, UTF-8 with BOM | First column read as `week_start`, not `﻿week_start` | No |
| A03_preamble_footer.csv | 1, 2 | 4 report lines above the header, a TOTAL row at the end | Header on line 5, 156 rows, TOTAL dropped as a summary row | No |
| A04_multisheet.xlsx | 1 | 3 sheets, data header on row 3 | Suggests `Weekly_Data`, header row 3 | No |
| A05_mixed_dates.csv | 2 | ISO, dd/mm/yyyy, dd-Mon-yy, dd.mm.yyyy mixed by row | 156 unique dates, 2023-01-02 to 2025-12-22 | No |
| A06_ambiguous_dates_monthly.csv | 2, 6 | Every date is 01/MM/YYYY | Prefers day-first (even monthly series) and asks to confirm | No |
| A07_messy_column_names.csv | 3 | Real-world names, index column, free-text notes | Roles as in the answer key; asks about "Fees" and "Payout" columns | No |
| A08_spend_exposure_pairs.csv | 3, 4 | Name-matched pairs, a mismatched pair, reach and frequency, organic | Asks before pairing `FB spend` with `Meta impressions` | No |
| A09_revenue_kpi.csv | 5 | KPI is premium in ₹ | Revenue KPI, no revenue-per-unit prompt | No |
| A10_leads_no_revenue.csv | 5 | KPI is quote requests, no money column | Asks for value per quote, or ROI in quotes | No |
| A11_daily.csv | 6 | 731 days | Daily detected; Phase B makes 105 weeks | No |
| A12_monthly.csv | 2, 6 | `YYYY-MM`, 36 months | Monthly detected; warns about step 14 | No |
| A13_regions_long.csv | 3, 6 | 5 states × 156 weeks with population | Region and population roles; asks regional vs national | No |
| A14_regions_wide.csv | 3, 6 | `_TN`, `_KA`, `_MH` column suffixes | Offers to reshape to long format | No |
| A15_bad_dates.csv | 2 | 9 unparseable dates (lines 27, 73, 89, 94, 98, 101, 109, 145, 153) | Lists all 9; offers fixes for Excel serial and ISO week | Yes |
| A16_no_media.csv | 3 | No media columns | Stops: no media channel | Yes |
| A17_empty.csv | 1 | 0 bytes | Stops: empty file | Yes |
| A17_header_only.csv | 1 | Header, no rows | Stops: 0 data rows | Yes |
| A18_indian_number_formats.csv | 3, B7 | `1,20,000`, `₹4.50 Cr`, `12.6 L`, `-`, `12%`, `(1,234)` | Read as numbers before roles are suggested; 2 negatives flagged | No |

## Two traps worth testing first

- **A05:** `pd.to_datetime(x, dayfirst=True)` also swaps ISO dates, so `2023-01-02` becomes 1 February and a duplicate week appears (155 unique instead of 156). Day-first must apply only to slash and dot dates.
- **A06:** Every date parses under both conventions, so the tool can't decide from validity alone. The even spacing of the day-first reading is the evidence, and the user should still confirm.
