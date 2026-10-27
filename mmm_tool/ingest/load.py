"""Step 1: load file.

Detects encoding, delimiter, sheet, header row and preamble rows, and returns
the table with every cell as it appeared in the file.
"""
from __future__ import annotations

import csv
import io
from datetime import date
from pathlib import Path

import pandas as pd
from charset_normalizer import from_bytes

from .keywords import load_keywords, words_of
from .types import Answers, Decision, StepResult, Verdict
from .values import is_null_token, to_number

STEP = 1
NAME = "Load file"
EXCEL_EXT = {".xlsx", ".xlsm", ".xls"}
TEXT_EXT = {".csv", ".tsv", ".txt"}
HEADER_SCAN_ROWS = 30


# Characters a Windows-1252 export plausibly contains outside ASCII.
CP1252_PLAUSIBLE = set("£€°§©®±µ·–—‘’“”•…×÷")


def detect_encoding(raw: bytes) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    # Excel on Windows saves CSV as cp1252. Statistical detectors misread
    # short runs of accented letters as CJK, so try cp1252 first.
    try:
        extra = [ch for ch in raw.decode("cp1252") if ord(ch) > 127]
        if all(ch.isalpha() or ch in CP1252_PLAUSIBLE for ch in extra):
            return "cp1252"
    except UnicodeDecodeError:
        pass
    best = from_bytes(raw).best()
    return best.encoding if best is not None else "latin-1"


def detect_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:50])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        # Sniffer gives up on files with preamble lines; take the delimiter
        # that splits the most lines into the same number of fields.
        best, best_score = ",", -1
        for delim in ",;\t|":
            counts = [line.count(delim) for line in sample.splitlines() if line.strip()]
            if not counts or max(counts) == 0:
                continue
            mode = max(set(counts), key=counts.count)
            score = counts.count(mode) * mode
            if score > best_score:
                best, best_score = delim, score
        return best


def _is_texty(value) -> bool:
    return not is_null_token(value) and pd.isna(to_number(value)) and not isinstance(value, date)


def detect_header_row(grid: pd.DataFrame) -> int:
    """Index of the header row: the first row that is mostly text labels and
    is as wide as the data, followed by rows that hold numbers."""
    width = int((~grid.map(is_null_token)).sum(axis=1).max())
    for r in range(min(HEADER_SCAN_ROWS, len(grid) - 1)):
        row = grid.iloc[r]
        filled = [v for v in row if not is_null_token(v)]
        if len(filled) < max(2, 0.6 * width):
            continue
        if sum(_is_texty(v) for v in filled) < 0.8 * len(filled):
            continue
        below = grid.iloc[r + 1 : r + 6]
        numeric_cells = below.map(lambda v: not pd.isna(to_number(v))).to_numpy().sum()
        if numeric_cells >= len(below):
            return r
    return 0


def _unique_names(raw_names: list) -> list[str]:
    names, seen = [], {}
    for i, n in enumerate(raw_names):
        name = "" if is_null_token(n) else str(n).strip()
        name = name or f"unnamed_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        names.append(name)
    return names


def summary_label(row: pd.Series) -> str | None:
    """The summary word ('TOTAL', 'Grand total') labelling a row, if any. Only
    text cells count, so a number never makes a row a summary."""
    pattern = load_keywords().summary_rows
    if pattern is None:
        return None
    for v in row:
        if isinstance(v, str) and not is_null_token(v) and pd.isna(to_number(v)) and pattern.fullmatch(words_of(v)):
            return v.strip()
    return None


def grid_to_table(grid: pd.DataFrame) -> tuple[pd.DataFrame, int, list[int], list[dict]]:
    """Return (table, header row number in the file, file line of each table
    row, summary rows dropped from the end)."""
    filled = ~grid.map(is_null_token).all(axis=1)
    rows = grid[filled]
    file_row = list(rows.index)
    rows = rows.reset_index(drop=True)
    header = detect_header_row(rows)
    table = rows.iloc[header + 1 :].copy()
    table.columns = _unique_names(list(rows.iloc[header]))
    blank_cols = [c for c in table.columns if c.startswith("unnamed_") and table[c].map(is_null_token).all()]
    table = table.drop(columns=blank_cols)
    # Trailing TOTAL / average rows are report summaries, not weeks.
    summaries = []
    while len(table) and len(summaries) < 3:
        label = summary_label(table.iloc[-1])
        if label is None:
            break
        summaries.insert(0, {"line": file_row[table.index[-1]] + 1, "label": label})
        table = table.iloc[:-1]
    lines = [file_row[i] + 1 for i in table.index]
    return table.reset_index(drop=True), file_row[header] + 1, lines, summaries


def _sheet_score(grid: pd.DataFrame) -> int:
    """Rough size of the data block; the biggest sheet is the default."""
    if grid.empty:
        return 0
    return int((~grid.map(is_null_token)).to_numpy().sum())


def load_file(path: str | Path, answers: Answers) -> tuple[pd.DataFrame | None, StepResult]:
    path = Path(path)
    ext = path.suffix.lower()
    details: dict = {"file": path.name}
    decisions: list[Decision] = []
    warnings: list[str] = []

    if ext in EXCEL_EXT:
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
        details["sheets"] = list(sheets)
        scores = {name: _sheet_score(g) for name, g in sheets.items()}
        best = max(scores, key=scores.get)
        sheet = best
        if len(sheets) > 1:
            d = Decision(
                id="sheet",
                step=STEP,
                question="This workbook has several sheets. Which one holds the weekly data?",
                options=list(sheets),
                default=best,
                context={"filled_cells": scores},
            )
            decisions.append(d)
            chosen = answers.ask(d)
            sheet = chosen if chosen is not None else best
            if sheet not in sheets:
                return None, StepResult(STEP, NAME, Verdict.FAIL, f"Sheet '{sheet}' is not in the workbook.", details)
        details["sheet"] = sheet
        grid = sheets[sheet]
    elif ext in TEXT_EXT:
        raw = path.read_bytes()
        encoding = detect_encoding(raw)
        text = raw.decode(encoding)
        delimiter = "\t" if ext == ".tsv" else detect_delimiter(text)
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        width = max((len(r) for r in rows), default=0)
        grid = pd.DataFrame([r + [None] * (width - len(r)) for r in rows], dtype=object)
        details.update(encoding=encoding, delimiter=delimiter)
    else:
        return None, StepResult(STEP, NAME, Verdict.FAIL, f"Unsupported file type '{ext}'. Upload a CSV or Excel file.", details)

    if grid.empty or grid.map(is_null_token).all().all():
        what = "0 bytes" if ext in TEXT_EXT and not path.stat().st_size else "only blank cells"
        return None, StepResult(STEP, NAME, Verdict.FAIL, f"The file is empty ({what}). Export the data again.", details)

    table, header_row, lines, summaries = grid_to_table(grid)
    n_preamble = header_row - 1  # every line above the header, blank ones included
    preamble = grid.iloc[: header_row - 1]
    details.update(
        header_row=header_row,
        preamble_rows=n_preamble,
        data_lines=lines,
        summary_rows_dropped=summaries,
        preamble_text=[
            " ".join(str(v) for v in r if not is_null_token(v))
            for r in preamble.itertuples(index=False)
            if not all(is_null_token(v) for v in r)
        ],
        rows=len(table),
        columns=list(table.columns),
        preview=[["" if is_null_token(v) else str(v) for v in r] for r in table.head(5).itertuples(index=False)],
    )
    if len(table.columns) < 2:
        return None, StepResult(STEP, NAME, Verdict.FAIL, "No data table found: fewer than 2 columns.", details)
    if table.empty:
        return None, StepResult(STEP, NAME, Verdict.FAIL,
                                f"Header found on row {header_row} but 0 data rows below it.", details)

    parts = [f"{len(table):,} rows × {len(table.columns)} columns"]
    if "encoding" in details:
        parts.append(f"{details['encoding']}, '{details['delimiter']}'-delimited")
    if "sheet" in details:
        parts.append(f"sheet '{details['sheet']}'")
    if n_preamble:
        parts.append(f"header on row {header_row}, {n_preamble} preamble line(s) skipped")
    if summaries:
        parts.append(", ".join(f"'{x['label']}' row on line {x['line']} dropped as a summary" for x in summaries))
    verdict = Verdict.PASS
    if any(c.startswith("unnamed_") for c in table.columns):
        verdict = Verdict.WARN
        warnings.append("Some columns have no header; they are named unnamed_N.")
    return table, StepResult(STEP, NAME, verdict, "; ".join(parts) + ".", details, warnings, decisions)
