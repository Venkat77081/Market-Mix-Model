"""The file's currency, and how money is written in messages and on screen.

Detected in step 5 from the money columns' names (TV Spend (₹), spend_usd)
and the symbols in their cells ($1,200, €45), confirmed or changed by the
user (decision "currency"). With no sign of a currency, amounts are shown
as plain numbers and ROI as "return per unit of spend". The grouping
follows the currency: lakh and crore (1,20,000; 1.2 Cr) for INR, thousands
and millions (120,000; 1.2M) otherwise.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

# code: symbol, a word for one unit ("the next rupee"), grouping, and the words or
# signs that point to it in a column name or cell.
CURRENCIES: dict[str, dict[str, Any]] = {
    "INR": {"symbol": "₹", "unit": "rupee", "grouping": "indian", "signs": [r"₹", r"\binr\b", r"\brs\b\.?", r"\brupees?\b"]},
    "USD": {"symbol": "$", "unit": "dollar", "grouping": "international", "signs": [r"\$", r"\busd\b", r"\bdollars?\b"]},
    "EUR": {"symbol": "€", "unit": "euro", "grouping": "international", "signs": [r"€", r"\beur\b", r"\beuros?\b"]},
    "GBP": {"symbol": "£", "unit": "pound", "grouping": "international", "signs": [r"£", r"\bgbp\b", r"\bpounds?\b"]},
    "JPY": {"symbol": "¥", "unit": "yen", "grouping": "international", "signs": [r"¥", r"\bjpy\b", r"\byen\b"]},
    "AUD": {"symbol": "A$", "unit": "dollar", "grouping": "international", "signs": [r"a\$", r"\baud\b"]},
    "CAD": {"symbol": "C$", "unit": "dollar", "grouping": "international", "signs": [r"c\$", r"\bcad\b"]},
    "SGD": {"symbol": "S$", "unit": "dollar", "grouping": "international", "signs": [r"s\$", r"\bsgd\b"]},
    "AED": {"symbol": "AED ", "unit": "dirham", "grouping": "international", "signs": [r"\baed\b", r"\bdirhams?\b"]},
    "BRL": {"symbol": "R$", "unit": "real", "grouping": "international", "signs": [r"r\$", r"\bbrl\b"]},
}
NONE: dict[str, Any] = {"code": None, "symbol": "", "unit": "unit of spend", "grouping": "international"}
_INDIAN_GROUPING = re.compile(r"\d,\d\d,\d{3}\b")          # 1,20,000


def get(code: str | None) -> dict[str, Any]:
    """The currency for a code ("INR"), or the plain one for None / unknown."""
    if code and code in CURRENCIES:
        c = CURRENCIES[code]
        return {"code": code, "symbol": c["symbol"], "unit": c["unit"], "grouping": c["grouping"]}
    return dict(NONE)


def of(kpi: dict | None) -> dict[str, Any]:
    """The currency recorded in step 5 (kpi["currency"]), or the plain one."""
    cur = (kpi or {}).get("currency")
    return cur if isinstance(cur, dict) else get(cur)


def detect(df: pd.DataFrame, columns: list[str]) -> str | None:
    """The currency the money columns point to: names first, then cell text. None if nothing does."""
    votes: dict[str, int] = {}
    from .ingest.keywords import words_of

    names = " ".join(words_of(c) for c in columns)          # "Spend ($)" → "spend usd", "tv_inr" → "tv inr"
    for code, c in CURRENCIES.items():
        for sign in c["signs"]:
            votes[code] = votes.get(code, 0) + 3 * len(re.findall(sign, names))
    for col in columns:
        if col not in df.columns or pd.api.types.is_numeric_dtype(df[col]):
            continue
        text = " ".join(df[col].dropna().astype(str).head(200)).lower()
        for code, c in CURRENCIES.items():
            for sign in c["signs"]:
                votes[code] += len(re.findall(sign, text))
        if _INDIAN_GROUPING.search(text):
            votes["INR"] += 1
    # "$" also matches A$, C$, S$ and R$: prefer the more specific sign when it is there.
    for code in ("AUD", "CAD", "SGD", "BRL"):
        if votes.get(code):
            votes["USD"] = max(0, votes["USD"] - votes[code])
    best = max(votes, key=votes.get) if votes else None
    return best if best and votes[best] > 0 else None


def group(x: float, cur: dict, decimals: int = 0) -> str:
    """1234567 → '12,34,567' (indian) or '1,234,567'."""
    s = f"{abs(x):,.{decimals}f}"
    if cur.get("grouping") == "indian":
        whole, _, frac = s.replace(",", "").partition(".")
        if len(whole) > 3:
            head, tail = whole[:-3], whole[-3:]
            head = ",".join([head[max(0, i - 2):i] for i in range(len(head), 0, -2)][::-1])
            whole = f"{head},{tail}"
        s = whole + (f".{frac}" if frac else "")
    return ("-" if x < 0 else "") + s


def money(x: float, cur: dict, decimals: int = 0) -> str:
    """₹9,310 / $9,310 / 9,310 (no currency)."""
    return f"{'-' if x < 0 else ''}{cur.get('symbol', '')}{group(abs(x), cur, decimals)}"


def money_short(x: float, cur: dict) -> str:
    """Large amounts: ₹2.3 Cr / ₹4.5 L for INR, $2.3M / $45K otherwise."""
    sym, a = cur.get("symbol", ""), abs(x)
    if cur.get("grouping") == "indian":
        for size, word in ((1e7, " Cr"), (1e5, " L")):
            if a >= size:
                return f"{sym}{x / size:,.1f}{word}"
    else:
        for size, word in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
            if a >= size:
                return f"{sym}{x / size:,.1f}{word}"
    return money(x, cur)


def unit(cur: dict) -> str:
    """'₹', '$', or 'unit of spend' with no currency: 'orders per ₹' / 'orders per unit of spend'."""
    return (cur.get("symbol") or "").strip() or "unit of spend"


def per(cur: dict) -> str:
    """The ROI unit: '₹ per ₹', or 'return per unit of spend'."""
    u = (cur.get("symbol") or "").strip()
    return f"{u} per {u}" if u else "return per unit of spend"


def returned_per(cur: dict) -> str:
    """'₹ returned per ₹ spent', or 'return per unit of spend'."""
    u = (cur.get("symbol") or "").strip()
    return f"{u} returned per {u} spent" if u else "return per unit of spend"
