"""Keyword lists for role suggestions, loaded from YAML.

Built-in lists live in role_keywords.yaml next to this file. The user's own
file (MMM_ROLES_FILE, default ./roles.yaml) is layered on top: its rules are
checked first, and its words and aliases are added to the built-in ones. Both
files are re-read when they change, so edits apply on the next run.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .types import ROLES

BUILTIN = Path(__file__).with_name("role_keywords.yaml")
CONFIDENCE = ("high", "medium", "low")


def user_file() -> Path:
    return Path(os.environ.get("MMM_ROLES_FILE", "roles.yaml"))


def words_of(name: str) -> str:
    """'TV_Spend (₹)' -> 'tv spend'; 'youtubeReach' -> 'youtube reach'."""
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(name)).lower()
    return " ".join(re.findall(r"[a-z0-9]+", s))


def _pattern(entries: list[str]) -> re.Pattern | None:
    phrases = [words_of(e) for e in entries if words_of(e)]
    if not phrases:
        return None
    alts = "|".join(re.escape(p) for p in sorted(set(phrases), key=len, reverse=True))
    return re.compile(rf"(?<![a-z0-9])(?:{alts})(?![a-z0-9])")


@dataclass
class Rule:
    role: str
    words: re.Pattern
    also_needs: re.Pattern | None = None
    unless: re.Pattern | None = None
    confidence: str = "medium"
    reason: str = ""
    origin: str = "built-in"

    def matches(self, w: str) -> bool:
        return (
            bool(self.words.search(w))
            and (self.also_needs is None or bool(self.also_needs.search(w)))
            and (self.unless is None or not self.unless.search(w))
        )


@dataclass
class Keywords:
    rules: list[Rule] = field(default_factory=list)
    strip: set[str] = field(default_factory=set)
    aliases: dict[str, str] = field(default_factory=dict)
    kpi_count: re.Pattern | None = None
    kpi_money: re.Pattern | None = None
    agg_mean: re.Pattern | None = None
    agg_sum: re.Pattern | None = None
    money_units: dict[str, float] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    def match(self, column: str, text: bool = False) -> Rule | None:
        """First matching rule. Text columns only take geo rules; numeric
        columns take every other rule."""
        w = words_of(column)
        return next((r for r in self.rules if (r.role == "geo") == text and r.matches(w)), None)


class KeywordError(ValueError):
    pass


def _read(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise KeywordError(f"{path} is not valid YAML: {e}") from e
    if not isinstance(data, dict):
        raise KeywordError(f"{path} should hold keys like 'rules' and 'channel_aliases'.")
    return data


def _rules(data: dict, origin: str, path: Path) -> list[Rule]:
    out = []
    for i, r in enumerate(data.get("rules") or [], 1):
        where = f"{path}, rule {i}"
        if not isinstance(r, dict) or not r.get("role") or not r.get("words"):
            raise KeywordError(f"{where}: each rule needs 'role' and 'words'.")
        if r["role"] not in ROLES or r["role"] == "date":
            raise KeywordError(f"{where}: unknown role '{r['role']}'. Use one of {[x for x in ROLES if x != 'date']}.")
        conf = r.get("confidence", "medium")
        if conf not in CONFIDENCE:
            raise KeywordError(f"{where}: confidence must be high, medium or low.")
        out.append(
            Rule(
                role=r["role"],
                words=_pattern([str(x) for x in r["words"]]),
                also_needs=_pattern([str(x) for x in r.get("also_needs") or []]),
                unless=_pattern([str(x) for x in r.get("unless") or []]),
                confidence=conf,
                reason=r.get("reason") or f"{r['role']} by name ({origin} words)",
                origin=origin,
            )
        )
    return out


_cache: dict = {}


def load_keywords(user_path: Path | None = None) -> Keywords:
    """Built-in keywords with the user's file layered on top (cached by mtime)."""
    user_path = Path(user_path) if user_path else user_file()
    stamp = (BUILTIN.stat().st_mtime, str(user_path), user_path.stat().st_mtime if user_path.is_file() else None)
    if _cache.get("stamp") == stamp:
        return _cache["kw"]

    layers = [("built-in", BUILTIN, _read(BUILTIN))]
    if user_path.is_file():
        layers.append(("your", user_path, _read(user_path)))

    kw = Keywords()
    count, money, mean_w, sum_w = [], [], [], []
    for origin, path, data in reversed(layers):  # user rules first
        kw.rules += _rules(data, origin, path)
    for origin, path, data in layers:
        kw.strip |= {words_of(x) for x in data.get("channel_words_to_strip") or []}
        kw.aliases.update({words_of(k).replace(" ", "_"): words_of(v).replace(" ", "_") for k, v in (data.get("channel_aliases") or {}).items()})
        count += [str(x) for x in data.get("kpi_count_words") or []]
        money += [str(x) for x in data.get("kpi_money_words") or []]
        mean_w += [str(x) for x in data.get("aggregation_mean_words") or []]
        sum_w += [str(x) for x in data.get("aggregation_sum_words") or []]
        for unit, rupees in (data.get("money_units") or {}).items():
            try:
                kw.money_units[words_of(unit)] = float(rupees)
            except (TypeError, ValueError) as e:
                raise KeywordError(f"{path}: money_units.{unit} must be a number of ₹.") from e
        kw.sources.append(str(path))
    kw.kpi_count, kw.kpi_money = _pattern(count), _pattern(money)
    kw.agg_mean, kw.agg_sum = _pattern(mean_w), _pattern(sum_w)
    _cache.update(stamp=stamp, kw=kw)
    return kw
