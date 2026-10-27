"""Wide-format regions, checked at the start of step 3, before role mapping.

An export sometimes puts each region in its own columns: policies_TN,
tv_spend_TN, policies_KA, tv_spend_KA... Read column by column, one region's
outcome becomes the KPI and the other regions' spend explains its sales.

Detection is by shape only: two or more column stems that each appear with
the same set of two or more tags, as a suffix or a prefix, joined by "_", "-"
or a space. A tag shared by only one stem is not a region. Tags that are
known metric or role words (spend, grps, index...) or plain numbers (years)
are not regions either, and neither are tags whose stems are bare metric
words (tv_spend + meta_spend: tv and meta are channels). Those words come
from role_keywords.yaml.

The user then chooses:
  reshape  one region column and one column per stem (the default)
  sum      add the regions into one national series; non-additive stems
           are combined the way the user says
  keep     leave the columns as they are, with a warning
"""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

from .keywords import Keywords, load_keywords, words_of
from .types import Answers, Decision
from .values import numeric_view

SEP = re.compile(r"[_\- ]+")
ADDITIVE = {"kpi", "media", "media_spend", "rf_spend", "reach", "organic_media", "population"}
RULES = ("sum", "mean", "max")


def _split(col: str, mode: str) -> tuple[str, str] | None:
    """('tv_spend', 'TN') for 'tv_spend_TN' (suffix) or 'TN_tv_spend' (prefix)."""
    seps = list(SEP.finditer(col.strip()))
    if not seps:
        return None
    m = seps[-1] if mode == "suffix" else seps[0]
    left, right = col.strip()[: m.start()], col.strip()[m.end():]
    if not left or not right:
        return None
    return (left, right) if mode == "suffix" else (right, left)


def _is_region_tag(tag: str, kw: Keywords) -> bool:
    w = words_of(tag)
    if not w or w.isdigit() or w in kw.strip:
        return False
    return kw.match(tag) is None and kw.match(tag, text=True) is None


def detect_wide_regions(columns: list[str], exclude: set[str] = frozenset(), kw: Keywords | None = None) -> dict | None:
    kw = kw or load_keywords()
    best = None
    for mode in ("suffix", "prefix"):
        tags_of: dict[str, set[str]] = defaultdict(set)
        col_of: dict[tuple[str, str], str] = {}
        order: list[str] = []
        for col in columns:
            if col in exclude:
                continue
            split = _split(col, mode)
            if split is None or not _is_region_tag(split[1], kw):
                continue
            stem, tag = split
            tags_of[stem].add(tag)
            col_of[(stem, tag)] = col
            if tag not in order:
                order.append(tag)
        groups: dict[frozenset, list[str]] = defaultdict(list)
        for stem, tags in tags_of.items():
            if len(tags) >= 2:
                groups[frozenset(tags)].append(stem)
        for tags, stems in groups.items():
            if len(stems) < 2:
                continue
            # A stem that is only a metric word ("spend", "impressions") means the
            # tags are channels (tv_spend, meta_spend), not regions.
            if any(set(words_of(st).split()) <= kw.strip for st in stems):
                continue
            size = len(stems) * len(tags)
            if best is None or size > best["size"]:
                tag_list = [t for t in order if t in tags]
                best = {
                    "mode": mode, "tags": tag_list, "stems": stems, "size": size,
                    "columns": {s: {t: col_of[(s, t)] for t in tag_list} for s in stems},
                }
    return best


def wide_columns(info: dict) -> list[str]:
    return [c for by_tag in info["columns"].values() for c in by_tag.values()]


def to_long(df: pd.DataFrame, info: dict, date_col: str, region_col: str) -> pd.DataFrame:
    others = [c for c in df.columns if c not in set(wide_columns(info))]
    parts = []
    for tag in info["tags"]:
        part = df[others].copy()
        part.insert(others.index(date_col) + 1, region_col, tag)
        for stem in info["stems"]:
            part[stem] = df[info["columns"][stem][tag]].to_numpy()
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values([date_col, region_col], kind="stable", key=lambda s: s if s.name != region_col else s.map(info["tags"].index)).reset_index(drop=True)


def combine_rule(stem: str, df: pd.DataFrame, cols: list[str], kw: Keywords) -> tuple[str | None, str]:
    """(rule, reason); rule None means the user is asked, with a suggestion in reason."""
    rule = kw.match(stem)
    if rule is not None and rule.role in ADDITIVE:
        return "sum", f"{rule.role.replace('_', ' ')}: added across regions"
    values = pd.concat([numeric_view(df[c]) for c in cols])
    if set(values.dropna().unique()) <= {0.0, 1.0}:
        return None, "max"
    w = words_of(stem)
    if kw.agg_sum is not None and kw.agg_sum.search(w):
        return None, "sum"
    return None, "mean"


def to_national(df: pd.DataFrame, info: dict, rules: dict[str, str]) -> pd.DataFrame:
    out = df.drop(columns=wide_columns(info))
    for stem in info["stems"]:
        block = pd.concat([numeric_view(df[c]) for c in info["columns"][stem].values()], axis=1)
        if rules[stem] == "sum":
            out[stem] = block.sum(axis=1, min_count=1)
        elif rules[stem] == "max":
            out[stem] = block.max(axis=1)
        else:
            out[stem] = block.mean(axis=1)
    return out


def resolve_wide_regions(df: pd.DataFrame, date_col: str, answers: Answers, step: int) -> tuple[pd.DataFrame, dict, list[Decision], list[str]]:
    """Detect and handle wide regions. Returns (table, info, decisions,
    warnings); info["action"] is None while the choice is pending."""
    kw = load_keywords()
    info = detect_wide_regions(list(df.columns), {date_col}, kw)
    if info is None:
        return df, {}, [], []
    cols = wide_columns(info)
    tags, stems = info["tags"], info["stems"]
    d = Decision(
        id="wide_regions",
        step=step,
        question=(f"{len(stems)} column groups ({', '.join(stems)}) repeat for {len(tags)} regions "
                  f"({', '.join(tags)}). Reshape to one row per week and region, add the regions into one "
                  "national series, or keep the columns as they are?"),
        options=["reshape", "sum", "keep"],
        default="reshape",
        blocking=True,
        context={"tags": tags, "stems": stems, "mode": info["mode"], "columns": info["columns"]},
    )
    decisions, warnings = [d], []
    action = answers.ask(d)
    info["action"] = action
    if action is None:
        return df, info, decisions, warnings

    if action == "reshape":
        dn = Decision(id="region_column", step=step, question="Name the new region column.",
                      options=[], default="region", context={"taken": [c for c in df.columns if c not in cols]})
        decisions.append(dn)
        name = str(answers.ask(dn) or "region").strip() or "region"
        if name in dn.context["taken"] or name in stems:
            dn.blocking, dn.answered, info["action"] = True, False, None
            info["problem"] = f"'{name}' is already a column; choose another name for the region column."
            return df, info, decisions, warnings
        info["region_column"] = name
        out = to_long(df, info, date_col, name)
        info.update(rows_before=len(df), rows_after=len(out))
        return out, info, decisions, warnings

    if action == "sum":
        fixed, ask = {}, {}
        for stem in stems:
            rule, hint = combine_rule(stem, df, list(info["columns"][stem].values()), kw)
            if rule:
                fixed[stem] = rule
            else:
                ask[stem] = hint
        rules = dict(fixed)
        if ask:
            dc = Decision(id="wide_regions_combine", step=step,
                          question=f"These columns can't simply be added across regions: {', '.join(ask)}. Combine each by sum, mean or max?",
                          options=list(RULES), default=ask, blocking=True, context={"added": fixed})
            decisions.append(dc)
            chosen = answers.ask(dc)
            if chosen is None:
                info["action"] = None
                return df, info, decisions, warnings
            bad = {k: v for k, v in chosen.items() if k not in ask or v not in RULES}
            if bad:
                dc.answered, info["action"] = False, None
                info["problem"] = f"Unknown column or rule in {bad}; rules are sum, mean or max."
                return df, info, decisions, warnings
            rules.update({**ask, **chosen})
        info["rules"] = rules
        out = to_national(df, info, rules)
        info.update(rows_before=len(df), rows_after=len(out))
        return out, info, decisions, warnings

    if action == "keep":
        warnings.append(
            f"Region columns kept side by side ({', '.join(tags)}): each region's spend will explain the other "
            "regions' sales, and only one region's outcome can be the KPI."
        )
        return df, info, decisions, warnings

    d.answered, info["action"] = False, None
    info["problem"] = f"'{action}' isn't an option; choose reshape, sum or keep."
    return df, info, decisions, warnings
