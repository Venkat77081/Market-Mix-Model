"""Step 4: pair spend with exposure.

Matches tv_spend with tv_grps, search_spend with search_clicks, and groups
reach, frequency and rf_spend into reach-and-frequency channels. A spend
column with no exposure is used as its own exposure, which is Meridian's
recommended fallback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .roles import channel_of
from .types import Answers, Decision, StepResult, Verdict
from .values import numeric_view

STEP = 4
NAME = "Pair spend with exposure"

# Preferred exposure metric when a channel has several (impressions first,
# as Meridian recommends; clicks last since they partly measure response).
EXPOSURE_PREFERENCE = ("impression", "impr", "imps", "grp", "trp", "view", "spot", "insertion", "click")


@dataclass
class Channel:
    name: str
    kind: str  # "media" or "rf"
    spend: str
    exposure: str | None = None  # media channels
    reach: str | None = None  # rf channels
    frequency: str | None = None
    exposure_is_spend: bool = False


def _rank(col: str) -> int:
    low = col.lower()
    return next((i for i, key in enumerate(EXPOSURE_PREFERENCE) if key in low), len(EXPOSURE_PREFERENCE))


def _group(roles: dict[str, str], role: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for col, r in roles.items():
        if r == role:
            out.setdefault(channel_of(col), []).append(col)
    return out


def propose_pairs(roles: dict[str, str]) -> tuple[dict[str, dict], list[str]]:
    """Return ({channel: pairing}, notes). A pairing names spend and exposure
    (or reach/frequency) columns."""
    spend, exposure = _group(roles, "media_spend"), _group(roles, "media")
    rf_spend, reach, freq = _group(roles, "rf_spend"), _group(roles, "reach"), _group(roles, "frequency")
    pairs: dict[str, dict] = {}
    notes: list[str] = []

    for ch, cols in spend.items():
        for i, col in enumerate(cols):
            name = ch if len(cols) == 1 else f"{ch}_{i + 1}"
            options = sorted(exposure.get(ch, []), key=_rank)
            pairs[name] = {"kind": "media", "spend": col, "exposure": options[0] if options else None}

    # Reach and frequency come in as rf_spend, or as a spend column whose
    # channel also has reach and frequency columns.
    for ch in set(rf_spend) | (set(reach) & set(freq)):
        spend_col = (rf_spend.get(ch) or [None])[0]
        if spend_col is None and ch in pairs:
            spend_col = pairs.pop(ch)["spend"]
            notes.append(f"'{spend_col}' is the spend for the {ch} reach-and-frequency channel.")
        if spend_col and reach.get(ch) and freq.get(ch):
            pairs[ch] = {"kind": "rf", "spend": spend_col, "reach": reach[ch][0], "frequency": freq[ch][0]}
    return pairs, notes


VALUE_PAIR_CORRELATION = 0.7  # exposure that moves this closely with a spend column is offered as its pair


def value_match(df: pd.DataFrame | None, col: str, candidates: list[Channel]) -> tuple[Channel, float] | None:
    """The spend-only channel whose spend moves most closely with this exposure column."""
    if df is None or col not in df.columns:
        return None
    x = numeric_view(df[col])
    scored = [(c, numeric_view(df[c.spend]).corr(x)) for c in candidates if c.spend in df.columns]
    scored = [(c, r) for c, r in scored if r == r and r > VALUE_PAIR_CORRELATION]
    return max(scored, key=lambda t: t[1]) if scored else None


def pair_channels(roles: dict[str, str], answers: Answers, df: pd.DataFrame | None = None) -> tuple[list[Channel], dict[str, str], StepResult]:
    """Return channels, the possibly updated roles, and the step result. With
    the table, exposure that no spend column matches by name is offered to the
    spend column it moves with (FB spend and Meta impressions)."""
    roles = dict(roles)
    proposed, notes = propose_pairs(roles)
    d = Decision(
        id="pairs",
        step=STEP,
        question="Confirm which exposure column goes with each spend column.",
        options=[c for c, r in roles.items() if r in ("media", "reach", "frequency")],
        default=proposed,
    )
    decisions = [d]
    chosen = answers.ask(d)
    pairs = {**proposed, **(chosen or {})}
    warnings: list[str] = []

    channels: list[Channel] = []
    used: set[str] = set()
    for name, p in sorted(pairs.items()):
        if p["kind"] == "rf":
            channels.append(Channel(name, "rf", p["spend"], reach=p["reach"], frequency=p["frequency"]))
            used.update([p["spend"], p["reach"], p["frequency"]])
            roles[p["spend"]], roles[p["reach"]], roles[p["frequency"]] = "rf_spend", "reach", "frequency"
        else:
            exp = p.get("exposure")
            channels.append(Channel(name, "media", p["spend"], exposure=exp or p["spend"], exposure_is_spend=exp is None))
            used.update(c for c in (p["spend"], exp) if c)
            if exp:
                roles[exp] = "media"

    # Leftover exposure, reach or frequency columns. A second exposure metric
    # for a paired channel is dropped; one with no spend at all can't get an
    # ROI, so the user can keep it as unpaid media or drop it.
    by_name = {c.name: c for c in channels}
    for col, role in list(roles.items()):
        if role not in ("media", "reach", "frequency", "rf_spend") or col in used:
            continue
        twin = by_name.get(channel_of(col))
        if twin is not None:
            roles[col] = "excluded"
            notes.append(f"'{col}' is a second exposure metric for {twin.name}; '{twin.exposure or twin.reach}' is used instead.")
            continue
        match = value_match(df, col, [c for c in channels if c.kind == "media" and c.exposure_is_spend]) if role == "media" else None
        if match:
            ch, corr = match
            dp = Decision(
                id=f"pair:{col}",
                step=STEP,
                question=(f"'{col}' has no spend column with a matching name, but it moves with '{ch.spend}' "
                          f"(correlation {corr:.2f}). Pair them as one channel?"),
                options=[ch.name, "organic_media", "excluded"],
                default=ch.name,
                blocking=True,
                context={"spend": ch.spend, "exposure": col, "correlation": round(float(corr), 3)},
            )
            decisions.append(dp)
            choice = answers.ask(dp)
            if choice == ch.name:
                ch.exposure, ch.exposure_is_spend = col, False
                used.add(col)
                notes.append(f"'{ch.spend}' paired with '{col}' by their values (correlation {corr:.2f}).")
            elif choice in ("organic_media", "excluded"):
                roles[col] = choice
            continue
        unpaired = Decision(
            id=f"unpaired:{col}",
            step=STEP,
            question=f"'{col}' has no matching spend column, so it can't get an ROI. Model it as unpaid media or drop it?",
            options=["organic_media", "excluded"],
            default="organic_media" if role in ("media", "reach") else "excluded",
        )
        decisions.append(unpaired)
        answer = answers.ask(unpaired) or unpaired.default
        roles[col] = answer
        warnings.append(f"'{col}' has no spend column; set to {answer}.")

    spend_only = [c.name for c in channels if c.exposure_is_spend]
    if spend_only:
        notes.append(f"No exposure column for {spend_only}; spend is used as exposure.")
    details = {
        "channels": [asdict(c) for c in channels],
        "notes": notes,
    }
    paired = sum(1 for c in channels if c.kind == "media" and not c.exposure_is_spend)
    rf = sum(1 for c in channels if c.kind == "rf")
    summary = f"{len(channels)} paid channels: {paired} paired with exposure, {len(spend_only)} spend-only"
    if rf:
        summary += f", {rf} reach-and-frequency"
    verdict = Verdict.WARN if warnings else Verdict.PASS
    return channels, roles, StepResult(STEP, NAME, verdict, summary + ".", details, warnings, decisions)
