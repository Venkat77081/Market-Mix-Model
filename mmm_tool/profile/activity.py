"""Step 15: channel activity.

A channel active in under 10% of weeks, or with under 1% of total spend,
gives the model almost nothing to learn from. The user merges such channels
into "Other" (the default for two or more), drops them, or keeps them with a
badge. A lone rare channel defaults to drop (under 1% of spend) or keep.
"""
from __future__ import annotations

from ..ingest.types import Answers, Decision, StepResult, Verdict
from .context import RARE_ACTIVE_SHARE, RARE_SPEND_SHARE, CCtx

STEP = 15
NAME = "Channel activity"
CHOICES = ("other", "drop", "keep")


def default_choices(rare: list[dict]) -> dict[str, str]:
    """Two or more rare channels merge into Other. A lone one is dropped if it
    is under 1% of spend (merging it alone would only rename it), else kept."""
    if len(rare) > 1:
        return {r["channel"]: "other" for r in rare}
    r = rare[0]
    return {r["channel"]: "drop" if r["spend_share"] < RARE_SPEND_SHARE else "keep"}


def check_activity(ctx: CCtx, answers: Answers) -> StepResult:
    spend = ctx.spend()
    total = float(spend.to_numpy().sum())
    rows = []
    for name in spend.columns:
        s = spend[name]
        rows.append({"channel": name, "active_weeks": int((s > 0).sum()), "weeks": len(s),
                     "active_share": round(float((s > 0).mean()), 3),
                     "spend": float(s.sum()), "spend_share": round(float(s.sum() / total), 4) if total else 0.0})
    rare = [r for r in rows if r["active_share"] < RARE_ACTIVE_SHARE or r["spend_share"] < RARE_SPEND_SHARE]
    details = {"channels": rows, "rare": [r["channel"] for r in rare]}
    if not rare:
        return StepResult(STEP, NAME, Verdict.PASS,
                          f"All {len(rows)} channels are active in at least {RARE_ACTIVE_SHARE:.0%} of weeks "
                          f"with at least {RARE_SPEND_SHARE:.0%} of spend.", details)

    reasons = {r["channel"]: (f"on in {r['active_weeks']} of {r['weeks']} weeks, {r['spend_share']:.1%} of spend") for r in rare}
    d = Decision(
        id="rare_channels",
        step=STEP,
        question=("These channels are rarely on or tiny: " + "; ".join(f"{c} ({why})" for c, why in reasons.items())
                  + ". Merge each into an 'Other' channel, drop it, or keep it?"),
        options=list(CHOICES),
        default=default_choices(rare),
        blocking=True,
        context={"channels": reasons},
    )
    chosen = answers.ask(d)
    if chosen is None:
        return StepResult(STEP, NAME, Verdict.NEEDS_INPUT,
                          f"{len(rare)} channel(s) are rarely on: {', '.join(reasons)}. Choose what to do with each.",
                          details, [], [d])
    chosen = {**d.default, **chosen}
    bad = {c: v for c, v in chosen.items() if c not in reasons or v not in CHOICES}
    if bad:
        d.answered = False
        return StepResult(STEP, NAME, Verdict.NEEDS_INPUT, f"Unknown channel or choice in {bad}; use other, drop or keep.",
                          details, [], [d])

    by_name = {c.name: c for c in ctx.channels}
    to_other = [by_name[c] for c, v in chosen.items() if v == "other"]
    warnings, done = [], []
    for c, v in chosen.items():
        if v == "drop":
            ctx.drop_channel(by_name[c])
            done.append(f"{c} dropped")
            warnings.append(f"'{c}' dropped; if it drove sales, the baseline now absorbs that effect.")
        elif v == "keep":
            ctx.badge(c, "rarely_on")
            warnings.append(f"'{c}' kept although it is {reasons[c]}; its ROI range will be wide.")
    if len(to_other) == 1:
        warnings.append(f"Only '{to_other[0].name}' was merged into 'Other', which just renames it; it is still rarely on.")
    if to_other:
        merged = ctx.merge_channels(to_other, "other")
        done.append(f"{', '.join(c.name for c in to_other)} merged into '{merged.name}' ({merged.spend})")
    if not ctx.channels:
        return StepResult(STEP, NAME, Verdict.FAIL, "No paid channel is left; keep at least one.", details, warnings, [d])
    details["actions"] = chosen
    return StepResult(STEP, NAME, Verdict.WARN, f"{len(rare)} rarely-on channel(s): " + "; ".join(done) + ".",
                      details, warnings, [d])
