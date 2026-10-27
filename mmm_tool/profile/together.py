"""Step 17: channels moving together.

Two channels whose weekly spend correlates above 0.7 can't be told apart:
the model splits their joint effect arbitrarily. For each such pair the user
combines them into one channel (the default) or keeps both with a badge.
Channels with a variance inflation factor above 10 that aren't in such a pair
(several channels moving together) get a warning.
"""
from __future__ import annotations

from ..ingest.types import Answers, Decision, StepResult, Verdict
from .context import COLLINEAR_CORR, COLLINEAR_VIF, CCtx, vif

STEP = 17
NAME = "Channels moving together"


def check_together(ctx: CCtx, answers: Answers) -> StepResult:
    spend = ctx.spend()
    names = list(spend.columns)
    corr = spend.corr().fillna(0.0)
    pairs = [(a, b, round(float(corr.at[a, b]), 3)) for i, a in enumerate(names) for b in names[i + 1:]
             if corr.at[a, b] > COLLINEAR_CORR]
    pairs.sort(key=lambda p: -p[2])
    vifs = {k: round(v, 1) for k, v in vif(spend).items()} if len(names) > 2 else {}
    details = {"correlation": {a: {b: round(float(corr.at[a, b]), 3) for b in names} for a in names},
               "vif": vifs, "pairs": [{"a": a, "b": b, "correlation": r} for a, b, r in pairs]}

    decisions, warnings, done = [], [], []
    by_name = {c.name: c for c in ctx.channels}
    merged: set[str] = set()
    pending = False
    for a, b, r in pairs:
        d = Decision(
            id=f"together:{a}|{b}",
            step=STEP,
            question=(f"'{a}' and '{b}' move together (correlation {r:.2f}), so the model can't tell their effects "
                      "apart. Combine them into one channel, or keep both?"),
            options=["combine", "keep"],
            default="combine",
            blocking=True,
            context={"a": a, "b": b, "correlation": r},
        )
        decisions.append(d)
        choice = answers.ask(d)
        if choice is None:
            pending = True
            continue
        if choice == "combine":
            if a in merged or b in merged:
                warnings.append(f"'{a}' and '{b}' not combined: one of them was already combined with another channel.")
                continue
            new = ctx.merge_channels([by_name[a], by_name[b]], f"{a}_{b}")
            merged |= {a, b}
            done.append(f"{a} + {b} → {new.name}")
        else:
            for x, y in ((a, b), (b, a)):
                ctx.badge(x, "moves_with")
            warnings.append(f"'{a}' and '{b}' kept separate although they move together (correlation {r:.2f}); "
                            "their ROI ranges will be wide and may trade off against each other.")
    if pending:
        return StepResult(STEP, NAME, Verdict.NEEDS_INPUT,
                          f"{len(pairs)} pair(s) of channels move together; choose whether to combine each.",
                          details, warnings, decisions)

    in_pairs = {x for a, b, _ in pairs for x in (a, b)}
    high = {c: v for c, v in vifs.items() if v > COLLINEAR_VIF and c not in in_pairs}
    for c, v in high.items():
        warnings.append(f"'{c}' is largely explained by the other channels together (VIF {v:g}, over {COLLINEAR_VIF}).")
    if not pairs and not high:
        top = max((abs(corr.at[a, b]) for i, a in enumerate(names) for b in names[i + 1:]), default=0.0)
        return StepResult(STEP, NAME, Verdict.PASS,
                          f"No two channels move together (highest correlation {top:.2f}"
                          + (f", highest VIF {max(vifs.values()):.1f}" if vifs else "") + ").", details)
    summary = "; ".join(done) if done else f"{len(pairs)} pair(s) kept separate" if pairs else "no pairs above 0.7"
    if high:
        summary += f"; high VIF for {', '.join(high)}"
    return StepResult(STEP, NAME, Verdict.WARN if warnings else Verdict.PASS, summary + ".", details, warnings, decisions)
