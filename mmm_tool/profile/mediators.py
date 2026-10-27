"""Step 18: marketing-caused variables.

A control that rises one or two weeks after media spend (branded search,
website visits) is caused by marketing. Used as a control it takes the credit
that belongs to the media. Any control whose correlation with a channel's
spend one or two weeks earlier is above 0.6 is flagged, and the user confirms
excluding it (the default) or keeps it.
"""
from __future__ import annotations

from ..ingest.types import Answers, Decision, StepResult, Verdict
from .context import MEDIATOR_LAG_CORR, MEDIATOR_LAGS, CCtx

STEP = 18
NAME = "Marketing-caused variables"


def _is_flag(s) -> bool:
    return set(s.dropna().unique()) <= {0.0, 1.0}


def check_mediators(ctx: CCtx, answers: Answers) -> StepResult:
    controls = [c for c in ctx.cols("control", "non_media_treatment") if not _is_flag(ctx.df[c]) and c != "kpi_imputed"]
    spend = ctx.spend()
    spend["all media"] = spend.sum(axis=1)
    level = ctx.weekly_mean(controls) if controls else None
    found = []
    for col in controls:
        best = (0.0, None, None)
        for ch in spend.columns:
            for lag in MEDIATOR_LAGS:
                r = level[col].corr(spend[ch].shift(lag))
                if r == r and r > best[0]:
                    best = (float(r), ch, lag)
        if best[0] > MEDIATOR_LAG_CORR:
            found.append({"column": col, "channel": best[1], "lag": best[2], "correlation": round(best[0], 3)})
    details = {"checked": controls, "flagged": found, "threshold": MEDIATOR_LAG_CORR}
    if not found:
        return StepResult(STEP, NAME, Verdict.PASS,
                          f"None of the {len(controls)} control(s) rises after media spend.", details)

    decisions, warnings, pending, excluded = [], [], False, []
    for f in found:
        d = Decision(
            id=f"mediator:{f['column']}",
            step=STEP,
            question=(f"'{f['column']}' rises {f['lag']} week(s) after {f['channel']} spend (correlation "
                      f"{f['correlation']:.2f}), so marketing probably causes it. As a control it would take media's "
                      "credit. Exclude it?"),
            options=["exclude", "keep"],
            default="exclude",
            blocking=True,
            context=f,
        )
        decisions.append(d)
        choice = answers.ask(d)
        if choice is None:
            pending = True
        elif choice == "exclude":
            ctx.roles[f["column"]] = "excluded"
            ctx.df = ctx.df.drop(columns=[f["column"]])
            excluded.append(f["column"])
        else:
            warnings.append(f"'{f['column']}' kept as a control although it follows {f['channel']} spend; "
                            f"{f['channel']}'s ROI will be understated.")
    if pending:
        return StepResult(STEP, NAME, Verdict.NEEDS_INPUT,
                          f"{len(found)} control(s) look caused by marketing; confirm whether to exclude each.",
                          details, warnings, decisions)
    summary = f"{len(found)} control(s) follow media spend" + (f"; excluded: {', '.join(excluded)}" if excluded else "; kept")
    return StepResult(STEP, NAME, Verdict.WARN if warnings or excluded else Verdict.PASS, summary + ".", details, warnings, decisions)
