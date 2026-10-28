"""State shared by the Phase E steps, and the settings from fit.yaml."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import yaml

SETTINGS_FILE = Path(__file__).with_name("fit.yaml")


@lru_cache(maxsize=1)
def settings() -> dict[str, Any]:
    return yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))


def sampler_settings(sampler: str | dict) -> dict[str, Any]:
    """A named sampler from fit.yaml ("quick", "full") or explicit numbers."""
    if isinstance(sampler, dict):
        out = {"label": "Custom", **sampler}
    else:
        presets = settings()["samplers"]
        if sampler not in presets:
            raise ValueError(f"Unknown sampler '{sampler}'; choose {', '.join(presets)}.")
        out = dict(presets[sampler])
    for k in ("n_chains", "n_adapt", "n_burnin", "n_keep"):
        out[k] = int(out[k])
    if out["n_chains"] < 2:
        raise ValueError("R-hat (step 30) needs at least 2 chains.")
    return out


# Custom sampler (step 29): hard minimums, and the recommended ones below which an amber hint shows.
CUSTOM_MIN = {"n_chains": 2, "n_adapt": 100, "n_burnin": 100, "n_keep": 100}
CUSTOM_RECOMMENDED = {"n_chains": 4, "n_keep": 500}
CUSTOM_LABEL = {"n_chains": "chains", "n_adapt": "adapt", "n_burnin": "burn-in", "n_keep": "kept"}


def check_custom(sampler: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(errors, hints) for the Custom sampler the user typed."""
    errors, hints = [], []
    for k, lo in CUSTOM_MIN.items():
        try:
            v = int(sampler.get(k))
        except (TypeError, ValueError):
            errors.append(f"{CUSTOM_LABEL[k]} must be a whole number")
            continue
        if v < lo:
            errors.append(f"{CUSTOM_LABEL[k]} must be at least {lo}" + (" (R-hat needs two chains to compare)" if k == "n_chains" else ""))
    if not errors:
        if int(sampler["n_chains"]) < CUSTOM_RECOMMENDED["n_chains"]:
            hints.append("Fewer than 4 chains: R-hat (step 30) has less to compare.")
        if int(sampler["n_keep"]) < CUSTOM_RECOMMENDED["n_keep"]:
            hints.append("Fewer than 500 kept draws: ROI ranges will be rougher.")
    return errors, hints


@dataclass
class FitOptions:
    sampler: str | dict = "quick"
    stability: bool = False   # step 34: three extra fits
    placebo: bool = False     # step 35: one extra fit
    seed: int | None = None
    continue_unconverged: bool = False   # step 30 failed and the user chose "Continue anyway"
    seed_source: str = "default"         # "user" when the seed was typed in

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "FitOptions":
        d = d or {}
        return cls(sampler=d.get("sampler", "quick"), stability=bool(d.get("stability")),
                   placebo=bool(d.get("placebo")), seed=d.get("seed"),
                   continue_unconverged=bool(d.get("continue_unconverged")), seed_source=d.get("seed_source", "default"))

    def to_dict(self) -> dict[str, Any]:
        return {"sampler": self.sampler, "stability": self.stability, "placebo": self.placebo, "seed": self.seed,
                "continue_unconverged": self.continue_unconverged, "seed_source": self.seed_source}


Progress = Callable[[int, str, float], None]


@dataclass
class ECtx:
    d: Any                     # complete Phase D result
    options: FitOptions
    out_dir: Path | None
    progress: Progress
    sampler: dict[str, Any] = field(default_factory=dict)
    seed: int = 7
    badges: dict[str, list[str]] = field(default_factory=dict)
    mmm: Any = None            # the fit with the holdout (steps 28-35)
    metrics: Any = None        # per-channel ROI and contributions from that fit
    gates: dict[int, bool] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)   # step 36: where the final model is saved
    fit_sampler: dict[str, Any] = field(default_factory=dict)   # what the later refits use (step 30 may lengthen warm-up)
    not_converged: bool = False                                 # step 30 failed and the user continued anyway
    log: list[dict[str, Any]] = field(default_factory=list)     # Phase E's decision-log lines
    final: Any = None          # step 36: the fit on every week
    timings: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.sampler = sampler_settings(self.options.sampler)
        self.fit_sampler = dict(self.sampler)
        self.seed = int(self.options.seed if self.options.seed is not None else settings()["seed"])
        self.badges = copy.deepcopy(dict(self.d.badges))

    @property
    def config(self) -> dict[str, Any]:
        return self.d.config

    @property
    def cfg(self) -> dict[str, Any]:
        return settings()

    @property
    def confidence(self) -> float:
        return float(settings()["confidence"])

    def badge(self, channel: str, badge: str) -> None:
        got = self.badges.setdefault(channel, [])
        if badge not in got:
            got.append(badge)
