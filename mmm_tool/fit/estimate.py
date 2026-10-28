"""How long a fit will take on this machine.

Every completed fit is logged (fit_times.jsonl): weeks, regions, channels, the
sampler settings, CPU cores and seconds. The estimate for one fit is

    seconds = a + b * size * draws_per_chain * ceil(chains / cores)

with size = weeks x regions x channels (paid, unpaid and reach-frequency) and
draws_per_chain = adapt + burn-in + kept; a and b are fitted by least squares on
this machine's log. a, the start-up cost (building and compiling the model), is
large, so small runs don't get proportionally faster.

With fewer than 3 logged runs a calibration (two fits of the bundled sample:
a short one and the Quick preset, done once and kept in the log) sets a and b,
and the estimate is labelled rough.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

MIN_RUNS = 3


def log_path() -> Path:
    env = os.environ.get("MMM_FIT_LOG")
    return Path(env) if env else Path.home() / ".cache" / "mmm_tool" / "fit_times.jsonl"


def cores() -> int:
    return os.cpu_count() or 1


def records(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or log_path()
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def record(n_weeks: int, n_geos: int, n_channels: int, sampler: dict, seconds: float, calibration: bool = False,
           path: Path | None = None) -> None:
    path = path or log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"n_weeks": n_weeks, "n_geos": n_geos, "n_channels": n_channels, "chains": int(sampler["n_chains"]),
           "adapt": int(sampler["n_adapt"]), "burnin": int(sampler["n_burnin"]), "kept": int(sampler["n_keep"]),
           "cores": cores(), "seconds": round(float(seconds), 2), "calibration": calibration, "at": time.time()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def work(n_weeks: int, n_geos: int, n_channels: int, chains: int, draws: int, n_cores: int) -> float:
    return n_weeks * n_geos * n_channels * draws * math.ceil(chains / max(1, n_cores))


def _x(r: dict) -> float:
    return work(r["n_weeks"], r["n_geos"], r["n_channels"], r["chains"], r["adapt"] + r["burnin"] + r["kept"], r["cores"])


def fit_ab(rows: list[dict]) -> tuple[float, float] | None:
    """Least squares a + b x, with a and b kept at zero or above. None if the log can't say."""
    pts = [(_x(r), float(r["seconds"])) for r in rows]
    if len({x for x, _ in pts}) < 2:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    b = max(0.0, sum((x - mx) * (y - my) for x, y in pts) / sxx) if sxx else 0.0
    a = max(0.0, my - b * mx)
    return a, b


def model(path: Path | None = None) -> dict[str, Any]:
    """a, b and how they were found, for this machine's core count."""
    rows = [r for r in records(path) if r.get("cores") == cores()]
    real = [r for r in rows if not r.get("calibration")]
    ab = fit_ab(real) if len(real) >= MIN_RUNS else None
    basis = f"from {len(real)} past runs" if ab else None
    if ab is None:
        ab = fit_ab(rows)                    # calibration runs, with any real ones
        basis = "rough" if ab else None
    return {"a": ab[0] if ab else None, "b": ab[1] if ab else None, "basis": basis, "runs": len(real), "cores": cores(),
            "calibrated": any(r.get("calibration") for r in rows)}


def estimate(n_weeks: int, n_geos: int, n_channels: int, sampler: dict, fits: int = 1, path: Path | None = None) -> dict | None:
    m = model(path)
    if m["a"] is None:
        return None
    per = m["a"] + m["b"] * work(n_weeks, n_geos, n_channels, int(sampler["n_chains"]),
                                 int(sampler["n_adapt"]) + int(sampler["n_burnin"]) + int(sampler["n_keep"]), m["cores"])
    return {"seconds": per * fits, "per_fit": per, "basis": m["basis"], "a": m["a"], "b": m["b"]}


def text(seconds: float, basis: str) -> str:
    mins = seconds / 60
    when = f"about {max(1, round(mins))} min" if mins >= 1 else f"about {max(10, round(seconds, -1)):.0f} s"
    return f"{when} ({basis})"


def model_size(mmm) -> tuple[int, int, int]:
    """(weeks, regions, channels) of a Meridian model: paid, unpaid and reach-frequency channels."""
    data = mmm.input_data
    n = lambda dim: len(getattr(data, dim).values) if getattr(data, dim, None) is not None else 0  # noqa: E731
    return n("time"), n("geo"), n("media_channel") + n("rf_channel") + n("organic_media_channel") + n("organic_rf_channel")


def calibrate(path: Path | None = None) -> dict[str, Any]:
    """Two fits of the bundled sample (a short one and the Quick preset), logged as calibration."""
    import tempfile

    from .. import samples
    from ..pipeline import run_pipeline
    from . import model as m
    from .context import sampler_settings

    with tempfile.TemporaryDirectory() as tmp:
        r = run_pipeline(samples.write_clean(Path(tmp)), {}, accept_defaults=True)
        cfg = {**r.d.config, "holdout": None}
        for sampler in ({"n_chains": 2, "n_adapt": 50, "n_burnin": 50, "n_keep": 50}, sampler_settings("quick")):
            mmm = m.build(r.d, cfg)
            mmm.sample_prior(50, seed=1)
            t = time.time()
            m.sample_posterior(mmm, sampler, 1)
            record(*model_size(mmm), sampler, time.time() - t, calibration=True, path=path)
    return model(path)
