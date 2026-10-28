"""Turn Phase D's config into Meridian objects (needs google-meridian).

    input_data = build_input_data(result_d)
    model_spec = build_model_spec(result_d.config, input_data)
    mmm = meridian.model.model.Meridian(input_data=input_data, model_spec=model_spec)

Priors use Meridian's own backend distributions (backend.tfd), which Meridian
requires; plain TensorFlow Probability distributions are refused.
"""
from __future__ import annotations

from typing import Any

import numpy as np

OPEN_BOUND_SDS = 10.0       # where a one-sided control prior (step 21) is cut on its open side


def build_input_data(result):
    """InputData from a complete Phase C or D result (its clean weekly table)."""
    from meridian.data import load

    m = result.coord_to_columns()
    ctc = dict(m["coord_to_columns"])
    df = result.table.copy()
    date_col = ctc["time"]
    df[date_col] = df[date_col].dt.strftime("%Y-%m-%d")
    if "population" not in ctc and "geo" in ctc:
        ctc.pop("geo")
    loader = load.DataFrameDataLoader(
        df=df,
        coord_to_columns=load.CoordToColumns(**ctc),
        kpi_type=m["kpi_type"],
        media_to_channel=m.get("media_to_channel"),
        media_spend_to_channel=m.get("media_spend_to_channel"),
        reach_to_channel=m.get("reach_to_channel"),
        frequency_to_channel=m.get("frequency_to_channel"),
        rf_spend_to_channel=m.get("rf_spend_to_channel"),
    )
    return loader.load()


def _channels(input_data, dim: str) -> list[str]:
    coord = getattr(input_data, dim, None)
    return [str(x) for x in coord.values] if coord is not None else []


def _calibration(spec_mod, config: dict[str, Any], channels: list[str]):
    entries = [
        spec_mod.ChannelCalibrationSpec(
            channels=[ch],
            date_ranges=[spec_mod.DateRange(p["start"], p["end"]) for p in config["calibration"][ch]["periods"]],
        )
        for ch in channels if ch in config.get("calibration", {})
    ]
    return spec_mod.CalibrationSpec(entries) if entries else None


def _vector(names: list[str], values: dict[str, Any], key, default) -> np.ndarray:
    """One float64 entry per channel (or control) in Meridian's order, looked up by name."""
    return np.array([key(values[n]) if n in values else default for n in names], dtype=np.float64)


def build_model_spec(config: dict[str, Any], input_data):
    """Meridian's ModelSpec from Phase D's config. Every per-channel vector follows
    input_data's own channel order (media_channel, rf_channel, control_variable), built by
    name; distributions are Meridian's backend.tfd with float64 arrays."""
    from meridian import backend, constants
    from meridian.model import prior_distribution
    from meridian.model import spec as spec_mod

    from .choices import direction_bounds, meridian_defaults

    tfd = backend.tfd
    media = _channels(input_data, "media_channel")
    rf = _channels(input_data, "rf_channel")
    kind = config.get("prior_type", "roi")
    priors = config["roi_priors"]
    defaults = meridian_defaults()
    family, d1, d2 = defaults[kind]
    names = {"roi": (constants.ROI_M, constants.ROI_RF), "mroi": (constants.MROI_M, constants.MROI_RF),
             "contribution": (constants.CONTRIBUTION_M, constants.CONTRIBUTION_RF)}[kind]

    def treatment(chans: list[str], name: str):
        p1 = _vector(chans, priors, lambda p: p.get("p1", p.get("mu")), d1)
        p2 = _vector(chans, priors, lambda p: p.get("p2", p.get("sigma")), d2)
        return tfd.LogNormal(p1, p2, name=name) if family == "lognormal" else tfd.Beta(p1, p2, name=name)

    prior_args = {}
    if media:
        prior_args[names[0]] = treatment(media, names[0])
    if rf:
        prior_args[names[1]] = treatment(rf, names[1])

    # Carryover: one Beta on alpha, per channel; Beta(1, 1) (= Meridian's Uniform(0, 1)) where unknown.
    carry = config.get("carryover_prior") or {}
    if any(v.get("speed", "dont_know") != "dont_know" for v in carry.values()):
        for chans, name in ((media, constants.ALPHA_M), (rf, constants.ALPHA_RF)):
            if chans:
                a = _vector(chans, carry, lambda v: v["beta"][0], 1.0)
                b = _vector(chans, carry, lambda v: v["beta"][1], 1.0)
                prior_args[name] = tfd.Beta(a, b, name=name)

    # Control directions: a TruncatedNormal on each control's coefficient, at Meridian's default scale.
    dirs = config.get("control_direction") or {}
    controls = _channels(input_data, "control_variable")
    if controls and any(dirs.get(c, "either") != "either" for c in controls):
        # The open side is cut at 10 prior sds (mass beyond it ~1e-23): TFP samples a TruncatedNormal
        # through a sigmoid between low and high, which turns an infinite bound into NaN draws.
        scale = float(defaults["gamma_c_scale"])
        far = OPEN_BOUND_SDS * scale
        bounds = [direction_bounds(dirs.get(c, "either")) for c in controls]
        prior_args[constants.GAMMA_C] = tfd.TruncatedNormal(
            loc=np.zeros(len(controls), dtype=np.float64),
            scale=np.full(len(controls), scale, dtype=np.float64),
            low=np.array([max(lo, -far) for lo, _ in bounds], dtype=np.float64),
            high=np.array([min(hi, far) for _, hi in bounds], dtype=np.float64), name=constants.GAMMA_C)

    saturation = {ch: v for ch, v in (config.get("saturation") or {}).items() if ch in media + rf and v == "none"}
    holdout = config.get("holdout")
    return spec_mod.ModelSpec(
        prior=prior_distribution.PriorDistribution(**prior_args),
        media_prior_type=kind if media else None,
        rf_prior_type=kind if rf else None,
        max_lag=config["max_lag"],
        knots=config["knots"],
        adstock_decay_spec={ch: shape for ch, shape in config["adstock"].items() if ch in media + rf},
        saturation_spec=saturation if saturation else constants.HILL,
        roi_calibration=_calibration(spec_mod, config, media) if kind == "roi" else None,
        rf_roi_calibration=_calibration(spec_mod, config, rf) if kind == "roi" else None,
        holdout=spec_mod.HoldoutSpec([spec_mod.DateRange(holdout["start"], holdout["end"])]) if holdout else None,
    )
