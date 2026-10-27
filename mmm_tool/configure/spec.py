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


def build_model_spec(config: dict[str, Any], input_data):
    from meridian import backend, constants
    from meridian.model import prior_distribution
    from meridian.model import spec as spec_mod

    media = _channels(input_data, "media_channel")
    rf = _channels(input_data, "rf_channel")
    priors = config["roi_priors"]

    def lognormal(names: list[str], name: str):
        mu = np.array([priors[n]["mu"] for n in names], dtype=np.float64)
        sigma = np.array([priors[n]["sigma"] for n in names], dtype=np.float64)
        return backend.tfd.LogNormal(mu, sigma, name=name)

    prior_args = {}
    if media:
        prior_args["roi_m"] = lognormal(media, constants.ROI_M)
    if rf:
        prior_args["roi_rf"] = lognormal(rf, constants.ROI_RF)
    holdout = config.get("holdout")
    return spec_mod.ModelSpec(
        prior=prior_distribution.PriorDistribution(**prior_args),
        media_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI if media else None,
        rf_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI if rf else None,
        max_lag=config["max_lag"],
        knots=config["knots"],
        adstock_decay_spec={ch: shape for ch, shape in config["adstock"].items() if ch in media + rf},
        roi_calibration=_calibration(spec_mod, config, media),
        rf_roi_calibration=_calibration(spec_mod, config, rf),
        holdout=spec_mod.HoldoutSpec([spec_mod.DateRange(holdout["start"], holdout["end"])]) if holdout else None,
    )
