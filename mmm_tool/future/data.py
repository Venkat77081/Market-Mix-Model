"""build_future_data: Meridian's DataTensors for a future plan.

Time axis: the max_lag lead-in weeks, then the plan weeks. The lead-in copies the weeks named by
period.lead_in unchanged, so carryover from earlier spend reaches the plan; results are only ever
reported for the plan weeks. Plan weeks copy the template weeks, then for each paid channel c:

  multipliers   spend = template spend x m_c (every week, so the shape stays);
                units = template units x m_c / (1 + cost change_c); reach likewise, frequency kept
  spend         a planned amount per week (an upload, or actual spend in Tracking), split across
                regions in the template's shares; units = template units x planned / template
                spend (the template's cost per unit where it spent nothing), / (1 + cost change)

A channel whose media units are its spend (no exposure column) ignores the cost change.
Controls, organic media and non-media levers are the template's, unchanged (they don't move the
incremental media outcome, which is all that is used here). Revenue per KPI is the template's,
or one number per region when given.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class FutureDataError(ValueError):
    """The future data can't be built as asked: says which channel and week."""


def channels(mmm) -> list[str]:
    """Paid channels in Meridian's order: media channels, then reach-and-frequency channels."""
    d = mmm.input_data
    out = []
    if d.media is not None:
        out += [str(c) for c in d.media.media_channel.values]
    if d.reach is not None:
        out += [str(c) for c in d.reach.rf_channel.values]
    return out


def spend_only(mmm) -> dict[str, bool]:
    """Channels whose media units are their spend (the cost change can't apply to them)."""
    d, out = mmm.input_data, {}
    if d.media is not None:
        off = d.media.shape[1] - d.media_spend.shape[1] if np.ndim(d.media_spend) == 3 else 0
        for i, c in enumerate(str(x) for x in d.media.media_channel.values):
            m = np.asarray(d.media.values[:, off:, i], float)
            s = np.asarray(d.media_spend.values[..., i], float) if np.ndim(d.media_spend) == 3 else None
            out[c] = bool(s is not None and m.shape == s.shape and np.allclose(m, s))
    if d.reach is not None:
        out.update({str(c): False for c in d.reach.rf_channel.values})
    return out


def weeks(mmm) -> list[str]:
    return [str(t)[:10] for t in mmm.input_data.time.values]


def _time_axis(arr, n_times: int) -> int:
    """Offset of the time dimension: media arrays may carry extra history (media_time)."""
    return arr.shape[1] - n_times


def template_spend(mmm, template: list[str]) -> dict[str, list[float]]:
    """Each paid channel's national spend in each template week."""
    d, ws = mmm.input_data, weeks(mmm)
    idx = [ws.index(w) for w in template]
    out = {}
    for arr, dim in ((d.media_spend, "media_channel"), (d.rf_spend, "rf_channel")):
        if arr is None:
            continue
        x = np.asarray(arr.values, float)
        if x.ndim != 3:
            raise FutureDataError("This model's spend isn't given week by week, so a future plan can't be built from it.")
        for i, c in enumerate(str(v) for v in arr[dim].values):
            out[c] = x[:, idx, i].sum(axis=0).tolist()
    return out


def build_future_data(mmm, resolved: dict[str, Any], mult: dict[str, float] | None = None,
                      spend: dict[str, list[float]] | None = None, cost: dict[str, float] | None = None,
                      rpk: dict[str, float] | None = None, units: dict[str, list[float]] | None = None):
    """DataTensors for resolved = period.resolve(...): lead-in + plan weeks. Give multipliers (per
    channel, 1.0 = the template) or spend (per channel, one amount per plan week); not both.
    units: actual media units per plan week (Tracking; None for a week without actuals), used
    instead of the scaled template's."""
    from meridian.analysis import tensors

    d, ws = mmm.input_data, weeks(mmm)
    plan, tmpl, lead_src, lead_dates = resolved["plan"], resolved["template"], resolved["lead_src"], resolved["lead_dates"]
    P, L = len(plan), len(lead_dates)
    try:
        src = [ws.index(w) for w in lead_src + tmpl]
    except ValueError as e:
        raise FutureDataError(f"A template or lead-in week isn't in the model's data: {e}") from e
    paid, only = channels(mmm), spend_only(mmm)
    mult = {c: float((mult or {}).get(c, 1.0)) for c in paid}
    cost = {c: float((cost or {}).get(c, 0.0) or 0.0) for c in paid}
    for c, v in cost.items():
        if v <= -1:
            raise FutureDataError(f"'{c}': a cost change must be above −100%.")
    unknown = sorted((set(spend or {}) | set(units or {})) - set(paid))
    if unknown:
        raise FutureDataError(f"Not channels of this model: {unknown}.")
    n = len(ws)

    def take(arr):
        if arr is None:
            return None
        x = np.asarray(arr.values, float)
        off = _time_axis(x, n)
        return x[:, [off + i for i in src], ...].copy()

    kw: dict[str, Any] = {}
    for name in ("media", "media_spend", "reach", "frequency", "rf_spend", "organic_media", "organic_reach",
                 "organic_frequency", "non_media_treatments", "controls", "revenue_per_kpi"):
        arr = getattr(d, name, None)
        if arr is not None:
            if name in ("media_spend", "rf_spend") and np.ndim(arr) != 3:
                raise FutureDataError("This model's spend isn't given week by week, so a future plan can't be built from it.")
            kw[name] = take(arr)
    plan_sl = slice(L, L + P)

    def scale(units_key: str, spend_key: str, names: list[str], unit_cost: bool):
        if kw.get(spend_key) is None:
            return
        for i, c in enumerate(names):
            s0 = kw[spend_key][:, plan_sl, i].copy()             # template spend, (geo, P)
            u0 = kw[units_key][:, plan_sl, i].copy() if kw.get(units_key) is not None else None
            if spend and c in spend:
                want = np.asarray(spend[c], float)
                if want.shape != (P,):
                    raise FutureDataError(f"'{c}': planned spend needs {P} weekly amounts, one per plan week; got {want.size}.")
                if (want < 0).any():
                    w = plan[int(np.argmax(want < 0))]
                    raise FutureDataError(f"'{c}': planned spend is negative in the week of {w}.")
                wk_total = s0.sum(axis=0)
                share = np.where(wk_total > 0, s0 / np.where(wk_total > 0, wk_total, 1), np.nan)
                period_share = s0.sum(axis=1) / s0.sum() if s0.sum() > 0 else np.full(s0.shape[0], 1 / s0.shape[0])
                share = np.where(np.isnan(share), period_share[:, None], share)
                s1 = share * want[None, :]
                if u0 is not None:
                    per_unit = np.where(s0 > 0, u0 / np.where(s0 > 0, s0, 1), np.nan)    # units per unit of spend
                    avg = (u0.sum() / s0.sum()) if s0.sum() > 0 else 1.0
                    per_unit = np.where(np.isnan(per_unit), avg, per_unit)
                    u1 = s1 * per_unit
                else:
                    u1 = None
            else:
                s1 = s0 * mult[c]
                u1 = u0 * mult[c] if u0 is not None else None
            kw[spend_key][:, plan_sl, i] = s1
            if u1 is None:
                continue
            if only.get(c, False):
                u1 = s1                                           # its units are its spend
            else:
                if unit_cost:
                    u1 = u1 / (1 + cost[c])
                if units and c in units:                          # actual units where known (None: the plan's)
                    uu = np.asarray([np.nan if x is None else x for x in units[c]], float)
                    if uu.shape != (P,):
                        raise FutureDataError(f"'{c}': media units need {P} weekly amounts; got {uu.size}.")
                    tot = s1.sum(axis=0, keepdims=True)
                    geo = np.where(tot > 0, s1 / np.where(tot > 0, tot, 1), 1 / s1.shape[0])
                    known = ~np.isnan(uu)
                    u1 = np.where(known[None, :], geo * np.nan_to_num(uu)[None, :], u1)
            kw[units_key][:, plan_sl, i] = u1

    if d.media is not None:
        scale("media", "media_spend", [str(c) for c in d.media.media_channel.values], True)
    if d.reach is not None:
        scale("reach", "rf_spend", [str(c) for c in d.reach.rf_channel.values], True)
    if rpk and kw.get("revenue_per_kpi") is not None:
        geos = [str(g) for g in d.geo.values]
        for gi, g in enumerate(geos):
            if g in rpk and rpk[g] is not None:
                kw["revenue_per_kpi"][gi, plan_sl] = float(rpk[g])
    # Checks: shapes, no gaps, no negative spend, every channel present.
    for k, v in kw.items():
        if np.isnan(v).any():
            pos = np.argwhere(np.isnan(v))[0]
            raise FutureDataError(f"'{k}' has no value in the week of {(lead_dates + plan)[pos[1]]}.")
    for key, names in (("media_spend", [str(c) for c in d.media.media_channel.values] if d.media is not None else []),
                       ("rf_spend", [str(c) for c in d.reach.rf_channel.values] if d.reach is not None else [])):
        if kw.get(key) is None:
            continue
        if kw[key].shape[2] != len(names):
            raise FutureDataError("Every paid channel must be in the plan.")
        bad = np.argwhere(kw[key] < 0)
        if len(bad):
            g, t, i = bad[0]
            raise FutureDataError(f"'{names[i]}' has negative spend in the week of {(lead_dates + plan)[t]}.")
    kw["time"] = lead_dates + plan
    planned = {}
    for key, names in (("media_spend", [str(c) for c in d.media.media_channel.values] if d.media is not None else []),
                       ("rf_spend", [str(c) for c in d.reach.rf_channel.values] if d.reach is not None else [])):
        if kw.get(key) is not None:
            for i, c in enumerate(names):
                planned[c] = kw[key][:, plan_sl, i].sum(axis=0).tolist()
    info = {"plan": plan, "lead_in": lead_dates, "template": tmpl, "spend": planned,
            "total_spend": float(sum(sum(v) for v in planned.values()))}
    return tensors.DataTensors(**kw), info
