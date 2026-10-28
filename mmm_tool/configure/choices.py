"""Prior choices the user makes in steps 21, 23 and 25, and the maths behind them.

Everything here works on plain numbers, so Phase D can show what a choice
means without Meridian installed; spec.py turns the same numbers into
Meridian's float64 distributions, in Meridian's own channel and control order.

Carryover (step 23)   "share of the effect still working next week"
    Don't know  Meridian's default for the decay rate alpha: Uniform(0, 1), i.e. Beta(1, 1)
    Fast        about 20%      Medium  about 40%      Slow  about 70%
  For geometric decay next week's share is alpha itself, so the presets are
  Beta(2, 8), Beta(4, 6) and Beta(7, 3). Meridian's binomial decay uses
  weights (1 - lag / W) ** (1 / alpha - 1) over a window of W = max_lag + 1
  weeks (adstock_hill.py, _map_alpha_for_binomial_decay), so next week's share
  is (1 - 1 / W) ** (1 / alpha - 1): the presets are converted through that.

Prior type (step 25)  one choice for all paid media
    roi           average return: LogNormal from a 90% range
    mroi          return on the next rupee: LogNormal from a 90% range
    contribution  share of sales: a Beta whose 5th and 95th percentiles match the range

Control direction (step 21)  Either / Raises sales / Lowers sales, a
    TruncatedNormal on the control's coefficient (gamma_c).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

Z90 = 1.6448536269514722            # a 90% range spans +/- 1.645 standard deviations

# Meridian 2.1's defaults, used to show what a blank choice means when Meridian
# isn't installed. When it is, spec.py and meridian_defaults() read the real ones.
FALLBACK_DEFAULTS = {
    "roi": ("lognormal", 0.2, 0.9), "mroi": ("lognormal", 0.0, 0.5), "contribution": ("beta", 1.0, 99.0),
    "gamma_c_scale": 5.0,
}

# --- carryover ------------------------------------------------------------------

SPEEDS = {"dont_know": None, "fast": 0.2, "medium": 0.4, "slow": 0.7}
GEOMETRIC_BETA = {"dont_know": (1.0, 1.0), "fast": (2.0, 8.0), "medium": (4.0, 6.0), "slow": (7.0, 3.0)}
PRESET_CONCENTRATION = 10.0          # the presets' a + b: how sure "about 20%" is
SPEED_LABEL = {"dont_know": "don't know", "fast": "fast (about 20% left next week)",
               "medium": "medium (about 40% left next week)", "slow": "slow (about 70% left next week)"}


def next_week_share(alpha, shape: str, max_lag: int):
    """Share of a week's effect still working the week after, for decay rate alpha."""
    a = np.asarray(alpha, float)
    if shape == "binomial":
        w = max_lag + 1
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(a > 0, (1 - 1 / w) ** (1 / np.clip(a, 1e-12, None) - 1), 0.0)
    return a


def alpha_for_share(share: float, shape: str, max_lag: int) -> float:
    """The decay rate alpha that leaves `share` of the effect next week."""
    if shape != "binomial":
        return share
    w = max_lag + 1
    return 1.0 / (1.0 + math.log(share) / math.log(1 - 1 / w))


def carryover_beta(speed: str, shape: str, max_lag: int) -> tuple[float, float]:
    """Beta(a, b) prior on alpha for a carryover speed and decay shape."""
    if speed == "dont_know" or SPEEDS.get(speed) is None:
        return GEOMETRIC_BETA["dont_know"]
    if shape != "binomial":
        return GEOMETRIC_BETA[speed]
    # The share-alpha map is strongly curved for binomial decay, so match the median
    # (quantiles map across exactly), not the mean: solve for the Beta whose median alpha
    # leaves the preset share next week.
    from scipy import optimize, stats

    target = alpha_for_share(SPEEDS[speed], shape, max_lag)
    c = PRESET_CONCENTRATION
    m = optimize.brentq(lambda m: stats.beta(c * m, c * (1 - m)).median() - target, 1e-4, 1 - 1e-4)
    return c * m, c * (1 - m)


def carryover_summary(a: float, b: float, shape: str, max_lag: int) -> dict[str, float]:
    """Median and 90% range of next week's share under Beta(a, b) on alpha."""
    from scipy import stats

    q = stats.beta(a, b).ppf([0.05, 0.5, 0.95])
    s = next_week_share(q, shape, max_lag)            # monotonic in alpha, so quantiles map across
    return {"low": float(s[0]), "median": float(s[1]), "high": float(s[2])}


# --- prior type ranges -----------------------------------------------------------

PRIOR_TYPES = {"roi": "average return (ROI)", "mroi": "return on the next rupee", "contribution": "share of sales"}


def lognormal_from_range(low: float, high: float) -> tuple[float, float]:
    """LogNormal(mu, sigma) whose 90% interval is [low, high]."""
    return math.log(math.sqrt(low * high)), math.log(high / low) / (2 * Z90)


def beta_from_range(low: float, high: float) -> tuple[float, float]:
    """Beta(a, b) whose 5th and 95th percentiles are low and high (0 < low < high < 1), solved
    numerically on log a and log b and checked: raises ValueError if no Beta matches."""
    from scipy import optimize, stats

    if not 0 < low < high < 1:
        raise ValueError("a share of sales must be between 0 and 1, low below high")
    mid = (low + high) / 2
    spread = (high - low) / (2 * Z90)
    var = min(spread ** 2, mid * (1 - mid) * 0.99)
    k = mid * (1 - mid) / var - 1                     # method-of-moments start
    x0 = np.log([max(mid * k, 1e-3), max((1 - mid) * k, 1e-3)])

    def err(x):
        a, b = np.exp(x)
        q = stats.beta(a, b).ppf([0.05, 0.95])
        return [math.log(q[0] / low), math.log(q[1] / high)]

    sol = optimize.least_squares(err, x0, bounds=([-10, -10], [12, 12]))
    a, b = (float(v) for v in np.exp(sol.x))
    q = stats.beta(a, b).ppf([0.05, 0.95])
    if not (abs(q[0] / low - 1) < 0.01 and abs(q[1] / high - 1) < 0.01):
        raise ValueError(f"no Beta has its 90% range at {low:g}-{high:g}")
    return a, b


def prior_from_range(kind: str, low: float, high: float) -> tuple[str, float, float]:
    """(family, p1, p2) for a prior type and a 90% range."""
    if kind == "contribution":
        return ("beta", *beta_from_range(low, high))
    if not 0 < low < high:
        raise ValueError("both ends above zero, low below high")
    return ("lognormal", *lognormal_from_range(low, high))


def prior_summary(family: str, p1: float, p2: float) -> dict[str, float]:
    """Median and 90% range of the built distribution."""
    from scipy import stats

    d = stats.lognorm(s=p2, scale=math.exp(p1)) if family == "lognormal" else stats.beta(p1, p2)
    lo, med, hi = d.ppf([0.05, 0.5, 0.95])
    return {"low": float(lo), "median": float(med), "high": float(hi)}


def meridian_defaults() -> dict[str, Any]:
    """Meridian's own default priors (read from PriorDistribution when installed)."""
    try:
        from meridian.model import prior_distribution
    except ImportError:
        return dict(FALLBACK_DEFAULTS)
    p = prior_distribution.PriorDistribution()
    par = lambda d, *names: tuple(float(np.asarray(d.parameters[n])) for n in names)  # noqa: E731
    return {
        "roi": ("lognormal", *par(p.roi_m, "loc", "scale")),
        "mroi": ("lognormal", *par(p.mroi_m, "loc", "scale")),
        "contribution": ("beta", *par(p.contribution_m, "concentration1", "concentration0")),
        "gamma_c_scale": par(p.gamma_c, "scale")[0],
    }


# --- diminishing returns -------------------------------------------------------------

NO_SATURATION_MAX_SHARE = 0.05       # hint "No" only for a channel under 5% of spend...
NO_SATURATION_MAX_PEAK = 2.0         # ...whose highest week is under 2x its typical spending week

NO_SATURATION_WARNING = ("With no diminishing returns, the optimizer will keep adding budget to this channel up to "
                         "its upper limit. Set a sensible limit in Plan (step 51).")


def saturation_hint(spend: np.ndarray, total_paid: float) -> bool:
    x = np.asarray(spend, float)
    on = x[x > 0]
    if not len(on) or total_paid <= 0:
        return False
    return x.sum() / total_paid < NO_SATURATION_MAX_SHARE and on.max() < NO_SATURATION_MAX_PEAK * np.median(on)


# --- control direction ------------------------------------------------------------------

DIRECTIONS = ("either", "raises", "lowers")
DIRECTION_LABEL = {"either": "either", "raises": "raises sales", "lowers": "lowers sales"}


def direction_bounds(direction: str) -> tuple[float, float]:
    return {"raises": (0.0, math.inf), "lowers": (-math.inf, 0.0)}.get(direction, (-math.inf, math.inf))
