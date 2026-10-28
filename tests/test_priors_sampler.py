"""Prior choices (steps 21, 23 and 25) and the sampler (steps 29, 30 and 36).

Property tests on synthetic data (tests/gen.py and the generated insurance
sample): nothing here depends on one file's numbers. Tests that fit with
Meridian take minutes and run only with MMM_FIT_TESTS=1.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gen
from mmm_tool import samples
from mmm_tool.configure import choices
from mmm_tool.fit import estimate, gates, job
from mmm_tool.fit import steps as fsteps
from mmm_tool.fit.context import FitOptions, check_custom, sampler_settings, settings
from mmm_tool.ingest.types import Verdict
from mmm_tool.pipeline import run_pipeline

try:
    import meridian  # noqa: F401
    HAVE_MERIDIAN = True
except ImportError:
    HAVE_MERIDIAN = False
needs_meridian = pytest.mark.skipif(not HAVE_MERIDIAN, reason="needs google-meridian")
fit_tests = pytest.mark.skipif(os.environ.get("MMM_FIT_TESTS") != "1" or not HAVE_MERIDIAN,
                               reason="set MMM_FIT_TESTS=1 to run Meridian fits")
SMALL = {"n_chains": 2, "n_adapt": 50, "n_burnin": 50, "n_keep": 50}


@pytest.fixture(scope="module")
def money_file(tmp_path_factory):
    """The generated insurance sample: it has a ₹ value per KPI, so every prior type is open."""
    return samples.write_clean(tmp_path_factory.mktemp("money"))


@pytest.fixture(scope="module")
def no_money_file(tmp_path_factory):
    """A synthetic file with a count KPI and no ₹ value per KPI."""
    return gen.write_knots_case(tmp_path_factory.mktemp("nomoney"), "flat_tv", 1)


def run(path, answers=None):
    r = run_pipeline(path, answers or {}, accept_defaults=True)
    assert r.d is not None and r.d.status == "complete", r.last.steps[-1].summary
    return r


def step(r, n):
    return next(s for s in r.d.steps if s.step == n)


def meridian_model(r, config=None):
    from meridian.model import model

    from mmm_tool.configure.spec import build_input_data, build_model_spec

    data = build_input_data(r.d)
    return model.Meridian(input_data=data, model_spec=build_model_spec(config or r.d.config, data)), data


def prior_draws(mmm, name, n=3000, seed=3):
    mmm.sample_prior(n, seed=seed)
    return np.asarray(mmm.inference_data.prior[name])


def log_lines(r, n):
    return [line for e in r.d.decision_log if e.get("step") == n for line in e.get("lines", [])]


# Custom sampler (step 29) ------------------------------------------------------------

@pytest.mark.parametrize("sampler, error", [
    ({"n_chains": 1, "n_adapt": 100, "n_burnin": 100, "n_keep": 100}, "chains"),
    ({"n_chains": 2, "n_adapt": 99, "n_burnin": 100, "n_keep": 100}, "adapt"),
    ({"n_chains": 2, "n_adapt": 100, "n_burnin": 99, "n_keep": 100}, "burn-in"),
    ({"n_chains": 2, "n_adapt": 100, "n_burnin": 100, "n_keep": 99}, "kept"),
    ({"n_chains": "two", "n_adapt": 100, "n_burnin": 100, "n_keep": 100}, "chains"),
    ({"n_adapt": 100, "n_burnin": 100, "n_keep": 100}, "chains"),
])
def test_custom_validation_refuses_below_the_minimums(sampler, error):
    errors, _ = check_custom(sampler)
    assert len(errors) == 1 and errors[0].startswith(error)


def test_custom_validation_hints():
    _, hints = check_custom({"n_chains": 2, "n_adapt": 100, "n_burnin": 100, "n_keep": 100})
    assert len(hints) == 2 and "4 chains" in hints[0] and "500" in hints[1]
    _, hints = check_custom({"n_chains": 3, "n_adapt": 100, "n_burnin": 100, "n_keep": 600})
    assert len(hints) == 1 and "4 chains" in hints[0]
    assert check_custom({"n_chains": 4, "n_adapt": 100, "n_burnin": 100, "n_keep": 500}) == ([], [])


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The API, with fits captured instead of started and a fit log that needs no calibration."""
    from fastapi.testclient import TestClient

    from mmm_tool.api import app as app_mod

    monkeypatch.setattr(app_mod, "RUNS_DIR", tmp_path / "runs")
    log = tmp_path / "fit_times.jsonl"
    monkeypatch.setenv("MMM_FIT_LOG", str(log))
    for s in (SMALL, {"n_chains": 2, "n_adapt": 300, "n_burnin": 300, "n_keep": 300}):
        estimate.record(52, 1, 3, s, 40 + s["n_keep"] / 10, calibration=True, path=log)
    started = []
    monkeypatch.setattr(app_mod.job, "start", lambda d, options, memory=None: started.append(options) or {"state": "queued"})
    c = TestClient(app_mod.app)
    c.started = started
    return c


def _new_run(client) -> str:
    run_id = client.post("/api/runs/sample/ins_clean").json()["run_id"]
    client.put(f"/api/runs/{run_id}/answers", json={"answers": {"roles": {}}})
    return run_id


@needs_meridian
def test_custom_validation_in_the_api(client):
    run_id = _new_run(client)
    bad = client.post(f"/api/runs/{run_id}/fit", json={"sampler": "custom", "custom": {"n_chains": 1, "n_adapt": 100, "n_burnin": 100, "n_keep": 100}})
    assert bad.status_code == 400 and "chains" in bad.json()["detail"]
    ok = client.post(f"/api/runs/{run_id}/fit", json={"sampler": "custom", "custom": {"n_chains": 2, "n_adapt": 100, "n_burnin": 100, "n_keep": 100}})
    assert ok.status_code == 200
    assert client.started[-1]["sampler"] == {"label": "Custom", "n_chains": 2, "n_adapt": 100, "n_burnin": 100, "n_keep": 100}


# Seed -------------------------------------------------------------------------------------

@needs_meridian
def test_seed_is_made_once_per_run_and_passed_to_the_fit(client):
    run_id = _new_run(client)
    e1 = client.get(f"/api/runs/{run_id}/fit-estimate").json()
    e2 = client.get(f"/api/runs/{run_id}/fit-estimate").json()
    assert isinstance(e1["seed"], int) and e1["seed"] == e2["seed"]                 # shown, and stable
    other = client.get(f"/api/runs/{_new_run(client)}/fit-estimate").json()["seed"]
    assert other != e1["seed"]                                                      # random per run
    client.post(f"/api/runs/{run_id}/fit", json={"sampler": "quick"})
    assert client.started[-1]["seed"] == e1["seed"] and client.started[-1]["seed_source"] == "default"
    client.post(f"/api/runs/{run_id}/fit", json={"sampler": "quick", "seed": 12345})
    assert client.started[-1]["seed"] == 12345 and client.started[-1]["seed_source"] == "user"


@fit_tests
def test_same_seed_reproduces_the_fit_exactly(money_file):
    from mmm_tool.fit import model as m

    r = run(money_file)
    cfg = {**r.d.config, "holdout": None}

    def fit(seed):
        mmm = m.build(r.d, cfg)
        mmm.sample_prior(100, seed=seed)
        m.sample_posterior(mmm, SMALL, seed)
        return mmm.inference_data

    a, b, c = fit(41), fit(41), fit(42)
    for name in ("roi_m", "alpha_m", "knot_values"):
        assert np.array_equal(np.asarray(a.posterior[name]), np.asarray(b.posterior[name])), name
        assert np.array_equal(np.asarray(a.prior[name]), np.asarray(b.prior[name])), name
    assert not np.array_equal(np.asarray(a.posterior["roi_m"]), np.asarray(c.posterior["roi_m"]))
    print(f"\nseed 41 twice: identical posterior; seed 42 differs (roi_m mean {float(a.posterior['roi_m'].mean()):.4f} "
          f"vs {float(c.posterior['roi_m'].mean()):.4f})")


# Run-time estimate -----------------------------------------------------------------------------

def _rows(a, b, sizes, calibration=False):
    rows = []
    for w, g, ch, s in sizes:
        x = estimate.work(w, g, ch, s["n_chains"], s["n_adapt"] + s["n_burnin"] + s["n_keep"], estimate.cores())
        rows.append({"n_weeks": w, "n_geos": g, "n_channels": ch, "chains": s["n_chains"], "adapt": s["n_adapt"],
                     "burnin": s["n_burnin"], "kept": s["n_keep"], "cores": estimate.cores(), "seconds": a + b * x,
                     "calibration": calibration})
    return rows


def test_estimate_recovers_a_and_b_and_labels_its_basis(tmp_path):
    log = tmp_path / "log.jsonl"
    sizes = [(156, 1, 9, SMALL), (104, 1, 5, {**SMALL, "n_keep": 200}), (156, 5, 9, {**SMALL, "n_adapt": 300})]
    a, b = 60.0, 2e-6
    cal = _rows(a, b, sizes[:2], calibration=True)
    log.write_text("".join(json.dumps(r) + "\n" for r in cal))
    m = estimate.model(log)
    assert m["basis"] == "rough" and m["runs"] == 0                  # under 3 real runs: the calibration, labelled rough
    log.write_text("".join(json.dumps(r) + "\n" for r in cal + _rows(a, b, sizes)))
    m = estimate.model(log)
    assert m["basis"] == "from 3 past runs"
    assert math.isclose(m["a"], a, rel_tol=1e-6) and math.isclose(m["b"], b, rel_tol=1e-6)
    e = estimate.estimate(156, 1, 9, {"n_chains": 4, "n_adapt": 500, "n_burnin": 500, "n_keep": 1000}, path=log)
    assert estimate.text(e["seconds"], e["basis"]).startswith("about ") and e["basis"] in estimate.text(e["seconds"], e["basis"])
    assert estimate.text(240, "from 12 past runs") == "about 4 min (from 12 past runs)"
    assert estimate.text(30, "rough") == "about 30 s (rough)"


def test_chains_beyond_the_cores_cost_a_second_round():
    n = estimate.cores()
    assert estimate.work(10, 1, 1, n + 1, 100, n) == 2 * estimate.work(10, 1, 1, n, 100, n)


@fit_tests
def test_estimate_within_half_after_three_runs(no_money_file, tmp_path, monkeypatch):
    """After 3 logged fits on this machine, the estimate of a 4th is within +/-50% of its real time."""
    from mmm_tool.fit import model as m

    log = tmp_path / "fit_times.jsonl"
    monkeypatch.setenv("MMM_FIT_LOG", str(log))
    r = run(no_money_file)
    cfg = {**r.d.config, "holdout": None}

    def timed(s):
        mmm = m.build(r.d, cfg)
        mmm.sample_prior(50, seed=1)
        t = time.time()
        m.sample_posterior(mmm, s, 1)
        return mmm, time.time() - t

    for s in ({**SMALL}, {**SMALL, "n_adapt": 150, "n_burnin": 150, "n_keep": 100}, {**SMALL, "n_adapt": 300, "n_burnin": 300, "n_keep": 200}):
        mmm, secs = timed(s)
        estimate.record(*estimate.model_size(mmm), s, secs)
    nxt = {**SMALL, "n_adapt": 200, "n_burnin": 200, "n_keep": 150}
    e = estimate.estimate(*estimate.model_size(mmm), nxt)
    _, actual = timed(nxt)
    print(f"\nestimate {e['seconds']:.0f} s ({e['basis']}; a {e['a']:.0f} s, b {e['b']:.2e}) vs actual {actual:.0f} s "
          f"({e['seconds'] / actual - 1:+.0%})")
    assert e["basis"] == "from 3 past runs"
    assert 0.5 * actual <= e["seconds"] <= 1.5 * actual


# Step 30 retry --------------------------------------------------------------------------------

class FakeModel:
    def sample_prior(self, n, seed=None):
        pass


@pytest.fixture
def fake(monkeypatch):
    """convergence() with R-hats scripted per fit; records every fit's sampler and seed."""
    state = SimpleNamespace(rhats=[], fits=[], builds=0, messages=[])

    def build(*a, **k):
        state.builds += 1
        return FakeModel()

    def fit(ctx, mmm, sampler, label):
        state.fits.append({"sampler": dict(sampler), "seed": ctx.seed, "label": label})
        return 1.0

    monkeypatch.setattr(fsteps.m, "rhat", lambda mmm: state.rhats.pop(0))
    monkeypatch.setattr(fsteps.m, "build", build)
    monkeypatch.setattr(fsteps.m, "rhat_detail", lambda mmm, limit: [{"text": "tv carryover (alpha_m)", "parameter": "alpha_m", "rhat": 1.5}])
    monkeypatch.setattr(fsteps, "_fit", fit)
    sampler = sampler_settings({"n_chains": 3, "n_adapt": 150, "n_burnin": 120, "n_keep": 400})

    def ctx(continue_unconverged=False):
        return SimpleNamespace(cfg=settings(), sampler=sampler, fit_sampler=dict(sampler), timings={"fit_holdout": 5.0},
                               progress=lambda n, msg, frac: state.messages.append(msg), seed=99, gates={}, log=[],
                               options=FitOptions(continue_unconverged=continue_unconverged), mmm=FakeModel(),
                               d=None, config={}, not_converged=False)

    state.ctx, state.sampler = ctx, sampler
    return state


def test_healthy_run_never_retries(fake):
    fake.rhats = [{"roi_m": 1.01, "alpha_m": 1.02, "eta_m": float("nan")}]
    s = fsteps.convergence(fake.ctx())
    assert s.verdict == Verdict.PASS and len(s.details["attempts"]) == 1
    assert fake.fits == [] and fake.builds == 0 and fake.messages == []


def test_nan_rhat_is_never_a_failure():
    assert gates.converged({"roi_m": 1.02, "eta_m": float("nan"), "xi_n": float("nan")}, 1.1)["ok"]
    res = gates.converged({"roi_m": float("nan"), "alpha_m": 1.3}, 1.1)
    assert not res["ok"] and res["worst"]["parameter"] == "alpha_m"


def test_one_retry_with_doubled_warm_up_then_pass(fake):
    fake.rhats = [{"alpha_m": 1.4, "eta_m": float("nan")}, {"alpha_m": 1.03, "eta_m": float("nan")}]
    ctx = fake.ctx()
    s = fsteps.convergence(ctx)
    assert s.verdict == Verdict.PASS and ctx.gates[30]
    at = s.details["attempts"]
    assert [a["attempt"] for a in at] == [1, 2] and [a["ok"] for a in at] == [False, True]
    retry = fake.fits[0]["sampler"]
    first = fake.sampler
    assert retry["n_adapt"] == 2 * first["n_adapt"] and retry["n_burnin"] == 2 * first["n_burnin"]
    assert retry["n_chains"] == first["n_chains"] and retry["n_keep"] == first["n_keep"] and fake.fits[0]["seed"] == ctx.seed
    assert ctx.fit_sampler == retry                                     # later refits use the settings that converged
    assert fake.messages and fake.messages[0].startswith("Retrying with longer warm-up")


def test_never_more_than_one_retry_and_both_attempts_recorded(fake):
    fake.rhats = [{"alpha_m": 1.4}, {"alpha_m": 1.3}]
    ctx = fake.ctx()
    s = fsteps.convergence(ctx)
    assert s.verdict == Verdict.FAIL and not ctx.gates[30] and len(fake.fits) == 1 and fake.rhats == []
    at = s.details["attempts"]
    assert [(a["attempt"], a["rhat"], a["parameter"]) for a in at] == [(1, 1.4, "alpha_m"), (2, 1.3, "alpha_m")]
    assert s.details["above"][0]["text"] == "tv carryover (alpha_m)"
    sug = s.details["suggestions"]                                      # in the order asked for
    assert [k in t for k, t in zip(("chains", "knots", "Combine", "priors"), sug)] == [True] * 4
    assert "Continue anyway" in s.warnings[0]


def test_continue_anyway_marks_the_run(fake):
    fake.rhats = [{"alpha_m": 1.4}, {"alpha_m": 1.3}]
    ctx = fake.ctx(continue_unconverged=True)
    s = fsteps.convergence(ctx)
    assert ctx.not_converged and "did not converge" in s.summary
    assert ctx.log[-1]["text"] == "continue although the model did not converge"


def test_final_fit_uses_the_same_retry_rule(fake):
    ctx = fake.ctx()
    fake.rhats = [{"alpha_m": 1.01}]
    _, at, res = fsteps._fit_converged(ctx, lambda: fsteps.m.build(), fake.sampler, "fit_final", 36)
    assert len(at) == 1 and res["ok"] and fake.builds == 1
    fake.rhats = [{"alpha_m": 1.5}, {"alpha_m": 1.6}]
    _, at, res = fsteps._fit_converged(ctx, lambda: fsteps.m.build(), fake.sampler, "fit_final", 36)
    assert len(at) == 2 and not res["ok"] and fake.builds == 3 and fake.rhats == []
    assert at[1]["sampler"]["n_adapt"] == 2 * fake.sampler["n_adapt"]


@fit_tests
def test_unconverged_run_waits_then_continues(money_file, tmp_path):
    from mmm_tool.fit import run_phase_e

    r = run(money_file)
    bad = {"n_chains": 2, "n_adapt": 10, "n_burnin": 10, "n_keep": 50}
    e = run_phase_e(r.d, {"sampler": bad, "seed": 11}, out_dir=tmp_path / "fit")
    at = e.steps[-1].details["attempts"]
    print("\n" + "; ".join(f"attempt {a['attempt']}: {a['sampler']['n_adapt']} adapt, {a['sampler']['n_burnin']} burn-in, "
                           f"R-hat {a['rhat']:.2f} ({a['parameter']})" for a in at))
    assert e.waiting and [s.step for s in e.steps] == [28, 29, 30]
    assert [(a["sampler"]["n_adapt"], a["sampler"]["n_burnin"]) for a in at] == [(10, 10), (20, 20)]
    e2 = run_phase_e(r.d, {"sampler": bad, "seed": 11, "continue_unconverged": True}, out_dir=tmp_path / "fit")
    assert e2.not_converged and not e2.waiting and e2.steps[-1].step == 36
    assert "fit_holdout" in e2.timings and e2.timings["fit_holdout"] == e.timings["fit_holdout"]   # no refit of the holdout
    texts = [x["text"] for x in e2.decision_log]
    assert "seed 11" in texts and "continue although the model did not converge" in texts


# Carryover priors (step 23) -----------------------------------------------------------------------

def test_carryover_presets_mean_the_stated_share():
    """Geometric: next week's share is alpha, and the Beta's mean is the share. Binomial: the share
    is a curved function of alpha, so the converted Beta puts its median share there."""
    for speed, share in (("fast", 0.2), ("medium", 0.4), ("slow", 0.7)):
        for max_lag in (4, 8, 13):
            a, b = choices.carryover_beta(speed, "geometric", max_lag)
            assert math.isclose(a / (a + b), share) and a + b == choices.PRESET_CONCENTRATION
            a, b = choices.carryover_beta(speed, "binomial", max_lag)
            med = choices.carryover_summary(a, b, "binomial", max_lag)["median"]
            assert abs(med - share) < 0.005, (max_lag, speed, med)
    assert choices.carryover_beta("dont_know", "binomial", 8) == (1.0, 1.0)        # Uniform(0, 1)


def test_binomial_share_matches_meridians_weights():
    """next_week_share agrees with Meridian's own binomial weights (lag 1 over lag 0)."""
    pytest.importorskip("meridian")
    from meridian.model import adstock_hill

    for alpha in (0.1, 0.3, 0.6, 0.9):
        for max_lag in (4, 8, 13):
            mapped = float(np.asarray(adstock_hill._map_alpha_for_binomial_decay(np.float64(alpha))))
            w = (1 - np.arange(max_lag + 1) / (max_lag + 1)) ** mapped
            assert math.isclose(w[1] / w[0], float(choices.next_week_share(alpha, "binomial", max_lag)), rel_tol=1e-6)


@needs_meridian
def test_carryover_prior_means_and_stable_channel_order(money_file):
    r0 = run(money_file)
    chans = list(r0.d.config["channels"]) if isinstance(r0.d.config["channels"], list) else list(r0.d.config["adstock"])
    fast, med, slow = chans[1], chans[3], chans[-1]
    speeds = {fast: "fast", med: "medium", slow: "slow"}
    r = run(money_file, {"carryover_speed": speeds})
    mmm, data = meridian_model(r)
    order = [str(c) for c in data.media_channel.values]
    alpha = prior_draws(mmm, "alpha_m").reshape(-1, len(order))
    means = {c: float(alpha[:, order.index(c)].mean()) for c in order}
    print("\nalpha_m prior means:", {c: round(v, 3) for c, v in means.items()})
    for c, want in ((fast, 0.2), (med, 0.4), (slow, 0.7)):
        assert abs(means[c] - want) < 0.02, (c, means[c])
    assert all(abs(means[c] - 0.5) < 0.02 for c in order if c not in speeds)
    assert any(f"{fast} fades fast" == line for line in log_lines(r, 23))
    # Shuffle the UI table (answer order and config order): the vectors still follow input_data.media_channel.
    from mmm_tool.configure.spec import build_model_spec

    base = build_model_spec(r.d.config, data).prior.alpha_m
    for seed in range(3):
        rnd = random.Random(seed)
        shuffled = dict(rnd.sample(list(speeds.items()), len(speeds)))
        r2 = run(money_file, {"carryover_speed": shuffled})
        cfg = {**r2.d.config, "carryover_prior": dict(rnd.sample(list(r2.d.config["carryover_prior"].items()), len(order)))}
        p = build_model_spec(cfg, data).prior.alpha_m
        assert np.array_equal(np.asarray(p.concentration1), np.asarray(base.concentration1))
        assert np.array_equal(np.asarray(p.concentration0), np.asarray(base.concentration0))
    c1 = np.asarray(base.concentration1)
    assert c1.dtype == np.float64 and c1[order.index(fast)] == choices.GEOMETRIC_BETA["fast"][0]


@needs_meridian
def test_binomial_channel_gets_the_converted_prior(money_file):
    ch = list(run(money_file).d.config["adstock"])[0]
    r = run(money_file, {"adstock_shape": {ch: "binomial"}, "carryover_speed": {ch: "fast"}})
    mmm, data = meridian_model(r)
    order = [str(c) for c in data.media_channel.values]
    alpha = prior_draws(mmm, "alpha_m").reshape(-1, len(order))[:, order.index(ch)]
    share = choices.next_week_share(alpha, "binomial", r.d.config["max_lag"])
    print(f"\nbinomial {ch}: median share next week {np.median(share):.3f}")
    assert abs(float(np.median(share)) - 0.2) < 0.03


# Prior type (step 25) ---------------------------------------------------------------------------------

def test_lognormal_and_beta_from_a_range():
    for lo, hi in ((1, 3), (0.5, 4), (2, 2.5), (0.05, 0.4)):
        mu, sigma = choices.lognormal_from_range(lo, hi)
        assert math.isclose(mu, math.log(math.sqrt(lo * hi))) and math.isclose(sigma, math.log(hi / lo) / (2 * 1.645), rel_tol=1e-3)
        s = choices.prior_summary("lognormal", mu, sigma)
        assert math.isclose(s["low"], lo, rel_tol=1e-6) and math.isclose(s["high"], hi, rel_tol=1e-6)
    for lo, hi in ((0.01, 0.03), (0.02, 0.10), (0.1, 0.4), (0.001, 0.02)):
        a, b = choices.beta_from_range(lo, hi)
        s = choices.prior_summary("beta", a, b)
        assert abs(s["low"] / lo - 1) < 0.01 and abs(s["high"] / hi - 1) < 0.01
    with pytest.raises(ValueError):
        choices.beta_from_range(0.2, 0.1)


@needs_meridian
@pytest.mark.parametrize("kind, low, high", [("roi", 1, 3), ("mroi", 1, 3), ("contribution", 0.01, 0.03)])
def test_each_prior_type_samples_its_range(money_file, kind, low, high):
    ch = list(run(money_file).d.config["adstock"])[2]
    r = run(money_file, {"prior_type": kind, "roi_ranges": {ch: {"low": low, "high": high}}})
    assert r.d.config["prior_type"] == kind
    mmm, data = meridian_model(r)
    order = [str(c) for c in data.media_channel.values]
    name = {"roi": "roi_m", "mroi": "mroi_m", "contribution": "contribution_m"}[kind]
    draws = prior_draws(mmm, name, n=6000).reshape(-1, len(order))
    q5, q95 = np.quantile(draws[:, order.index(ch)], [0.05, 0.95])
    print(f"\n{kind} {ch}: range {low}-{high} -> sampled 5th {q5:.4g}, 95th {q95:.4g}")
    assert abs(q5 / low - 1) < 0.05 and abs(q95 / high - 1) < 0.05
    other = next(c for c in order if c != ch)                    # a blank channel keeps Meridian's default
    fam, p1, p2 = choices.meridian_defaults()[kind]
    want = choices.prior_summary(fam, p1, p2)["median"]
    assert abs(np.median(draws[:, order.index(other)]) / want - 1) < 0.1
    row = step(r, 25).details["priors"][ch]
    assert abs(row["low"] / low - 1) < 0.01 and abs(row["high"] / high - 1) < 0.01


def test_return_options_disabled_without_a_rupee_value(no_money_file):
    r = run(no_money_file)
    d = next(x for x in step(r, 25).decisions if x.id == "prior_type")
    assert d.options == ["contribution"] and d.default == "contribution" and d.context["why_disabled"]
    assert r.d.config["prior_type"] == "contribution"
    r2 = run_pipeline(no_money_file, {"prior_type": "roi"}, accept_defaults=True)
    s = next(s for s in r2.last.steps if s.step == 25)
    assert s.verdict == Verdict.NEEDS_INPUT and (r2.d is None or r2.d.status != "complete")


# Diminishing returns (step 23) --------------------------------------------------------------------------

def test_no_saturation_hint_rule():
    steady = np.full(100, 10.0)
    assert choices.saturation_hint(steady, total_paid=steady.sum() / 0.04)            # 4% of spend, steady
    assert not choices.saturation_hint(steady, total_paid=steady.sum() / 0.06)        # 6% of spend
    peaky = steady.copy()
    peaky[10] = 25.0                                                                  # a week 2.5x the typical one
    assert not choices.saturation_hint(peaky, total_paid=peaky.sum() / 0.04)


@needs_meridian
def test_saturation_off_gives_a_straight_curve_and_the_warning(money_file):
    from meridian.analysis import analyzer

    chans = list(run(money_file).d.config["adstock"])
    off, on = chans[4], chans[0]
    r = run(money_file, {"diminishing_returns": {off: "no"}})
    assert r.d.config["saturation"][off] == "none"
    s23 = step(r, 23)
    assert any(choices.NO_SATURATION_WARNING in w for w in s23.warnings)
    assert "Plan (step 51)" in choices.NO_SATURATION_WARNING
    mmm, _ = meridian_model(r)
    mmm.sample_prior(500, seed=2)
    rc = analyzer.Analyzer(mmm).response_curves(use_posterior=False, spend_multipliers=[0, 0.5, 1, 1.5, 2], use_kpi=True)
    y_off = rc["incremental_outcome"].sel(channel=off, metric="mean").values
    y_on = rc["incremental_outcome"].sel(channel=on, metric="mean").values
    print(f"\n{off} (no saturation) second differences: {np.round(np.diff(y_off, 2), 4)}; {on}: {np.round(np.diff(y_on, 2), 1)}")
    assert np.max(np.abs(np.diff(y_off, 2))) < 1e-6 * np.max(np.abs(y_off))         # straight
    assert np.all(np.diff(y_on, 2) < 0)                                               # bends over


# Control direction (steps 21 and 32) ------------------------------------------------------------------

def test_direction_suggestions_from_names(no_money_file):
    r = run(no_money_file)
    sug = step(r, 21).details["suggest_direction"]
    controls = list(r.d.config["control_direction"])
    for c in controls:
        low = c.lower()
        if any(w in low for w in ("price", "premium", "competitor")):
            assert sug.get(c) == "lowers", c
        elif any(w in low for w in ("season", "festive", "holiday")):
            assert sug.get(c) == "raises", c
    assert "lowers" in sug.values() and "raises" in sug.values()
    assert all(v == "either" for v in r.d.config["control_direction"].values())      # a suggestion, never applied


@needs_meridian
def test_direction_truncates_the_prior(money_file):
    from meridian.model import prior_distribution

    controls = list(run(money_file).d.config["control_direction"])
    low_c, high_c, free = controls[0], controls[1], controls[2]
    r = run(money_file, {"control_direction": {low_c: "lowers", high_c: "raises"}})
    assert f"{low_c} lowers sales" in log_lines(r, 21)
    mmm, data = meridian_model(r)
    order = [str(c) for c in data.control_variable.values]
    g = prior_draws(mmm, "gamma_c").reshape(-1, len(order))
    assert g[:, order.index(low_c)].max() <= 0 and g[:, order.index(high_c)].min() >= 0
    assert g[:, order.index(free)].min() < 0 < g[:, order.index(free)].max()
    from mmm_tool.configure.spec import build_model_spec

    scale = np.asarray(build_model_spec(r.d.config, data).prior.gamma_c.scale)
    assert scale.dtype == np.float64
    assert np.allclose(scale, float(np.asarray(prior_distribution.PriorDistribution().gamma_c.parameters["scale"])))


def _posterior_ctx(draws: dict, direction: dict):
    names = list(draws)
    g = np.stack([draws[c] for c in names], axis=-1).reshape(2, -1, len(names))
    mmm = SimpleNamespace(inference_data=SimpleNamespace(posterior={"gamma_c": g}),
                          input_data=SimpleNamespace(control_variable=SimpleNamespace(values=np.array(names))))
    return SimpleNamespace(mmm=mmm, config={"control_direction": direction}, cfg=settings())


def test_direction_check_flags_draws_piled_at_the_limit():
    rng = np.random.default_rng(0)
    ctx = _posterior_ctx({"held": -np.abs(rng.normal(0, 0.01, 400)), "fine": rng.normal(1.0, 0.2, 400) + 0.5,
                          "free": rng.normal(0, 1, 400)}, {"held": "lowers", "fine": "raises", "free": "either"})
    out = {x["control"]: x for x in fsteps.direction_check(ctx)}
    assert set(out) == {"held", "fine"} and out["held"]["flagged"] and not out["fine"]["flagged"]


@fit_tests
def test_step32_flags_a_forced_wrong_direction(tmp_path):
    """Synthetic data where a control truly raises sales, forced to 'Lowers': step 32 flags it."""
    from mmm_tool.fit import model as m

    df = gen.make_knots_case("flat_tv", 3)
    rng = np.random.default_rng(3)
    z = rng.normal(0, 1, len(df))
    df["demand_index"] = np.round(100 + 10 * z, 2)
    df["new_policies"] = df["new_policies"] + np.round(400 * z)                    # strong positive effect
    path = tmp_path / "demand.csv"
    df.assign(week_start=df["week_start"].dt.strftime("%Y-%m-%d")).to_csv(path, index=False)
    r = run(path, {"control_direction": {"demand_index": "lowers"}})
    mmm = m.build(r.d, {**r.d.config, "holdout": None})
    mmm.sample_prior(100, seed=5)
    m.sample_posterior(mmm, {"n_chains": 2, "n_adapt": 200, "n_burnin": 200, "n_keep": 200}, 5)
    out = {x["control"]: x for x in fsteps.direction_check(SimpleNamespace(mmm=mmm, config=r.d.config, cfg=settings()))}
    print(f"\ndemand_index forced 'lowers': median {out['demand_index']['median']:.4f}, "
          f"{out['demand_index']['share_at_limit']:.0%} of draws at the limit")
    assert out["demand_index"]["flagged"]


# Everywhere: config, decision log, rerun ----------------------------------------------------------------

def test_every_choice_reaches_config_log_and_marks_a_rerun(money_file):
    r0 = run(money_file)
    base = job.inputs_hash(r0.d)
    ch = list(r0.d.config["adstock"])[0]
    ctl = list(r0.d.config["control_direction"])[0]
    cases = [
        ({"carryover_speed": {ch: "slow"}}, lambda c: c["carryover_prior"][ch]["speed"] == "slow", 23, f"{ch} fades slow"),
        ({"diminishing_returns": {ch: "no"}}, lambda c: c["saturation"][ch] == "none", 23, f"{ch} has no diminishing returns"),
        ({"prior_type": "mroi"}, lambda c: c["prior_type"] == "mroi", 25, "priors set on return on the next rupee"),
        ({"control_direction": {ctl: "raises"}}, lambda c: c["control_direction"][ctl] == "raises", 21, f"{ctl} raises sales"),
    ]
    for answers, in_config, n, line in cases:
        r = run(money_file, answers)
        assert in_config(r.d.config), answers
        assert job.inputs_hash(r.d) != base, answers                                 # steps 28+ need a rerun
        assert any(line in x for x in log_lines(r, n)), (answers, log_lines(r, n))
