import numpy as np
import pytest
from scipy.integrate import quad

from ofdmest import (
    DelayProfile,
    LinearInterpolation,
    OfdmGrid,
    PilotMap,
    WienerFilter,
    best_uniform_span,
    comb,
    comb_error_matched,
    comb_error_uniform_design,
    equivalent_snr_db,
    exact_error,
    link_quality,
    mean_pilot_spacing_hz,
    rayleigh_rate,
    rule_span,
    rule_wiener,
    uniform_span_rule,
)
from ofdmest.cli import main


def wide_grid():
    return OfdmGrid(1200, 15e3, fft_size=2048, cp_duration_s=144 / (15000 * 2048))


def test_span_rule_values_and_limits():
    spread, spacing = 100e-9, 90e3
    g = 10 ** 1.5
    unbounded = spread * np.log(g / (spacing * spread))
    assert uniform_span_rule(spread, spacing, 15.0) == pytest.approx(unbounded)
    assert uniform_span_rule(spread, spacing, 15.0, bandwidth_hz=18e6) == pytest.approx(unbounded - 1 / 18e6)
    assert uniform_span_rule(spread, spacing, 15.0, bandwidth_hz=1e6) == pytest.approx(0.5 * unbounded)
    assert uniform_span_rule(spread, spacing, 15.0, max_span_s=300e-9) == pytest.approx(300e-9)
    assert uniform_span_rule(spread, 5e6, 15.0) == pytest.approx(1 / 5e6)
    assert uniform_span_rule(spread, 90e3, -30.0) == 0.0
    with pytest.raises(ValueError):
        uniform_span_rule(0.0, spacing, 15.0)


def test_rule_span_respects_the_caps_of_the_grid():
    grid = OfdmGrid(64, 312.5e3, fft_size=64, cp_duration_s=16 / (64 * 312.5e3))
    pilots = comb(64, 8, 4)
    alias_free = 1 / mean_pilot_spacing_hz(grid, pilots)
    span = rule_span(grid, pilots, 130e-9, 30.0)
    taps = int(span / grid.sample_period_s + 1e-9) + 1
    assert taps * grid.sample_period_s <= alias_free + 1e-15
    assert rule_span(grid, pilots, 130e-9, 30.0, max_delay_s=100e-9) == pytest.approx(100e-9)
    wide = wide_grid()
    lte = comb(1200, 6, 0)
    assert rule_span(wide, lte, 100e-9, 15.0) == pytest.approx(uniform_span_rule(100e-9, 90e3, 15.0, 18e6))
    assert rule_span(wide, lte, 100e-9, 15.0, band_correction=False) == pytest.approx(uniform_span_rule(100e-9, 90e3, 15.0))


def test_unbounded_comb_error_has_its_minimum_at_the_rule_span():
    decay, spacing, snr = 100e-9, 90e3, 20.0
    spans = np.linspace(2, 20, 3601) * decay
    errors = [comb_error_uniform_design(w, decay, spacing, snr) for w in spans]
    best = spans[int(np.argmin(errors))]
    assert best == pytest.approx(uniform_span_rule(decay, spacing, snr), rel=0.01)
    b = spacing * decay / 10 ** (snr / 10)
    assert comb_error_matched(decay, spacing, snr) == pytest.approx(b * np.log(1 + 1 / b))
    assert min(errors) > comb_error_matched(decay, spacing, snr)


def test_closed_forms_track_the_exact_error_on_a_wide_band():
    grid = wide_grid()
    pilots = comb(1200, 6, 0)
    spread = 150e-9
    profile = DelayProfile.exponential(spread, grid.sample_period_s)
    decay = -grid.sample_period_s / np.log(profile.powers[1] / profile.powers[0])
    spacing = mean_pilot_spacing_hz(grid, pilots)
    assert spacing == pytest.approx(90e3)
    exact = exact_error(WienerFilter(), grid, pilots, profile, 15.0).total_db
    model = 10 * np.log10(comb_error_matched(decay, spacing, 15.0))
    assert abs(exact - model) < 0.6


def test_rule_design_is_close_to_the_best_uniform_span():
    grid = wide_grid()
    pilots = comb(1200, 6, 0)
    spread = 100e-9
    profile = DelayProfile.exponential(spread, grid.sample_period_s)
    for snr in (5.0, 25.0):
        rule = exact_error(rule_wiener(grid, pilots, spread, snr), grid, pilots, profile, snr).total
        _, best = best_uniform_span(grid, pilots, profile, snr, max_span_s=16 * spread)
        matched = exact_error(WienerFilter(), grid, pilots, profile, snr).total
        assert best <= rule
        assert 10 * np.log10(rule / best) < 0.2
        assert 10 * np.log10(rule / matched) < 1.0


def test_rayleigh_rate_and_its_inverse():
    for snr in (0.01, 1.0, 31.6, 1e4):
        numeric, _ = quad(lambda x: np.log2(1 + snr * x) * np.exp(-x), 0, np.inf)
        assert rayleigh_rate(snr)[0] == pytest.approx(numeric, rel=1e-6)
    assert rayleigh_rate(1e-4)[0] == pytest.approx(1e-4 / np.log(2), rel=1e-3)
    assert equivalent_snr_db(float(rayleigh_rate(10 ** 1.2)[0])) == pytest.approx(12.0, abs=1e-6)


def test_link_quality_of_an_all_pilot_identity_estimator():
    class Identity(LinearInterpolation):
        name = "identity"

        def matrix(self, grid, pilots, point=None):
            return np.eye(grid.n_active)

    grid = OfdmGrid(32, 15e3, fft_size=32, cp_duration_s=4 / (32 * 15e3))
    pilots = PilotMap(np.arange(32), "all tones")
    profile = DelayProfile.exponential(grid.sample_period_s, grid.sample_period_s, max_delay_s=grid.cp_duration_s)
    var = 10 ** -1.0
    link = link_quality(Identity(), grid, pilots, profile, 10.0)
    assert np.allclose(link.effective_snr, 1 / (var * (2 + var)))
    assert np.allclose(link.perfect_snr, 1 / var)
    assert link.snr_loss_db > 0


def test_link_quality_agrees_with_simulated_statistics():
    grid = OfdmGrid(96, 15e3, fft_size=128, cp_duration_s=9 / (128 * 15e3))
    pilots = comb(96, 6, 2)
    profile = DelayProfile.exponential(1.5 * grid.sample_period_s, grid.sample_period_s, max_delay_s=grid.cp_duration_s)
    snr = 12.0
    var = 10 ** (-snr / 10)
    rng = np.random.default_rng(3)
    n = 40000
    taps = (rng.standard_normal((n, profile.n_taps)) + 1j * rng.standard_normal((n, profile.n_taps))) * np.sqrt(profile.powers / 2)
    H = taps @ profile.frequency_response_basis(np.arange(96), grid.spacing_hz).T
    noise = (rng.standard_normal((n, pilots.count)) + 1j * rng.standard_normal((n, pilots.count))) * np.sqrt(var / 2)
    for est in (LinearInterpolation(), WienerFilter(DelayProfile.uniform(grid.cp_duration_s, grid.sample_period_s), var)):
        A = est.matrix(grid, pilots)
        estimate = (H[:, pilots.indices] + noise) @ A.T
        cross = np.mean(H * estimate.conj(), axis=0)
        power = np.mean(np.abs(estimate) ** 2, axis=0)
        signal = np.abs(cross) ** 2 / power
        simulated = signal / (np.mean(np.abs(H) ** 2, axis=0) - signal + var)
        link = link_quality(est, grid, pilots, profile, snr)
        assert np.mean(simulated) == pytest.approx(np.mean(link.effective_snr), rel=0.03)


def test_cli_span_command(capsys):
    assert main(["span", "--grid", "lte:25", "--profile", "exp:100ns", "--snr", "15", "--search"]) == 0
    text = capsys.readouterr().out
    assert "rule_span_ns" in text and "best_error_db" in text
    assert main(["error", "--grid", "lte:25", "--profile", "exp:100ns", "--snr", "15", "--estimators", "linear,wiener"]) == 0
    assert "snr_loss_db" in capsys.readouterr().out
