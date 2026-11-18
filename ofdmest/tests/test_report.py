import json

import pytest

from ofdmest import (
    DelayProfile,
    OfdmGrid,
    WienerFilter,
    comb,
    compare,
    compare_from_specs,
    exact_error,
    link_quality,
    rule_span,
    rule_wiener,
    standard_estimators,
)


def small_grid():
    return OfdmGrid(240, 15e3, fft_size=256, cp_duration_s=18 / (256 * 15e3))


def rows_by_name(result):
    return {row["estimator"]: row for row in result["estimators"]}


def test_compare_agrees_with_the_separate_functions():
    grid, pilots = small_grid(), comb(240, 6, 0)
    profile = DelayProfile.exponential(120e-9, grid.sample_period_s, max_delay_s=1.8e-6)
    result = compare(grid, pilots, profile, 15.0)
    rows = rows_by_name(result)
    matched = exact_error(WienerFilter(), grid, pilots, profile, 15.0)
    rule = rule_wiener(grid, pilots, profile.rms_delay_spread_s, 15.0)
    assert rows["Wiener (matched)"]["error_db"] == pytest.approx(matched.total_db)
    assert rows["Wiener (matched)"]["loss_db"] == 0.0
    assert rows["Wiener (uniform, span rule)"]["error_db"] == pytest.approx(exact_error(rule, grid, pilots, profile, 15.0).total_db)
    assert rows["Wiener (uniform, span rule)"]["snr_loss_db"] == pytest.approx(link_quality(rule, grid, pilots, profile, 15.0).snr_loss_db)
    span = rule_span(grid, pilots, profile.rms_delay_spread_s, 15.0)
    taps = int(span / grid.sample_period_s + 1e-9) + 1
    assert result["rule"]["value_ns"] == pytest.approx(1e9 * span)
    assert result["rule"]["taps"] == taps == rule.design_profile.n_taps
    assert result["rule"]["width_ns"] == pytest.approx(1e9 * taps * grid.sample_period_s)
    assert result["rule"]["width_over_spread"] == pytest.approx(taps * grid.sample_period_s / profile.rms_delay_spread_s)
    assert result["matched_error_db"] == pytest.approx(matched.total_db)
    assert all(row["loss_db"] >= -1e-9 for row in result["estimators"])
    assert json.loads(json.dumps(result)) == result


def test_compare_uses_the_design_spread_and_the_delay_cap():
    grid, pilots = small_grid(), comb(240, 6, 0)
    profile = DelayProfile.exponential(120e-9, grid.sample_period_s, max_delay_s=1.8e-6)
    plain = compare(grid, pilots, profile, 20.0)
    narrow = compare(grid, pilots, profile, 20.0, design_spread_s=60e-9)
    capped = compare(grid, pilots, profile, 20.0, max_delay_s=300e-9)
    assert narrow["design_spread_ns"] == pytest.approx(60.0)
    assert narrow["rule"]["value_ns"] < plain["rule"]["value_ns"]
    assert capped["rule"]["value_ns"] == pytest.approx(300.0)
    with pytest.raises(ValueError):
        compare(grid, pilots, DelayProfile.single_tap(), 20.0)


def test_best_width_is_not_worse_than_the_rule():
    grid, pilots = small_grid(), comb(240, 6, 0)
    profile = DelayProfile.exponential(120e-9, grid.sample_period_s, max_delay_s=1.8e-6)
    result = compare(grid, pilots, profile, 10.0, search_best=True)
    rows = rows_by_name(result)
    assert result["best"]["width_ns"] == pytest.approx(1e9 * result["best"]["taps"] * grid.sample_period_s)
    assert rows["Wiener (uniform, best width)"]["loss_db"] <= rows["Wiener (uniform, span rule)"]["loss_db"] + 1e-9
    assert list(rows)[-1] == "Wiener (matched)"


def test_estimators_follow_the_grid_and_the_map():
    with_prefix = small_grid()
    without_prefix = OfdmGrid(240, 15e3, fft_size=256)
    names = [e.name for e in standard_estimators(with_prefix, comb(240, 6, 0), 120e-9, 15.0)]
    assert names == ["nearest pilot", "linear (constant edges)", "cubic spline (constant edges)", "DFT interpolation", "Wiener (uniform, prefix-wide)", "Wiener (uniform, span rule)", "Wiener (matched)"]
    names = [e.name for e in standard_estimators(without_prefix, comb(240, 6, 0), 120e-9, 15.0)]
    assert names == ["nearest pilot", "linear (constant edges)", "cubic spline (constant edges)", "Wiener (uniform, span rule)", "Wiener (matched)"]
    paired = compare_from_specs("nr:6:30:2", "exp:60ns", 15.0)
    assert "DFT least squares (ridge)" in rows_by_name(paired)


def test_compare_from_specs_accepts_text_and_profiles():
    text = compare_from_specs("lte:25", "exp:100ns", 10.0)
    grid = OfdmGrid(300, 15e3, fft_size=512, cp_duration_s=144 / (15000 * 2048))
    profile = DelayProfile.exponential(100e-9, grid.sample_period_s, max_delay_s=1.5e-6)
    given = compare_from_specs("lte:25", profile, 10.0)
    assert text["grid"] == "LTE 25 PRB" and text["pilots"] == 50
    for a, b in zip(text["estimators"], given["estimators"]):
        assert a["error_db"] == pytest.approx(b["error_db"])
    long_profile = DelayProfile([0.0, 100e-9, 9e-6], [0.6, 0.3, 0.1])
    trimmed = compare_from_specs("lte:25", long_profile, 10.0, cap_at_max_delay=True)
    assert trimmed["profile_max_delay_ns"] == pytest.approx(100.0)
    assert trimmed["rule"]["value_ns"] <= 100.0 + 1e-9
    combed = compare_from_specs("custom:64:312500:64:0.8us", "exp:74ns", 10.0, map_spec="centred:8")
    assert combed["pilots"] == 8 and combed["subcarriers"] == 64
