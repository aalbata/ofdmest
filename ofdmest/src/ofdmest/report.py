"""Comparison of the standard estimators at one operating point, returned as plain data.

:func:`compare` evaluates the interpolators, the transform-domain estimator and three Wiener
designs (prefix-wide, span rule and matched) on one grid, pilot map, delay profile and SNR.
The result is a dictionary of numbers, strings and lists that converts directly to JSON, so
that a program or a browser page can display it without knowing the classes of the package.
:func:`compare_from_specs` accepts the text specifications of the command line.
"""
from __future__ import annotations

import numpy as np

from .analysis import exact_error, link_quality
from .channel import DelayProfile, OfdmGrid
from .design import best_uniform_span, mean_pilot_spacing_hz, rule_span, uniform_design, uniform_span_rule
from .estimators import CubicSplineInterpolation, DftInterpolation, DftLeastSquares, Estimator, LinearInterpolation, NearestPilot, WienerFilter, robust_wiener
from .pilots import PilotMap

RULE_NAME = "Wiener (uniform, span rule)"
PREFIX_NAME = "Wiener (uniform, prefix-wide)"
BEST_NAME = "Wiener (uniform, best width)"
MATCHED_NAME = "Wiener (matched)"


def standard_estimators(grid: OfdmGrid, pilots: PilotMap, design_spread_s: float, snr_db: float, max_delay_s: float | None = None) -> list[Estimator]:
    """Interpolators, the transform-domain estimator and the Wiener designs for one operating point.

    The transform-domain estimator and the prefix-wide Wiener design need the cyclic prefix
    of the grid and are left out when it is not set. On a map that is not a uniform comb the
    ridge-regularized tap fit replaces the transform-domain estimator.
    """
    out: list[Estimator] = [NearestPilot(), LinearInterpolation(), CubicSplineInterpolation()]
    if grid.cp_duration_s is not None:
        steps = np.diff(pilots.indices)
        if np.all(steps == steps[0]):
            out.append(DftInterpolation())
        else:
            out.append(DftLeastSquares(ridge=1e-3, name="DFT least squares (ridge)"))
        out.append(robust_wiener(grid, name=PREFIX_NAME))
    span = rule_span(grid, pilots, design_spread_s, snr_db, max_delay_s)
    out.append(uniform_design(grid, span, RULE_NAME))
    out.append(WienerFilter(name=MATCHED_NAME))
    return out


def compare(
    grid: OfdmGrid,
    pilots: PilotMap,
    profile: DelayProfile,
    snr_db: float,
    design_spread_s: float | None = None,
    max_delay_s: float | None = None,
    search_best: bool = False,
) -> dict:
    """Exact error and link-level measures of the standard estimators at one SNR.

    ``design_spread_s`` is the RMS delay spread given to the span rule; the default is that
    of ``profile``. ``max_delay_s`` caps the rule width when the largest delay of the channel
    is known. ``search_best`` adds the uniform design whose width has the smallest exact
    error. A design profile of ``n`` equal taps on the sample grid has the width ``n`` sample
    periods; ``value_ns`` is the value of the rule before it is realized in taps. Errors
    are in decibels, ``loss_db`` is the error above that of the matched design,
    ``snr_loss_db`` the SNR that the estimation error costs and ``rate_bits`` the mean
    achievable rate per subcarrier use.
    """
    spread = profile.rms_delay_spread_s if design_spread_s is None else float(design_spread_s)
    if not spread > 0:
        raise ValueError("the design delay spread must be positive; a single-tap profile needs design_spread_s")
    spacing = mean_pilot_spacing_hz(grid, pilots)
    span = rule_span(grid, pilots, spread, snr_db, max_delay_s)
    taps = int(np.floor(span / grid.sample_period_s + 1e-9)) + 1
    estimators = standard_estimators(grid, pilots, spread, snr_db, max_delay_s)
    best = None
    if search_best:
        best_span, _ = best_uniform_span(grid, pilots, profile, snr_db)
        best_taps = int(np.floor(best_span / grid.sample_period_s + 1e-9)) + 1
        best = {"taps": best_taps, "width_ns": best_taps * grid.sample_period_s * 1e9}
        estimators.insert(len(estimators) - 1, uniform_design(grid, best_span, BEST_NAME))
    matched_db = exact_error(estimators[-1], grid, pilots, profile, snr_db).total_db
    rows = []
    for est in estimators:
        try:
            res = exact_error(est, grid, pilots, profile, snr_db)
            link = link_quality(est, grid, pilots, profile, snr_db)
        except ValueError as err:
            rows.append({"estimator": est.name, "note": str(err)})
            continue
        rows.append({
            "estimator": est.name,
            "error_db": res.total_db,
            "loss_db": res.total_db - matched_db,
            "snr_loss_db": link.snr_loss_db,
            "rate_bits": link.rate_bits,
            "real_multiplications": int(res.real_multiplications),
        })
    return {
        "snr_db": float(snr_db),
        "pilots": int(pilots.count),
        "subcarriers": int(grid.n_active),
        "overhead_percent": 100.0 * pilots.overhead(grid),
        "pilot_spacing_khz": spacing / 1e3,
        "band_mhz": grid.n_active * grid.spacing_hz / 1e6,
        "sample_period_ns": grid.sample_period_s * 1e9,
        "cyclic_prefix_ns": None if grid.cp_duration_s is None else grid.cp_duration_s * 1e9,
        "profile": profile.name,
        "profile_spread_ns": profile.rms_delay_spread_s * 1e9,
        "profile_max_delay_ns": profile.max_delay_s * 1e9,
        "design_spread_ns": spread * 1e9,
        "rule": {
            "value_ns": span * 1e9,
            "taps": taps,
            "width_ns": taps * grid.sample_period_s * 1e9,
            "width_over_spread": taps * grid.sample_period_s / spread,
            "unbounded_comb_width_ns": uniform_span_rule(spread, spacing, snr_db) * 1e9,
        },
        "best": best,
        "matched_error_db": matched_db,
        "estimators": rows,
    }


def compare_from_specs(
    grid_spec: str,
    profile_spec: str | DelayProfile,
    snr_db: float,
    map_spec: str | None = None,
    window: int | None = None,
    design_spread_s: float | None = None,
    cap_at_max_delay: bool = False,
    search_best: bool = False,
) -> dict:
    """:func:`compare` for a grid, map and profile given in the notation of the command line.

    ``grid_spec`` is for example ``lte:100`` or ``nr:51:30:1``, ``map_spec`` for example
    ``comb:6`` (default: the standard map of the grid) and ``profile_spec`` for example
    ``exp:132ns``, or a :class:`DelayProfile`. Taps of a given profile beyond the cyclic
    prefix are dropped and the powers renormalized. ``cap_at_max_delay`` caps the rule width
    at the largest delay of the profile.
    """
    from .cli import parse_grid, parse_map, parse_profile

    grid, default_map, label = parse_grid(grid_spec, window)
    pilots = parse_map(map_spec, grid, default_map)
    if isinstance(profile_spec, DelayProfile):
        profile = profile_spec
        if grid.cp_duration_s is not None and profile.max_delay_s > grid.cp_duration_s:
            keep = profile.delays_s <= grid.cp_duration_s
            profile = DelayProfile(profile.delays_s[keep], profile.powers[keep], name=profile.name)
    else:
        profile = parse_profile(profile_spec, grid)
    out = compare(grid, pilots, profile, snr_db, design_spread_s, profile.max_delay_s if cap_at_max_delay else None, search_best)
    out["grid"] = label
    out["map"] = pilots.name
    return out
