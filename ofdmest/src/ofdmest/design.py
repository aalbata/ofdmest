"""Design span of a Wiener filter built on a uniform delay profile.

A Wiener filter designed on a uniform profile of width ``W`` needs no knowledge of the
profile shape, only the width. On an unbounded pilot comb with pilots ``F`` hertz apart the
filter passes every delay inside the window with the gain ``1 / (1 + x)``, ``x = W F / g``,
and blocks every delay outside it, where ``g`` is the signal-to-noise ratio. For an
exponential profile with decay constant ``d`` the error is therefore

    e(W) = x / (1 + x) + exp(-W / d) (1 + 2 x) / (1 + x)^2,

close to ``W F / g + exp(-W / d)`` when ``x`` is small: the noise admitted by the window
grows in proportion to its width and the channel power left outside it falls
exponentially. The minimum is at ``W = d ln(g / (F d))``. The matched filter reaches
``b ln(1 + 1 / b)`` with ``b = F d / g``. A band of width ``B`` resolves delays no finer
than ``1 / B``, so a window of nominal width ``W`` acts as one of width ``W + 1 / B``; the
span rule subtracts that cell, by at most half of the unbounded-comb span.
"""
from __future__ import annotations

import numpy as np

from .analysis import exact_error
from .channel import DelayProfile, OfdmGrid
from .estimators import WienerFilter
from .pilots import PilotMap


def mean_pilot_spacing_hz(grid: OfdmGrid, pilots: PilotMap) -> float:
    """Width of the analysed band divided by the number of pilots, in hertz."""
    return grid.n_active * grid.spacing_hz / pilots.count


def comb_error_uniform_design(span_s: float, decay_s: float, pilot_spacing_hz: float, snr_db: float) -> float:
    """Error of the uniform design of width ``span_s`` on an unbounded comb, exponential channel."""
    if span_s < 0 or decay_s <= 0 or pilot_spacing_hz <= 0:
        raise ValueError("span_s must be non-negative, decay_s and pilot_spacing_hz positive")
    x = span_s * pilot_spacing_hz / 10.0 ** (snr_db / 10.0)
    outside = np.exp(-span_s / decay_s)
    return float(x / (1.0 + x) + outside * (1.0 + 2.0 * x) / (1.0 + x) ** 2)


def comb_error_matched(decay_s: float, pilot_spacing_hz: float, snr_db: float) -> float:
    """Error of the matched Wiener filter on an unbounded comb, exponential channel."""
    if decay_s <= 0 or pilot_spacing_hz <= 0:
        raise ValueError("decay_s and pilot_spacing_hz must be positive")
    b = pilot_spacing_hz * decay_s / 10.0 ** (snr_db / 10.0)
    return float(b * np.log1p(1.0 / b))


def uniform_span_rule(
    rms_delay_spread_s: float,
    pilot_spacing_hz: float,
    snr_db: float,
    bandwidth_hz: float | None = None,
    max_span_s: float | None = None,
) -> float:
    """Width in seconds of the uniform design profile for a channel of given RMS delay spread.

    The unbounded-comb span is ``s ln(g / (F s))`` with ``s`` the RMS delay spread, ``F`` the
    pilot spacing in hertz and ``g`` the signal-to-noise ratio. When ``bandwidth_hz`` is
    given, one resolution cell ``1 / B`` is subtracted, limited to half of that span. The
    result is capped by ``max_span_s`` (for example the cyclic prefix) and by the delay range
    ``1 / F`` that the comb resolves without aliasing, and is never negative.
    """
    if rms_delay_spread_s <= 0 or pilot_spacing_hz <= 0:
        raise ValueError("rms_delay_spread_s and pilot_spacing_hz must be positive")
    ratio = 10.0 ** (snr_db / 10.0) / (pilot_spacing_hz * rms_delay_spread_s)
    span = rms_delay_spread_s * np.log(ratio) if ratio > 1.0 else 0.0
    if bandwidth_hz is not None:
        if bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be positive")
        span -= min(1.0 / bandwidth_hz, 0.5 * span)
    cap = 1.0 / pilot_spacing_hz
    if max_span_s is not None:
        cap = min(cap, max_span_s)
    return float(max(min(span, cap), 0.0))


def uniform_design(grid: OfdmGrid, span_s: float, name: str | None = None) -> WienerFilter:
    """Wiener filter designed on a uniform profile of width ``span_s`` at the actual noise level."""
    profile = DelayProfile.uniform(span_s, grid.sample_period_s)
    return WienerFilter(profile, None, name=name or f"Wiener (uniform to {span_s*1e9:.0f} ns)")


def rule_span(grid: OfdmGrid, pilots: PilotMap, rms_delay_spread_s: float, snr_db: float, max_delay_s: float | None = None, band_correction: bool = True) -> float:
    """Width from :func:`uniform_span_rule` for a grid and pilot map, with the caps of the grid.

    The width is capped by the cyclic prefix, by ``max_delay_s`` when the largest delay of
    the channel is known, and so that the taps of the design profile, one sample period
    apart, stay inside the alias-free range of the comb. ``band_correction=False`` leaves
    out the resolution-cell term.
    """
    spacing = mean_pilot_spacing_hz(grid, pilots)
    band = grid.n_active * grid.spacing_hz if band_correction else None
    span = uniform_span_rule(rms_delay_spread_s, spacing, snr_db, band)
    limits = [1.0 / spacing - grid.sample_period_s]
    if grid.cp_duration_s is not None:
        limits.append(grid.cp_duration_s)
    if max_delay_s is not None:
        limits.append(max_delay_s)
    return float(max(min(span, *limits), 0.0))


def rule_wiener(grid: OfdmGrid, pilots: PilotMap, rms_delay_spread_s: float, snr_db: float, max_delay_s: float | None = None, name: str | None = None) -> WienerFilter:
    """Wiener filter designed on the uniform profile whose width follows :func:`rule_span`."""
    span = rule_span(grid, pilots, rms_delay_spread_s, snr_db, max_delay_s)
    return uniform_design(grid, span, name or f"Wiener (uniform, span rule, {span*1e9:.0f} ns)")


def best_uniform_span(grid: OfdmGrid, pilots: PilotMap, profile: DelayProfile, snr_db: float, max_span_s: float | None = None) -> tuple[float, float]:
    """Width of the uniform design profile with the smallest exact error, and that error.

    Every width on the sample grid up to ``max_span_s`` is evaluated; the default limit is
    the smallest of the cyclic prefix, the alias-free range of the comb and thirty times
    the RMS delay spread of ``profile``.
    """
    limits = [1.0 / mean_pilot_spacing_hz(grid, pilots), 30.0 * max(profile.rms_delay_spread_s, grid.sample_period_s)]
    if grid.cp_duration_s is not None:
        limits.append(grid.cp_duration_s)
    if max_span_s is not None:
        limits.append(max_span_s)
    n_max = int(np.floor(min(limits) / grid.sample_period_s + 1e-9)) + 1
    best_span, best_error = 0.0, np.inf
    for n in range(1, n_max + 1):
        span = (n - 1) * grid.sample_period_s
        error = exact_error(uniform_design(grid, span), grid, pilots, profile, snr_db).total
        if error < best_error:
            best_span, best_error = span, error
    return float(best_span), float(best_error)
