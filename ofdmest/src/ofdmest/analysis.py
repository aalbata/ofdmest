"""Exact expected error of linear pilot-aided estimators.

With ``H`` the channel over the active tones, ``S`` the selection of the pilot tones and
``w`` the noise on the pilots, a linear estimator gives ``H_hat = A (S H + w)``. Its error
covariance is

    C = (A S - I) R (A S - I)^H + A diag(sigma_p^2) A^H,

where ``R`` is the channel covariance over the active tones. The per-tone mean squared
error is the diagonal of ``C`` divided by the per-tone channel power, and the total error
is its mean over the active tones. Only second-order statistics enter, so the result is
exact for any zero-mean channel with that covariance.

The same statistics give a link-level measure. Writing ``H_m = c_m H_hat_m + e_m`` with
``e_m`` uncorrelated with the estimate, a receiver that detects the data with the estimate
sees the signal power ``s_m = |E{H_m conj(H_hat_m)}|^2 / E{|H_hat_m|^2}`` and the residual
``R_mm - s_m`` in addition to the noise, so the effective signal-to-noise ratio on tone
``m`` is ``s_m / (R_mm - s_m + sigma_m^2)``. For a Rayleigh-fading channel the mean of
``log2(1 + effective SNR)`` over the fading is a rate that the link can achieve when the
estimation error is treated as noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import exp1

from .channel import DelayProfile, OfdmGrid
from .estimators import Estimator, OperatingPoint, noise_variance_from_snr
from .pilots import PilotMap


@dataclass(frozen=True)
class ErrorResult:
    """Exact error of one estimator at one operating point."""

    estimator: str
    per_tone: np.ndarray
    bias_per_tone: np.ndarray
    noise_per_tone: np.ndarray
    real_multiplications: int

    @property
    def total(self) -> float:
        """Normalised mean squared error averaged over the active tones."""
        return float(np.mean(self.per_tone))

    @property
    def total_db(self) -> float:
        return float(10.0 * np.log10(self.total))

    @property
    def bias_total(self) -> float:
        return float(np.mean(self.bias_per_tone))

    @property
    def noise_total(self) -> float:
        return float(np.mean(self.noise_per_tone))


def error_from_matrix(A: np.ndarray, grid: OfdmGrid, pilots: PilotMap, point: OperatingPoint) -> tuple[np.ndarray, np.ndarray]:
    """Per-tone channel-mismatch and noise contributions to the error for a given matrix."""
    pos = pilots.positions_in(grid)
    act = grid.active_indices
    if A.shape != (act.size, pilots.count):
        raise ValueError("A must have one row per active tone and one column per pilot")
    var = point.noise_variance
    if var.size == 1:
        var = np.broadcast_to(var, (grid.n_subcarriers,))
    if var.shape != (grid.n_subcarriers,):
        raise ValueError("noise variance must be a scalar or have one entry per subcarrier")
    R = point.profile.covariance(grid.n_subcarriers, grid.spacing_hz)
    Ra = R[np.ix_(act, act)]
    X = A @ Ra[pos, :]
    diag_bRb = np.einsum("ik,ik->i", X[:, pos], A.conj()).real
    diag_bR = np.diag(X).real
    diag_R = np.diag(Ra).real
    bias = (diag_bRb - 2.0 * diag_bR + diag_R) / diag_R
    noise = (np.abs(A) ** 2 * var[pilots.indices][None, :]).sum(axis=1) / diag_R
    return np.maximum(bias, 0.0), noise


def error_components(A: np.ndarray, grid: OfdmGrid, pilots: PilotMap, profile: DelayProfile, relative_noise: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-tone channel-mismatch error and per-tone noise gain of a fixed matrix.

    The noise gain is the noise contribution for unit noise variance on every pilot (scaled
    tone by tone by ``relative_noise`` when given). For a matrix that does not depend on the
    SNR, the error at SNR ``s`` is ``bias + 10**(-s/10) * noise_gain``.
    """
    rel = np.ones(grid.n_subcarriers) if relative_noise is None else np.asarray(relative_noise, dtype=float)
    point = OperatingPoint(profile, rel)
    return error_from_matrix(A, grid, pilots, point)


def exact_error(estimator: Estimator, grid: OfdmGrid, pilots: PilotMap, profile: DelayProfile, snr_db: float, extra_noise_db: np.ndarray | None = None) -> ErrorResult:
    """Exact per-tone and total error of ``estimator`` at the given operating point."""
    point = OperatingPoint(profile, noise_variance_from_snr(snr_db, grid.n_subcarriers, extra_noise_db))
    A = estimator.matrix(grid, pilots, point)
    bias, noise = error_from_matrix(A, grid, pilots, point)
    return ErrorResult(estimator.name, bias + noise, bias, noise, estimator.real_multiplications(grid, pilots))


def ideal_error_db(grid: OfdmGrid, pilots: PilotMap, profile: DelayProfile, snr_db: float, extra_noise_db: np.ndarray | None = None) -> float:
    """Error of the matched Wiener filter, the best any linear estimator can reach."""
    from .estimators import WienerFilter

    return exact_error(WienerFilter(), grid, pilots, profile, snr_db, extra_noise_db).total_db


def rayleigh_rate(mean_snr: np.ndarray | float) -> np.ndarray:
    """Mean of ``log2(1 + a X)`` for ``X`` exponential with unit mean, ``a`` the mean SNR."""
    a = np.atleast_1d(np.asarray(mean_snr, dtype=float))
    out = np.zeros_like(a)
    pos = a > 0
    x = 1.0 / a[pos]
    small = x < 500.0
    val = np.empty_like(x)
    val[small] = np.exp(x[small]) * exp1(x[small])
    xl = x[~small]
    val[~small] = (1.0 - 1.0 / xl + 2.0 / xl**2) / xl
    out[pos] = val / np.log(2.0)
    return out


def equivalent_snr_db(rate_bits: float) -> float:
    """SNR in dB at which a Rayleigh-fading tone with a perfectly known channel reaches ``rate_bits``."""
    if rate_bits <= 0:
        return -np.inf
    return float(brentq(lambda s: float(rayleigh_rate(10.0 ** (s / 10.0))[0]) - rate_bits, -80.0, 120.0))


@dataclass(frozen=True)
class LinkQuality:
    """Effective SNR and achievable rate of detection with one estimator at one operating point."""

    estimator: str
    effective_snr: np.ndarray
    perfect_snr: np.ndarray

    @property
    def rate_bits(self) -> float:
        """Mean achievable rate in bits per subcarrier use, averaged over the active tones."""
        return float(np.mean(rayleigh_rate(self.effective_snr)))

    @property
    def perfect_rate_bits(self) -> float:
        """Mean rate with a perfectly known channel at the same noise levels."""
        return float(np.mean(rayleigh_rate(self.perfect_snr)))

    @property
    def snr_loss_db(self) -> float:
        """SNR that the estimation error costs, as the difference of the rate-equivalent SNRs."""
        return equivalent_snr_db(self.perfect_rate_bits) - equivalent_snr_db(self.rate_bits)


def link_quality(estimator: Estimator, grid: OfdmGrid, pilots: PilotMap, profile: DelayProfile, snr_db: float, extra_noise_db: np.ndarray | None = None) -> LinkQuality:
    """Effective SNR per active tone when data are detected with the estimate of ``estimator``.

    The estimate is applied to data on every active tone, with data noise of the per-tone
    variance of the operating point and independent of the pilot noise, as for the data
    symbols of a static channel that reuse the estimate of the pilot symbol.
    """
    point = OperatingPoint(profile, noise_variance_from_snr(snr_db, grid.n_subcarriers, extra_noise_db))
    A = estimator.matrix(grid, pilots, point)
    pos = pilots.positions_in(grid)
    act = grid.active_indices
    var = point.noise_variance
    R = profile.covariance(grid.n_subcarriers, grid.spacing_hz)
    Ra = R[np.ix_(act, act)]
    cross = np.einsum("ik,ik->i", Ra[:, pos], A.conj())
    Rpp = Ra[np.ix_(pos, pos)] + np.diag(var[pilots.indices])
    power = np.einsum("ik,kl,il->i", A, Rpp, A.conj()).real
    diag_R = np.diag(Ra).real
    signal = np.where(power > 0, np.abs(cross) ** 2 / np.where(power > 0, power, 1.0), 0.0)
    signal = np.minimum(signal, diag_R)
    effective = signal / (diag_R - signal + var[act])
    return LinkQuality(estimator.name, effective, diag_R / var[act])
