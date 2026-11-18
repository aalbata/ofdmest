"""Exact error analysis and design of linear pilot-aided OFDM channel estimators."""
from .analysis import ErrorResult, LinkQuality, equivalent_snr_db, error_components, error_from_matrix, exact_error, ideal_error_db, link_quality, rayleigh_rate
from .channel import DelayProfile, OfdmGrid, indoor_factory_delay_spread_s
from .design import best_uniform_span, comb_error_matched, comb_error_uniform_design, mean_pilot_spacing_hz, rule_span, rule_wiener, uniform_design, uniform_span_rule
from .estimators import (
    CubicSplineInterpolation,
    DftInterpolation,
    DftLeastSquares,
    Estimator,
    LinearInterpolation,
    NearestPilot,
    OperatingPoint,
    WienerFilter,
    noise_variance_from_snr,
    robust_wiener,
)
from .pilots import (
    DVBT2_SCATTERED_PATTERNS,
    PilotMap,
    comb,
    dvbt2_scattered,
    lte_crs,
    nr_dmrs_type1,
    nr_dmrs_type2,
    uniform_anchored,
    uniform_centred,
)
from .report import compare, compare_from_specs, standard_estimators
from .simulate import SimulationResult, simulate

__version__ = "0.3.0"

__all__ = [
    "CubicSplineInterpolation",
    "DVBT2_SCATTERED_PATTERNS",
    "DelayProfile",
    "DftInterpolation",
    "DftLeastSquares",
    "ErrorResult",
    "Estimator",
    "LinearInterpolation",
    "LinkQuality",
    "NearestPilot",
    "OfdmGrid",
    "OperatingPoint",
    "PilotMap",
    "SimulationResult",
    "WienerFilter",
    "best_uniform_span",
    "comb",
    "comb_error_matched",
    "comb_error_uniform_design",
    "compare",
    "compare_from_specs",
    "dvbt2_scattered",
    "equivalent_snr_db",
    "error_components",
    "error_from_matrix",
    "exact_error",
    "ideal_error_db",
    "indoor_factory_delay_spread_s",
    "link_quality",
    "lte_crs",
    "mean_pilot_spacing_hz",
    "noise_variance_from_snr",
    "nr_dmrs_type1",
    "nr_dmrs_type2",
    "rayleigh_rate",
    "robust_wiener",
    "rule_span",
    "rule_wiener",
    "simulate",
    "standard_estimators",
    "uniform_anchored",
    "uniform_centred",
    "uniform_design",
    "uniform_span_rule",
]
