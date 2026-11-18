"""Span study: width of the uniform design profile of a Wiener filter.

For each grid, factory profile and SNR, the exact error of the uniform design is computed
for every width on the sample grid, which gives the best width, and at the widths given by
the span rule of ofdmest.design (with and without the resolution-cell term, and capped
at the largest delay of the profile), by eight times the RMS delay spread and by the
cyclic prefix. The closed-form errors of the
unbounded comb are stored next to the exact ones. Writes results/span.csv with one row per
cell and results/span_curves.csv with the full error-against-width curves of the cells
shown in the figure.

The column ``development`` marks the cells that were inspected while the resolution-cell
term and its limit were set (exponential profiles at the three NIST delay spreads, SNRs of
5, 15 and 25 dB, on the NR, LTE and DVB-T2 PP1, PP3 and PP7 maps); all other cells were
evaluated afterwards with the rule unchanged.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ofdmest import WienerFilter, comb_error_matched, comb_error_uniform_design, exact_error, mean_pilot_spacing_hz, robust_wiener, rule_span, uniform_centred, uniform_design
from common import RESULTS, SNR_DB, grids, proprietary_grid, site_profiles

DEVELOPMENT_GRIDS = ("NR 20 MHz, DM-RS type 1", "NR 20 MHz, DM-RS type 2", "LTE 20 MHz, CRS", "DVB-T2 8K, PP1 (spacing 3)", "DVB-T2 8K, PP3 (spacing 6)", "DVB-T2 8K, PP7 (spacing 24)")
DEVELOPMENT_PROFILES = ("NIST steam plant (36 ns)", "NIST machine shop (74 ns)", "NIST automotive plant (132 ns)")
DEVELOPMENT_SNR = (5.0, 15.0, 25.0)
CURVE_GRIDS = ("LTE 20 MHz, CRS", "NR 20 MHz, DM-RS type 1")
MAX_FACTOR = 24.0


def taps_for(span_s: float, sample_period_s: float) -> int:
    """Number of equal taps of the design profile of width ``span_s``."""
    return int(np.floor(span_s / sample_period_s + 1e-9)) + 1


def main():
    all_grids = dict(grids())
    link = proprietary_grid()
    all_grids["Proprietary link, 8 centred pilots"] = (link, uniform_centred(link.n_subcarriers, 8))
    rows, curves = [], []
    t0 = time.perf_counter()
    for label, (grid, pilots) in all_grids.items():
        spacing = mean_pilot_spacing_hz(grid, pilots)
        band = grid.n_active * grid.spacing_hz
        ts = grid.sample_period_s
        for pname, profile in site_profiles(grid).items():
            spread = profile.rms_delay_spread_s
            limit = min(MAX_FACTOR * spread, grid.cp_duration_s, 1.0 / spacing)
            n_max = max(taps_for(limit, ts), 2)
            for snr in SNR_DB:
                matched = exact_error(WienerFilter(), grid, pilots, profile, snr).total
                errors = np.array([exact_error(uniform_design(grid, (n - 1) * ts), grid, pilots, profile, snr).total for n in range(1, n_max + 1)])
                best = int(np.argmin(errors))
                spans = {
                    "rule": rule_span(grid, pilots, spread, snr),
                    "unbounded": rule_span(grid, pilots, spread, snr, band_correction=False),
                    "capped": rule_span(grid, pilots, spread, snr, max_delay_s=profile.max_delay_s),
                    "eight": min(8.0 * spread, grid.cp_duration_s),
                }
                row = {
                    "grid": label, "profile": pname, "measured": "measured" in pname, "rms_delay_spread_ns": spread * 1e9, "snr_db": snr,
                    "pilot_spacing_khz": spacing / 1e3, "band_times_spread": band * spread, "sample_period_ns": ts * 1e9,
                    "development": label in DEVELOPMENT_GRIDS and pname in DEVELOPMENT_PROFILES and snr in DEVELOPMENT_SNR,
                    "matched_db": 10 * np.log10(matched), "best_taps": best + 1, "best_width_ns": (best + 1) * ts * 1e9,
                    "best_loss_db": 10 * np.log10(errors[best] / matched), "searched_taps": n_max,
                }
                for key, span in spans.items():
                    n = taps_for(span, ts)
                    error = errors[n - 1] if n <= n_max else exact_error(uniform_design(grid, span), grid, pilots, profile, snr).total
                    row[f"{key}_span_ns"] = span * 1e9
                    row[f"{key}_taps"] = n
                    row[f"{key}_loss_db"] = 10 * np.log10(error / matched)
                row["prefix_loss_db"] = 10 * np.log10(exact_error(robust_wiener(grid), grid, pilots, profile, snr).total / matched)
                row["model_matched_db"] = 10 * np.log10(comb_error_matched(spread, spacing, snr))
                width = taps_for(spans["rule"], ts) * ts
                row["model_rule_db"] = 10 * np.log10(comb_error_uniform_design(width + 1.0 / band, spread, spacing, snr))
                row["rule_error_db"] = row["matched_db"] + row["rule_loss_db"]
                rows.append(row)
                if label in CURVE_GRIDS and pname in DEVELOPMENT_PROFILES and snr in DEVELOPMENT_SNR:
                    for n, error in enumerate(errors, start=1):
                        width = n * ts
                        curves.append({
                            "grid": label, "profile": pname, "rms_delay_spread_ns": spread * 1e9, "snr_db": snr, "taps": n, "width_ns": width * 1e9,
                            "error_db": 10 * np.log10(error), "matched_db": 10 * np.log10(matched),
                            "model_unbounded_db": 10 * np.log10(comb_error_uniform_design(width, spread, spacing, snr)),
                            "model_band_db": 10 * np.log10(comb_error_uniform_design(width + 1.0 / band, spread, spacing, snr)),
                        })
        print(f"{label:36s} done, elapsed {time.perf_counter()-t0:6.1f} s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "span.csv", index=False)
    pd.DataFrame(curves).to_csv(RESULTS / "span_curves.csv", index=False)
    print(f"wrote {RESULTS / 'span.csv'} with {len(df)} rows and {RESULTS / 'span_curves.csv'} with {len(curves)} rows")


if __name__ == "__main__":
    main()
