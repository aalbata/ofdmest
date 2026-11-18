"""Which delay spread enters the span rule on the measured profiles.

The measured NIST profiles have two delay-spread figures: the RMS delay spread of the
averaged profile over all kept samples, and the smaller mean RMS delay spread that NIST
reports for the site, computed per record over the samples within 15 dB of the peak. For
each measured profile, NR and LTE map and SNR, the Wiener filter is designed by the span
rule with either figure, with and without the cap at the largest delay of the profile, and
its exact loss relative to the matched filter is written to results/spread_choice.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ofdmest import WienerFilter, exact_error, rule_span, uniform_design
from common import RESULTS, SNR_DB, grids, measured_profiles

GRIDS = ("NR 20 MHz, DM-RS type 1", "NR 20 MHz, DM-RS type 2", "LTE 20 MHz, CRS")
REPORTED_NS = {"NIST steam plant": 36.1, "NIST machine shop": 74.3, "NIST automotive plant": 131.8}


def main():
    rows = []
    all_grids = grids()
    for label in GRIDS:
        grid, pilots = all_grids[label]
        for name, profile in measured_profiles(grid).items():
            site = name.split(",")[0]
            spreads = {"averaged profile": profile.rms_delay_spread_s, "reported for the site": REPORTED_NS[site] * 1e-9}
            for snr in SNR_DB:
                matched = exact_error(WienerFilter(), grid, pilots, profile, snr).total
                for statistic, spread in spreads.items():
                    row = {"grid": label, "profile": name, "snr_db": snr, "statistic": statistic, "delay_spread_ns": spread * 1e9, "largest_delay_ns": profile.max_delay_s * 1e9}
                    for key, cap in (("rule", None), ("capped", profile.max_delay_s)):
                        span = rule_span(grid, pilots, spread, snr, max_delay_s=cap)
                        error = exact_error(uniform_design(grid, span), grid, pilots, profile, snr).total
                        row[f"{key}_span_ns"] = span * 1e9
                        row[f"{key}_loss_db"] = 10 * np.log10(error / matched)
                    rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "spread_choice.csv", index=False)
    print(f"wrote {RESULTS / 'spread_choice.csv'} with {len(df)} rows")


if __name__ == "__main__":
    main()
