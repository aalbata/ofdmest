"""Rate study: what the estimation error costs a link, without a receiver simulation.

For each grid, factory profile and SNR, the effective SNR of detection with each estimator
is computed exactly from the second-order statistics, and from it the achievable rate of a
Rayleigh-fading link that treats the estimation error as noise and the SNR loss relative to
a perfectly known channel. The loss predicted from the estimation error alone,
10 log10((1 + g e) / (1 - e)) with g the SNR and e the error, is stored next to it. Writes
results/rate.csv.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ofdmest import exact_error, link_quality, rule_wiener
from common import RESULTS, SNR_DB, estimator_set, grids, site_profiles

GRIDS = ("NR 20 MHz, DM-RS type 1", "NR 20 MHz, DM-RS type 2", "LTE 20 MHz, CRS", "DVB-T2 8K, PP3 (spacing 6)")
RULE_NAME = "Wiener (uniform, span rule)"


def main():
    rows = []
    t0 = time.perf_counter()
    all_grids = grids()
    for label in GRIDS:
        grid, pilots = all_grids[label]
        for pname, profile in site_profiles(grid).items():
            base = estimator_set(grid, pilots, profile)
            for snr in SNR_DB:
                g = 10.0 ** (snr / 10.0)
                ests = base + [rule_wiener(grid, pilots, profile.rms_delay_spread_s, snr, name=RULE_NAME)]
                for est in ests:
                    error = exact_error(est, grid, pilots, profile, snr).total
                    link = link_quality(est, grid, pilots, profile, snr)
                    rows.append({
                        "grid": label, "profile": pname, "rms_delay_spread_ns": profile.rms_delay_spread_s * 1e9, "snr_db": snr, "estimator": est.name,
                        "error_db": 10 * np.log10(error), "rate_bits": link.rate_bits, "perfect_rate_bits": link.perfect_rate_bits,
                        "snr_loss_db": link.snr_loss_db, "predicted_loss_db": 10 * np.log10((1.0 + g * error) / (1.0 - error)) if error < 1 else np.nan,
                    })
        print(f"{label:32s} done, elapsed {time.perf_counter()-t0:6.1f} s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "rate.csv", index=False)
    print(f"wrote {RESULTS / 'rate.csv'} with {len(df)} rows")


if __name__ == "__main__":
    main()
