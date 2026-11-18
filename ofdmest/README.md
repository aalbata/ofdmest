# ofdmest

Exact error analysis and design of linear pilot-aided OFDM channel estimators.

A pilot-aided OFDM receiver turns a few pilot measurements into a channel estimate for
every subcarrier. The estimator is the receiver designer's choice: nearest pilot, linear or
cubic spline interpolation, a transform-domain (DFT) estimator, or a Wiener filter built on
an assumed delay profile. Every one of these is a linear map from the pilot observations to
the estimate, so its expected error follows exactly from the pilot map, the second-order
channel statistics (the power-delay profile) and the per-tone noise level. This package

- computes that error per subcarrier, in milliseconds, for standard pilot maps (5G NR
  DM-RS, LTE CRS, DVB-T2 scattered pilots) and custom maps;
- computes what the error costs a link: the effective SNR of detection with the estimate,
  the achievable rate that follows from it and the SNR loss relative to a perfectly known
  channel;
- designs the Wiener filter from the RMS delay spread alone, with a uniform profile whose
  width follows a closed-form span rule, and finds the best width by exact search;
- checks the exact values against Monte Carlo simulation, compares the computational cost
  of the estimators, and optimises free pilot maps for proprietary links, including maps
  that avoid tones hit by narrowband interference.

The analysis covers one OFDM symbol (block fading within the symbol, cyclic prefix longer
than the delay span, one antenna) and uses second-order statistics only. Bit error rates are
measured by simulation, not computed in closed form.

## Installation

```
pip install -e .
```

Requirements: Python 3.12, NumPy, SciPy, pandas and Matplotlib at the versions pinned in
`pyproject.toml`. Tested on Windows 11 with Python 3.12.10.

## First result in one minute

Exact error, rate and SNR loss of six estimators on a 20 MHz LTE grid with a 200 ns
exponential delay profile:

```
ofdmest error --grid lte:100 --profile exp:200ns --snr 0,15,30
```

Width of the uniform design profile of a Wiener filter for a hall with a 132 ns delay
spread, with the exact error of that design, of the matched filter, of the prefix-wide
design and of the best width found by search:

```
ofdmest span --grid lte:100 --profile exp:132ns --snr 5,15,25 --search
```

Cheapest estimator within 1 dB of the matched Wiener filter in a factory hall (3GPP indoor
factory model, line of sight, 48,000 m3 hall with 14,560 m2 of surface):

```
ofdmest recommend --grid nr:51:30 --profile inf:48000:14560:los --snr 5,20
```

Error over a delay-spread and SNR grid, written to a CSV file:

```
ofdmest sweep --grid lte:100 --ds 20ns:2us:12 --snr 0:30:7 --out sweep.csv
```

Check the exact values against Monte Carlo simulation and report the QPSK bit error rate:

```
ofdmest validate --grid custom:64:312500:64:0.8us --map centred:8 --profile exp:100ns --snr 15
```

Optimise an 8-pilot map on a 64-tone proprietary link with an interferer on tones 20 to 27:

```
ofdmest search --grid custom:64:312500:64:0.8us --pilots 8 --profile exp:200ns --snr 10,20 --interference 20:28:20
```

## Using the library

```python
from ofdmest import (OfdmGrid, DelayProfile, lte_crs, LinearInterpolation, WienerFilter,
                     exact_error, link_quality, rule_wiener)

grid = OfdmGrid(1200, 15e3, fft_size=2048, cp_duration_s=4.69e-6)
pilots = lte_crs(100)
profile = DelayProfile.exponential(132e-9, grid.sample_period_s)
designs = (LinearInterpolation(), rule_wiener(grid, pilots, 132e-9, snr_db=15.0), WienerFilter())
for est in designs:
    res = exact_error(est, grid, pilots, profile, snr_db=15.0)
    link = link_quality(est, grid, pilots, profile, snr_db=15.0)
    print(est.name, round(res.total_db, 2), "dB error,", round(link.snr_loss_db, 2), "dB SNR loss")
```

`exact_error` returns the per-tone error as well, split into the channel-mismatch part
(what the estimator cannot follow) and the noise part (what it lets through). A measured
delay profile is given as `DelayProfile(delays_s, powers)`; tones hit by interference are
given as an array of extra noise levels in dB.

`compare` evaluates the interpolators, the transform-domain estimator and the prefix-wide,
span-rule and matched Wiener designs at one operating point and returns the widths, errors,
SNR losses, rates and operation counts as a dictionary of plain numbers.
`compare_from_specs` does the same from the text notation of the command line, for example
`compare_from_specs("lte:100", "exp:132ns", 15.0)`. The browser application of the
repository, <https://aalbata.github.io/ofdmest/>, shows this comparison without any
installation.

## The span rule

A Wiener filter designed on a uniform profile needs only the width of that profile. For a
channel with RMS delay spread `s`, pilots `F` hertz apart, a band of `B` hertz and a
signal-to-noise ratio `g`, `uniform_span_rule` returns

```
W = s ln(g / (F s)) - 1/B
```

capped by the cyclic prefix and by the alias-free range `1/F` of the comb. The subtraction
of `1/B` is limited to half of the first term, a safeguard for bands too narrow to resolve
the profile; in that case `best_uniform_span` finds the width by exact search.
`rule_wiener` builds the filter, and `comb_error_uniform_design` and `comb_error_matched`
give the closed-form errors on an unbounded comb from which the rule follows.

## Grid presets of the command line

| Preset | Meaning |
|---|---|
| `nr:<prb>[:<scs_khz>[:<type>]]` | 5G NR with the given resource blocks, subcarrier spacing (default 30 kHz) and DM-RS type 1 or 2 (3GPP TS 38.211) |
| `lte:<prb>` | LTE with cell-specific reference signals of port 0 (3GPP TS 36.211) |
| `dvbt2:<fft_k>:<pattern>[:<gi>]` | DVB-T2 scattered pilots PP1 to PP8 of one symbol (ETSI EN 302 755); large symbols are analysed over a window set with `--window` |
| `custom:<n>:<spacing_hz>:<fft>[:<cp>]` | any grid; give the map with `--map` |

Estimator names: `nearest`, `linear` (constant edges) or `linear:linear`, `spline`,
`dft[:window]`, `ridge[:lambda]`, `robust[:span]` (Wiener with a uniform profile), `wiener`
(matched) or `wiener:<delay spread>[:<design SNR>]` (mismatched design).

## Reproducing the study

The `experiments` folder holds the scripts that produced every table and figure of the
accompanying paper; `reproduce.py` regenerates them from the shipped result files (fast
path) or from scratch (`--full`).

## Authors and licence

Ali Al Bataineh and Lakshmana Kumar Ramasamy, Higher Colleges of Technology, Abu Dhabi, United
Arab Emirates. MIT License. Please cite the accompanying paper and this repository when you use
the package.
