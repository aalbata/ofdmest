# Exact error analysis and design of pilot-aided OFDM channel estimators

Repository: <https://github.com/aalbata/ofdmest>

Browser application: <https://aalbata.github.io/ofdmest/>

This repository holds the `ofdmest` library and the scripts, result files and figures of a
study of pilot-aided OFDM channel estimation in factory channels. The study derives in
closed form how wide the uniform design profile of a Wiener filter should be for a channel
of known delay spread, and what an estimation error costs a link in signal-to-noise ratio,
and tests both with the library. A browser application gives the same computations to users
who do not program.

A pilot-aided OFDM receiver turns a few pilot measurements into a channel estimate for every
subcarrier. The standards fix the pilot positions; the estimator (nearest pilot, linear or
spline interpolation, a transform-domain estimator, or a Wiener filter built on an assumed
delay profile) is the designer's choice. Every one of these is a linear map from the pilot
observations to the estimate, so its expected error follows exactly from the pilot map, the
second-order channel statistics and the per-tone noise level. The library computes that
error and the link-level measures that follow from it, and designs the Wiener filter from
the RMS delay spread by a span rule. The study applies it to the 5G NR, LTE and DVB-T2 pilot
maps at the delay spreads measured by NIST in three factory halls, on profiles measured in
the same halls, and at the delay spreads of the 3GPP indoor factory model.

## Layout

| Path | Content |
|---|---|
| `ofdmest/` | The library (installable package, tests, package README) |
| `app/` | The browser application (page, bridge to the library, check scripts) |
| `index.html` | Forwards to `app/` when the repository is served as a website |
| `experiments/` | Scripts that produced every result file, figure and table of the paper |
| `results/` | Result files written by the experiment scripts (CSV) |
| `figures/` | Figures of the paper (PDF and PNG), produced by `experiments/make_figures.py`; the screenshot of the application is captured by `app/check/check_page.mjs` |
| `tables/` | Tables and inline numbers of the paper (LaTeX), produced by `experiments/make_tables.py` |
| `reproduce.py` | Regenerates figures and tables from the shipped results, or everything from scratch with `--full` |

## Installation and reproduction

Tested on Windows 11 with Python 3.12.10. From the repository root:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e ofdmest
python -m pytest ofdmest/tests -q
python reproduce.py
```

`python reproduce.py` rebuilds the figures and the paper's tables and inline numbers from the
shipped result files in under a minute. `python reproduce.py --full` reruns every experiment
first; it took one and three quarter hours on a laptop with an Intel Core Ultra 7 155U processor and writes the result files anew. All
experiments are deterministic given their recorded random-number seeds and the four
linear-algebra threads that `reproduce.py` sets, except `results/cost.csv` and the
`ga_seconds` column of `results/pilot_maps.csv`, whose measured run times depend on the
machine.

## Browser application

<https://aalbata.github.io/ofdmest/> runs the library in a web browser. Nothing is installed
and no programming is needed. Choose a pilot map (5G NR, LTE, DVB-T2, a 64-subcarrier link or
a custom grid), give the delay profile of the channel (an RMS delay spread, the size of a
hall for the 3GPP indoor factory model, one of the three measured NIST profiles, or a CSV
file with your own profile) and a list of SNR values. The page returns

- the width of the uniform design profile from the span rule, in nanoseconds and in taps;
- for each estimator the exact error, the error above the matched Wiener filter, the SNR
  loss, the achievable rate and the multiplications per symbol;
- charts of the error and of the SNR loss against SNR, and a CSV file with all results.

Optional settings give the rule a delay spread different from that of the profile, cap the
width at the largest delay of the profile, and search the width with the smallest exact
error.

Every value is computed by the `ofdmest` package itself, which the page runs in Pyodide
314.0.8, a WebAssembly build of Python 3.14.2 with NumPy 2.4.6 and SciPy 1.18.0. These
versions differ from the ones pinned for the desktop package. The page loads Pyodide from
cdn.jsdelivr.net and the package source from this repository. The computation runs in the
browser tab, so the inputs, including a profile file, stay on the user's computer. The
results are exact values from second-order statistics; nothing is simulated or precomputed.
The 20 MHz LTE map took about 13 seconds per SNR value in Microsoft Edge on a laptop with
an Intel Core Ultra 7 155U processor, the other maps took less, and the width search takes
several times longer.

To serve the page from a local copy, run `python -m http.server 8000` in the repository
root and open <http://localhost:8000/>.

The agreement of the page with the desktop library is checked by the scripts in
`app/check`. `make_reference.py` computes seven requests with the library under CPython
3.12.10, and `check_page.mjs` replays them through the page in a Chromium-based browser and
writes `results/app_check.json`. In the recorded run (Microsoft Edge 157 on Windows 11) the
364 numbers compared differ by at most 2.1e-11. The same script captured
`figures/app_screenshot.png`.

## Results

Error of four estimators at the factory profiles, from `results/regime.csv` (normalised mean
squared error of the channel estimate in dB, lower is better; 20 MHz LTE grid, cell-specific
reference signals, 15 dB SNR, exponential profiles with the NIST delay spreads):

| Profile | Matched Wiener | Wiener, uniform profile by the span rule | Wiener, uniform over the prefix | Linear interpolation |
|---|---|---|---|---|
| Steam plant, 36 ns | -29.1 | -28.7 | -18.3 | -16.7 |
| Machine shop, 74 ns | -26.8 | -26.3 | -18.4 | -16.7 |
| Automotive plant, 132 ns | -24.9 | -24.3 | -18.4 | -16.7 |

Loss of the uniform Wiener design relative to the matched filter for four ways of choosing
the width of its profile, from `results/span.csv` (dB, median / maximum over the profiles
and seven SNRs from 0 to 30 dB):

| Map and profiles | Cells | Best width (exact search) | Span rule | Eight times the delay spread | Cyclic prefix |
|---|---|---|---|---|---|
| NR type 1, seven exponential profiles | 49 | 0.5 / 0.6 | 0.5 / 0.7 | 1.1 / 5.4 | 6.5 / 9.4 |
| NR type 2, seven exponential profiles | 49 | 0.5 / 0.6 | 0.5 / 0.7 | 1.2 / 4.2 | 6.8 / 9.4 |
| LTE, seven exponential profiles | 49 | 0.5 / 0.6 | 0.5 / 0.7 | 1.2 / 4.0 | 9.4 / 11.9 |
| NR and LTE, three measured profiles | 63 | 0.5 / 1.3 | 1.1 / 2.7 | 1.2 / 2.2 | 6.9 / 10.2 |
| DVB-T2 window, four maps | 280 | 0.2 / 0.7 | 0.3 / 0.9 | 0.7 / 1.8 | 16.8 / 22.9 |
| 64-subcarrier link | 70 | 0.6 / 11.3 | 1.0 / 12.0 | 4.7 / 17.7 | 9.4 / 28.5 |

On the measured profiles the span rule loses more above 15 dB SNR, because it assumes an
exponential tail and the measured profiles end about six to seven delay spreads after the
first arrival; capped at the largest measured delay it loses at most 1.3 dB. An
underestimated delay spread is the costly error: with the smaller delay spreads that NIST
reports for the sites, computed over the samples within 15 dB of the peak, the rule loses up
to 5.4 dB at the steam plant, and up to 1.8 and 2.1 dB at the machine shop and the automotive
plant against 2.0 and 2.5 dB with the delay spread of the averaged profile
(`results/spread_choice.csv`). On the 64-subcarrier link the pilot comb does not resolve the
profiles, and the exact search (`best_uniform_span`) should set the width. The term `1/B` of
the rule is an empirical correction for the finite band.

SNR loss that the estimation error costs relative to a perfectly known channel, from
`results/rate.csv` (LTE grid, automotive-plant profile, 15 dB SNR): nearest pilot 3.1 dB,
linear interpolation 2.3 dB, Wiener over the prefix 1.7 dB, Wiener by the span rule 0.49 dB,
matched Wiener 0.44 dB.

Exact error against Monte Carlo (`results/validation.csv`): 408 cells, 96.3 percent of the
exact values inside the 95 percent interval of the simulation, largest relative deviation
2.3 percent.

## Measured factory profiles

`results/nist_profiles/` holds the average power-delay profiles derived from the public NIST
industrial wireless channel-sounder measurements (doi:10.18434/T44S3N), one run per site at
2.245 GHz with vertical polarisation, produced by `experiments/process_nist.py` from the
original MATLAB 7.3 files (1.5 to 5.8 GB each, not shipped). Reproducing those files needs
the `h5py` package (version 3.16.0 was used) and the three NIST files named in
`results/nist_profiles/README.md`; everything else in the study runs from the shipped
profiles.

Statistics of the three runs (from the `_stats.csv` files):

| Site | Records used | RMS delay spread, 15 dB window, after outlier test (ns) | Reported by NIST for the run (ns) | RMS delay spread, all kept samples (ns) | RMS delay spread of the averaged profile (ns) | Taps |
|---|---|---|---|---|---|---|
| Steam plant | 12,815 | 32.6 | 35.4 | 44.8 | 54.8 | 76 |
| Machine shop | 11,099 | 65.3 | 64.9 | 82.4 | 93.4 | 53 |
| Automotive plant | 34,881 | 116.7 | 135.3 | 184.3 | 150.0 | 73 |

## Result file columns

`results/regime.csv`: `grid` and `map` (pilot map), `n_tones`, `n_pilots`, `profile_kind`
(`sweep` for the exponential delay-spread sweep, `site` for the factory profiles), `profile`,
`rms_delay_spread_ns`, `snr_db`, `estimator`, `error` (normalised mean squared error over the
active tones) and `error_db`, `bias` (part of the error the estimator cannot follow) and
`noise` (part let through from the pilot noise), `edge_error` (mean over the two band-edge
tones), `interior_error` (mean over the central half of the band), `real_multiplications`
per OFDM symbol.

`results/span.csv`: one row per grid, factory profile and SNR. `measured` (profile from the
NIST measurements), `rms_delay_spread_ns`, `pilot_spacing_khz` (mean pilot spacing),
`band_times_spread` (band width times RMS delay spread), `sample_period_ns`, `development`
(cell inspected while the band term of the rule was set), `matched_db` (error of the matched
filter), `best_taps`, `best_width_ns` and `best_loss_db` (width with the smallest exact error
among `searched_taps` widths, and its loss relative to the matched filter). For each way of
choosing the width (`rule`: the span rule; `unbounded`: the rule without the band term;
`capped`: the rule capped at the largest delay of the profile; `eight`: eight times the RMS
delay spread) the columns `<name>_span_ns`, `<name>_taps` and `<name>_loss_db`.
`prefix_loss_db` (uniform design over the cyclic prefix), `model_matched_db` and
`model_rule_db` (closed-form errors on an unbounded comb for the matched filter and for the
rule width increased by one resolution cell), `rule_error_db`.

`results/span_curves.csv`: `grid`, `profile`, `rms_delay_spread_ns`, `snr_db`, `taps` and
`width_ns` of the uniform design, `error_db` (exact), `matched_db`, `model_unbounded_db` and
`model_band_db` (closed form at the width and at the width plus one resolution cell).

`results/spread_choice.csv`: span rule on the measured profiles with two delay-spread
figures. `grid`, `profile`, `snr_db`, `statistic` (`averaged profile`: RMS delay spread of
the averaged profile; `reported for the site`: mean RMS delay spread that NIST reports),
`delay_spread_ns`, `largest_delay_ns`, and for the rule (`rule`) and the rule capped at the
largest delay (`capped`) the columns `<name>_span_ns` and `<name>_loss_db`.

`results/rate.csv`: `grid`, `profile`, `rms_delay_spread_ns`, `snr_db`, `estimator`,
`error_db`, `rate_bits` (mean achievable rate per subcarrier use when the estimation error
is treated as noise, Rayleigh fading), `perfect_rate_bits` (the same with a perfectly known
channel), `snr_loss_db` (difference of the SNRs at which a perfectly known channel reaches
the two rates), `predicted_loss_db` (10 log10((1 + g e) / (1 - e)) from the SNR g and the
error e).

`results/mismatch.csv`: `grid`, `profile`, `rms_delay_spread_ns`, `snr_db`, `design`
(design profile of the Wiener filter), `design_scale_ns` (span or delay spread of the design
profile), `error_db`, `matched_db` (error of the matched filter), `loss_db` (difference),
`bias`, `noise`.

`results/validation.csv`: `map`, `n_tones`, `n_pilots`, `rms_delay_spread_ns`, `snr_db`,
`estimator`, `exact_mse`, `mc_mse` (Monte Carlo mean over 10,000 symbols),
`mc_half_width_95`, `relative_error_percent`, `inside_95`, `mc_ber` (QPSK bit error rate).

`results/ber.csv` and `results/ber_16qam.csv`: `grid`, `profile`, `rms_delay_spread_ns`,
`snr_db`, `estimator`, `exact_error_db`, `mc_error_db`, `mc_half_width_db`, `modulation`,
`ber`, `bit_errors`, `bits`, `n_symbols`, `seed`.

`results/cost.csv`: `grid`, `n_tones`, `n_pilots`, `estimator`, `real_multiplications`,
`design_ms` (time to build the estimator matrix), `dense_apply_us_per_symbol`,
`structured_apply_us_per_symbol` (where a structured implementation exists), `machine`.

`results/per_tone.csv`: `estimator`, `subcarrier`, `error`, `bias`, `noise` for the LTE grid,
automotive-plant profile, 15 dB SNR.

`results/app_check.json`: outcome of the check of the browser application. `page_runtime`
(versions of Python, NumPy, SciPy, Pyodide, the package and the browser in the page),
`reference_runtime` (versions under CPython), `cases`, `numbers_compared`,
`largest_absolute_difference`, `tolerance` (relative, with a floor of one unit) and
`differences_beyond_tolerance`.

`results/gaps.csv` and `results/recommendations.csv`: summaries written by
`experiments/make_tables.py` from `regime.csv` (gaps to the matched filter over the
delay-spread sweep; cheapest estimator within 1 dB of the matched filter at each factory
cell).

`results/pilot_maps_small.csv` and `results/pilot_maps.csv`: a demonstration of the pilot-map
search of the library on a 64-tone link, not reported in the paper. `pilot_maps_small.csv`:
`n_tones`, `n_pilots`, `rms_delay_spread_samples`, `snr_db`, `map` (anchored or centred rule),
`indices`, `error_db`, `optimum_db` and `optimum_indices` (exhaustive optimum),
`maps_enumerated`. `pilot_maps.csv`: `rms_delay_spread_ns`, `scenario`, `map`, `indices`,
`objective_db` (mean exact error over 10 and 20 dB SNR), `ga_seconds`, `ga_maps_evaluated`.

## Licence

MIT. Please cite the accompanying paper and this repository when you use the code.
