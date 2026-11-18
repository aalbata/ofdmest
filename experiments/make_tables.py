"""Tables and inline numbers of the paper, generated from the result files only.

Writes tables/numbers.tex (one macro per number quoted in the text) and the table files in
tables/. When the paper sources are present next to this folder, every macro they use must
be defined here; the script checks that and stops otherwise.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
TABLES.mkdir(exist_ok=True)
PAPER = ROOT / "paper"

NR1, NR2, LTE = "NR 20 MHz, DM-RS type 1", "NR 20 MHz, DM-RS type 2", "LTE 20 MHz, CRS"
WIDE = [NR1, NR2, LTE]
T2 = ["DVB-T2 8K, PP1 (spacing 3)", "DVB-T2 8K, PP3 (spacing 6)", "DVB-T2 8K, PP5 (spacing 12)", "DVB-T2 8K, PP7 (spacing 24)"]
LINK = "Proprietary link, 8 centred pilots"
GSHORT = {NR1: "NR type 1", NR2: "NR type 2", LTE: "LTE CRS"}
STEAM, SHOP, AUTO = "NIST steam plant (36 ns)", "NIST machine shop (74 ns)", "NIST automotive plant (132 ns)"
PSHORT = {STEAM: "Steam plant, 36 ns", SHOP: "Machine shop, 74 ns", AUTO: "Automotive plant, 132 ns"}
LINEAR, SPLINE, NEAREST, DFT = "linear (constant edges)", "cubic spline (constant edges)", "nearest pilot", "DFT interpolation"
MATCHED, RULE = "Wiener (matched)", "Wiener (uniform, span rule)"
SHORT = {NEAREST: "Nearest pilot", LINEAR: "Linear", SPLINE: "Spline", DFT: "Transform domain", "DFT least squares (ridge)": "Tap fit (ridge)", MATCHED: "Wiener, matched", RULE: "Wiener, span rule"}
VALIDATION_MAPS = {"proprietary anchored": "64-subcarrier, anchored", "proprietary centred": "64-subcarrier, centred"}


def short(name: str) -> str:
    if name.startswith("Wiener (uniform to"):
        return "Wiener, prefix"
    return SHORT.get(name, name)


def is_prefix(names: pd.Series) -> pd.Series:
    return names.str.startswith("Wiener (uniform to")


def fmt(v: float, nd: int = 1) -> str:
    return f"{v:.{nd}f}"


def minus(text: str) -> str:
    """Typeset the sign of negative numbers as a mathematical minus."""
    return re.sub(r"(?<![\w.$-])-(?=\d)", "$-$", text)


def table(name: str, caption: str, label: str, spec: str, header: str, body: list[str], wide: bool = False, tabcolsep: float = 3.5, placement: str = "t") -> None:
    env = "table*" if wide else "table"
    body = [minus(line) for line in body]
    text = (f"\\begin{{{env}}}[{placement}]\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\centering\n\\footnotesize\n\\setlength{{\\tabcolsep}}{{{tabcolsep}pt}}\n"
            f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header}\n\\midrule\n" + "\n".join(body) + f"\n\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")
    (TABLES / f"{name}.tex").write_text(text)


def macros_from_validation(v: pd.DataFrame, out: dict) -> None:
    out["ValCells"] = str(len(v))
    out["ValInsidePct"] = fmt(100 * v.inside_95.mean(), 1)
    out["ValMaxRelPct"] = fmt(v.relative_error_percent.abs().max(), 1)
    out["ValMedianRelPct"] = fmt(v.relative_error_percent.abs().median(), 2)
    out["ValMaps"] = str(v["map"].nunique())
    out["ValEstimators"] = str(int(v.groupby("map").estimator.nunique().max()))
    out["ValSymbols"] = "10,000"
    rows = []
    for m, g in v.groupby("map", sort=False):
        rows.append(f"{VALIDATION_MAPS.get(m, m)} & {g.n_pilots.iloc[0]}/{g.n_tones.iloc[0]} & {len(g)} & {100*g.inside_95.mean():.0f} & {g.relative_error_percent.abs().max():.2f} \\\\")
    table("validation", "Exact error against Monte Carlo simulation with 10,000 symbols per cell, with the pilots per analysed subcarriers, the number of cells, the share of cells in percent whose exact value lies inside the 95 percent interval of the simulation, and the largest relative deviation in percent",
          "tab:validation", "lcccc", "Map & Pilots & Cells & Inside & Dev. \\\\", rows, tabcolsep=1.8)


def macros_from_regime(r: pd.DataFrame, out: dict) -> None:
    sweep = r[r.profile_kind == "sweep"]
    sites = r[r.profile_kind == "site"]
    # Share of site cells in which the span-rule design is within 1 dB of the matched filter.
    rec_rows = []
    for (grid, prof, snr), h in sites.groupby(["grid", "profile", "snr_db"], sort=False):
        ref = h[h.estimator == MATCHED].error_db.iloc[0]
        ok = h[h.error_db <= ref + 1.0].sort_values(["real_multiplications", "error_db"])
        rec_rows.append({"grid": grid, "profile": prof, "snr_db": snr, "cheapest_within_1db": ok.iloc[0].estimator, "matched_db": ref,
                         "rule_db": h[h.estimator == RULE].error_db.iloc[0], "prefix_db": h[is_prefix(h.estimator)].error_db.iloc[0], "linear_db": h[h.estimator == LINEAR].error_db.iloc[0]})
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(RESULTS / "recommendations.csv", index=False)
    out["SiteCells"] = str(len(rec))
    out["RuleWithinOneDbPct"] = fmt(100 * (rec.rule_db <= rec.matched_db + 1.0).mean(), 0)
    out["PrefixWithinOneDbPct"] = fmt(100 * (rec.prefix_db <= rec.matched_db + 1.0).mean(), 0)
    # Site table: matched Wiener, span-rule design, prefix-wide design and linear interpolation.
    lines = []
    rule_losses, cp_losses, lin_gaps = [], [], []
    for prof in (STEAM, SHOP, AUTO):
        for snr in (5.0, 15.0, 25.0):
            cells = []
            for grid in WIDE:
                h = sites[(sites.grid == grid) & (sites.profile == prof) & (sites.snr_db == snr)]
                matched = float(h[h.estimator == MATCHED].error_db.iloc[0])
                rule = float(h[h.estimator == RULE].error_db.iloc[0])
                prefix = float(h[is_prefix(h.estimator)].error_db.iloc[0])
                linear = float(h[h.estimator == LINEAR].error_db.iloc[0])
                rule_losses.append(rule - matched)
                cp_losses.append(prefix - matched)
                lin_gaps.append(linear - matched)
                cells.append(f"{matched:.1f} & {rule:.1f} & {prefix:.1f} & {linear:.1f}")
            lines.append(f"{PSHORT[prof]} & {snr:.0f} & " + " & ".join(cells) + " \\\\")
    header = " & & " + " & ".join("\\multicolumn{4}{c}{" + GSHORT[g] + "}" for g in WIDE) + " \\\\\n\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-14}\nProfile & SNR (dB) & M & R & P & L & M & R & P & L & M & R & P & L \\\\"
    table("sites", "Error in dB at the exponential profiles with the NIST delay spreads for the matched design (M), the uniform design by the span rule (R), the prefix-wide design (P) and linear interpolation (L)",
          "tab:sites", "ll rrrr rrrr rrrr", header, lines, wide=True, tabcolsep=3)
    out["SiteRuleLossMin"] = fmt(min(rule_losses))
    out["SiteRuleLossMax"] = fmt(max(rule_losses))
    out["SiteCpLossMin"] = fmt(min(cp_losses))
    out["SiteCpLossMax"] = fmt(max(cp_losses))
    out["SiteLinearGapMin"] = fmt(min(lin_gaps))
    out["SiteLinearGapMax"] = fmt(max(lin_gaps))
    interp = sites[(sites.snr_db <= 25.0) & sites.estimator.isin([LINEAR, SPLINE]) & sites.grid.isin(WIDE)]
    out["InterpBiasShareMaxPct"] = fmt(100 * (interp.bias / (interp.bias + interp.noise)).max(), 1)

    def err(grid, prof, snr, est):
        h = r[(r.grid == grid) & (r.profile == prof) & (r.snr_db == snr) & (r.estimator == est)]
        return float(h.error_db.iloc[0])

    out["LteAutoLinearFifteen"] = fmt(err(LTE, AUTO, 15, LINEAR))
    out["LteAutoMatchedFifteen"] = fmt(err(LTE, AUTO, 15, MATCHED))
    out["LteSteamMatchedFifteen"] = fmt(err(LTE, STEAM, 15, MATCHED))
    out["NrOneShopLinearFifteen"] = fmt(err(NR1, SHOP, 15, LINEAR))
    out["NrOneShopMatchedFifteen"] = fmt(err(NR1, SHOP, 15, MATCHED))
    out["LteAutoNoiseGainLinear"] = fmt(10 ** (1.5 + err(LTE, AUTO, 15, LINEAR) / 10), 2)
    out["LteAutoNoiseGainMatched"] = fmt(10 ** (1.5 + err(LTE, AUTO, 15, MATCHED) / 10), 2)
    # Gaps to the matched filter over the delay-spread sweep (kept as a result file).
    lines = []
    for grid in WIDE + T2:
        g = sweep[sweep.grid == grid]
        for snr in (5.0, 15.0, 25.0):
            h = g[g.snr_db == snr]
            ref = h[h.estimator == MATCHED].set_index("rms_delay_spread_ns").error_db
            lin = h[h.estimator == LINEAR].set_index("rms_delay_spread_ns").error_db
            uni = h[is_prefix(h.estimator)].set_index("rms_delay_spread_ns").error_db
            rule = h[h.estimator == RULE].set_index("rms_delay_spread_ns").error_db
            dft = h[h.estimator == DFT].set_index("rms_delay_spread_ns").error_db if (h.estimator == DFT).any() else None
            lines.append((grid, snr, float((lin - ref).min()), float((lin - ref).max()), float((uni - ref).min()), float((uni - ref).max()), float((rule - ref).min()), float((rule - ref).max()), float((dft - ref).max()) if dft is not None else np.nan))
    gap = pd.DataFrame(lines, columns=["grid", "snr_db", "linear_gap_min_db", "linear_gap_max_db", "prefix_gap_min_db", "prefix_gap_max_db", "rule_gap_min_db", "rule_gap_max_db", "dft_gap_max_db"])
    gap.to_csv(RESULTS / "gaps.csv", index=False)
    wide = gap[gap.grid.isin(WIDE)]
    out["SweepPoints"] = str(sweep.groupby("grid").rms_delay_spread_ns.nunique().iloc[0])
    three = sweep
    keys = ["grid", "snr_db", "rms_delay_spread_ns"]
    rule_gap = (three[three.estimator == RULE].set_index(keys).error_db - three[three.estimator == MATCHED].set_index(keys).error_db).reset_index()
    out["SweepRuleGapMaxLong"] = fmt(rule_gap[rule_gap.rms_delay_spread_ns >= 19.0].error_db.max())
    out["SweepRuleGapMaxShort"] = fmt(rule_gap[rule_gap.rms_delay_spread_ns < 19.0].error_db.max())
    out["SweepPrefixGapMaxWide"] = fmt(wide.prefix_gap_max_db.max(), 0)
    out["SweepPrefixGapMinWide"] = fmt(wide.prefix_gap_min_db.min())
    out["SweepLinearGapMaxWide"] = fmt(wide.linear_gap_max_db.max(), 0)
    high = sweep[(sweep.snr_db >= 15.0) & sweep.grid.isin([NR1, LTE])]
    dft_gap = (high[high.estimator == DFT].set_index(keys).error_db - high[high.estimator == MATCHED].set_index(keys).error_db).groupby(level=[0, 1]).max()
    out["SweepDftGapMinHighSnr"] = fmt(dft_gap.min(), 0)
    out["SweepDftGapMaxHighSnr"] = fmt(dft_gap.max(), 0)
    fifteen = sites[sites.grid.isin(WIDE) & (sites.snr_db == 15.0)]
    cell = ["grid", "profile"]
    near_lin = fifteen[fifteen.estimator == NEAREST].set_index(cell).error_db - fifteen[fifteen.estimator == LINEAR].set_index(cell).error_db
    out["NearestMinusLinearMin"] = fmt(near_lin.min())
    out["NearestMinusLinearMax"] = fmt(near_lin.max())


def macros_from_curves(cv: pd.DataFrame, sp: pd.DataFrame, out: dict) -> None:
    """Agreement of the closed forms with the exact error on the falling side of the curves."""
    wide_for_low, narrow_for_high = [], []
    for (grid, profile), g in cv.groupby(["grid", "profile"]):
        cells = sp[(sp.grid == grid) & (sp.profile == profile)].set_index("snr_db")
        low, high = g[g.snr_db == 5.0].set_index("taps"), g[g.snr_db == 25.0].set_index("taps")
        taps_low, taps_high = int(cells.loc[5.0].rule_taps), int(cells.loc[25.0].rule_taps)
        wide_for_low.append(low.loc[taps_high].error_db - low.loc[taps_high].matched_db)
        narrow_for_high.append(high.loc[taps_low].error_db - high.loc[taps_low].matched_db)
    out["HighSnrWidthAtLowSnrMin"] = fmt(min(wide_for_low))
    out["HighSnrWidthAtLowSnrMax"] = fmt(max(wide_for_low))
    out["LowSnrWidthAtHighSnrMin"] = fmt(min(narrow_for_high))
    out["LowSnrWidthAtHighSnrMax"] = fmt(max(narrow_for_high))
    for tag, profile in (("Auto", AUTO), ("Steam", STEAM)):
        band, plain = [], []
        for _, h in cv[cv.profile == profile].groupby(["grid", "snr_db"]):
            best = h.loc[h.error_db.idxmin(), "width_ns"]
            fall = h[(h.width_ns < best) & (h.width_ns >= 2 * h.rms_delay_spread_ns)]
            band.append((fall.error_db - fall.model_band_db).abs().max())
            plain.append((fall.error_db - fall.model_unbounded_db).abs().max())
        out[f"CurveBandDev{tag}"] = fmt(max(band))
        out[f"CurvePlainDev{tag}"] = fmt(max(plain))


def macros_from_choice(ch: pd.DataFrame, out: dict) -> None:
    """Span rule on the measured profiles with the delay spread reported for the site."""
    site = ch[ch.statistic == "reported for the site"]
    out["SurveyLossMaxWideMeas"] = fmt(site.rule_loss_db.max())
    out["SurveyLossMedianWideMeas"] = fmt(site.rule_loss_db.median())
    for tag, name in (("Steam", "steam plant"), ("Shop", "machine shop"), ("Auto", "automotive plant")):
        g = ch[ch.profile.str.contains(name)]
        out[f"SurveyLossMax{tag}"] = fmt(g[g.statistic == "reported for the site"].rule_loss_db.max())
        out[f"ProfileLossMax{tag}"] = fmt(g[g.statistic == "averaged profile"].rule_loss_db.max())
    rule_width = ch[(ch.statistic == "averaged profile")]
    beyond = rule_width[rule_width.rule_span_ns > rule_width.largest_delay_ns].groupby("profile").snr_db.min()
    out["RuleBeyondEndFromMin"] = fmt(beyond.min(), 0)
    out["RuleBeyondEndFromMax"] = fmt(beyond.max(), 0)


def macros_from_span(sp: pd.DataFrame, out: dict) -> None:
    sp = sp.copy()
    sp["rule_vs_best"] = sp.rule_loss_db - sp.best_loss_db
    wide = sp[sp.grid.isin(WIDE)]
    narrow = sp[sp.grid.isin(T2)]
    link = sp[sp.grid == LINK]
    expo = wide[~wide.measured]
    meas = wide[wide.measured]
    out["SpanCells"] = str(len(sp))
    out["SpanCellsDev"] = str(int(sp.development.sum()))
    out["SpanCellsEval"] = str(int((~sp.development).sum()))
    out["SpanCellsWideExp"] = str(len(expo))
    out["SpanCellsWideExpEval"] = str(int((~expo.development).sum()))
    out["SpanCellsWideMeas"] = str(len(meas))
    out["SpanCellsNarrow"] = str(len(narrow))
    out["SpanCellsLink"] = str(len(link))
    for tag, g in (("WideExp", expo), ("WideMeas", meas), ("Wide", wide), ("Narrow", narrow), ("Link", link)):
        for col, name in (("rule_loss_db", "Rule"), ("best_loss_db", "Best"), ("eight_loss_db", "Eight"), ("prefix_loss_db", "Prefix"), ("unbounded_loss_db", "Unbounded"), ("capped_loss_db", "Capped")):
            out[f"{name}LossMax{tag}"] = fmt(g[col].max())
            out[f"{name}LossMin{tag}"] = fmt(g[col].min())
            out[f"{name}LossMedian{tag}"] = fmt(g[col].median())
        out[f"RuleVsBestMax{tag}"] = fmt(g.rule_vs_best.max(), 2)
    out["RuleVsBestMaxWideExpEval"] = fmt(expo[~expo.development].rule_vs_best.max(), 2)
    out["RuleVsBestMaxWideExpDev"] = fmt(expo[expo.development].rule_vs_best.max(), 2)
    out["RuleLossMaxWideMeasLowSnr"] = fmt(meas[meas.snr_db <= 15.0].rule_loss_db.max())
    out["RuleLossMaxWideMeasHighSnr"] = fmt(meas[meas.snr_db > 15.0].rule_loss_db.max())
    out["CappedVsBestMaxWideMeas"] = fmt((meas.capped_loss_db - meas.best_loss_db).max(), 1)
    out["RuleTapsWithinOnePctWideExp"] = fmt(100 * ((expo.rule_taps - expo.best_taps).abs() <= 1).mean(), 0)
    below = pd.concat([expo.matched_db - expo.model_matched_db, expo.rule_error_db - expo.model_rule_db])
    out["ModelBelowMin"] = fmt(below.min())
    out["ModelBelowMax"] = fmt(below.max())
    loss_dev = ((expo.model_rule_db - expo.model_matched_db) - expo.rule_loss_db).abs()
    out["ModelLossDevMax"] = fmt(loss_dev.max(), 2)
    out["BandTimesSpreadMinWide"] = fmt(wide.band_times_spread.min(), 1)
    out["BandTimesSpreadMaxNarrow"] = fmt(narrow.band_times_spread.max(), 2)
    out["NarrowSafeguardPct"] = fmt(100 * ((narrow.unbounded_span_ns - narrow.rule_span_ns) < 1e9 / (narrow.band_times_spread / (narrow.rms_delay_spread_ns * 1e-9)) - 1e-6).mean(), 0)
    out["MeasuredEndMin"] = fmt((meas.capped_span_ns.where(meas.snr_db == 30.0) / meas.rms_delay_spread_ns).min(), 0)
    out["MeasuredEndMax"] = fmt((meas.capped_span_ns.where(meas.snr_db == 30.0) / meas.rms_delay_spread_ns).max(), 0)
    cells_per_ns = expo.band_times_spread / expo.rms_delay_spread_ns
    offset = (expo.unbounded_span_ns - expo.best_width_ns) * cells_per_ns
    out["BandOffsetMean"] = fmt(offset.mean(), 2)
    out["BandOffsetMin"] = fmt(offset.min())
    out["BandOffsetMax"] = fmt(offset.max())
    out["EightLossMaxWideMeasHighSnr"] = fmt(meas[meas.snr_db > 15.0].eight_loss_db.max())
    out["EightLossZeroDbMaxWideExp"] = fmt(expo[expo.snr_db == 0.0].eight_loss_db.max())
    out["LinkRuleWithinOneDbOfBestPct"] = fmt(100 * (link.rule_vs_best <= 1.0).mean(), 0)
    auto = sp[(sp.grid == LTE) & (sp.profile == AUTO)].set_index("snr_db")
    for snr, word in ((5.0, "Five"), (15.0, "Fifteen"), (25.0, "TwentyFive")):
        row = auto.loc[snr]
        out[f"RuleFactor{word}"] = fmt(row.rule_taps * row.sample_period_ns / row.rms_delay_spread_ns)
        out[f"BestFactor{word}"] = fmt(row.best_width_ns / row.rms_delay_spread_ns)
        out[f"EightLossAuto{word}"] = fmt(row.eight_loss_db)
        out[f"RuleLossAuto{word}"] = fmt(row.rule_loss_db)
    body = []
    groups = [("NR type 1, exponential", expo[expo.grid == NR1]), ("NR type 2, exponential", expo[expo.grid == NR2]), ("LTE CRS, exponential", expo[expo.grid == LTE]),
              ("NR and LTE, measured", meas), ("DVB-T2, four maps", narrow), ("64-subcarrier link", link)]
    for name, g in groups:
        cells = [f"{g[c].median():.1f} & {g[c].max():.1f}" for c in ("best_loss_db", "rule_loss_db", "eight_loss_db", "prefix_loss_db")]
        body.append(f"{name} & {len(g)} & " + " & ".join(cells) + " \\\\")
    header = " & & \\multicolumn{2}{c}{Best width} & \\multicolumn{2}{c}{Span rule} & \\multicolumn{2}{c}{Eight times} & \\multicolumn{2}{c}{Prefix} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\nMap and profiles & Cells & Med. & Max & Med. & Max & Med. & Max & Med. & Max \\\\"
    table("span", "Loss in dB of the uniform design relative to the matched design, as median and maximum over the factory profiles and seven SNRs, for four choices of the width of the design profile",
          "tab:span", "lr rr rr rr rr", header, body, wide=True, tabcolsep=5)


def macros_from_rate(rt: pd.DataFrame, out: dict) -> None:
    rt = rt.copy()
    rt["simple_db"] = 10 * np.log10(1 + 10 ** (rt.snr_db / 10) * 10 ** (rt.error_db / 10))
    rt["dev"] = (rt.simple_db - rt.snr_loss_db).abs()
    wide = rt[rt.grid.isin(WIDE)]
    auto = rt[(rt.grid == LTE) & (rt.profile == AUTO) & (rt.snr_db == 15.0)].set_index("estimator")
    names = {"Linear": LINEAR, "Matched": MATCHED, "Rule": RULE, "Nearest": NEAREST, "Spline": SPLINE, "Dft": DFT}
    for key, est in names.items():
        out[f"RateLoss{key}"] = fmt(auto.loc[est].snr_loss_db, 2 if key in ("Matched", "Rule") else 1)
    out["RateLossPrefix"] = fmt(auto[is_prefix(pd.Series(auto.index, index=auto.index))].snr_loss_db.iloc[0])
    out["RateBitsPerfect"] = fmt(auto.loc[MATCHED].perfect_rate_bits, 2)
    out["RateBitsMatched"] = fmt(auto.loc[MATCHED].rate_bits, 2)
    out["RateBitsLinear"] = fmt(auto.loc[LINEAR].rate_bits, 2)
    out["RateGainMatchedVsLinear"] = fmt(auto.loc[LINEAR].snr_loss_db - auto.loc[MATCHED].snr_loss_db)
    out["RateGainRuleVsLinear"] = fmt(auto.loc[LINEAR].snr_loss_db - auto.loc[RULE].snr_loss_db)
    for tag, grid in (("Lte", LTE), ("NrOne", NR1)):
        med = wide[wide.grid == grid].groupby(["estimator", "snr_db"]).snr_loss_db.median()
        worse = [snr for snr in sorted(wide.snr_db.unique()) if med[(DFT, snr)] > med[(LINEAR, snr)]]
        stays = [snr for snr in worse if all(med[(DFT, t)] > med[(LINEAR, t)] for t in sorted(wide.snr_db.unique()) if t >= snr)]
        out[f"TransformWorseFrom{tag}"] = fmt(min(stays), 0)
    table_cells = wide[wide.profile.isin([STEAM, SHOP, AUTO]) & wide.snr_db.isin([5.0, 15.0, 25.0])]
    cell = ["grid", "profile", "snr_db"]
    gain = table_cells[table_cells.estimator == LINEAR].set_index(cell).snr_loss_db - table_cells[table_cells.estimator == MATCHED].set_index(cell).snr_loss_db
    out["RateGainMin"] = fmt(gain.min())
    out["RateGainMax"] = fmt(gain.max())
    mid = wide[wide.snr_db >= 5.0]
    for key, est in (("Linear", LINEAR), ("Matched", MATCHED), ("Rule", RULE)):
        g = mid[mid.estimator == est]
        out[f"RateLoss{key}MinWide"] = fmt(g.snr_loss_db.min())
        out[f"RateLoss{key}MaxWide"] = fmt(g.snr_loss_db.max())
    g = mid[is_prefix(mid.estimator)]
    out["RateLossPrefixMinWide"] = fmt(g.snr_loss_db.min())
    out["RateLossPrefixMaxWide"] = fmt(g.snr_loss_db.max())
    out["RatePredDevWienerMax"] = fmt(mid[mid.estimator.isin([MATCHED, RULE])].dev.max(), 2)
    out["RatePredDevLinearMax"] = fmt(mid[mid.estimator == LINEAR].dev.max(), 1)
    out["RateRuleVsMatchedMaxWide"] = fmt((mid[mid.estimator == RULE].set_index(["grid", "profile", "snr_db"]).snr_loss_db - mid[mid.estimator == MATCHED].set_index(["grid", "profile", "snr_db"]).snr_loss_db).max(), 2)
    order = [NEAREST, LINEAR, SPLINE, "transform", "prefix", RULE, MATCHED]
    labels = {NEAREST: "Nearest pilot", LINEAR: "Linear", SPLINE: "Cubic spline", "transform": "Transform domain", "prefix": "Wiener, prefix", RULE: "Wiener, span rule", MATCHED: "Wiener, matched"}
    body = []
    for est in order:
        cells = []
        for grid in WIDE:
            for snr in (5.0, 15.0, 25.0):
                g = wide[(wide.grid == grid) & (wide.snr_db == snr)]
                if est == "prefix":
                    g = g[is_prefix(g.estimator)]
                elif est == "transform":
                    g = g[g.estimator.isin([DFT, "DFT least squares (ridge)"])]
                else:
                    g = g[g.estimator == est]
                cells.append(f"{g.snr_loss_db.median():.1f}")
        body.append(f"{labels[est]} & " + " & ".join(cells) + " \\\\")
    header = "Estimator & " + " & ".join("\\multicolumn{3}{c}{" + GSHORT[g] + "}" for g in WIDE) + " \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\nSNR (dB) & 5 & 15 & 25 & 5 & 15 & 25 & 5 & 15 & 25 \\\\"
    table("rate", "SNR loss in dB caused by the estimation error, relative to a perfectly known channel, as the median over the ten factory profiles. The transform-domain row is the ridge tap fit on the NR type 2 map",
          "tab:rate", "l rrr rrr rrr", header, body, wide=True, tabcolsep=5)


def macros_from_mismatch(m: pd.DataFrame, out: dict) -> None:
    sub = m[m.profile.str.startswith("exponential")]

    def worst(design):
        return float(sub[sub.design == design].loss_db.max())

    def median(design):
        return float(sub[sub.design == design].loss_db.median())

    out["MisHalfWorst"] = fmt(worst("exponential x0.5"))
    out["MisQuarterWorst"] = fmt(worst("exponential x0.25"))
    out["MisDoubleWorst"] = fmt(worst("exponential x2"))
    out["MisDoubleMedian"] = fmt(median("exponential x2"))
    out["MisHalfMedian"] = fmt(median("exponential x0.5"))
    wide = sub[sub.grid.isin(WIDE)]
    out["MisUniFourMedianMin"] = fmt(min(wide[(wide.grid == g) & (wide.design == "uniform x4")].loss_db.median() for g in WIDE), 0)
    out["MisUniFourMedianMax"] = fmt(max(wide[(wide.grid == g) & (wide.design == "uniform x4")].loss_db.median() for g in WIDE), 0)
    out["MisUniSixteenMedian"] = fmt(wide[wide.design == "uniform x16"].loss_db.median(), 0)
    designs = ["exponential x0.25", "exponential x0.5", "exponential x2", "exponential x4", "uniform x4", "uniform x8", "uniform x16", "uniform to CP"]
    labels = {"exponential x0.25": "Exponential, 0.25", "exponential x0.5": "Exponential, 0.5", "exponential x2": "Exponential, 2", "exponential x4": "Exponential, 4",
              "uniform x4": "Uniform, 4", "uniform x8": "Uniform, 8", "uniform x16": "Uniform, 16", "uniform to CP": "Uniform, prefix"}
    grids = WIDE + ["DVB-T2 8K, PP3 (spacing 6)"]
    gshort = dict(GSHORT)
    gshort["DVB-T2 8K, PP3 (spacing 6)"] = "DVB-T2 PP3"
    body = []
    for d in designs:
        cells = []
        for grid in grids:
            g = sub[(sub.grid == grid) & (sub.design == d)]
            cells.append(f"{g.loss_db.median():.1f} & {g.loss_db.max():.1f}")
        body.append(f"{labels[d]} & " + " & ".join(cells) + " \\\\")
    header = " & \\multicolumn{2}{c}{" + "} & \\multicolumn{2}{c}{".join(gshort[g] for g in grids) + "} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\nDesign & Med. & Max & Med. & Max & Med. & Max & Med. & Max \\\\"
    table("mismatch", "Loss in dB of a mismatched Wiener design relative to the matched design, as median and maximum over a sweep of exponential channels and the SNRs 5, 15 and 25 dB. The design profile is exponential with the actual delay spread times the factor shown, uniform over that multiple of the delay spread, or uniform over the cyclic prefix",
          "tab:mismatch", "l cc cc cc cc", header, body, wide=True, tabcolsep=6)


def macros_from_ber(b: pd.DataFrame, out: dict, prefix: str, column: str) -> None:
    lte = b[(b.grid == LTE) & (b.profile == AUTO)]

    def ber(est, snr):
        return float(lte[(lte.estimator == est) & (lte.snr_db == snr)][column].iloc[0])

    def snr_at(est, target=1e-2):
        g = lte[lte.estimator == est].sort_values("snr_db")
        x, y = g.snr_db.to_numpy(), np.log10(g[column].to_numpy())
        return float(np.interp(np.log10(target), y[::-1], x[::-1]))

    prefix_name = lte[is_prefix(lte.estimator)].estimator.iloc[0]
    out[f"{prefix}LinearTwenty"] = f"{ber(LINEAR, 20):.4f}"
    out[f"{prefix}MatchedTwenty"] = f"{ber(MATCHED, 20):.4f}"
    out[f"{prefix}RuleTwenty"] = f"{ber(RULE, 20):.4f}"
    out[f"{prefix}SnrLinearAtOnePct"] = fmt(snr_at(LINEAR))
    out[f"{prefix}SnrMatchedAtOnePct"] = fmt(snr_at(MATCHED))
    out[f"{prefix}SnrRuleAtOnePct"] = fmt(snr_at(RULE))
    out[f"{prefix}SnrPrefixAtOnePct"] = fmt(snr_at(prefix_name))
    out[f"{prefix}SnrGainMatchedVsLinear"] = fmt(snr_at(LINEAR) - snr_at(MATCHED))
    out[f"{prefix}SnrGainRuleVsLinear"] = fmt(snr_at(LINEAR) - snr_at(RULE))
    if prefix == "Ber":
        out["BerDftThirty"] = f"{ber(DFT, 30):.4f}"
        out["BerMatchedThirty"] = f"{ber(MATCHED, 30):.4f}"
        out["BerCells"] = str(len(b))
        out["BerInsidePct"] = fmt(100 * (abs(b.mc_error_db - b.exact_error_db) <= b.mc_half_width_db).mean(), 0)
        out["BerMaxDevDb"] = f"{(b.mc_error_db - b.exact_error_db).abs().max():.2f}"


def macros_from_per_tone(t: pd.DataFrame, out: dict) -> None:
    def db(est, tone):
        g = t[t.estimator == est].sort_values("subcarrier")
        return 10 * np.log10(float(g.error.iloc[tone]))

    def tones_to_interior(est):
        g = t[t.estimator == est].sort_values("subcarrier")
        e = 10 * np.log10(g.error.to_numpy())
        interior = np.median(e[300:900])
        return next(i for i in range(600) if abs(e[i] - interior) < 1.0)

    out["EdgeDftFirst"] = fmt(db(DFT, 0))
    out["EdgeDftTones"] = str(tones_to_interior(DFT))
    out["EdgeMatchedFirst"] = fmt(db(MATCHED, 0))
    out["EdgeMatchedCentre"] = fmt(db(MATCHED, 600))
    out["EdgeMatchedTones"] = str(tones_to_interior(MATCHED))
    out["EdgeRuleFirst"] = fmt(db(RULE, 0))
    out["EdgeRuleCentre"] = fmt(db(RULE, 600))
    out["EdgeLinearLast"] = fmt(db(LINEAR, -1))
    lin = t[t.estimator == LINEAR].sort_values("subcarrier").error.to_numpy()[300:900]
    out["EdgeLinearOscMin"] = fmt(10 * np.log10(lin.max()))
    out["EdgeLinearOscMax"] = fmt(10 * np.log10(lin.min()))


def macros_from_nist(out: dict) -> None:
    """Numbers from the NIST processing statistics."""
    folder = RESULTS / "nist_profiles"
    names = {"steam_plant": "Steam", "machine_shop": "Shop", "automotive_plant": "Auto"}
    reported = {"machine_shop": 64.9, "steam_plant": 35.4, "automotive_plant": 135.3}
    for key, short_name in names.items():
        st = pd.read_csv(folder / f"nist_profile_{key}_stats.csv").iloc[0]
        out[f"Nist{short_name}Records"] = f"{int(st.records_used):,}"
        out[f"Nist{short_name}RmsFifteen"] = fmt(st.rms_delay_spread_15db_grubbs_mean_ns)
        out[f"Nist{short_name}PeakToFloor"] = fmt(st.peak_to_floor_db_mean, 0)
        out[f"Nist{short_name}RmsProfile"] = fmt(st.average_profile_rms_delay_spread_ns, 0)
        out[f"Nist{short_name}ChipNs"] = fmt(1e3 / float(st.sample_rate_mhz) * float(st.samples_per_chip), 0)
        out[f"Nist{short_name}Reported"] = fmt(reported[key])


def table_measured_vs_exponential(r: pd.DataFrame, out: dict) -> None:
    """Error on the measured profiles against the exponential site profiles, NR type 1 and LTE, 15 dB."""
    sites = r[(r.profile_kind == "site") & (r.snr_db == 15.0)]
    pairs = [("NIST steam plant", "steam"), ("NIST machine shop", "shop"), ("NIST automotive plant", "automotive")]
    ests = [LINEAR, DFT, MATCHED, RULE]
    lines = []
    for prefix, label in pairs:
        for grid, gname in ((NR1, "NR type 1"), (LTE, "LTE CRS")):
            g = sites[sites.grid == grid]
            measured = g[g.profile.str.startswith(prefix + ", measured")]
            expo = g[g.profile.str.startswith(prefix + " (")]
            cells = []
            for est in ests:
                cells.append(f"{measured[measured.estimator == est].error_db.iloc[0]:.1f} & {expo[expo.estimator == est].error_db.iloc[0]:.1f}")
            lines.append(f"{label.capitalize()} ({measured.rms_delay_spread_ns.iloc[0]:.0f} / {expo.rms_delay_spread_ns.iloc[0]:.0f} ns) & {gname} & " + " & ".join(cells) + " \\\\")
    groups = {"Interp": [NEAREST, LINEAR, SPLINE], "Wiener": [MATCHED], "Dft": [DFT], "Rule": [RULE]}
    for gname, members in groups.items():
        vals = []
        for prefix, _ in pairs:
            for grid in WIDE:
                g = sites[sites.grid == grid]
                measured = g[g.profile.str.startswith(prefix + ", measured")]
                expo = g[g.profile.str.startswith(prefix + " (")]
                for est in members:
                    a, b = measured[measured.estimator == est], expo[expo.estimator == est]
                    if not a.empty and not b.empty:
                        vals.append(abs(float(a.error_db.iloc[0]) - float(b.error_db.iloc[0])))
                if gname == "Wiener":
                    a, b = measured[is_prefix(measured.estimator)], expo[is_prefix(expo.estimator)]
                    vals.append(abs(float(a.error_db.iloc[0]) - float(b.error_db.iloc[0])))
        out[f"MeasVsExp{gname}Max"] = fmt(max(vals), 2 if gname == "Interp" else 1)
    header = " & & \\multicolumn{2}{c}{Linear} & \\multicolumn{2}{c}{Transform} & \\multicolumn{2}{c}{Wiener, matched} & \\multicolumn{2}{c}{Wiener, span rule} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\nSite & Map & Meas. & Exp. & Meas. & Exp. & Meas. & Exp. & Meas. & Exp. \\\\"
    table("measured", "Error in dB at 15 dB SNR on the measured factory profiles (Meas.) and on the exponential profiles with the delay spreads reported by NIST for the sites (Exp.). The RMS delay spreads of the two profiles follow the site name",
          "tab:measured", "ll rr rr rr rr", header, lines, wide=True, tabcolsep=3.5)


def macros_from_cost(c: pd.DataFrame, out: dict) -> None:
    lte = c[c.grid == LTE].set_index("estimator")
    out["CostLinearMults"] = f"{int(lte.loc[LINEAR].real_multiplications):,}"
    out["CostWienerMults"] = f"{int(lte.loc[MATCHED].real_multiplications):,}"
    out["CostDftMults"] = f"{int(lte.loc[DFT].real_multiplications):,}"
    out["CostRatioWienerLinear"] = f"{lte.loc[MATCHED].real_multiplications / lte.loc[LINEAR].real_multiplications:.0f}"
    out["CostDenseMinUs"] = fmt(lte.dense_apply_us_per_symbol.min(), 0)
    out["CostDenseMaxUs"] = fmt(lte.dense_apply_us_per_symbol.max(), 0)
    wiener = lte[lte.index.str.startswith("Wiener")]
    out["CostWienerDesignMinMs"] = fmt(wiener.design_ms.min(), 0)
    out["CostWienerDesignMaxMs"] = fmt(wiener.design_ms.max(), 0)
    body = []
    for grid, gname in ((NR1, "NR type 1"), (LTE, "LTE CRS")):
        for _, row in c[c.grid == grid].iterrows():
            s = "n/a" if np.isnan(row.structured_apply_us_per_symbol) else f"{row.structured_apply_us_per_symbol:.0f}"
            body.append(f"{gname} & {short(row.estimator)} & {int(row.real_multiplications):,} & {row.design_ms:.1f} & {row.dense_apply_us_per_symbol:.0f} & {s} \\\\")
    table("cost", "Cost per OFDM symbol as real multiplications of a straightforward implementation (Mult.), design time of the matrix, and measured application time per symbol as a stored matrix (Dense) and as a structured implementation where one exists (Struct.)",
          "tab:cost", "llrrrr", "Map & Estimator & Mult. & Design & Dense & Struct. \\\\\n & & & (ms) & (\\si{\\micro\\second}) & (\\si{\\micro\\second}) \\\\", body, tabcolsep=1.6, placement="b")


def setup_table() -> None:
    body = [
        "NR, DM-RS type 1 & 30 & 612 & 306 & 60 & 18.4 & 2.34 \\\\",
        "NR, DM-RS type 2 & 30 & 612 & 204 & 90 & 18.4 & 2.34 \\\\",
        "LTE, CRS port 0 & 15 & 1200 & 200 & 90 & 18.0 & 4.69 \\\\",
        "DVB-T2 PP1, GI 1/4 & 1.116 & 1024 & 342 & 3.3 & 1.14 & 224 \\\\",
        "DVB-T2 PP3, GI 1/8 & 1.116 & 1024 & 171 & 6.7 & 1.14 & 112 \\\\",
        "DVB-T2 PP5, GI 1/16 & 1.116 & 1024 & 86 & 13.3 & 1.14 & 56 \\\\",
        "DVB-T2 PP7, GI 1/128 & 1.116 & 1024 & 43 & 26.6 & 1.14 & 7 \\\\",
        "64-subcarrier link & 312.5 & 64 & 8 & 2500 & 20.0 & 0.8 \\\\",
    ]
    table("setup", "Grids and pilot maps of the study, with DVB-T2 in 8K mode, the subcarrier spacing, the active subcarriers analysed, the pilots, the mean pilot spacing $F$, the band $B$, and the cyclic prefix (CP) or guard interval (GI)",
          "tab:setup", "lrrrrrr", "Grid and map & $\\Delta f$ (kHz) & $N$ & $K$ & $F$ (kHz) & $B$ (MHz) & CP (\\si{\\micro\\second}) \\\\", body, wide=True, tabcolsep=7)


def macros_from_app_check(out: dict) -> None:
    """Agreement of the browser application with the package, from results/app_check.json."""
    path = RESULTS / "app_check.json"
    if not path.exists():
        return
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["differences_beyond_tolerance"]:
        raise SystemExit("the check of the browser application reports differences")
    out["AppCheckCases"] = str(record["cases"])
    out["AppCheckNumbers"] = str(record["numbers_compared"])
    exponent = int(np.ceil(np.log10(record["largest_absolute_difference"])))
    out["AppCheckBound"] = f"$10^{{{{-}}{abs(exponent)}}}$" if exponent < 0 else f"$10^{{{exponent}}}$"


def main() -> None:
    out = {}
    test_count = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", str(ROOT / "ofdmest" / "tests")], capture_output=True, text=True, cwd=ROOT / "ofdmest")
    out["TestCount"] = str(sum(1 for line in test_count.stdout.splitlines() if "::" in line))
    macros_from_validation(pd.read_csv(RESULTS / "validation.csv"), out)
    regime = pd.read_csv(RESULTS / "regime.csv")
    macros_from_regime(regime, out)
    macros_from_nist(out)
    table_measured_vs_exponential(regime, out)
    macros_from_span(pd.read_csv(RESULTS / "span.csv"), out)
    macros_from_curves(pd.read_csv(RESULTS / "span_curves.csv"), pd.read_csv(RESULTS / "span.csv"), out)
    macros_from_choice(pd.read_csv(RESULTS / "spread_choice.csv"), out)
    macros_from_rate(pd.read_csv(RESULTS / "rate.csv"), out)
    macros_from_mismatch(pd.read_csv(RESULTS / "mismatch.csv"), out)
    macros_from_ber(pd.read_csv(RESULTS / "ber.csv"), out, "Ber", "qpsk_ber")
    macros_from_ber(pd.read_csv(RESULTS / "ber_16qam.csv"), out, "BerSixteen", "ber")
    macros_from_per_tone(pd.read_csv(RESULTS / "per_tone.csv"), out)
    macros_from_cost(pd.read_csv(RESULTS / "cost.csv"), out)
    macros_from_app_check(out)
    setup_table()
    lines = [f"\\newcommand{{\\{k}}}{{{minus(v)}}}" for k, v in sorted(out.items())]
    (TABLES / "numbers.tex").write_text("% Generated by experiments/make_tables.py from the result files. Do not edit.\n" + "\n".join(lines) + "\n")
    if PAPER.exists():
        text = "".join(t.read_text(encoding="utf-8") for t in PAPER.glob("*.tex"))
        used = set(re.findall(r"\\([A-Z][A-Za-z]+)\{\}", text))
        missing = sorted(used - set(out))
        if missing:
            raise SystemExit(f"macros used in the paper but not generated: {missing}")
        unused = sorted(set(out) - used)
        print(f"{len(used)} macros used in the paper, {len(unused)} generated but not used")
    print(f"wrote {TABLES / 'numbers.tex'} with {len(out)} macros and {len(list(TABLES.glob('*.tex'))) - 1} tables")


if __name__ == "__main__":
    main()
