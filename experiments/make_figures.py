"""Figures of the study.

The data plots are produced from the result files only. The two method figures, the flow
of the study and the trade-off behind the span rule, are drawn from fixed parameters and
the closed forms of the package.

Every figure is a vector PDF sized to one text column (3.45 in wide) or to the page width,
with serif fonts of 7 pt, thin lines and a colour per estimator that is the same in every
figure. A PNG copy of each figure is written for inspection.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ofdmest import comb_error_matched, comb_error_uniform_design

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGURES = HERE.parent / "figures"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"], "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "lines.linewidth": 0.9, "lines.markersize": 3, "axes.linewidth": 0.5, "grid.linewidth": 0.4, "grid.linestyle": ":",
    "legend.frameon": False, "pdf.fonttype": 42, "figure.dpi": 150,
})
COLUMN = 3.45
DOUBLE = 7.1

STYLE = {
    "nearest pilot": ("#7f7f7f", "o", ":"),
    "linear (constant edges)": ("#1f77b4", "s", "-"),
    "cubic spline (constant edges)": ("#17becf", "^", "-"),
    "DFT interpolation": ("#2ca02c", "v", "--"),
    "DFT least squares (ridge)": ("#2ca02c", "v", "--"),
    "Wiener (uniform to 2344 ns)": ("#ff7f0e", "D", "-."),
    "Wiener (uniform to 4688 ns)": ("#ff7f0e", "D", "-."),
    "Wiener (uniform to 112000 ns)": ("#ff7f0e", "D", "-."),
    "Wiener (matched)": ("#d62728", "x", "-"),
    "Wiener (uniform, span rule)": ("#9467bd", "P", "-"),
}
LABEL = {
    "nearest pilot": "nearest pilot", "linear (constant edges)": "linear", "cubic spline (constant edges)": "cubic spline",
    "DFT interpolation": "transform domain", "DFT least squares (ridge)": "tap fit, ridge", "Wiener (matched)": "Wiener, matched",
    "Wiener (uniform, span rule)": "Wiener, span rule",
}


def label_of(name: str) -> str:
    if name.startswith("Wiener (uniform to"):
        return "Wiener, prefix-wide"
    return LABEL.get(name, name)


def style_of(name: str):
    return STYLE.get(name, ("#000000", ".", "-"))


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", dpi=600, metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{name}.png", dpi=220, metadata={"Software": None})
    plt.close(fig)


def fig_regime(df: pd.DataFrame) -> None:
    """Error versus delay spread at 15 dB for four grids (two by two panels)."""
    panels = ["NR 20 MHz, DM-RS type 1", "NR 20 MHz, DM-RS type 2", "LTE 20 MHz, CRS", "DVB-T2 8K, PP3 (spacing 6)"]
    titles = ["NR DM-RS type 1 (spacing 2)", "NR DM-RS type 2 (pairs)", "LTE CRS (spacing 6)", "DVB-T2 PP3 (spacing 6)"]
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE, 1.9), sharey=False)
    for ax, grid, title in zip(axes.ravel(), panels, titles):
        sub = df[(df.grid == grid) & (df.profile_kind == "sweep") & (df.snr_db == 15.0)]
        for est in sub.estimator.unique():
            s = sub[sub.estimator == est].sort_values("rms_delay_spread_ns")
            c, m, ls = style_of(est)
            ax.semilogx(s.rms_delay_spread_ns, s.error_db, color=c, marker=m, linestyle=ls, label=label_of(est), markevery=2)
        ax.set_title(title, pad=2)
        ax.grid(True, which="both")
        ax.set_xlabel("RMS delay spread (ns)")
        ax.set_ylabel("Error (dB)")
    for ax in axes[1:]:
        ax.set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, columnspacing=1.0, handlelength=1.6, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=0.8)
    save(fig, "regime_delay_spread")


def fig_span(curves: pd.DataFrame, span: pd.DataFrame) -> None:
    """Error against the width of the uniform design, and loss of three ways of choosing it."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.0))
    ax = axes[0]
    sub = curves[(curves.grid == "LTE 20 MHz, CRS") & (curves.profile == "NIST automotive plant (132 ns)")]
    cells = span[(span.grid == "LTE 20 MHz, CRS") & (span.profile == "NIST automotive plant (132 ns)")].set_index("snr_db")
    for snr, c, m in zip((5.0, 15.0, 25.0), ("#1f77b4", "#2ca02c", "#d62728"), ("o", "s", "^")):
        g = sub[sub.snr_db == snr].sort_values("taps")
        x = g.width_ns / g.rms_delay_spread_ns
        ax.plot(x, g.error_db, color=c, marker=m, linestyle="none", markevery=2, label=f"exact, {snr:.0f} dB")
        ax.plot(x, g.model_band_db, color=c, linestyle="-", linewidth=0.8)
        ax.plot(x, g.model_unbounded_db, color=c, linestyle=":", linewidth=0.8)
        row = cells.loc[snr]
        ax.axvline(row.rule_taps * row.sample_period_ns / row.rms_delay_spread_ns, color=c, linestyle="--", linewidth=0.5)
    ax.plot([], [], color="k", linestyle="-", linewidth=0.8, label="Equation (8) at $W + 1/B$")
    ax.plot([], [], color="k", linestyle=":", linewidth=0.8, label="Equation (8) at $W$")
    ax.plot([], [], color="k", linestyle="--", linewidth=0.5, label="width by the span rule")
    ax.set_xlim(0, 16)
    ax.set_ylim(-36, 7)
    ax.set_xlabel("Width $W$ of the uniform design / RMS delay spread")
    ax.set_ylabel("Error (dB)")
    ax.set_title("LTE CRS, automotive plant (132 ns)", pad=2)
    ax.grid(True)
    ax.legend(ncol=2, loc="upper right", handlelength=1.6, columnspacing=0.8, labelspacing=0.25)
    ax = axes[1]
    wide = span[span.grid.isin(["NR 20 MHz, DM-RS type 1", "NR 20 MHz, DM-RS type 2", "LTE 20 MHz, CRS"])]
    series = (("best_loss_db", "best width (search)", "#7f7f7f", "o"), ("rule_loss_db", "span rule", "#9467bd", "P"),
              ("eight_loss_db", "eight times the delay spread", "#17becf", "^"), ("prefix_loss_db", "cyclic prefix", "#ff7f0e", "D"))
    for column, label, c, m in series:
        g = wide[~wide.measured].groupby("snr_db")[column].max()
        ax.plot(g.index, g.values, color=c, marker=m, linestyle="-", label=label)
        g = wide[wide.measured].groupby("snr_db")[column].max()
        ax.plot(g.index, g.values, color=c, marker=m, linestyle="--", markerfacecolor="none")
    ax.plot([], [], color="k", linestyle="-", label="exponential profiles")
    ax.plot([], [], color="k", linestyle="--", label="measured profiles")
    ax.set_yscale("log")
    ax.set_ylim(0.15, 110)
    ax.set_yticks([0.2, 0.5, 1, 2, 5, 10])
    ax.set_yticklabels(["0.2", "0.5", "1", "2", "5", "10"])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Largest loss versus matched (dB)")
    ax.set_title("NR and LTE maps, ten factory profiles", pad=2)
    ax.grid(True, which="both")
    ax.legend(ncol=3, loc="upper center", handlelength=1.8, columnspacing=0.8, labelspacing=0.25)
    fig.tight_layout(w_pad=1.2)
    save(fig, "design_span")


def fig_rate(df: pd.DataFrame) -> None:
    """SNR that the estimation error costs, LTE grid, automotive-plant profile."""
    sub = df[(df.grid == "LTE 20 MHz, CRS") & (df.profile == "NIST automotive plant (132 ns)")]
    fig, ax = plt.subplots(figsize=(COLUMN, 1.9))
    for est in sub.estimator.unique():
        s = sub[sub.estimator == est].sort_values("snr_db")
        c, m, ls = style_of(est)
        ax.plot(s.snr_db, s.snr_loss_db, color=c, marker=m, linestyle=ls, label=label_of(est))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("SNR loss (dB)")
    ax.set_ylim(0, 10.5)
    ax.grid(True)
    ax.legend(ncol=2, loc="upper left", handlelength=1.8, columnspacing=0.8, labelspacing=0.25)
    fig.tight_layout()
    save(fig, "snr_loss_lte_automotive")


def fig_mismatch(df: pd.DataFrame) -> None:
    """Loss of a mismatched Wiener design relative to the matched design, LTE grid, 15 dB."""
    sub = df[(df.grid == "LTE 20 MHz, CRS") & (df.snr_db == 15.0) & df.profile.str.startswith("exponential")]
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 2.2), sharey=True)
    exp_designs = ["exponential x0.25", "exponential x0.5", "exponential x2", "exponential x4", "exponential x8"]
    uni_designs = ["uniform x2", "uniform x4", "uniform x8", "uniform x16", "uniform to CP"]
    markers = ["o", "s", "^", "v", "D"]
    labels = {"exponential x0.25": "exp., 0.25 x", "exponential x0.5": "exp., 0.5 x", "exponential x2": "exp., 2 x", "exponential x4": "exp., 4 x", "exponential x8": "exp., 8 x",
              "uniform x2": "unif., 2 x", "uniform x4": "unif., 4 x", "uniform x8": "unif., 8 x", "uniform x16": "unif., 16 x", "uniform to CP": "unif., prefix"}
    for ax, designs, title in zip(axes, (exp_designs, uni_designs), ("Exponential design", "Uniform design")):
        for d, mk in zip(designs, markers):
            s = sub[sub.design == d].sort_values("rms_delay_spread_ns")
            ax.semilogx(s.rms_delay_spread_ns, s.loss_db, marker=mk, label=labels[d])
        ax.set_title(title, pad=2)
        ax.set_xlabel("Actual RMS delay spread (ns)")
        ax.grid(True, which="both")
    axes[0].set_ylabel("Loss versus matched (dB)")
    handles = [h for ax in axes for h in ax.get_legend_handles_labels()[0]]
    names = [l for ax in axes for l in ax.get_legend_handles_labels()[1]]
    fig.legend(handles, names, loc="lower center", ncol=5, handlelength=1.2, columnspacing=0.8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=0.6)
    save(fig, "wiener_mismatch")


def fig_ber(df: pd.DataFrame) -> None:
    """QPSK bit error rate versus SNR on the LTE grid, automotive-plant profile."""
    sub = df[(df.grid == "LTE 20 MHz, CRS") & (df.profile == "NIST automotive plant (132 ns)")]
    fig, ax = plt.subplots(figsize=(COLUMN, 2.2))
    for est in sub.estimator.unique():
        s = sub[sub.estimator == est].sort_values("snr_db")
        c, m, ls = style_of(est)
        ax.semilogy(s.snr_db, s.qpsk_ber, color=c, marker=m, linestyle=ls, label=label_of(est))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("QPSK bit error rate")
    ax.grid(True, which="both")
    ax.legend(ncol=1, loc="lower left", handlelength=1.8, labelspacing=0.2)
    fig.tight_layout()
    save(fig, "ber_lte_automotive")


def fig_per_tone(df: pd.DataFrame) -> None:
    """Per-subcarrier error along the LTE band, automotive-plant profile, 15 dB."""
    fig, ax = plt.subplots(figsize=(COLUMN, 2.35))
    for est in df.estimator.unique():
        s = df[df.estimator == est].sort_values("subcarrier")
        c, m, ls = style_of(est)
        ax.plot(s.subcarrier, 10 * np.log10(s.error), color=c, linestyle=ls, label=label_of(est), linewidth=0.8)
    ax.set_xlabel("Subcarrier index")
    ax.set_ylabel("Error (dB)")
    ax.set_xlim(0, 1199)
    ax.grid(True)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.97, bottom=0.44)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.30), columnspacing=1.0, handlelength=1.8, labelspacing=0.25)
    save(fig, "per_tone_lte_automotive")


def fig_profiles() -> None:
    """Measured average power-delay profiles of the three NIST runs."""
    folder = RESULTS / "nist_profiles"
    files = [("steam_plant", "steam plant"), ("machine_shop", "machine shop"), ("automotive_plant", "automotive plant")]
    if not all((folder / f"nist_profile_{k}.csv").exists() for k, _ in files):
        return
    fig, ax = plt.subplots(figsize=(COLUMN, 1.8))
    for (key, label), c, m in zip(files, ["#1f77b4", "#2ca02c", "#d62728"], ["o", "s", "^"]):
        t = pd.read_csv(folder / f"nist_profile_{key}.csv")
        ax.plot(t.delay_s * 1e9, 10 * np.log10(t.power / t.power.max()), color=c, marker=m, markersize=2, linewidth=0.8, label=label)
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Power relative to peak (dB)")
    ax.set_ylim(-32, 2)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    save(fig, "measured_profiles")


def fig_concept() -> None:
    """Profile, noise level and design window (a); the two terms of the error and their sum (b).

    Drawn from the closed forms for the pilot spacing of the LTE map (90 kHz), a delay spread
    of 132 ns and an SNR of 15 dB. Delays and widths are in units of the delay spread.
    """
    spacing, decay, snr_db = 90e3, 132e-9, 15.0
    beta = spacing * decay / 10.0 ** (snr_db / 10.0)
    best = float(np.log(1.0 / beta))
    w = np.linspace(0.0, 14.0, 561)
    violet = "#9467bd"
    fig, (a, b) = plt.subplots(1, 2, figsize=(COLUMN, 1.8))
    a.axvspan(0.0, best, color=violet, alpha=0.16, linewidth=0)
    a.semilogy(w, np.exp(-w), color="#1f77b4")
    a.axhline(beta, color="#7f7f7f", linestyle="--")
    a.axvline(best, color=violet, linewidth=0.7)
    a.text(0.5 * best, 3e-6, "design\nwindow", ha="center", va="center", fontsize=6)
    a.text(3.4, 0.22, "channel\nprofile", ha="left", va="center", fontsize=6)
    a.text(13.8, beta * 1.9, "noise level", ha="right", va="bottom", fontsize=6)
    a.text(best + 0.35, 0.25, r"$W^{\star}$", ha="left", va="center")
    a.set_xlim(0, 14)
    a.set_ylim(1e-6, 2)
    a.set_xlabel("Delay / delay spread")
    a.set_ylabel("Normalized power density")
    a.text(0.96, 0.95, "(a)", transform=a.transAxes, ha="right", va="top")

    total = np.array([comb_error_uniform_design(v * decay, decay, spacing, snr_db) for v in w])
    db = lambda v: 10.0 * np.log10(np.maximum(v, 1e-12))
    b.plot(w, db(w * beta), color="#7f7f7f", linestyle="--", label="noise admitted")
    b.plot(w, db(np.exp(-w)), color="#1f77b4", linestyle="-.", label="channel left out")
    b.plot(w, db(total), color=violet, label="uniform design")
    b.axhline(db(comb_error_matched(decay, spacing, snr_db)), color="#d62728", linestyle=":", label="matched design")
    b.axvline(best, color=violet, linewidth=0.7)
    b.text(best + 0.35, -3.0, r"$W^{\star}$", ha="left", va="center")
    b.set_xlim(0, 14)
    b.set_ylim(-36, 0)
    b.set_xlabel("Width / delay spread")
    b.set_ylabel("Error (dB)")
    b.grid(True)
    b.text(0.96, 0.95, "(b)", transform=b.transAxes, ha="right", va="top")
    handles, labels = b.get_legend_handles_labels()
    fig.subplots_adjust(left=0.135, right=0.985, top=0.97, bottom=0.36, wspace=0.50)
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.0), columnspacing=0.9, handlelength=1.6, handletextpad=0.5)
    save(fig, "design_concept")


SHAPE_COLOURS = {
    "process": ("#B8E3DA", "#5FB3A3", "#3F8F80"),
    "alternative": ("#D6D3F2", "#8F89DD", "#6C66B8"),
    "secondary": ("#C6E0FF", "#66AAFF", "#4A86D6"),
    "data": ("#FFF2B3", "#FFD54F", "#D6B03A"),
    "document": ("#E6D5EE", "#A780BF", "#8A63A3"),
}


def _shape_vertices(kind: str, x: float, y: float, w: float, h: float) -> np.ndarray:
    """Outline of a flowchart shape with lower left corner (x, y), width w and height h."""
    if kind == "data":
        s = 0.10
        return np.array([(x + s, y + h), (x + w, y + h), (x + w - s, y), (x, y)])
    if kind == "alternative":
        s = 0.11
        return np.array([(x + s, y + h), (x + w - s, y + h), (x + w, y + h / 2), (x + w - s, y), (x + s, y), (x, y + h / 2)])
    if kind == "secondary":
        r = h / 2
        t = np.linspace(-np.pi / 2, np.pi / 2, 25)
        right = np.column_stack([x + w - r + r * np.cos(t), y + r + r * np.sin(t)])
        left = np.column_stack([x + r - r * np.cos(t), y + r - r * np.sin(t)])
        return np.vstack([right, left])
    if kind == "document":
        t = np.linspace(0.0, 1.0, 41)
        wave = np.column_stack([x + w * (1 - t), y + 0.035 * np.sin(2 * np.pi * t)])
        return np.vstack([[(x, y + h), (x + w, y + h)], wave])
    return np.array([(x, y + h), (x + w, y + h), (x + w, y), (x, y)])


def _shape(ax, kind: str, cx: float, cy: float, w: float, h: float, text: str) -> None:
    """Draw one flowchart shape centred on (cx, cy) with a vertical gradient and its label."""
    top, bottom, edge = SHAPE_COLOURS[kind]
    x, y = cx - w / 2, cy - h / 2
    outline = plt.Polygon(_shape_vertices(kind, x, y, w, h), closed=True, facecolor="none", edgecolor=edge, linewidth=0.6, zorder=3)
    ramp = np.linspace(0.0, 1.0, 64)[:, None, None]
    colours = (1 - ramp) * np.array(matplotlib.colors.to_rgb(top)) + ramp * np.array(matplotlib.colors.to_rgb(bottom))
    image = ax.imshow(colours, extent=(x - 0.01, x + w + 0.01, y - 0.05, y + h + 0.01), origin="upper", aspect="auto", interpolation="bicubic", zorder=2)
    ax.add_patch(outline)
    image.set_clip_path(outline)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=6.5, color="black", linespacing=1.15, zorder=4)


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], head: bool = True) -> None:
    style = "-|>,head_length=0.45,head_width=0.2" if head else "-"
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle=style, color="#222222", linewidth=0.6, shrinkA=0, shrinkB=0), zorder=1)


def fig_overview() -> None:
    """Flow of the study from the inputs through the estimators to the exact error and the link-level measure.

    Parallelograms are inputs, wavy-bottom shapes are files, the rectangle with the span rule
    and the two rectangles of the analysis are the steps of this study, hexagons are the
    estimators it is compared with, and the rounded shape is the check by simulation.
    """
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.52))
    ax.set_axis_off()
    rows = [2.18, 1.60, 1.02, 0.44]
    h = 0.44
    _shape(ax, "data", 0.86, rows[0], 1.62, h, "Pilot map\nNR, LTE, DVB-T2, 64 subcarriers")
    _shape(ax, "data", 0.86, rows[1], 1.62, h, "SNR\n0 to 30 dB")
    _shape(ax, "data", 0.86, rows[2], 1.62, h, "Delay profile\nexponential or measured")
    _shape(ax, "document", 0.86, rows[3], 1.50, h - 0.06, "NIST factory\nmeasurements")
    _arrow(ax, (0.86, rows[3] + (h - 0.06) / 2), (0.86, rows[2] - h / 2))

    designs = [
        ("alternative", "Interpolators and\ntransform-domain estimator"),
        ("alternative", "Wiener filter,\nprefix-wide profile"),
        ("process", "Wiener filter, width from the\nspan rule (delay spread, SNR)"),
        ("alternative", "Wiener filter, matched\nto the actual profile"),
    ]
    left_bus, right_bus = 1.86, 3.84
    for cy in rows[:3]:
        _arrow(ax, (1.62, cy), (left_bus, cy), head=False)
    _arrow(ax, (left_bus, rows[2]), (left_bus, rows[0]), head=False)
    _arrow(ax, (left_bus, rows[2]), (left_bus, rows[3]), head=False)
    for cy, (kind, text) in zip(rows, designs):
        _shape(ax, kind, 2.85, cy, 1.70, h, text)
        _arrow(ax, (left_bus, cy), (2.0, cy))
        _arrow(ax, (3.70, cy), (right_bus, cy), head=False)
    _arrow(ax, (right_bus, rows[0]), (right_bus, rows[3]), head=False)
    _arrow(ax, (right_bus, rows[1]), (4.02, rows[1]))

    _shape(ax, "process", 4.66, rows[1], 1.28, 0.62, "Exact error\nper subcarrier")
    _shape(ax, "process", 6.32, rows[1], 1.42, 0.62, "Link-level measure\neffective SNR, rate,\nSNR loss")
    _shape(ax, "secondary", 4.66, rows[3], 1.28, h, "Monte Carlo check,\nbit error rate")
    _shape(ax, "document", 6.32, rows[3], 1.42, h - 0.06, "Result files,\ntables and figures")
    top = rows[0] + 0.31
    _arrow(ax, (right_bus, rows[0]), (right_bus, top), head=False)
    _arrow(ax, (right_bus, top), (6.32, top), head=False)
    _arrow(ax, (6.32, top), (6.32, rows[1] + 0.31))
    _arrow(ax, (4.66, rows[1] - 0.31), (4.66, rows[3] + h / 2))
    _arrow(ax, (6.32, rows[1] - 0.31), (6.32, rows[3] + (h - 0.06) / 2))
    _arrow(ax, (5.30, rows[3]), (5.61, rows[3]))
    ax.set_xlim(0.0, DOUBLE)
    ax.set_ylim(0.06, 2.58)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save(fig, "study_overview")


def main() -> None:
    fig_overview()
    fig_concept()
    regime = pd.read_csv(RESULTS / "regime.csv")
    fig_regime(regime)
    if (RESULTS / "span.csv").exists():
        fig_span(pd.read_csv(RESULTS / "span_curves.csv"), pd.read_csv(RESULTS / "span.csv"))
    if (RESULTS / "rate.csv").exists():
        fig_rate(pd.read_csv(RESULTS / "rate.csv"))
    if (RESULTS / "per_tone.csv").exists():
        fig_per_tone(pd.read_csv(RESULTS / "per_tone.csv"))
    fig_profiles()
    if (RESULTS / "mismatch.csv").exists():
        fig_mismatch(pd.read_csv(RESULTS / "mismatch.csv"))
    if (RESULTS / "ber.csv").exists():
        fig_ber(pd.read_csv(RESULTS / "ber.csv"))
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
