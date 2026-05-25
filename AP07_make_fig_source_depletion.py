#!/usr/bin/env python3
"""
AP07_make_fig_source_depletion.py

Plot-only script for the corrected local-BL-perpendicular AP07 result.
It reads the finished AP07 CSV output and overwrites the Fig. 4 PDF.

It does not rerun the sampling or transition calculation.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


RUN_LABEL = "AP07_320x20_localPerp_meanTheta_n30000"
PRIMARY_LAG = 1
FIT_MIN = 32
FIT_MAX = 160

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "04_outputs" / "AP07_320_timewindow_source_depletion" / RUN_LABEL
FIGDIR = ROOT / "05_figures" / "AP07_320_timewindow_source_depletion"
FIGDIR.mkdir(parents=True, exist_ok=True)

INFILE = OUTDIR / "AP07_reconstruction_cube_mean_sem.csv"
OUTPDF = FIGDIR / f"{RUN_LABEL}_S2_lag{PRIMARY_LAG}.pdf"
OUTCSV = OUTDIR / "AP07_fit_slopes_clean_plot.csv"


def fit_h(r, y):
    r = np.asarray(r, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (r >= FIT_MIN) & (r <= FIT_MAX) & np.isfinite(y) & (y > 0)
    if mask.sum() < 3:
        return np.nan
    h, _ = np.polyfit(np.log(r[mask]), np.log(y[mask]), 1)
    return float(h)


def anchored_guide(r, y, h):
    r = np.asarray(r, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (r >= FIT_MIN) & (r <= FIT_MAX) & np.isfinite(y) & (y > 0)
    rr_fit = r[mask]
    yy_fit = y[mask]

    order = np.argsort(rr_fit)
    rr_fit = rr_fit[order]
    yy_fit = yy_fit[order]

    r0 = math.sqrt(FIT_MIN * FIT_MAX)
    y0 = float(np.exp(np.interp(np.log(r0), np.log(rr_fit), np.log(yy_fit))))

    rr = np.array([FIT_MIN, FIT_MAX], dtype=float)
    yy = y0 * (rr / r0) ** h
    return rr, yy


def main():
    if not INFILE.exists():
        raise FileNotFoundError(f"Missing input CSV: {INFILE}")

    df = pd.read_csv(INFILE)
    df = df[df["lag"] == PRIMARY_LAG].copy()

    panels = [
        ("+", r"$(S_2^+)^{1/2}$"),
        ("-", r"$(S_2^-)^{1/2}$"),
        ("+-", r"$(S_2^{+-})^{1/2}$"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))
    slope_rows = []

    for ax, (alpha, ylabel) in zip(axes, panels):
        g = df[df["alpha"] == alpha].sort_values("r").copy()
        if g.empty:
            raise RuntimeError(f"No rows for alpha={alpha}")

        r = g["r"].to_numpy(dtype=float)
        yd = g["direct_mean"].to_numpy(dtype=float)
        ed = g["direct_sem"].to_numpy(dtype=float)
        ys = g["sd_mean"].to_numpy(dtype=float)
        es = g["sd_sem"].to_numpy(dtype=float)

        h_direct = fit_h(r, yd)
        h_sd = fit_h(r, ys)

        slope_rows.append(
            {
                "lag": PRIMARY_LAG,
                "alpha": alpha,
                "h_direct": h_direct,
                "h_source_depletion": h_sd,
                "fit_min": FIT_MIN,
                "fit_max": FIT_MAX,
            }
        )

        ax.set_xscale("log")
        ax.set_yscale("log")

        direct_line, = ax.plot(r, yd, "o-", linewidth=1.35, markersize=4.0, label="direct")
        ax.errorbar(r, yd, yerr=ed, fmt="none", ecolor="black", elinewidth=0.9, capsize=3, capthick=0.9)

        sd_line, = ax.plot(r, ys, "s-", linewidth=1.25, markersize=3.8, label="source--depletion")
        ax.errorbar(r, ys, yerr=es, fmt="none", ecolor="black", elinewidth=0.9, capsize=3, capthick=0.9)

        rr14, yy14 = anchored_guide(r, yd, 1.0 / 4.0)
        rr13, yy13 = anchored_guide(r, yd, 1.0 / 3.0)

        h14_line, = ax.plot(rr14, yy14, "--", linewidth=1.1, label=r"$h=1/4$")
        h13_line, = ax.plot(rr13, yy13, ":", linewidth=1.4, label=r"$h=1/3$")

        ax.set_xlabel(r"$r$")
        ax.set_ylabel(ylabel)

        ax.set_xticks([32, 64, 128, 192])
        ax.set_xticklabels(["32", "64", "128", "192"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(direction="in", which="both", top=True, right=True)

        ax.legend(
            handles=[direct_line, sd_line, h14_line, h13_line],
            frameon=False,
            fontsize=7.5,
            loc="upper left",
            handlelength=2.1,
        )

    fig.tight_layout(w_pad=1.4)
    fig.savefig(OUTPDF, bbox_inches="tight")
    plt.close(fig)

    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(OUTCSV, index=False)

    print(f"Wrote: {OUTPDF}")
    print(f"Wrote: {OUTCSV}")
    print(slopes.to_string(index=False))


if __name__ == "__main__":
    main()