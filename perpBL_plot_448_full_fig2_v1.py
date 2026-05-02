#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path("/home/idies/workspace/Storage/elenceq/mhd_work/jhtdb_mhd1024")
IN_JSON = BASE / "processed/perpBL_v1/smoke/perpBL_448_full_v1_summary.json"
FIGDIR = BASE / "processed/perpBL_v1/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = FIGDIR / "fig_448_full_mean_angle_perpBL_v1.pdf"
OUT_PNG = FIGDIR / "fig_448_full_mean_angle_perpBL_v1.png"

with open(IN_JSON, "r") as f:
    data = json.load(f)

rows = data["summary"]

r = np.array([row["r"] for row in rows], dtype=float)
theta_all = np.array([row["mean_angle_all_deg"] for row in rows], dtype=float)
theta_topA = np.array([row["mean_angle_top10_A_deg"] for row in rows], dtype=float)
theta_topj = np.array([row["mean_angle_top10_j_deg"] for row in rows], dtype=float)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.35, 2.35))

line_all, = ax.plot(
    r, theta_all,
    marker="o",
    linewidth=1.35,
    markersize=3.0,
    label="all points",
)

line_A, = ax.plot(
    r, theta_topA,
    marker="s",
    linewidth=1.35,
    markersize=3.0,
    label=r"top 10% by $A_r$",
)

line_j, = ax.plot(
    r, theta_topj,
    marker="^",
    linewidth=1.35,
    markersize=3.0,
    label=r"top 10% by $|j|$",
)

baseline = ax.axhline(
    57.2957795,
    linestyle="--",
    linewidth=0.9,
    color="black",
    alpha=0.85,
)

baseline_handle = Line2D(
    [0], [0],
    color="black",
    linestyle="--",
    linewidth=0.9,
    label="random 3D baseline",
)

ax.set_xlabel(r"$r$")
ax.set_ylabel("mean angle (deg)")

ax.set_xlim(28, 198)
ax.set_ylim(34, 59)

ax.set_xticks([32, 64, 96, 128, 160, 192])

# Put legend outside/below so it never covers data.
handles = [line_all, line_A, line_j, baseline_handle]
ax.legend(
    handles=handles,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.19),
    ncol=2,
    handlelength=2.0,
    columnspacing=1.0,
    labelspacing=0.3,
    borderpad=0.0,
)

fig.tight_layout(pad=0.35)
fig.subplots_adjust(bottom=0.29)

fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUT_PDF)
print("Saved:", OUT_PNG)