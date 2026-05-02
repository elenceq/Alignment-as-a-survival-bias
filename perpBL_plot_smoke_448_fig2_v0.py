#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path("/home/idies/workspace/Storage/elenceq/mhd_work/jhtdb_mhd1024")
IN_JSON = BASE / "processed/perpBL_v1/smoke/perpBL_smoke_448_v0_summary.json"
FIGDIR = BASE / "processed/perpBL_v1/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = FIGDIR / "fig_smoke_448_mean_angle_perpBL_v0.pdf"
OUT_PNG = FIGDIR / "fig_smoke_448_mean_angle_perpBL_v0.png"

with open(IN_JSON, "r") as f:
    data = json.load(f)

rows = data["summary"]

r = np.array([row["r"] for row in rows], dtype=float)
theta_all = np.array([row["mean_angle_all_deg"] for row in rows], dtype=float)
theta_topA = np.array([row["mean_angle_top10_A_deg"] for row in rows], dtype=float)
theta_topj = np.array([row["mean_angle_top10_j_deg"] for row in rows], dtype=float)

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.45, 2.55))

ax.plot(r, theta_all, marker="o", linewidth=1.5, markersize=3.5, label="all points")
ax.plot(r, theta_topA, marker="s", linewidth=1.5, markersize=3.5, label=r"top 10% by $A_r$")
ax.plot(r, theta_topj, marker="^", linewidth=1.5, markersize=3.5, label=r"top 10% by $|j|$")

ax.axhline(57.2957795, linestyle="--", linewidth=0.9, color="black", alpha=0.8)

baseline_handle = Line2D(
    [0], [0],
    color="black",
    linestyle="--",
    linewidth=0.9,
    label=r"random 3D baseline"
)

handles, labels = ax.get_legend_handles_labels()
handles.append(baseline_handle)

ax.set_xlabel(r"$r$")
ax.set_ylabel("mean angle (deg)")

ax.set_xlim(25, 200)
ax.set_ylim(30, 60)

ax.legend(
    handles=handles,
    frameon=False,
    loc="lower right",
    handlelength=2.2,
    borderpad=0.2,
    labelspacing=0.35,
)

fig.tight_layout(pad=0.5)
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUT_PDF)
print("Saved:", OUT_PNG)