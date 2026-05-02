#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path("/home/idies/workspace/Storage/elenceq/mhd_work/jhtdb_mhd1024")

ENSEMBLE_NPZ = BASE / "processed/perpBL_v1/ensemble/perpBL_320_final15_mean_angle_summary_v1.npz"
REF448_JSON = BASE / "processed/perpBL_v1/smoke/perpBL_448_full_v1_summary.json"

FIGDIR = BASE / "processed/perpBL_v1/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = FIGDIR / "fig_448_vs_320_mean_angle_perpBL_v1.pdf"
OUT_PNG = FIGDIR / "fig_448_vs_320_mean_angle_perpBL_v1.png"

# -------------------------
# Load 320^3 ensemble summary
# -------------------------

E = np.load(ENSEMBLE_NPZ, allow_pickle=True)

r320 = E["r"].astype(float)

all_mean = E["all_mean"].astype(float)
all_sem = E["all_sem"].astype(float)

topA_mean = E["topA_mean"].astype(float)
topA_sem = E["topA_sem"].astype(float)

topj_mean = E["topj_mean"].astype(float)
topj_sem = E["topj_sem"].astype(float)

# -------------------------
# Load 448^3 reference summary
# -------------------------

with open(REF448_JSON, "r") as f:
    D = json.load(f)

rows = D["summary"]

r448 = np.array([row["r"] for row in rows], dtype=float)
all_448 = np.array([row["mean_angle_all_deg"] for row in rows], dtype=float)
topA_448 = np.array([row["mean_angle_top10_A_deg"] for row in rows], dtype=float)
topj_448 = np.array([row["mean_angle_top10_j_deg"] for row in rows], dtype=float)

if not np.allclose(r320, r448):
    raise RuntimeError("r-lists do not match between 320 ensemble and 448 reference.")

r = r320

# -------------------------
# Plot
# -------------------------

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

fig, ax = plt.subplots(figsize=(3.45, 2.85))

# 15-cube ensemble: solid lines with SEM.
line_all, = ax.plot(
    r,
    all_mean,
    color="C0",
    marker="o",
    linewidth=1.45,
    markersize=3.0,
    label="all points",
)

line_A, = ax.plot(
    r,
    topA_mean,
    color="C1",
    marker="s",
    linewidth=1.45,
    markersize=3.0,
    label=r"top 10% by $A_r$",
)

line_j, = ax.plot(
    r,
    topj_mean,
    color="C2",
    marker="^",
    linewidth=1.45,
    markersize=3.0,
    label=r"top 10% by $|j|$",
)

for mean, sem in [
    (all_mean, all_sem),
    (topA_mean, topA_sem),
    (topj_mean, topj_sem),
]:
    ax.errorbar(
        r,
        mean,
        yerr=sem,
        fmt="none",
        ecolor="black",
        elinewidth=0.65,
        capsize=1.8,
        capthick=0.65,
        label="_nolegend_",
        zorder=5,
    )

# 448^3 reference: same colors, dashed open-marker style.
ax.plot(
    r448,
    all_448,
    color="C0",
    marker="o",
    linewidth=1.15,
    markersize=3.0,
    linestyle="--",
    markerfacecolor="white",
    label="_nolegend_",
)

ax.plot(
    r448,
    topA_448,
    color="C1",
    marker="s",
    linewidth=1.15,
    markersize=3.0,
    linestyle="--",
    markerfacecolor="white",
    label="_nolegend_",
)

ax.plot(
    r448,
    topj_448,
    color="C2",
    marker="^",
    linewidth=1.15,
    markersize=3.0,
    linestyle="--",
    markerfacecolor="white",
    label="_nolegend_",
)

ax.axhline(
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

ensemble_handle = Line2D(
    [0], [0],
    color="black",
    linestyle="-",
    marker="o",
    linewidth=1.35,
    markersize=3.0,
    label=r"15 cubes",
)

ref_handle = Line2D(
    [0], [0],
    color="black",
    linestyle="--",
    marker="o",
    markerfacecolor="white",
    linewidth=1.15,
    markersize=3.0,
    label=r"$448^3$ reference",
)

ax.set_xlabel(r"$r$")
ax.set_ylabel("mean angle (deg)")

ax.set_xlim(28, 198)
ax.set_xticks([32, 64, 96, 128, 160, 192])
ax.set_ylim(30, 60)

# One outside legend: group identity + curve meaning.
handles = [
    line_all,
    line_A,
    line_j,
    baseline_handle,
    ensemble_handle,
    ref_handle,
]

ax.legend(
    handles=handles,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.23),
    ncol=2,
    handlelength=2.0,
    columnspacing=0.9,
    labelspacing=0.35,
    borderpad=0.0,
)

fig.tight_layout(pad=0.35)
fig.subplots_adjust(bottom=0.34)

fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUT_PDF)
print("Saved:", OUT_PNG)

print("\n448 vs 320 comparison:")
for i, rr in enumerate(r.astype(int)):
    print(
        f"r={rr:3d}  "
        f"320 all={all_mean[i]:.3f}, 448 all={all_448[i]:.3f};  "
        f"320 topA={topA_mean[i]:.3f}, 448 topA={topA_448[i]:.3f};  "
        f"320 topj={topj_mean[i]:.3f}, 448 topj={topj_448[i]:.3f}"
    )