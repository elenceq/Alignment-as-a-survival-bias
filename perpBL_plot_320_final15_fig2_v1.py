#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path("/home/idies/workspace/Storage/elenceq/mhd_work/jhtdb_mhd1024")

STEMS_FILE = BASE / "processed/perpBL_v1/config/perpBL_320_final15_stems_v1.txt"
PER_CUBE = BASE / "processed/perpBL_v1/per_cube"
FIGDIR = BASE / "processed/perpBL_v1/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = FIGDIR / "fig_320_final15_mean_angle_perpBL_v1_clean.pdf"
OUT_PNG = FIGDIR / "fig_320_final15_mean_angle_perpBL_v1_clean.png"
OUT_NPZ = BASE / "processed/perpBL_v1/ensemble/perpBL_320_final15_mean_angle_summary_v1.npz"
OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)

stems = [
    line.strip()
    for line in STEMS_FILE.read_text().splitlines()
    if line.strip()
]

r_ref = None
all_curves = []
topA_curves = []
topj_curves = []

for stem in stems:
    js = PER_CUBE / f"perpBL_320_final15_v1_{stem}_summary.json"
    if not js.exists():
        raise FileNotFoundError(js)

    with open(js, "r") as f:
        data = json.load(f)

    rows = data["summary"]
    r = np.array([row["r"] for row in rows], dtype=float)

    if r_ref is None:
        r_ref = r
    else:
        if not np.allclose(r, r_ref):
            raise RuntimeError(f"r-list mismatch in {stem}")

    all_curves.append([row["mean_angle_all_deg"] for row in rows])
    topA_curves.append([row["mean_angle_top10_A_deg"] for row in rows])
    topj_curves.append([row["mean_angle_top10_j_deg"] for row in rows])

r = r_ref
all_curves = np.asarray(all_curves, dtype=float)
topA_curves = np.asarray(topA_curves, dtype=float)
topj_curves = np.asarray(topj_curves, dtype=float)

def mean_sem(a):
    mean = np.nanmean(a, axis=0)
    sem = np.nanstd(a, axis=0, ddof=1) / np.sqrt(a.shape[0])
    return mean, sem

all_mean, all_sem = mean_sem(all_curves)
topA_mean, topA_sem = mean_sem(topA_curves)
topj_mean, topj_sem = mean_sem(topj_curves)

np.savez(
    OUT_NPZ,
    stems=np.array(stems),
    r=r,
    all_curves=all_curves,
    topA_curves=topA_curves,
    topj_curves=topj_curves,
    all_mean=all_mean,
    all_sem=all_sem,
    topA_mean=topA_mean,
    topA_sem=topA_sem,
    topj_mean=topj_mean,
    topj_sem=topj_sem,
)

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

fig, ax = plt.subplots(figsize=(3.45, 2.75))

# Use fixed group colors; individual curves are pale and unobtrusive.
c_all = "C0"
c_A = "C1"
c_j = "C2"

for y in all_curves:
    ax.plot(r, y, color=c_all, linewidth=0.55, alpha=0.13)

for y in topA_curves:
    ax.plot(r, y, color=c_A, linewidth=0.55, alpha=0.20, linestyle="--")

for y in topj_curves:
    ax.plot(r, y, color=c_j, linewidth=0.55, alpha=0.13, linestyle=":")

# Bold ensemble means.
line_all, = ax.plot(
    r, all_mean,
    color=c_all,
    marker="o",
    linewidth=1.55,
    markersize=3.1,
    label="all points",
)

line_A, = ax.plot(
    r, topA_mean,
    color=c_A,
    marker="s",
    linewidth=1.55,
    markersize=3.1,
    label=r"top 10% by $A_r$",
)

line_j, = ax.plot(
    r, topj_mean,
    color=c_j,
    marker="^",
    linewidth=1.55,
    markersize=3.1,
    label=r"top 10% by $|j|$",
)

# SEM error bars, deliberately excluded from legend.
for mean, sem in [(all_mean, all_sem), (topA_mean, topA_sem), (topj_mean, topj_sem)]:
    ax.errorbar(
        r, mean, yerr=sem,
        fmt="none",
        ecolor="black",
        elinewidth=0.65,
        capsize=1.8,
        capthick=0.65,
        label="_nolegend_",
        zorder=5,
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

ax.set_xlabel(r"$r$")
ax.set_ylabel("mean angle (deg)")

ax.set_xlim(28, 198)
ax.set_xticks([32, 64, 96, 128, 160, 192])
ax.set_ylim(24, 60)

handles = [line_all, line_A, line_j, baseline_handle]
ax.legend(
    handles=handles,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.23),
    ncol=2,
    handlelength=2.0,
    columnspacing=1.0,
    labelspacing=0.35,
    borderpad=0.0,
)

fig.tight_layout(pad=0.35)
fig.subplots_adjust(bottom=0.30)

fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUT_PDF)
print("Saved:", OUT_PNG)
print("Saved:", OUT_NPZ)

print("\nEnsemble means:")
for i, rr in enumerate(r.astype(int)):
    print(
        f"r={rr:3d}  "
        f"all={all_mean[i]:6.3f}±{all_sem[i]:.3f}  "
        f"topA={topA_mean[i]:6.3f}±{topA_sem[i]:.3f}  "
        f"topj={topj_mean[i]:6.3f}±{topj_sem[i]:.3f}"
    )