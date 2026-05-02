#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path("/home/idies/workspace/Storage/elenceq/mhd_work/jhtdb_mhd1024")

IN_NPZ = BASE / "processed/perpBL_v1/smoke/perpBL_448_full_v1_results.npz"

FIGDIR = BASE / "processed/perpBL_v1/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = FIGDIR / "fig_448_full_weighting_covariance_perpBL_v1.pdf"
OUT_PNG = FIGDIR / "fig_448_full_weighting_covariance_perpBL_v1.png"

RNG_SEED = 24680

data = np.load(IN_NPZ)

r = data["r_list"].astype(float)
sin_theta = data["sin_theta"].astype(float)
A = data["A"].astype(float)

rng = np.random.RandomState(RNG_SEED)

mean_sin = []
weighted_sin = []
shuffled_weighted_sin = []
norm_cov = []
shuffled_norm_cov = []

for i in range(len(r)):
    s = sin_theta[i]
    a = A[i]

    mask = np.isfinite(s) & np.isfinite(a) & (a > 0)
    s = s[mask]
    a = a[mask]

    a_shuf = a.copy()
    rng.shuffle(a_shuf)

    mean_sin_i = np.mean(s)
    weighted_sin_i = np.sum(a * s) / np.sum(a)
    shuffled_weighted_sin_i = np.sum(a_shuf * s) / np.sum(a_shuf)

    cov_i = np.mean(a * s) - np.mean(a) * np.mean(s)
    cov_shuf_i = np.mean(a_shuf * s) - np.mean(a_shuf) * np.mean(s)

    mean_a = np.mean(a)

    mean_sin.append(mean_sin_i)
    weighted_sin.append(weighted_sin_i)
    shuffled_weighted_sin.append(shuffled_weighted_sin_i)
    norm_cov.append(cov_i / mean_a)
    shuffled_norm_cov.append(cov_shuf_i / mean_a)

mean_sin = np.array(mean_sin)
weighted_sin = np.array(weighted_sin)
shuffled_weighted_sin = np.array(shuffled_weighted_sin)
norm_cov = np.array(norm_cov)
shuffled_norm_cov = np.array(shuffled_norm_cov)

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

fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.75))

# -------------------------
# Left panel
# -------------------------
ax = axes[0]

line_w, = ax.plot(
    r, weighted_sin,
    marker="s",
    linewidth=1.35,
    markersize=3.0,
    label=r"$\langle A_r\sin\theta_r\rangle/\langle A_r\rangle$",
)

line_shuf, = ax.plot(
    r, shuffled_weighted_sin,
    marker="^",
    linewidth=1.35,
    markersize=3.0,
    label="shuffled weighted null",
)

line_unw, = ax.plot(
    r, mean_sin,
    marker="o",
    linewidth=1.35,
    markersize=3.0,
    label=r"$\langle\sin\theta_r\rangle$",
)

ax.axhline(
    np.pi / 4,
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
    label="random folded baseline",
)

ax.set_xlabel(r"$r$")
ax.set_ylabel("sine-based alignment measure")
ax.set_xlim(28, 198)
ax.set_xticks([32, 64, 96, 128, 160, 192])
ax.set_ylim(0.64, 0.80)

# Legend outside, below left panel.
ax.legend(
    handles=[baseline_handle, line_w, line_shuf, line_unw],
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),
    ncol=2,
    handlelength=2.0,
    columnspacing=1.0,
    labelspacing=0.35,
    borderpad=0.0,
)

# -------------------------
# Right panel
# -------------------------
ax = axes[1]

line_cov, = ax.plot(
    r, norm_cov,
    marker="o",
    linewidth=1.35,
    markersize=3.0,
    label=r"$\mathrm{Cov}(A_r,\sin\theta_r)/\langle A_r\rangle$",
)

line_cov_shuf, = ax.plot(
    r, shuffled_norm_cov,
    marker="^",
    linewidth=1.35,
    markersize=3.0,
    label="shuffled covariance null",
)

ax.axhline(
    0.0,
    linestyle="--",
    linewidth=0.9,
    color="black",
    alpha=0.85,
)

ax.set_xlabel(r"$r$")
ax.set_ylabel("normalized covariance")
ax.set_xlim(28, 198)
ax.set_xticks([32, 64, 96, 128, 160, 192])

ymin = min(np.nanmin(norm_cov), np.nanmin(shuffled_norm_cov), 0.0)
ymax = max(np.nanmax(norm_cov), np.nanmax(shuffled_norm_cov), 0.0)
pad = 0.12 * max(abs(ymin), abs(ymax), 1e-3)
ax.set_ylim(ymin - pad, ymax + pad)

# Legend outside, below right panel.
ax.legend(
    handles=[line_cov, line_cov_shuf],
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),
    ncol=1,
    handlelength=2.0,
    labelspacing=0.35,
    borderpad=0.0,
)

fig.tight_layout(pad=0.45, w_pad=1.0)
fig.subplots_adjust(bottom=0.32)

fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUT_PDF)
print("Saved:", OUT_PNG)

print()
print("Values:")
for i, rr in enumerate(r.astype(int)):
    print(
        f"r={rr:3d}  "
        f"<sin>={mean_sin[i]:.6f}  "
        f"<A sin>/<A>={weighted_sin[i]:.6f}  "
        f"shuf={shuffled_weighted_sin[i]:.6f}  "
        f"cov/<A>={norm_cov[i]:.6e}  "
        f"shuf cov/<A>={shuffled_norm_cov[i]:.6e}"
    )