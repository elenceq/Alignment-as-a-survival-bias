#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path("/home/idies/workspace/Storage/elenceq/mhd_work/jhtdb_mhd1024")

STEMS_FILE = BASE / "processed/perpBL_v1/config/perpBL_320_final15_stems_v1.txt"
PER_CUBE = BASE / "processed/perpBL_v1/per_cube"
FIGDIR = BASE / "processed/perpBL_v1/figures"
ENSEMBLE_DIR = BASE / "processed/perpBL_v1/ensemble"

FIGDIR.mkdir(parents=True, exist_ok=True)
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = FIGDIR / "fig_320_final15_weighting_covariance_perpBL_v1.pdf"
OUT_PNG = FIGDIR / "fig_320_final15_weighting_covariance_perpBL_v1.png"
OUT_NPZ = ENSEMBLE_DIR / "perpBL_320_final15_weighting_covariance_summary_v1.npz"

RNG_SEED = 24680

stems = [
    line.strip()
    for line in STEMS_FILE.read_text().splitlines()
    if line.strip()
]

r_ref = None

mean_sin_curves = []
weighted_sin_curves = []
shuffled_weighted_sin_curves = []
norm_cov_curves = []
shuffled_norm_cov_curves = []

rng = np.random.RandomState(RNG_SEED)

for stem in stems:
    npz = PER_CUBE / f"perpBL_320_final15_v1_{stem}.npz"
    if not npz.exists():
        raise FileNotFoundError(npz)

    data = np.load(npz)

    r = data["r_list"].astype(float)
    sin_theta = data["sin_theta"].astype(float)
    A = data["A"].astype(float)

    if r_ref is None:
        r_ref = r
    else:
        if not np.allclose(r, r_ref):
            raise RuntimeError(f"r-list mismatch in {stem}")

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

    mean_sin_curves.append(mean_sin)
    weighted_sin_curves.append(weighted_sin)
    shuffled_weighted_sin_curves.append(shuffled_weighted_sin)
    norm_cov_curves.append(norm_cov)
    shuffled_norm_cov_curves.append(shuffled_norm_cov)

r = r_ref

mean_sin_curves = np.asarray(mean_sin_curves, dtype=float)
weighted_sin_curves = np.asarray(weighted_sin_curves, dtype=float)
shuffled_weighted_sin_curves = np.asarray(shuffled_weighted_sin_curves, dtype=float)
norm_cov_curves = np.asarray(norm_cov_curves, dtype=float)
shuffled_norm_cov_curves = np.asarray(shuffled_norm_cov_curves, dtype=float)

def mean_sem(a):
    mean = np.nanmean(a, axis=0)
    sem = np.nanstd(a, axis=0, ddof=1) / np.sqrt(a.shape[0])
    return mean, sem

mean_sin_mean, mean_sin_sem = mean_sem(mean_sin_curves)
weighted_sin_mean, weighted_sin_sem = mean_sem(weighted_sin_curves)
shuffled_weighted_sin_mean, shuffled_weighted_sin_sem = mean_sem(shuffled_weighted_sin_curves)

norm_cov_mean, norm_cov_sem = mean_sem(norm_cov_curves)
shuffled_norm_cov_mean, shuffled_norm_cov_sem = mean_sem(shuffled_norm_cov_curves)

np.savez(
    OUT_NPZ,
    stems=np.array(stems),
    r=r,
    mean_sin_curves=mean_sin_curves,
    weighted_sin_curves=weighted_sin_curves,
    shuffled_weighted_sin_curves=shuffled_weighted_sin_curves,
    norm_cov_curves=norm_cov_curves,
    shuffled_norm_cov_curves=shuffled_norm_cov_curves,
    mean_sin_mean=mean_sin_mean,
    mean_sin_sem=mean_sin_sem,
    weighted_sin_mean=weighted_sin_mean,
    weighted_sin_sem=weighted_sin_sem,
    shuffled_weighted_sin_mean=shuffled_weighted_sin_mean,
    shuffled_weighted_sin_sem=shuffled_weighted_sin_sem,
    norm_cov_mean=norm_cov_mean,
    norm_cov_sem=norm_cov_sem,
    shuffled_norm_cov_mean=shuffled_norm_cov_mean,
    shuffled_norm_cov_sem=shuffled_norm_cov_sem,
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

fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.75))

# ============================================================
# Left panel: weighted/unweighted sine measure
# ============================================================

ax = axes[0]

# Faint individual-cube curves.
for y in weighted_sin_curves:
    ax.plot(r, y, color="C0", linewidth=0.55, alpha=0.18)
for y in shuffled_weighted_sin_curves:
    ax.plot(r, y, color="C1", linewidth=0.55, alpha=0.12)
for y in mean_sin_curves:
    ax.plot(r, y, color="C2", linewidth=0.55, alpha=0.18)

line_w, = ax.plot(
    r, weighted_sin_mean,
    color="C0",
    marker="s",
    linewidth=1.45,
    markersize=3.0,
    label=r"$\langle A_r\sin\theta_r\rangle/\langle A_r\rangle$",
)

line_shuf, = ax.plot(
    r, shuffled_weighted_sin_mean,
    color="C1",
    marker="^",
    linewidth=1.45,
    markersize=3.0,
    label="shuffled weighted null",
)

line_unw, = ax.plot(
    r, mean_sin_mean,
    color="C2",
    marker="o",
    linewidth=1.45,
    markersize=3.0,
    label=r"$\langle\sin\theta_r\rangle$",
)

for mean, sem in [
    (weighted_sin_mean, weighted_sin_sem),
    (shuffled_weighted_sin_mean, shuffled_weighted_sin_sem),
    (mean_sin_mean, mean_sin_sem),
]:
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

ymin = min(
    np.nanmin(weighted_sin_curves),
    np.nanmin(shuffled_weighted_sin_curves),
    np.nanmin(mean_sin_curves),
    np.pi / 4,
)
ymax = max(
    np.nanmax(weighted_sin_curves),
    np.nanmax(shuffled_weighted_sin_curves),
    np.nanmax(mean_sin_curves),
    np.pi / 4,
)
pad = 0.08 * (ymax - ymin)
ax.set_ylim(ymin - pad, ymax + pad)

ax.legend(
    handles=[baseline_handle, line_w, line_shuf, line_unw],
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),
    ncol=2,
    handlelength=2.0,
    columnspacing=0.9,
    labelspacing=0.35,
    borderpad=0.0,
)

# ============================================================
# Right panel: normalized covariance
# ============================================================

ax = axes[1]

for y in norm_cov_curves:
    ax.plot(r, y, color="C0", linewidth=0.55, alpha=0.18)
for y in shuffled_norm_cov_curves:
    ax.plot(r, y, color="C1", linewidth=0.55, alpha=0.12)

line_cov, = ax.plot(
    r, norm_cov_mean,
    color="C0",
    marker="o",
    linewidth=1.45,
    markersize=3.0,
    label=r"$\mathrm{Cov}(A_r,\sin\theta_r)/\langle A_r\rangle$",
)

line_cov_shuf, = ax.plot(
    r, shuffled_norm_cov_mean,
    color="C1",
    marker="^",
    linewidth=1.45,
    markersize=3.0,
    label="shuffled covariance null",
)

for mean, sem in [
    (norm_cov_mean, norm_cov_sem),
    (shuffled_norm_cov_mean, shuffled_norm_cov_sem),
]:
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

ymin = min(np.nanmin(norm_cov_curves), np.nanmin(shuffled_norm_cov_curves), 0.0)
ymax = max(np.nanmax(norm_cov_curves), np.nanmax(shuffled_norm_cov_curves), 0.0)
pad = 0.10 * max(abs(ymin), abs(ymax), 1.0e-3)
ax.set_ylim(ymin - pad, ymax + pad)

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
print("Saved:", OUT_NPZ)

print("\nEnsemble values:")
for i, rr in enumerate(r.astype(int)):
    print(
        f"r={rr:3d}  "
        f"<sin>={mean_sin_mean[i]:.6f}±{mean_sin_sem[i]:.6f}  "
        f"<A sin>/<A>={weighted_sin_mean[i]:.6f}±{weighted_sin_sem[i]:.6f}  "
        f"shuf={shuffled_weighted_sin_mean[i]:.6f}±{shuffled_weighted_sin_sem[i]:.6f}  "
        f"cov/<A>={norm_cov_mean[i]:.6e}±{norm_cov_sem[i]:.6e}  "
        f"shuf cov/<A>={shuffled_norm_cov_mean[i]:.6e}±{shuffled_norm_cov_sem[i]:.6e}"
    )