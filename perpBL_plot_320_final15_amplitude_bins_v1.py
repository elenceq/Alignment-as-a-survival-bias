#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

BASE = Path("/home/idies/workspace/Storage/elenceq/mhd_work/jhtdb_mhd1024")

STEMS_FILE = BASE / "processed/perpBL_v1/config/perpBL_320_final15_stems_v1.txt"
PER_CUBE = BASE / "processed/perpBL_v1/per_cube"
FIGDIR = BASE / "processed/perpBL_v1/figures"
ENSEMBLE_DIR = BASE / "processed/perpBL_v1/ensemble"

FIGDIR.mkdir(parents=True, exist_ok=True)
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = FIGDIR / "fig_320_final15_amplitude_bins_perpBL_v1.pdf"
OUT_PNG = FIGDIR / "fig_320_final15_amplitude_bins_perpBL_v1.png"
OUT_NPZ = ENSEMBLE_DIR / "perpBL_320_final15_amplitude_bins_summary_v1.npz"

# Amplitude percentile bins. Last bins isolate the strongest events.
BIN_EDGES = np.array([0, 50, 75, 90, 97, 99, 100], dtype=float)
BIN_LABELS = ["0-50", "50-75", "75-90", "90-97", "97-99", "99-100"]

stems = [
    line.strip()
    for line in STEMS_FILE.read_text().splitlines()
    if line.strip()
]


def mean_c_from_a(a):
    """
    Mean c for p(c) proportional to exp(a c), c in [0,1].
    """
    if abs(a) < 1.0e-7:
        return 0.5 + a / 12.0

    ea = np.exp(a)
    return (((a - 1.0) * ea) + 1.0) / (a * (ea - 1.0))


def fit_a_from_mean_c(m):
    """
    Fit a from mean c_abs under p(c) proportional to exp(a c), c in [0,1].
    """
    if not np.isfinite(m):
        return np.nan

    # Keep the fit inside the valid open interval.
    m = min(max(float(m), 1.0e-6), 1.0 - 1.0e-6)

    if abs(m - 0.5) < 1.0e-8:
        return 0.0

    def f(a):
        return mean_c_from_a(a) - m

    return float(brentq(f, -80.0, 80.0, maxiter=200))


r_ref = None
nbin = len(BIN_LABELS)

theta_bin_cubes = []
cmean_bin_cubes = []
a_bin_cubes = []
count_bin_cubes = []

for icube, stem in enumerate(stems, 1):
    npz = PER_CUBE / f"perpBL_320_final15_v1_{stem}.npz"
    if not npz.exists():
        raise FileNotFoundError(npz)

    print(f"Processing cube {icube:02d}: {stem}", flush=True)

    data = np.load(npz)

    r = data["r_list"].astype(float)
    q = data["q"].astype(float)
    theta_deg = data["theta_deg"].astype(float)
    A = data["A"].astype(float)

    if r_ref is None:
        r_ref = r
    else:
        if not np.allclose(r, r_ref):
            raise RuntimeError(f"r-list mismatch in {stem}")

    nr = len(r)

    theta_bin = np.full((nr, nbin), np.nan, dtype=float)
    cmean_bin = np.full((nr, nbin), np.nan, dtype=float)
    a_bin = np.full((nr, nbin), np.nan, dtype=float)
    count_bin = np.zeros((nr, nbin), dtype=int)

    for ir in range(nr):
        q_ir = q[ir]
        th_ir = theta_deg[ir]
        A_ir = A[ir]

        mask = np.isfinite(q_ir) & np.isfinite(th_ir) & np.isfinite(A_ir) & (A_ir > 0)

        qv = q_ir[mask]
        thv = th_ir[mask]
        Av = A_ir[mask]

        cabs = np.abs(qv)

        if Av.size == 0:
            continue

        cuts = np.percentile(Av, BIN_EDGES)

        for ib in range(nbin):
            lo = cuts[ib]
            hi = cuts[ib + 1]

            if ib == 0:
                m = (Av >= lo) & (Av <= hi)
            else:
                m = (Av > lo) & (Av <= hi)

            n = int(np.count_nonzero(m))
            count_bin[ir, ib] = n

            if n < 10:
                continue

            theta_bin[ir, ib] = float(np.mean(thv[m]))
            cmean_bin[ir, ib] = float(np.mean(cabs[m]))
            a_bin[ir, ib] = fit_a_from_mean_c(cmean_bin[ir, ib])

    theta_bin_cubes.append(theta_bin)
    cmean_bin_cubes.append(cmean_bin)
    a_bin_cubes.append(a_bin)
    count_bin_cubes.append(count_bin)

r = r_ref

theta_bin_cubes = np.asarray(theta_bin_cubes, dtype=float)
cmean_bin_cubes = np.asarray(cmean_bin_cubes, dtype=float)
a_bin_cubes = np.asarray(a_bin_cubes, dtype=float)
count_bin_cubes = np.asarray(count_bin_cubes, dtype=int)

theta_mean = np.nanmean(theta_bin_cubes, axis=0)
theta_sem = np.nanstd(theta_bin_cubes, axis=0, ddof=1) / np.sqrt(theta_bin_cubes.shape[0])

cmean_mean = np.nanmean(cmean_bin_cubes, axis=0)
cmean_sem = np.nanstd(cmean_bin_cubes, axis=0, ddof=1) / np.sqrt(cmean_bin_cubes.shape[0])

a_mean = np.nanmean(a_bin_cubes, axis=0)
a_sem = np.nanstd(a_bin_cubes, axis=0, ddof=1) / np.sqrt(a_bin_cubes.shape[0])

np.savez(
    OUT_NPZ,
    stems=np.array(stems),
    r=r,
    bin_edges=BIN_EDGES,
    bin_labels=np.array(BIN_LABELS),
    theta_bin_cubes=theta_bin_cubes,
    cmean_bin_cubes=cmean_bin_cubes,
    a_bin_cubes=a_bin_cubes,
    count_bin_cubes=count_bin_cubes,
    theta_mean=theta_mean,
    theta_sem=theta_sem,
    cmean_mean=cmean_mean,
    cmean_sem=cmean_sem,
    a_mean=a_mean,
    a_sem=a_sem,
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

fig = plt.figure(figsize=(6.85, 3.00))

ax1 = fig.add_axes([0.08, 0.20, 0.34, 0.68])
ax2 = fig.add_axes([0.53, 0.20, 0.34, 0.68])
cax1 = fig.add_axes([0.43, 0.24, 0.018, 0.60])
cax2 = fig.add_axes([0.88, 0.24, 0.018, 0.60])

# Transpose for plotting: y=amplitude bin, x=r.
Mtheta = theta_mean.T
Ma = a_mean.T

im1 = ax1.imshow(
    Mtheta,
    origin="lower",
    aspect="auto",
    interpolation="nearest",
)

im2 = ax2.imshow(
    Ma,
    origin="lower",
    aspect="auto",
    interpolation="nearest",
)

for ax in [ax1, ax2]:
    ax.set_xlabel(r"$r$")
    ax.set_xticks(np.arange(len(r)))
    ax.set_xticklabels([str(int(x)) for x in r], rotation=45, ha="right")

    ax.set_yticks(np.arange(nbin))
    ax.set_yticklabels(BIN_LABELS)

ax1.set_ylabel(r"$A_r$ percentile")
ax2.set_ylabel(r"$A_r$ percentile")

# Small panel labels, not titles.
ax1.text(
    0.04,
    0.96,
    "mean angle",
    transform=ax1.transAxes,
    ha="left",
    va="top",
    fontsize=8,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
)

ax2.text(
    0.04,
    0.96,
    r"fit $a(A_r)$",
    transform=ax2.transAxes,
    ha="left",
    va="top",
    fontsize=8,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
)

cb1 = fig.colorbar(im1, cax=cax1)
cb1.set_label("deg")

cb2 = fig.colorbar(im2, cax=cax2)
cb2.set_label(r"$a$")

fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUT_PDF)
print("Saved:", OUT_PNG)
print("Saved:", OUT_NPZ)

print("\nAmplitude-bin ensemble values:")
for ir, rr in enumerate(r.astype(int)):
    print(f"\nr = {rr}")
    for ib, lab in enumerate(BIN_LABELS):
        print(
            f"  A percentile {lab:>6s}: "
            f"theta={theta_mean[ir, ib]:6.3f}±{theta_sem[ir, ib]:.3f} deg, "
            f"a={a_mean[ir, ib]:7.3f}±{a_sem[ir, ib]:.3f}, "
            f"Nmedian={int(np.median(count_bin_cubes[:, ir, ib]))}"
        )