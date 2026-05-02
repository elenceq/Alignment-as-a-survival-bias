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

OUT_PDF = FIGDIR / "fig_320_final15_pdf_repr_perpBL_v1.pdf"
OUT_PNG = FIGDIR / "fig_320_final15_pdf_repr_perpBL_v1.png"
OUT_NPZ = ENSEMBLE_DIR / "perpBL_320_final15_pdf_repr_summary_v1.npz"

R_TARGETS = [32, 96, 192]

# Folded angle theta in degrees, from 0 to 90.
THETA_EDGES = np.linspace(0.0, 90.0, 46)
THETA_CENTERS = 0.5 * (THETA_EDGES[:-1] + THETA_EDGES[1:])
DTHETA = np.diff(THETA_EDGES)

stems = [
    line.strip()
    for line in STEMS_FILE.read_text().splitlines()
    if line.strip()
]


def pdf_hist(theta_deg, weights=None):
    """
    Return density per degree on THETA_EDGES.
    """
    mask = np.isfinite(theta_deg)
    theta_deg = theta_deg[mask]

    if weights is not None:
        weights = weights[mask]
        wmask = np.isfinite(weights) & (weights > 0)
        theta_deg = theta_deg[wmask]
        weights = weights[wmask]

    hist, _ = np.histogram(theta_deg, bins=THETA_EDGES, weights=weights)

    norm = np.sum(hist * DTHETA)
    if norm > 0:
        hist = hist / norm
    else:
        hist = np.full_like(THETA_CENTERS, np.nan, dtype=float)

    return hist.astype(float)


def mean_std(a):
    mean = np.nanmean(a, axis=0)
    std = np.nanstd(a, axis=0, ddof=1)
    return mean, std


ncube = len(stems)
nrsel = len(R_TARGETS)
nbin = len(THETA_CENTERS)

unw_pdfs = np.full((ncube, nrsel, nbin), np.nan, dtype=float)
Aw_pdfs = np.full((ncube, nrsel, nbin), np.nan, dtype=float)

for icube, stem in enumerate(stems):
    npz = PER_CUBE / f"perpBL_320_final15_v1_{stem}.npz"
    if not npz.exists():
        raise FileNotFoundError(npz)

    print(f"Processing cube {icube+1:02d}: {stem}", flush=True)

    data = np.load(npz)

    r = data["r_list"].astype(float)
    theta_all = data["theta_deg"].astype(float)
    A_all = data["A"].astype(float)

    for irsel, r_target in enumerate(R_TARGETS):
        ir = int(np.argmin(np.abs(r - r_target)))
        if int(r[ir]) != r_target:
            raise RuntimeError(
                f"Could not find r={r_target} in {stem}; nearest is {r[ir]}"
            )

        theta = theta_all[ir]
        A = A_all[ir]

        mask = np.isfinite(theta) & np.isfinite(A) & (A > 0)
        theta = theta[mask]
        A = A[mask]

        unw_pdfs[icube, irsel, :] = pdf_hist(theta, weights=None)
        Aw_pdfs[icube, irsel, :] = pdf_hist(theta, weights=A)

unw_mean = np.full((nrsel, nbin), np.nan, dtype=float)
unw_std = np.full((nrsel, nbin), np.nan, dtype=float)
Aw_mean = np.full((nrsel, nbin), np.nan, dtype=float)
Aw_std = np.full((nrsel, nbin), np.nan, dtype=float)

for irsel in range(nrsel):
    unw_mean[irsel], unw_std[irsel] = mean_std(unw_pdfs[:, irsel, :])
    Aw_mean[irsel], Aw_std[irsel] = mean_std(Aw_pdfs[:, irsel, :])

theta_rad = np.deg2rad(THETA_CENTERS)
# Random folded-angle density on [0, pi/2]: p(theta_rad)=sin(theta_rad).
# Convert to density per degree.
random_pdf_per_deg = np.sin(theta_rad) * (np.pi / 180.0)

np.savez(
    OUT_NPZ,
    stems=np.array(stems),
    r_targets=np.array(R_TARGETS, dtype=int),
    theta_edges=THETA_EDGES,
    theta_centers=THETA_CENTERS,
    unw_pdfs=unw_pdfs,
    Aw_pdfs=Aw_pdfs,
    unw_mean=unw_mean,
    unw_std=unw_std,
    Aw_mean=Aw_mean,
    Aw_std=Aw_std,
    random_pdf_per_deg=random_pdf_per_deg,
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

fig, axes = plt.subplots(1, 3, figsize=(6.85, 2.75), sharex=True, sharey=True)

for irsel, (ax, r_target) in enumerate(zip(axes, R_TARGETS)):
    # Faint individual cube curves.
    for y in unw_pdfs[:, irsel, :]:
        ax.plot(THETA_CENTERS, y, color="C0", linewidth=0.55, alpha=0.16)

    for y in Aw_pdfs[:, irsel, :]:
        ax.plot(THETA_CENTERS, y, color="C1", linewidth=0.55, alpha=0.16)

    # Random baseline.
    line_rand, = ax.plot(
        THETA_CENTERS,
        random_pdf_per_deg,
        color="black",
        linestyle="--",
        linewidth=0.9,
        label="random 3D baseline",
    )

    # Ensemble means.
    line_unw, = ax.plot(
        THETA_CENTERS,
        unw_mean[irsel],
        color="C0",
        linewidth=1.55,
        label="unweighted",
    )

    line_Aw, = ax.plot(
        THETA_CENTERS,
        Aw_mean[irsel],
        color="C1",
        linewidth=1.55,
        label=r"$A_r$-weighted",
    )

    # One-sigma ensemble spread bands.
    ax.fill_between(
        THETA_CENTERS,
        unw_mean[irsel] - unw_std[irsel],
        unw_mean[irsel] + unw_std[irsel],
        color="C0",
        alpha=0.12,
        linewidth=0,
    )

    ax.fill_between(
        THETA_CENTERS,
        Aw_mean[irsel] - Aw_std[irsel],
        Aw_mean[irsel] + Aw_std[irsel],
        color="C1",
        alpha=0.12,
        linewidth=0,
    )

    ax.set_xlim(0, 90)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$\theta$ (deg)")

    if irsel == 0:
        ax.set_ylabel("PDF")

    # Small in-panel label instead of title.
    ax.text(
        0.05,
        0.95,
        rf"$r={r_target}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.3),
    )

# Shared legend outside.
handles = [
    Line2D([0], [0], color="black", linestyle="--", linewidth=0.9, label="random 3D baseline"),
    Line2D([0], [0], color="C0", linewidth=1.55, label="unweighted"),
    Line2D([0], [0], color="C1", linewidth=1.55, label=r"$A_r$-weighted"),
]
fig.legend(
    handles=handles,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=3,
    handlelength=2.0,
    columnspacing=1.0,
    labelspacing=0.35,
    borderpad=0.0,
)

fig.tight_layout(pad=0.35, w_pad=0.8)
fig.subplots_adjust(bottom=0.28)

fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUT_PDF)
print("Saved:", OUT_PNG)
print("Saved:", OUT_NPZ)

print("\nRepresentative-scale PDF summary:")
for irsel, r_target in enumerate(R_TARGETS):
    # crude scalar summary: mean angle from the ensemble mean PDFs
    norm_unw = np.sum(unw_mean[irsel] * DTHETA)
    norm_Aw = np.sum(Aw_mean[irsel] * DTHETA)

    mean_theta_unw = np.sum(THETA_CENTERS * unw_mean[irsel] * DTHETA) / norm_unw
    mean_theta_Aw = np.sum(THETA_CENTERS * Aw_mean[irsel] * DTHETA) / norm_Aw

    print(
        f"r={r_target:3d}  "
        f"mean theta from unweighted PDF = {mean_theta_unw:.3f} deg,  "
        f"mean theta from A-weighted PDF = {mean_theta_Aw:.3f} deg"
    )