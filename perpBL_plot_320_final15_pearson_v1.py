#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE = Path("/home/idies/workspace/Storage/elenceq/mhd_work/jhtdb_mhd1024")

STEMS_FILE = BASE / "processed/perpBL_v1/config/perpBL_320_final15_stems_v1.txt"
PER_CUBE = BASE / "processed/perpBL_v1/per_cube"
FIGDIR = BASE / "processed/perpBL_v1/figures"
ENSEMBLE_DIR = BASE / "processed/perpBL_v1/ensemble"

FIGDIR.mkdir(parents=True, exist_ok=True)
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = FIGDIR / "fig_320_final15_pearson_matrices_perpBL_v1.pdf"
OUT_PNG = FIGDIR / "fig_320_final15_pearson_matrices_perpBL_v1.png"
OUT_NPZ = ENSEMBLE_DIR / "perpBL_320_final15_pearson_matrices_summary_v1.npz"

stems = [
    line.strip()
    for line in STEMS_FILE.read_text().splitlines()
    if line.strip()
]


def pearson_pairwise_matrix(X):
    """
    X shape: (nr, nevent). Pairwise-finite Pearson matrix.
    """
    nr = X.shape[0]
    R = np.full((nr, nr), np.nan, dtype=float)
    N = np.zeros((nr, nr), dtype=int)

    for i in range(nr):
        xi = X[i]
        for j in range(nr):
            xj = X[j]
            mask = np.isfinite(xi) & np.isfinite(xj)
            n = int(np.count_nonzero(mask))
            N[i, j] = n

            if n < 3:
                continue

            a = xi[mask].astype(float)
            b = xj[mask].astype(float)

            a = a - np.mean(a)
            b = b - np.mean(b)

            da = np.sqrt(np.sum(a * a))
            db = np.sqrt(np.sum(b * b))

            if da == 0.0 or db == 0.0:
                continue

            R[i, j] = np.sum(a * b) / (da * db)

    return R, N


r_ref = None
Rq_list = []
Rs_list = []
Nq_list = []
Ns_list = []

for stem in stems:
    npz = PER_CUBE / f"perpBL_320_final15_v1_{stem}.npz"
    if not npz.exists():
        raise FileNotFoundError(npz)

    data = np.load(npz)

    r = data["r_list"].astype(float)
    q = data["q"].astype(float)
    s = data["sin_theta"].astype(float)

    if r_ref is None:
        r_ref = r
    else:
        if not np.allclose(r, r_ref):
            raise RuntimeError(f"r-list mismatch in {stem}")

    print("Computing Pearson matrices:", stem, flush=True)

    Rq, Nq = pearson_pairwise_matrix(q)
    Rs, Ns = pearson_pairwise_matrix(s)

    Rq_list.append(Rq)
    Rs_list.append(Rs)
    Nq_list.append(Nq)
    Ns_list.append(Ns)

r = r_ref

Rq_arr = np.asarray(Rq_list, dtype=float)
Rs_arr = np.asarray(Rs_list, dtype=float)
Nq_arr = np.asarray(Nq_list, dtype=int)
Ns_arr = np.asarray(Ns_list, dtype=int)

Rq_mean = np.nanmean(Rq_arr, axis=0)
Rs_mean = np.nanmean(Rs_arr, axis=0)

Rq_sem = np.nanstd(Rq_arr, axis=0, ddof=1) / np.sqrt(Rq_arr.shape[0])
Rs_sem = np.nanstd(Rs_arr, axis=0, ddof=1) / np.sqrt(Rs_arr.shape[0])

np.savez(
    OUT_NPZ,
    stems=np.array(stems),
    r=r,
    Rq_arr=Rq_arr,
    Rs_arr=Rs_arr,
    Nq_arr=Nq_arr,
    Ns_arr=Ns_arr,
    Rq_mean=Rq_mean,
    Rs_mean=Rs_mean,
    Rq_sem=Rq_sem,
    Rs_sem=Rs_sem,
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

# Make the figure slightly wider and reserve space at the right for the colorbar.
fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.00))
fig.subplots_adjust(left=0.09, right=0.88, bottom=0.16, top=0.97, wspace=0.28)

vmin = 0.0
vmax = 1.0

im1 = axes[0].imshow(
    Rq_mean,
    origin="lower",
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",
    aspect="equal",
)

im2 = axes[1].imshow(
    Rs_mean,
    origin="lower",
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",
    aspect="equal",
)

for ax, label in [
    (axes[0], r"$q_r$"),
    (axes[1], r"$\sin\theta_r$"),
]:
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$r$")

    ax.set_xticks(np.arange(len(r)))
    ax.set_yticks(np.arange(len(r)))

    ax.set_xticklabels([str(int(x)) for x in r], rotation=45, ha="right")
    ax.set_yticklabels([str(int(x)) for x in r])

    ax.text(
        0.04,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
    )

# Dedicated colorbar axis completely outside the two panels.
cax = fig.add_axes([0.90, 0.22, 0.022, 0.56])
cbar = fig.colorbar(im2, cax=cax)
cbar.set_label("correlation")

fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUT_PDF)
print("Saved:", OUT_PNG)
print("Saved:", OUT_NPZ)

print("\nMean off-diagonal correlations:")
for name, M in [("q", Rq_mean), ("sin_theta", Rs_mean)]:
    off = M[~np.eye(M.shape[0], dtype=bool)]
    print(
        name,
        "mean offdiag =", np.nanmean(off),
        "min =", np.nanmin(off),
        "max =", np.nanmax(off),
    )