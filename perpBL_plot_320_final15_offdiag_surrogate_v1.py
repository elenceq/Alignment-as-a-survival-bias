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

OUT_PDF = FIGDIR / "fig_320_final15_offdiag_surrogate_perpBL_v1.pdf"
OUT_PNG = FIGDIR / "fig_320_final15_offdiag_surrogate_perpBL_v1.png"
OUT_NPZ = ENSEMBLE_DIR / "perpBL_320_final15_offdiag_surrogate_summary_v1.npz"

N_SHUFFLE = 20
RNG_SEED = 24680

stems = [
    line.strip()
    for line in STEMS_FILE.read_text().splitlines()
    if line.strip()
]


def pearson_matrix_fast(X):
    """
    Fast Pearson matrix for X with shape (nr, nevent).

    Assumes entries are finite or mostly finite. If NaNs appear, use
    pairwise finite masks.
    """
    X = np.asarray(X, dtype=float)
    nr = X.shape[0]

    if np.isfinite(X).all():
        Y = X - X.mean(axis=1, keepdims=True)
        denom = np.sqrt(np.sum(Y * Y, axis=1))
        denom[denom == 0.0] = np.nan
        Y = Y / denom[:, None]
        return Y @ Y.T

    R = np.full((nr, nr), np.nan, dtype=float)
    for i in range(nr):
        xi = X[i]
        for j in range(nr):
            xj = X[j]
            mask = np.isfinite(xi) & np.isfinite(xj)
            if np.count_nonzero(mask) < 3:
                continue
            a = xi[mask] - np.mean(xi[mask])
            b = xj[mask] - np.mean(xj[mask])
            da = np.sqrt(np.sum(a * a))
            db = np.sqrt(np.sum(b * b))
            if da > 0.0 and db > 0.0:
                R[i, j] = np.sum(a * b) / (da * db)
    return R


def mean_offdiag(R):
    mask = ~np.eye(R.shape[0], dtype=bool)
    return float(np.nanmean(R[mask]))


def full_shuffle_rows(X, rng):
    """
    Shuffle each scale row independently. This destroys cross-scale
    correspondence while preserving each scale's one-point distribution.
    """
    Y = np.empty_like(X)
    for i in range(X.shape[0]):
        Y[i] = X[i, rng.permutation(X.shape[1])]
    return Y


rng = np.random.RandomState(RNG_SEED)

real_q = []
real_s = []
fullshuffle_q = []
fullshuffle_s = []

r_ref = None

print("Computing mean off-diagonal cross-scale correlations")
print("N_SHUFFLE =", N_SHUFFLE)

for icube, stem in enumerate(stems, 1):
    npz = PER_CUBE / f"perpBL_320_final15_v1_{stem}.npz"
    if not npz.exists():
        raise FileNotFoundError(npz)

    print(f"\nCube {icube:02d}: {stem}", flush=True)

    data = np.load(npz)

    r = data["r_list"].astype(float)
    q = data["q"].astype(float)
    s = data["sin_theta"].astype(float)

    if r_ref is None:
        r_ref = r
    else:
        if not np.allclose(r, r_ref):
            raise RuntimeError(f"r-list mismatch in {stem}")

    Rq = pearson_matrix_fast(q)
    Rs = pearson_matrix_fast(s)

    rq_real = mean_offdiag(Rq)
    rs_real = mean_offdiag(Rs)

    q_null_vals = []
    s_null_vals = []

    for k in range(N_SHUFFLE):
        q_shuf = full_shuffle_rows(q, rng)
        s_shuf = full_shuffle_rows(s, rng)

        q_null_vals.append(mean_offdiag(pearson_matrix_fast(q_shuf)))
        s_null_vals.append(mean_offdiag(pearson_matrix_fast(s_shuf)))

    rq_null = float(np.mean(q_null_vals))
    rs_null = float(np.mean(s_null_vals))

    real_q.append(rq_real)
    real_s.append(rs_real)
    fullshuffle_q.append(rq_null)
    fullshuffle_s.append(rs_null)

    print(f"  q_r real mean offdiag           = {rq_real:.6f}")
    print(f"  q_r full-shuffle mean offdiag   = {rq_null:.6f}")
    print(f"  sin real mean offdiag           = {rs_real:.6f}")
    print(f"  sin full-shuffle mean offdiag   = {rs_null:.6f}")

real_q = np.asarray(real_q, dtype=float)
real_s = np.asarray(real_s, dtype=float)
fullshuffle_q = np.asarray(fullshuffle_q, dtype=float)
fullshuffle_s = np.asarray(fullshuffle_s, dtype=float)

def mean_sem(x):
    return float(np.nanmean(x)), float(np.nanstd(x, ddof=1) / np.sqrt(x.size))

labels = [
    r"$q_r$ real",
    r"$q_r$ full shuffle",
    r"$\sin\theta_r$ real",
    r"$\sin\theta_r$ full shuffle",
]

groups = [real_q, fullshuffle_q, real_s, fullshuffle_s]
means = np.array([mean_sem(g)[0] for g in groups])
sems = np.array([mean_sem(g)[1] for g in groups])

np.savez(
    OUT_NPZ,
    stems=np.array(stems),
    r=r_ref,
    N_SHUFFLE=np.array([N_SHUFFLE], dtype=int),
    real_q=real_q,
    real_s=real_s,
    fullshuffle_q=fullshuffle_q,
    fullshuffle_s=fullshuffle_s,
    means=means,
    sems=sems,
    labels=np.array(labels),
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

fig, ax = plt.subplots(figsize=(3.45, 2.55))

x = np.arange(4)

# Individual cube points, jittered.
rng_plot = np.random.RandomState(13579)
for i, g in enumerate(groups):
    jitter = 0.08 * rng_plot.randn(g.size)
    ax.plot(
        np.full(g.size, x[i]) + jitter,
        g,
        marker="o",
        linestyle="none",
        markersize=2.2,
        alpha=0.35,
        color="C0" if i in [0, 2] else "C1",
        label="_nolegend_",
    )

# Ensemble mean and SEM.
ax.errorbar(
    x,
    means,
    yerr=sems,
    fmt="o",
    markersize=4.0,
    color="black",
    ecolor="black",
    elinewidth=0.8,
    capsize=2.5,
    capthick=0.8,
    label="_nolegend_",
)

ax.axhline(0.0, linestyle="--", linewidth=0.9, color="black", alpha=0.85)

ax.set_ylabel("mean off-diagonal correlation")
ax.set_xticks(x)
ax.set_xticklabels(
    [
        r"$q_r$" + "\nreal",
        r"$q_r$" + "\nshuffle",
        r"$\sin\theta_r$" + "\nreal",
        r"$\sin\theta_r$" + "\nshuffle",
    ]
)

ymin = min(np.nanmin([np.nanmin(g) for g in groups]), 0.0)
ymax = max(np.nanmax([np.nanmax(g) for g in groups]), 0.0)
pad = 0.15 * max(abs(ymin), abs(ymax), 1.0e-3)
ax.set_ylim(ymin - pad, ymax + pad)

fig.tight_layout(pad=0.45)

fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nSaved:", OUT_PDF)
print("Saved:", OUT_PNG)
print("Saved:", OUT_NPZ)

print("\nEnsemble summary:")
for lab, m, e in zip(labels, means, sems):
    print(f"{lab:30s}  {m:.6f} ± {e:.6f}")