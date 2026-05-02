#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import brentq

BASE = Path("/home/idies/workspace/Storage/elenceq/mhd_work/jhtdb_mhd1024")

STEMS_FILE = BASE / "processed/perpBL_v1/config/perpBL_320_final15_stems_v1.txt"
PER_CUBE = BASE / "processed/perpBL_v1/per_cube"
FIGDIR = BASE / "processed/perpBL_v1/figures"
ENSEMBLE_DIR = BASE / "processed/perpBL_v1/ensemble"

FIGDIR.mkdir(parents=True, exist_ok=True)
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = FIGDIR / "fig_320_final15_fit_a_repr_perpBL_v1.pdf"
OUT_PNG = FIGDIR / "fig_320_final15_fit_a_repr_perpBL_v1.png"
OUT_NPZ = ENSEMBLE_DIR / "perpBL_320_final15_fit_a_repr_summary_v1.npz"

R_TARGETS = [32, 96, 192]
RNG_SEED = 24680
N_SHUFFLE = 20

# Broad bins, consistent with the old figure style.
BIN_EDGES = np.array([0, 50, 80, 95, 100], dtype=float)
BIN_LABELS = ["0-50%", "50-80%", "80-95%", "95-100%"]

stems = [
    line.strip()
    for line in STEMS_FILE.read_text().splitlines()
    if line.strip()
]


def mean_c_cosh_model(a):
    """
    Folded unsigned model:
        rho(theta) ∝ sin(theta) cosh(a cos(theta)), theta in [0, pi/2].

    With c = cos(theta) in [0,1], density is proportional to cosh(a c).
    This returns E[c].
    """
    a = float(a)

    if abs(a) < 1.0e-6:
        return 0.5 + (a * a) / 24.0

    return 1.0 - (np.cosh(a) - 1.0) / (a * np.sinh(a))


def fit_a_from_mean_c(mean_c):
    """
    Fit nonnegative a from mean c = |cos theta|.
    If sample noise gives mean_c <= 0.5, return a = 0.
    """
    if not np.isfinite(mean_c):
        return np.nan

    mean_c = float(mean_c)

    if mean_c <= 0.5:
        return 0.0

    mean_c = min(mean_c, 1.0 - 1.0e-8)

    def f(a):
        return mean_c_cosh_model(a) - mean_c

    return float(brentq(f, 0.0, 200.0, maxiter=300))


def compute_a_by_bins(q, A, bin_edges):
    """
    Compute fitted a in amplitude-percentile bins.
    """
    mask = np.isfinite(q) & np.isfinite(A) & (A > 0)
    q = q[mask]
    A = A[mask]

    cabs = np.abs(q)

    if A.size == 0:
        nbin = len(bin_edges) - 1
        return (
            np.full(nbin, np.nan, dtype=float),
            np.full(nbin, np.nan, dtype=float),
            np.full(nbin, np.nan, dtype=float),
            np.zeros(nbin, dtype=int),
        )

    cuts = np.percentile(A, bin_edges)

    a_vals = []
    mean_c_vals = []
    mean_theta_vals = []
    counts = []

    for ib in range(len(bin_edges) - 1):
        lo = cuts[ib]
        hi = cuts[ib + 1]

        if ib == 0:
            m = (A >= lo) & (A <= hi)
        else:
            m = (A > lo) & (A <= hi)

        n = int(np.count_nonzero(m))
        counts.append(n)

        if n < 10:
            a_vals.append(np.nan)
            mean_c_vals.append(np.nan)
            mean_theta_vals.append(np.nan)
            continue

        cmean = float(np.mean(cabs[m]))
        theta_mean = float(np.degrees(np.mean(np.arccos(np.clip(cabs[m], 0.0, 1.0)))))

        mean_c_vals.append(cmean)
        mean_theta_vals.append(theta_mean)
        a_vals.append(fit_a_from_mean_c(cmean))

    return (
        np.asarray(a_vals, dtype=float),
        np.asarray(mean_c_vals, dtype=float),
        np.asarray(mean_theta_vals, dtype=float),
        np.asarray(counts, dtype=int),
    )


def mean_sem(a):
    mean = np.nanmean(a, axis=0)
    sem = np.nanstd(a, axis=0, ddof=1) / np.sqrt(a.shape[0])
    return mean, sem


rng = np.random.RandomState(RNG_SEED)

ncube = len(stems)
nrsel = len(R_TARGETS)
nbin = len(BIN_LABELS)

real_a_curves = np.full((ncube, nrsel, nbin), np.nan, dtype=float)
shuf_a_curves = np.full((ncube, nrsel, nbin), np.nan, dtype=float)

real_c_curves = np.full((ncube, nrsel, nbin), np.nan, dtype=float)
shuf_c_curves = np.full((ncube, nrsel, nbin), np.nan, dtype=float)

real_theta_curves = np.full((ncube, nrsel, nbin), np.nan, dtype=float)
shuf_theta_curves = np.full((ncube, nrsel, nbin), np.nan, dtype=float)

count_curves = np.zeros((ncube, nrsel, nbin), dtype=int)

for icube, stem in enumerate(stems):
    npz = PER_CUBE / f"perpBL_320_final15_v1_{stem}.npz"
    if not npz.exists():
        raise FileNotFoundError(npz)

    print(f"Processing cube {icube+1:02d}: {stem}", flush=True)

    data = np.load(npz)

    r = data["r_list"].astype(float)
    q_all = data["q"].astype(float)
    A_all = data["A"].astype(float)

    for irsel, r_target in enumerate(R_TARGETS):
        ir = int(np.argmin(np.abs(r - r_target)))
        if int(r[ir]) != r_target:
            raise RuntimeError(
                f"Could not find r={r_target} in {stem}; nearest is {r[ir]}"
            )

        q = q_all[ir]
        A = A_all[ir]

        a_real, c_real, th_real, counts = compute_a_by_bins(q, A, BIN_EDGES)

        mask = np.isfinite(q) & np.isfinite(A) & (A > 0)
        qv = q[mask]
        Av = A[mask]

        a_shuf_list = []
        c_shuf_list = []
        th_shuf_list = []

        for _ in range(N_SHUFFLE):
            Ash = Av.copy()
            rng.shuffle(Ash)
            a_sh, c_sh, th_sh, _ = compute_a_by_bins(qv, Ash, BIN_EDGES)
            a_shuf_list.append(a_sh)
            c_shuf_list.append(c_sh)
            th_shuf_list.append(th_sh)

        a_shuf = np.nanmean(np.asarray(a_shuf_list, dtype=float), axis=0)
        c_shuf = np.nanmean(np.asarray(c_shuf_list, dtype=float), axis=0)
        th_shuf = np.nanmean(np.asarray(th_shuf_list, dtype=float), axis=0)

        real_a_curves[icube, irsel, :] = a_real
        shuf_a_curves[icube, irsel, :] = a_shuf

        real_c_curves[icube, irsel, :] = c_real
        shuf_c_curves[icube, irsel, :] = c_shuf

        real_theta_curves[icube, irsel, :] = th_real
        shuf_theta_curves[icube, irsel, :] = th_shuf

        count_curves[icube, irsel, :] = counts

real_a_mean = np.full((nrsel, nbin), np.nan, dtype=float)
real_a_sem = np.full((nrsel, nbin), np.nan, dtype=float)
shuf_a_mean = np.full((nrsel, nbin), np.nan, dtype=float)
shuf_a_sem = np.full((nrsel, nbin), np.nan, dtype=float)

real_theta_mean = np.full((nrsel, nbin), np.nan, dtype=float)
real_theta_sem = np.full((nrsel, nbin), np.nan, dtype=float)
shuf_theta_mean = np.full((nrsel, nbin), np.nan, dtype=float)
shuf_theta_sem = np.full((nrsel, nbin), np.nan, dtype=float)

for irsel in range(nrsel):
    real_a_mean[irsel], real_a_sem[irsel] = mean_sem(real_a_curves[:, irsel, :])
    shuf_a_mean[irsel], shuf_a_sem[irsel] = mean_sem(shuf_a_curves[:, irsel, :])

    real_theta_mean[irsel], real_theta_sem[irsel] = mean_sem(real_theta_curves[:, irsel, :])
    shuf_theta_mean[irsel], shuf_theta_sem[irsel] = mean_sem(shuf_theta_curves[:, irsel, :])

np.savez(
    OUT_NPZ,
    stems=np.array(stems),
    r_targets=np.array(R_TARGETS, dtype=int),
    bin_edges=BIN_EDGES,
    bin_labels=np.array(BIN_LABELS),
    real_a_curves=real_a_curves,
    shuf_a_curves=shuf_a_curves,
    real_c_curves=real_c_curves,
    shuf_c_curves=shuf_c_curves,
    real_theta_curves=real_theta_curves,
    shuf_theta_curves=shuf_theta_curves,
    count_curves=count_curves,
    real_a_mean=real_a_mean,
    real_a_sem=real_a_sem,
    shuf_a_mean=shuf_a_mean,
    shuf_a_sem=shuf_a_sem,
    real_theta_mean=real_theta_mean,
    real_theta_sem=real_theta_sem,
    shuf_theta_mean=shuf_theta_mean,
    shuf_theta_sem=shuf_theta_sem,
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

fig, axes = plt.subplots(1, 3, figsize=(6.85, 2.75), sharey=True)

x = np.arange(nbin)

for irsel, (ax, r_target) in enumerate(zip(axes, R_TARGETS)):
    # Faint individual cube curves.
    for y in real_a_curves[:, irsel, :]:
        ax.plot(x, y, color="C0", linewidth=0.60, alpha=0.22)

    for y in shuf_a_curves[:, irsel, :]:
        ax.plot(x, y, color="C1", linewidth=0.60, alpha=0.16, linestyle="--")

    # Bold ensemble means.
    line_real, = ax.plot(
        x,
        real_a_mean[irsel],
        color="C0",
        marker="o",
        linewidth=1.50,
        markersize=3.1,
        label="real",
    )

    line_shuf, = ax.plot(
        x,
        shuf_a_mean[irsel],
        color="C1",
        marker="s",
        linewidth=1.50,
        markersize=3.1,
        label="shuffled null",
    )

    # Black SEM error bars, excluded from legend.
    for mean, sem in [
        (real_a_mean[irsel], real_a_sem[irsel]),
        (shuf_a_mean[irsel], shuf_a_sem[irsel]),
    ]:
        ax.errorbar(
            x,
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

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlabel(r"$A_r$ quantile bin")

    if irsel == 0:
        ax.set_ylabel(r"alignment-bias fit parameter $a$")

    # Small in-panel label instead of a big title.
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

# Shared y-range.
ymin = min(np.nanmin(real_a_curves), np.nanmin(shuf_a_curves), 0.0)
ymax = max(np.nanmax(real_a_curves), np.nanmax(shuf_a_curves))
pad = 0.10 * max(ymax - ymin, 1.0)
for ax in axes:
    ax.set_ylim(max(0.0, ymin - pad), ymax + pad)

# One shared legend outside.
handles = [
    Line2D([0], [0], color="C0", marker="o", linewidth=1.50, markersize=3.1, label="real"),
    Line2D([0], [0], color="C1", marker="s", linewidth=1.50, markersize=3.1, label="shuffled null"),
]
fig.legend(
    handles=handles,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=2,
    handlelength=2.0,
    columnspacing=1.2,
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

print("\nFitted a(A_r) at representative separations:")
for irsel, r_target in enumerate(R_TARGETS):
    print(f"\n=== r = {r_target} ===")
    for ib, lab in enumerate(BIN_LABELS):
        print(
            f"{lab:>8s}  "
            f"real a={real_a_mean[irsel, ib]:.4f} ± {real_a_sem[irsel, ib]:.4f}  "
            f"shuffle a={shuf_a_mean[irsel, ib]:.4f} ± {shuf_a_sem[irsel, ib]:.4f}  "
            f"real theta={real_theta_mean[irsel, ib]:.3f} ± {real_theta_sem[irsel, ib]:.3f} deg"
        )