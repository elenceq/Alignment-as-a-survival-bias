#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import rankdata

BASE = Path("/home/idies/workspace/Storage/elenceq/mhd_work/jhtdb_mhd1024")

STEMS_FILE = BASE / "processed/perpBL_v1/config/perpBL_320_final15_stems_v1.txt"
PER_CUBE = BASE / "processed/perpBL_v1/per_cube"
FIGDIR = BASE / "processed/perpBL_v1/figures"
ENSEMBLE_DIR = BASE / "processed/perpBL_v1/ensemble"

FIGDIR.mkdir(parents=True, exist_ok=True)
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = FIGDIR / "fig_320_final15_decay_perpBL_v1.pdf"
OUT_PNG = FIGDIR / "fig_320_final15_decay_perpBL_v1.png"
OUT_NPZ = ENSEMBLE_DIR / "perpBL_320_final15_decay_summary_v1.npz"

stems = [
    line.strip()
    for line in STEMS_FILE.read_text().splitlines()
    if line.strip()
]


def pearson_pair(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 3:
        return np.nan

    a = x[mask].astype(float)
    b = y[mask].astype(float)

    a = a - np.mean(a)
    b = b - np.mean(b)

    da = np.sqrt(np.sum(a * a))
    db = np.sqrt(np.sum(b * b))

    if da == 0.0 or db == 0.0:
        return np.nan

    return float(np.sum(a * b) / (da * db))


def spearman_pair(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 3:
        return np.nan

    a = rankdata(x[mask])
    b = rankdata(y[mask])

    a = a - np.mean(a)
    b = b - np.mean(b)

    da = np.sqrt(np.sum(a * a))
    db = np.sqrt(np.sum(b * b))

    if da == 0.0 or db == 0.0:
        return np.nan

    return float(np.sum(a * b) / (da * db))


r_ref = None

# Per-cube, per-pair records.
records = []

for icube, stem in enumerate(stems, 1):
    npz = PER_CUBE / f"perpBL_320_final15_v1_{stem}.npz"
    if not npz.exists():
        raise FileNotFoundError(npz)

    print(f"Processing cube {icube:02d}: {stem}", flush=True)

    data = np.load(npz)

    r = data["r_list"].astype(float)
    q = data["q"].astype(float)
    s = data["sin_theta"].astype(float)

    if r_ref is None:
        r_ref = r
    else:
        if not np.allclose(r, r_ref):
            raise RuntimeError(f"r-list mismatch in {stem}")

    nr = len(r)

    for i in range(nr):
        for j in range(i + 1, nr):
            log_ratio = np.log2(r[j] / r[i])
            index_gap = j - i

            records.append({
                "cube": icube,
                "stem": stem,
                "i": i,
                "j": j,
                "r_i": int(r[i]),
                "r_j": int(r[j]),
                "index_gap": index_gap,
                "log2_ratio": log_ratio,
                "q_pearson": pearson_pair(q[i], q[j]),
                "q_spearman": spearman_pair(q[i], q[j]),
                "s_pearson": pearson_pair(s[i], s[j]),
                "s_spearman": spearman_pair(s[i], s[j]),
            })

r = r_ref
gaps = np.array(sorted(set(rec["index_gap"] for rec in records)), dtype=int)

def collect_by_gap(key):
    out = []
    sem = []
    xval = []

    for gap in gaps:
        vals = np.array([rec[key] for rec in records if rec["index_gap"] == gap], dtype=float)
        vals = vals[np.isfinite(vals)]

        # x-axis value: mean log2(r_j/r_i) over all pairs with this index gap.
        xs = np.array([rec["log2_ratio"] for rec in records if rec["index_gap"] == gap], dtype=float)

        out.append(np.mean(vals))
        sem.append(np.std(vals, ddof=1) / np.sqrt(vals.size))
        xval.append(np.mean(xs))

    return np.asarray(xval), np.asarray(out), np.asarray(sem)

x_qP, qP_mean, qP_sem = collect_by_gap("q_pearson")
x_qS, qS_mean, qS_sem = collect_by_gap("q_spearman")
x_sP, sP_mean, sP_sem = collect_by_gap("s_pearson")
x_sS, sS_mean, sS_sem = collect_by_gap("s_spearman")

np.savez(
    OUT_NPZ,
    stems=np.array(stems),
    r=r,
    gaps=gaps,
    records=np.array(records, dtype=object),
    x_qP=x_qP,
    qP_mean=qP_mean,
    qP_sem=qP_sem,
    qS_mean=qS_mean,
    qS_sem=qS_sem,
    sP_mean=sP_mean,
    sP_sem=sP_sem,
    sS_mean=sS_mean,
    sS_sem=sS_sem,
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

fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.75), sharey=True)

ax = axes[0]
line_qP = ax.errorbar(
    x_qP,
    qP_mean,
    yerr=qP_sem,
    marker="o",
    linewidth=1.45,
    markersize=3.0,
    capsize=2.0,
    label="Pearson",
)
line_qS = ax.errorbar(
    x_qS,
    qS_mean,
    yerr=qS_sem,
    marker="s",
    linewidth=1.45,
    markersize=3.0,
    capsize=2.0,
    label="Spearman",
)

ax.axhline(0.0, linestyle="--", linewidth=0.9, color="black", alpha=0.85)
ax.set_xlabel(r"$\log_2(r_j/r_i)$")
ax.set_ylabel("correlation")
ax.text(
    0.05,
    0.95,
    r"$q_r$",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.3),
)

ax = axes[1]
ax.errorbar(
    x_sP,
    sP_mean,
    yerr=sP_sem,
    marker="o",
    linewidth=1.45,
    markersize=3.0,
    capsize=2.0,
    label="Pearson",
)
ax.errorbar(
    x_sS,
    sS_mean,
    yerr=sS_sem,
    marker="s",
    linewidth=1.45,
    markersize=3.0,
    capsize=2.0,
    label="Spearman",
)

ax.axhline(0.0, linestyle="--", linewidth=0.9, color="black", alpha=0.85)
ax.set_xlabel(r"$\log_2(r_j/r_i)$")
ax.text(
    0.05,
    0.95,
    r"$s_r$",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.3),
)

handles = [
    Line2D([0], [0], color="C0", marker="o", linewidth=1.45, markersize=3.0, label="Pearson"),
    Line2D([0], [0], color="C1", marker="s", linewidth=1.45, markersize=3.0, label="Spearman"),
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

print("\nDecay summary:")
for k, gap in enumerate(gaps):
    print(
        f"gap={gap:2d}, mean log2 ratio={x_qP[k]:.3f}: "
        f"q Pearson={qP_mean[k]:.4f}±{qP_sem[k]:.4f}, "
        f"q Spearman={qS_mean[k]:.4f}±{qS_sem[k]:.4f}, "
        f"s Pearson={sP_mean[k]:.4f}±{sP_sem[k]:.4f}, "
        f"s Spearman={sS_mean[k]:.4f}±{sS_sem[k]:.4f}"
    )