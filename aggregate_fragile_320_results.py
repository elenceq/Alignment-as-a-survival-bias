from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import csv

BASE = Path("/home/idies/workspace/Storage/elenceq/mhd_work/jhtdb_mhd1024")
IN = BASE / "processed" / "fragile_alignment_exact_320"
OUT = BASE / "processed" / "fragile_alignment_exact_320" / "ensemble"
FIG = BASE / "processed" / "final_figures_new15"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

geo_paths = sorted(IN.glob("Cl_exact_compact_320_*.json"))
sto_paths = sorted(IN.glob("stochastic_compact_320_*.json"))

def sem(a, axis=0):
    a = np.asarray(a, dtype=float)
    return np.nanstd(a, axis=axis, ddof=1) / np.sqrt(np.sum(np.isfinite(a), axis=axis))

def get(row, names):
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(f"Missing keys {names}; available keys = {list(row.keys())}")

# ----------------------------
# Geometric aggregation
# ----------------------------
geo_rows = []
geo_stems = []
for p in geo_paths:
    d = json.loads(p.read_text())
    stem = p.name.replace("Cl_exact_compact_320_", "").replace(".json", "")
    geo_stems.append(stem)
    rows = d["rows"]
    geo_rows.append({
        "ell": [get(r, ["ell_grid", "ell"]) for r in rows],
        "C_plus": [get(r, ["C_plus"]) for r in rows],
        "C_minus": [get(r, ["C_minus"]) for r in rows],
        "C_avg": [get(r, ["C_avg"]) for r in rows],
    })

ell = np.asarray(geo_rows[0]["ell"], dtype=float)
for gr in geo_rows:
    assert np.allclose(np.asarray(gr["ell"], dtype=float), ell)

Cplus = np.asarray([gr["C_plus"] for gr in geo_rows], dtype=float)
Cminus = np.asarray([gr["C_minus"] for gr in geo_rows], dtype=float)
Cavg = np.asarray([gr["C_avg"] for gr in geo_rows], dtype=float)

geo_summary = {
    "n_cubes": len(geo_paths),
    "ell": ell.tolist(),
    "C_plus_mean": np.nanmean(Cplus, axis=0).tolist(),
    "C_plus_sem": sem(Cplus, axis=0).tolist(),
    "C_minus_mean": np.nanmean(Cminus, axis=0).tolist(),
    "C_minus_sem": sem(Cminus, axis=0).tolist(),
    "C_avg_mean": np.nanmean(Cavg, axis=0).tolist(),
    "C_avg_sem": sem(Cavg, axis=0).tolist(),
    "stems": geo_stems,
}

# ----------------------------
# Stochastic aggregation
# ----------------------------
sto_rows = []
sto_stems = []
for p in sto_paths:
    d = json.loads(p.read_text())
    stem = p.name.replace("stochastic_compact_320_", "").replace(".json", "")
    sto_stems.append(stem)
    rows = d["rows"]
    sto_rows.append({
        "r": [get(r, ["r"]) for r in rows],
        "theta_unw": [get(r, ["theta_unweighted_deg", "theta_unw", "theta_unweighted", "theta_all"]) for r in rows],
        "theta_A": [get(r, ["theta_A_weighted_deg", "theta_A", "theta_weighted", "theta_w"]) for r in rows],
        "theta_null": [get(r, ["theta_shuffled_null_deg", "theta_null", "theta_shuffle", "theta_shuffled"]) for r in rows],
        "cov": [get(r, ["cov_A_sintheta", "cov", "cov_real"]) for r in rows],
        "cov_null": [get(r, ["cov_A_sintheta_null", "nullcov", "cov_null", "null_cov", "cov_shuffle"]) for r in rows],
    })

r = np.asarray(sto_rows[0]["r"], dtype=float)
for sr in sto_rows:
    assert np.allclose(np.asarray(sr["r"], dtype=float), r)

theta_unw = np.asarray([sr["theta_unw"] for sr in sto_rows], dtype=float)
theta_A = np.asarray([sr["theta_A"] for sr in sto_rows], dtype=float)
theta_null = np.asarray([sr["theta_null"] for sr in sto_rows], dtype=float)
cov = np.asarray([sr["cov"] for sr in sto_rows], dtype=float)
cov_null = np.asarray([sr["cov_null"] for sr in sto_rows], dtype=float)

sto_summary = {
    "n_cubes": len(sto_paths),
    "r": r.tolist(),
    "theta_unw_mean": np.nanmean(theta_unw, axis=0).tolist(),
    "theta_unw_sem": sem(theta_unw, axis=0).tolist(),
    "theta_A_mean": np.nanmean(theta_A, axis=0).tolist(),
    "theta_A_sem": sem(theta_A, axis=0).tolist(),
    "theta_null_mean": np.nanmean(theta_null, axis=0).tolist(),
    "theta_null_sem": sem(theta_null, axis=0).tolist(),
    "cov_mean": np.nanmean(cov, axis=0).tolist(),
    "cov_sem": sem(cov, axis=0).tolist(),
    "cov_null_mean": np.nanmean(cov_null, axis=0).tolist(),
    "cov_null_sem": sem(cov_null, axis=0).tolist(),
    "stems": sto_stems,
}

summary = {"geometric": geo_summary, "stochastic": sto_summary}
json_path = OUT / "fragile_alignment_320_ensemble_summary.json"
json_path.write_text(json.dumps(summary, indent=2))

# CSV tables
geo_csv = OUT / "fragile_alignment_320_geometric_summary.csv"
with geo_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ell", "C_plus_mean", "C_plus_sem", "C_minus_mean", "C_minus_sem", "C_avg_mean", "C_avg_sem"])
    for i in range(len(ell)):
        w.writerow([
            ell[i],
            geo_summary["C_plus_mean"][i], geo_summary["C_plus_sem"][i],
            geo_summary["C_minus_mean"][i], geo_summary["C_minus_sem"][i],
            geo_summary["C_avg_mean"][i], geo_summary["C_avg_sem"][i],
        ])

sto_csv = OUT / "fragile_alignment_320_stochastic_summary.csv"
with sto_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["r", "theta_unw_mean", "theta_unw_sem", "theta_A_mean", "theta_A_sem", "theta_null_mean", "theta_null_sem", "cov_mean", "cov_sem", "cov_null_mean", "cov_null_sem"])
    for i in range(len(r)):
        w.writerow([
            r[i],
            sto_summary["theta_unw_mean"][i], sto_summary["theta_unw_sem"][i],
            sto_summary["theta_A_mean"][i], sto_summary["theta_A_sem"][i],
            sto_summary["theta_null_mean"][i], sto_summary["theta_null_sem"][i],
            sto_summary["cov_mean"][i], sto_summary["cov_sem"][i],
            sto_summary["cov_null_mean"][i], sto_summary["cov_null_sem"][i],
        ])

# ----------------------------
# Figure 1: geometric C_l ensemble
# ----------------------------
fig = plt.figure(figsize=(6.2, 4.6))
for arr in Cavg:
    plt.loglog(ell, arr, marker="o", lw=0.8, alpha=0.25)

plt.errorbar(ell, geo_summary["C_avg_mean"], yerr=geo_summary["C_avg_sem"],
             marker="o", lw=2.2, capsize=3, label=r"$C_\ell$ ensemble mean")

# h=1/3 reference, normalized to the mean curve at ell=80 if available
p = -5.0/3.0
ref = ell**p
idx = int(np.argmin(np.abs(ell - 80)))
ref = ref * (geo_summary["C_avg_mean"][idx] / ref[idx])
plt.loglog(ell, ref, ls="--", lw=1.8, label=r"reference $\ell^{-5/3}$")

plt.xlabel(r"coarse-graining scale $\ell$ (grid cells)")
plt.ylabel(r"control coefficient $C_\ell$")
plt.title(r"15 pressure-complete $320^3$ cubes: geometric diagnostic")
plt.legend(frameon=False)
plt.tight_layout()
for path in [FIG / "fragile320_Cl_ensemble.pdf", FIG / "fragile320_Cl_ensemble.png",
             OUT / "fragile320_Cl_ensemble.pdf", OUT / "fragile320_Cl_ensemble.png"]:
    fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)

# ----------------------------
# Figure 2: stochastic mean angles
# ----------------------------
fig = plt.figure(figsize=(6.4, 4.7))
plt.errorbar(r, sto_summary["theta_unw_mean"], yerr=sto_summary["theta_unw_sem"],
             marker="o", lw=2.2, capsize=3, label=r"unweighted $\langle\theta_r\rangle$")
plt.errorbar(r, sto_summary["theta_A_mean"], yerr=sto_summary["theta_A_sem"],
             marker="s", lw=2.2, capsize=3, label=r"$A_r$-weighted")
plt.errorbar(r, sto_summary["theta_null_mean"], yerr=sto_summary["theta_null_sem"],
             marker="^", lw=2.0, ls="--", capsize=3, label=r"shuffled-null weighted")
plt.axhline(57.29577951308232, ls=":", lw=1.8, label=r"random $57.3^\circ$")
plt.xlabel(r"separation $r$ (grid cells)")
plt.ylabel(r"mean unsigned angle (deg)")
plt.title(r"15 pressure-complete $320^3$ cubes: stochastic angle summary")
plt.legend(frameon=False)
plt.tight_layout()
for path in [FIG / "fragile320_stochastic_angles_ensemble.pdf", FIG / "fragile320_stochastic_angles_ensemble.png",
             OUT / "fragile320_stochastic_angles_ensemble.pdf", OUT / "fragile320_stochastic_angles_ensemble.png"]:
    fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)

# ----------------------------
# Figure 3: covariance
# ----------------------------
fig = plt.figure(figsize=(6.4, 4.5))
plt.errorbar(r, sto_summary["cov_mean"], yerr=sto_summary["cov_sem"],
             marker="D", lw=2.2, capsize=3, label=r"real $\mathrm{Cov}(A_r,\sin\theta_r)$")
plt.errorbar(r, sto_summary["cov_null_mean"], yerr=sto_summary["cov_null_sem"],
             marker="D", lw=2.0, ls="--", capsize=3, label=r"shuffled null")
plt.axhline(0.0, ls=":", lw=1.5)
plt.xlabel(r"separation $r$ (grid cells)")
plt.ylabel(r"covariance")
plt.title(r"15 pressure-complete $320^3$ cubes: angle--amplitude covariance")
plt.legend(frameon=False)
plt.tight_layout()
for path in [FIG / "fragile320_stochastic_covariance_ensemble.pdf", FIG / "fragile320_stochastic_covariance_ensemble.png",
             OUT / "fragile320_stochastic_covariance_ensemble.pdf", OUT / "fragile320_stochastic_covariance_ensemble.png"]:
    fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:")
print(" ", json_path)
print(" ", geo_csv)
print(" ", sto_csv)
print(" ", FIG / "fragile320_Cl_ensemble.pdf")
print(" ", FIG / "fragile320_stochastic_angles_ensemble.pdf")
print(" ", FIG / "fragile320_stochastic_covariance_ensemble.pdf")
