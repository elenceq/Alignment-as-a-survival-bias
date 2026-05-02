#!/usr/bin/env python3

from pathlib import Path
import itertools
import numpy as np
import json
from scipy.stats import rankdata

BASE = Path("/home/idies/workspace/Storage/elenceq/mhd_work/jhtdb_mhd1024")

STEMS_FILE = BASE / "processed/perpBL_v1/config/perpBL_320_final15_stems_v1.txt"
PER_CUBE = BASE / "processed/perpBL_v1/per_cube"
OUTDIR = BASE / "processed/perpBL_v1/tables"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_TXT = OUTDIR / "perpBL_tables_1_2_values_v1.txt"
OUT_TEX = OUTDIR / "perpBL_tables_1_2_latex_v1.tex"
OUT_NPZ = OUTDIR / "perpBL_tables_1_2_values_v1.npz"

BOOT = 20000
N_SHUFFLE = 20
NBIN = 10
RNG_SEED = 24680

PAIR_LIST = [(64, 96), (48, 160)]

stems = [
    line.strip()
    for line in STEMS_FILE.read_text().splitlines()
    if line.strip()
]


def pearson_matrix_fast(X):
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


def spearman_pair(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 3:
        return np.nan

    rx = rankdata(x[mask])
    ry = rankdata(y[mask])

    rx = rx - np.mean(rx)
    ry = ry - np.mean(ry)

    dx = np.sqrt(np.sum(rx * rx))
    dy = np.sqrt(np.sum(ry * ry))

    if dx == 0.0 or dy == 0.0:
        return np.nan

    return float(np.sum(rx * ry) / (dx * dy))


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


def mean_offdiag(R):
    mask = ~np.eye(R.shape[0], dtype=bool)
    return float(np.nanmean(R[mask]))


def full_shuffle_rows(X, rng):
    Y = np.empty_like(X)
    for i in range(X.shape[0]):
        Y[i] = X[i, rng.permutation(X.shape[1])]
    return Y


def percentile_bin_ids(values, nbin):
    values = np.asarray(values, dtype=float)
    ids = np.full(values.shape, -1, dtype=np.int16)

    mask = np.isfinite(values)
    v = values[mask]

    if v.size == 0:
        return ids

    edges = np.percentile(v, np.linspace(0.0, 100.0, nbin + 1))

    # Guard against duplicate percentile edges.
    for k in range(nbin):
        lo = edges[k]
        hi = edges[k + 1]

        if k == 0:
            m = mask & (values >= lo) & (values <= hi)
        else:
            m = mask & (values > lo) & (values <= hi)

        ids[m] = k

    # Any finite points missed by numerical equality go to nearest valid bin.
    missing = mask & (ids < 0)
    if np.any(missing):
        ids[missing] = np.clip(np.searchsorted(edges, values[missing], side="right") - 1, 0, nbin - 1)

    return ids


def bin_shuffle_rows(X, bin_values, rng, nbin):
    """
    Shuffle each row of X independently within percentile bins of bin_values.

    X shape: (nr, nevent)

    bin_values may be:
      - shape (nr, nevent), e.g. A_r per scale
      - shape (nevent,), e.g. |j| at midpoint repeated over directions
    """
    X = np.asarray(X)
    Y = np.empty_like(X)
    nr = X.shape[0]

    if bin_values.ndim == 1:
        bin_ids_shared = percentile_bin_ids(bin_values, nbin)

    for i in range(nr):
        if bin_values.ndim == 1:
            bin_ids = bin_ids_shared
        else:
            bin_ids = percentile_bin_ids(bin_values[i], nbin)

        y = X[i].copy()

        for b in range(nbin):
            idx = np.where(bin_ids == b)[0]
            if idx.size > 1:
                y[idx] = y[rng.permutation(idx)]

        Y[i] = y

    return Y


def bootstrap_ci(values, rng, nboot=BOOT):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan, np.nan

    n = values.size
    boots = np.empty(nboot, dtype=float)

    for k in range(nboot):
        idx = rng.randint(0, n, size=n)
        boots[k] = np.mean(values[idx])

    mean = float(np.mean(values))
    lo, hi = np.percentile(boots, [2.5, 97.5])

    return mean, float(lo), float(hi)


def exact_signflip_p(values):
    """
    Exact two-sided cube-level sign-flip randomization p-value for mean != 0.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = values.size
    obs = abs(np.mean(values))

    count = 0
    total = 2 ** n

    for signs in itertools.product([-1.0, 1.0], repeat=n):
        m = abs(np.mean(values * np.asarray(signs)))
        if m >= obs - 1.0e-15:
            count += 1

    return count / total


def fmt_ci(mean, lo, hi, nd=3):
    return f"{mean:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def fmt_ci_deg(mean, lo, hi):
    return f"{mean:.2f}\\degree\\ [{lo:.2f}, {hi:.2f}]\\degree"


def fmt_p(p):
    if p < 1.0e-3:
        return f"{p:.2e}".replace("e-0", r"\times 10^{-").replace("e-", r"\times 10^{-") + "}"
    return f"{p:.3f}"


rng = np.random.RandomState(RNG_SEED)

# ------------------------------------------------------------------
# Per-cube quantities for Table I
# ------------------------------------------------------------------

theta_all_cube = []
theta_topA_cube = []
theta_topj_cube = []

Rc_real_cube = []
Rs_real_cube = []

Rc_full_cube = []
Rs_full_cube = []

Rc_Abin_cube = []
Rs_Abin_cube = []

Rc_jbin_cube = []
Rs_jbin_cube = []

# ------------------------------------------------------------------
# Per-cube quantities for Table II
# ------------------------------------------------------------------

pair_rows = {}
for obs in ["c", "s"]:
    for pair in PAIR_LIST:
        pair_rows[(obs, pair, "pearson")] = []
        pair_rows[(obs, pair, "spearman")] = []

print("=" * 100)
print("Making corrected Tables I and II")
print("BOOT =", BOOT)
print("N_SHUFFLE =", N_SHUFFLE)
print("NBIN =", NBIN)
print("=" * 100)

for icube, stem in enumerate(stems, 1):
    print(f"\nCube {icube:02d}: {stem}", flush=True)

    npz = PER_CUBE / f"perpBL_320_final15_v1_{stem}.npz"
    js = PER_CUBE / f"perpBL_320_final15_v1_{stem}_summary.json"

    if not npz.exists():
        raise FileNotFoundError(npz)
    if not js.exists():
        raise FileNotFoundError(js)

    data = np.load(npz)

    r = data["r_list"].astype(float)
    q = data["q"].astype(float)
    s = data["sin_theta"].astype(float)
    A = data["A"].astype(float)
    j_rep = data["j_rep"].astype(float)

    with open(js, "r") as f:
        summary = json.load(f)

    rows = summary["summary"]

    theta_all_cube.append(np.mean([row["mean_angle_all_deg"] for row in rows]))
    theta_topA_cube.append(np.mean([row["mean_angle_top10_A_deg"] for row in rows]))
    theta_topj_cube.append(np.mean([row["mean_angle_top10_j_deg"] for row in rows]))

    Rq = pearson_matrix_fast(q)
    Rs = pearson_matrix_fast(s)

    Rc_real_cube.append(mean_offdiag(Rq))
    Rs_real_cube.append(mean_offdiag(Rs))

    # Full shuffle and bin shuffles.
    rc_full_vals = []
    rs_full_vals = []
    rc_Abin_vals = []
    rs_Abin_vals = []
    rc_jbin_vals = []
    rs_jbin_vals = []

    for ishuf in range(N_SHUFFLE):
        q_full = full_shuffle_rows(q, rng)
        s_full = full_shuffle_rows(s, rng)

        q_Abin = bin_shuffle_rows(q, A, rng, NBIN)
        s_Abin = bin_shuffle_rows(s, A, rng, NBIN)

        q_jbin = bin_shuffle_rows(q, j_rep, rng, NBIN)
        s_jbin = bin_shuffle_rows(s, j_rep, rng, NBIN)

        rc_full_vals.append(mean_offdiag(pearson_matrix_fast(q_full)))
        rs_full_vals.append(mean_offdiag(pearson_matrix_fast(s_full)))

        rc_Abin_vals.append(mean_offdiag(pearson_matrix_fast(q_Abin)))
        rs_Abin_vals.append(mean_offdiag(pearson_matrix_fast(s_Abin)))

        rc_jbin_vals.append(mean_offdiag(pearson_matrix_fast(q_jbin)))
        rs_jbin_vals.append(mean_offdiag(pearson_matrix_fast(s_jbin)))

    Rc_full_cube.append(float(np.mean(rc_full_vals)))
    Rs_full_cube.append(float(np.mean(rs_full_vals)))

    Rc_Abin_cube.append(float(np.mean(rc_Abin_vals)))
    Rs_Abin_cube.append(float(np.mean(rs_Abin_vals)))

    Rc_jbin_cube.append(float(np.mean(rc_jbin_vals)))
    Rs_jbin_cube.append(float(np.mean(rs_jbin_vals)))

    print(f"  Rc real/full/Abin/jbin = {Rc_real_cube[-1]:.6f}, {Rc_full_cube[-1]:.6f}, {Rc_Abin_cube[-1]:.6f}, {Rc_jbin_cube[-1]:.6f}")
    print(f"  Rs real/full/Abin/jbin = {Rs_real_cube[-1]:.6f}, {Rs_full_cube[-1]:.6f}, {Rs_Abin_cube[-1]:.6f}, {Rs_jbin_cube[-1]:.6f}")

    # Selected pair statistics for Table II.
    for pair in PAIR_LIST:
        r1, r2 = pair
        i1 = int(np.where(r.astype(int) == r1)[0][0])
        i2 = int(np.where(r.astype(int) == r2)[0][0])

        pair_rows[("c", pair, "pearson")].append(pearson_pair(q[i1], q[i2]))
        pair_rows[("c", pair, "spearman")].append(spearman_pair(q[i1], q[i2]))

        pair_rows[("s", pair, "pearson")].append(pearson_pair(s[i1], s[i2]))
        pair_rows[("s", pair, "spearman")].append(spearman_pair(s[i1], s[i2]))


# Convert to arrays.
theta_all_cube = np.asarray(theta_all_cube)
theta_topA_cube = np.asarray(theta_topA_cube)
theta_topj_cube = np.asarray(theta_topj_cube)

Rc_real_cube = np.asarray(Rc_real_cube)
Rs_real_cube = np.asarray(Rs_real_cube)

Rc_full_cube = np.asarray(Rc_full_cube)
Rs_full_cube = np.asarray(Rs_full_cube)

Rc_Abin_cube = np.asarray(Rc_Abin_cube)
Rs_Abin_cube = np.asarray(Rs_Abin_cube)

Rc_jbin_cube = np.asarray(Rc_jbin_cube)
Rs_jbin_cube = np.asarray(Rs_jbin_cube)

# ------------------------------------------------------------------
# Bootstrap Table I
# ------------------------------------------------------------------

rng_boot = np.random.RandomState(98765)

table1_items = [
    ("Mean folded angle, all points (inertial-range average)", theta_all_cube, "deg"),
    (r"Mean folded angle, top 10\% by $A_r$ (inertial-range average)", theta_topA_cube, "deg"),
    (r"Mean folded angle, top 10\% by $|j|$ (inertial-range average)", theta_topj_cube, "deg"),
    (r"Mean off-diagonal $R_c$ (real data)", Rc_real_cube, "num"),
    (r"Mean off-diagonal $R_s$ (real data)", Rs_real_cube, "num"),
    (r"Mean off-diagonal $R_c$ (full shuffle)", Rc_full_cube, "num"),
    (r"Mean off-diagonal $R_s$ (full shuffle)", Rs_full_cube, "num"),
    (r"Mean off-diagonal $R_c$ ($A_r$-bin shuffle)", Rc_Abin_cube, "num"),
    (r"Mean off-diagonal $R_s$ ($A_r$-bin shuffle)", Rs_Abin_cube, "num"),
    (r"Mean off-diagonal $R_c$ ($|j|$-bin shuffle)", Rc_jbin_cube, "num"),
    (r"Mean off-diagonal $R_s$ ($|j|$-bin shuffle)", Rs_jbin_cube, "num"),
]

table1_results = []
for label, values, kind in table1_items:
    mean, lo, hi = bootstrap_ci(values, rng_boot, BOOT)
    table1_results.append((label, mean, lo, hi, kind))

# ------------------------------------------------------------------
# Bootstrap Table II and sign-flip p-values
# ------------------------------------------------------------------

table2_results = []
for obs in ["c", "s"]:
    for pair in PAIR_LIST:
        P_vals = np.asarray(pair_rows[(obs, pair, "pearson")], dtype=float)
        S_vals = np.asarray(pair_rows[(obs, pair, "spearman")], dtype=float)

        P_mean, P_lo, P_hi = bootstrap_ci(P_vals, rng_boot, BOOT)
        S_mean, S_lo, S_hi = bootstrap_ci(S_vals, rng_boot, BOOT)

        pP = exact_signflip_p(P_vals)
        pS = exact_signflip_p(S_vals)

        table2_results.append(
            {
                "obs": obs,
                "pair": pair,
                "pearson_mean": P_mean,
                "pearson_lo": P_lo,
                "pearson_hi": P_hi,
                "spearman_mean": S_mean,
                "spearman_lo": S_lo,
                "spearman_hi": S_hi,
                "pP": pP,
                "pS": pS,
            }
        )

# Save arrays.
np.savez(
    OUT_NPZ,
    stems=np.array(stems),
    theta_all_cube=theta_all_cube,
    theta_topA_cube=theta_topA_cube,
    theta_topj_cube=theta_topj_cube,
    Rc_real_cube=Rc_real_cube,
    Rs_real_cube=Rs_real_cube,
    Rc_full_cube=Rc_full_cube,
    Rs_full_cube=Rs_full_cube,
    Rc_Abin_cube=Rc_Abin_cube,
    Rs_Abin_cube=Rs_Abin_cube,
    Rc_jbin_cube=Rc_jbin_cube,
    Rs_jbin_cube=Rs_jbin_cube,
    table1_results=np.array([(x[0], x[1], x[2], x[3], x[4]) for x in table1_results], dtype=object),
    table2_results=np.array(table2_results, dtype=object),
)

# ------------------------------------------------------------------
# Write plain text and LaTeX
# ------------------------------------------------------------------

lines = []
tex = []

lines.append("TABLE I: corrected local-BL-perpendicular values")
lines.append("")
tex.append(r"% Corrected local-\(B_L\)-perpendicular Table I rows")
tex.append(r"\begin{tabular}{lcc}")
tex.append(r"\hline\hline")
tex.append(r"Quantity & Ensemble mean & 95\% CI \\")
tex.append(r"\hline")

for label, mean, lo, hi, kind in table1_results:
    if kind == "deg":
        value = f"{mean:.2f} deg"
        ci = f"[{lo:.2f}, {hi:.2f}] deg"
        tex_val = f"{mean:.2f}\\degree"
        tex_ci = f"[{lo:.2f}, {hi:.2f}]\\degree"
    else:
        value = f"{mean:.3f}"
        ci = f"[{lo:.3f}, {hi:.3f}]"
        tex_val = f"{mean:.3f}"
        tex_ci = f"[{lo:.3f}, {hi:.3f}]"

    lines.append(f"{label:75s}  {value:15s}  {ci}")
    tex.append(f"{label} & {tex_val} & {tex_ci} \\\\")

tex.append(r"\hline\hline")
tex.append(r"\end{tabular}")
tex.append("")

lines.append("")
lines.append("TABLE II: corrected local-BL-perpendicular values")
lines.append("")

tex.append(r"% Corrected local-\(B_L\)-perpendicular Table II rows")
tex.append(r"\begin{tabular}{lccccccc}")
tex.append(r"\hline\hline")
tex.append(r"Observable & Scale pair & Pearson & Spearman & 95\% CI (Pearson) & 95\% CI (Spearman) & $p_P$ & $p_S$ \\")
tex.append(r"\hline")

for row in table2_results:
    obs_tex = r"$c_r$" if row["obs"] == "c" else r"$s_r$"
    obs_txt = "c_r" if row["obs"] == "c" else "s_r"
    pair = row["pair"]
    pair_tex = f"({pair[0]}, {pair[1]})"

    P = row["pearson_mean"]
    S = row["spearman_mean"]
    Pci = f"[{row['pearson_lo']:.3f}, {row['pearson_hi']:.3f}]"
    Sci = f"[{row['spearman_lo']:.3f}, {row['spearman_hi']:.3f}]"
    pP = row["pP"]
    pS = row["pS"]

    lines.append(
        f"{obs_txt:5s} {pair_tex:10s}  Pearson={P:.3f}  Spearman={S:.3f}  "
        f"CI_P={Pci}  CI_S={Sci}  pP={pP:.6g}  pS={pS:.6g}"
    )

    tex.append(
        f"{obs_tex} & {pair_tex} & {P:.3f} & {S:.3f} & {Pci} & {Sci} & "
        f"{pP:.2e} & {pS:.2e} \\\\"
    )

tex.append(r"\hline\hline")
tex.append(r"\end{tabular}")

OUT_TXT.write_text("\n".join(lines) + "\n")
OUT_TEX.write_text("\n".join(tex) + "\n")

print("\nSaved:", OUT_TXT)
print("Saved:", OUT_TEX)
print("Saved:", OUT_NPZ)

print("\n" + "\n".join(lines))
print("\nLaTeX written to:")
print(OUT_TEX)