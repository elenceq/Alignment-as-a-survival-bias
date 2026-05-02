#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


# ============================================================
# EXPLICIT SETTINGS: FINAL 15-CUBE LOCAL-PERPENDICULAR RUN
# ============================================================

BASE = Path("/home/idies/workspace/Storage/elenceq/mhd_work/jhtdb_mhd1024")
RAW = BASE / "raw"

STEMS_FILE = BASE / "processed/perpBL_v1/config/perpBL_320_final15_stems_v1.txt"

OUTDIR = BASE / "processed/perpBL_v1/per_cube"
OUTDIR.mkdir(parents=True, exist_ok=True)

R_LIST = np.array([32, 40, 48, 64, 80, 96, 128, 160, 192], dtype=np.float64)

N_SAMPLE = 100000
NPHI = 8
SEED0 = 12345

BL_DOWNSAMPLE = 4
CHUNK = 200000
EPS = 1.0e-30

# Set to 1 only if you want a one-cube test first.
# Set to None for all 15 cubes.
MAX_CUBES = None


# ============================================================
# UTILITIES
# ============================================================

def ensure_vec_last(arr):
    if arr.ndim != 4:
        raise ValueError("Expected 4D vector field, got shape %s" % (arr.shape,))
    if arr.shape[-1] == 3:
        return arr
    if arr.shape[0] == 3:
        return np.moveaxis(arr, 0, -1)
    raise ValueError("Cannot identify vector axis in shape %s" % (arr.shape,))


def unit_rows(v):
    n = np.linalg.norm(v, axis=1)
    ok = n > EPS
    out = np.zeros_like(v, dtype=np.float64)
    out[ok] = v[ok] / n[ok, None]
    return out, ok


def trilinear_vec(field, coords):
    nx, ny, nz = field.shape[:3]

    x = np.clip(coords[:, 0], 0.0, nx - 2.000001)
    y = np.clip(coords[:, 1], 0.0, ny - 2.000001)
    z = np.clip(coords[:, 2], 0.0, nz - 2.000001)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    xd = (x - x0).astype(np.float32)
    yd = (y - y0).astype(np.float32)
    zd = (z - z0).astype(np.float32)

    c000 = field[x0, y0, z0, :]
    c100 = field[x1, y0, z0, :]
    c010 = field[x0, y1, z0, :]
    c110 = field[x1, y1, z0, :]
    c001 = field[x0, y0, z1, :]
    c101 = field[x1, y0, z1, :]
    c011 = field[x0, y1, z1, :]
    c111 = field[x1, y1, z1, :]

    wx0 = (1.0 - xd)[:, None]
    wx1 = xd[:, None]
    wy0 = (1.0 - yd)[:, None]
    wy1 = yd[:, None]
    wz0 = (1.0 - zd)[:, None]
    wz1 = zd[:, None]

    c00 = c000 * wx0 + c100 * wx1
    c10 = c010 * wx0 + c110 * wx1
    c01 = c001 * wx0 + c101 * wx1
    c11 = c011 * wx0 + c111 * wx1

    c0 = c00 * wy0 + c10 * wy1
    c1 = c01 * wy0 + c11 * wy1

    return (c0 * wz0 + c1 * wz1).astype(np.float32)


def trilinear_vec_chunked(field, coords, chunk=CHUNK):
    out = np.empty((coords.shape[0], 3), dtype=np.float32)
    for i0 in range(0, coords.shape[0], chunk):
        i1 = min(i0 + chunk, coords.shape[0])
        out[i0:i1] = trilinear_vec(field, coords[i0:i1])
    return out


def block_average_vec(B, ds):
    if ds == 1:
        return B.astype(np.float32, copy=False)

    nx, ny, nz = B.shape[:3]
    nx2 = (nx // ds) * ds
    ny2 = (ny // ds) * ds
    nz2 = (nz // ds) * ds

    Bc = B[:nx2, :ny2, :nz2, :]
    Bds = Bc.reshape(
        nx2 // ds, ds,
        ny2 // ds, ds,
        nz2 // ds, ds,
        3
    ).mean(axis=(1, 3, 5))

    return Bds.astype(np.float32)


def compute_jmag(B):
    print("Computing |j| ...", flush=True)

    Bx = B[..., 0]
    By = B[..., 1]
    Bz = B[..., 2]

    jx = np.gradient(Bz, axis=1) - np.gradient(By, axis=2)
    jy = np.gradient(Bx, axis=2) - np.gradient(Bz, axis=0)
    jz = np.gradient(By, axis=0) - np.gradient(Bx, axis=1)

    jmag = np.sqrt(jx*jx + jy*jy + jz*jz).astype(np.float32)
    print("Computed |j|:", jmag.shape, flush=True)
    return jmag


def perp_basis(bhat):
    n = bhat.shape[0]

    ref = np.zeros((n, 3), dtype=np.float64)
    ref[:, 2] = 1.0

    close = np.abs(bhat[:, 2]) > 0.90
    ref[close, :] = np.array([1.0, 0.0, 0.0])

    e1 = np.cross(bhat, ref)
    e1, ok = unit_rows(e1)

    if np.any(~ok):
        bad = np.where(~ok)[0]
        ref[bad, :] = np.array([0.0, 1.0, 0.0])
        e1[bad, :] = np.cross(bhat[bad, :], ref[bad, :])
        e1[bad, :], _ = unit_rows(e1[bad, :])

    e2 = np.cross(bhat, e1)
    e2, _ = unit_rows(e2)

    return e1.astype(np.float32), e2.astype(np.float32)


def mean_top_fraction(values, score, frac=0.10):
    mask = np.isfinite(values) & np.isfinite(score)
    if np.count_nonzero(mask) == 0:
        return np.nan

    v = values[mask]
    s = score[mask]
    cutoff = np.quantile(s, 1.0 - frac)
    m = s >= cutoff

    if np.count_nonzero(m) == 0:
        return np.nan

    return float(np.mean(v[m]))


def weighted_mean(values, weights):
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if np.count_nonzero(mask) == 0:
        return np.nan
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def covariance(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) == 0:
        return np.nan
    xx = x[mask]
    yy = y[mask]
    return float(np.mean(xx * yy) - np.mean(xx) * np.mean(yy))


# ============================================================
# PER-CUBE RUN
# ============================================================

def run_one_cube(stem, cube_index):
    u_file = RAW / f"mhd1024_velocity_{stem}.npy"
    b_file = RAW / f"mhd1024_magneticfield_{stem}.npy"

    if not u_file.exists():
        raise FileNotFoundError(u_file)
    if not b_file.exists():
        raise FileNotFoundError(b_file)

    out_npz = OUTDIR / f"perpBL_320_final15_v1_{stem}.npz"
    out_json = OUTDIR / f"perpBL_320_final15_v1_{stem}_summary.json"

    if out_npz.exists() and out_json.exists():
        print("\nSKIPPING existing cube:", stem, flush=True)
        print("  existing:", out_npz, flush=True)
        return

    seed = SEED0 + 1000 * cube_index

    print("\n" + "=" * 100, flush=True)
    print(f"CUBE {cube_index:02d}: {stem}", flush=True)
    print("=" * 100, flush=True)
    print("U_FILE:", u_file, flush=True)
    print("B_FILE:", b_file, flush=True)
    print("OUT_NPZ:", out_npz, flush=True)
    print("R_LIST:", R_LIST.astype(int).tolist(), flush=True)
    print("N_SAMPLE:", N_SAMPLE, flush=True)
    print("NPHI:", NPHI, flush=True)
    print("SEED:", seed, flush=True)
    print("BL_DOWNSAMPLE:", BL_DOWNSAMPLE, flush=True)
    print("=" * 100, flush=True)

    print("Loading U ...", flush=True)
    U = ensure_vec_last(np.load(u_file)).astype(np.float32, copy=False)
    print("U shape:", U.shape, U.dtype, flush=True)

    print("Loading B ...", flush=True)
    B = ensure_vec_last(np.load(b_file)).astype(np.float32, copy=False)
    print("B shape:", B.shape, B.dtype, flush=True)

    if U.shape != B.shape:
        raise RuntimeError("U and B shapes differ.")

    nx, ny, nz = B.shape[:3]

    if (nx, ny, nz) != (320, 320, 320):
        raise RuntimeError(f"Expected 320^3 cube, got {(nx, ny, nz)} for {stem}")

    rmax = float(np.max(R_LIST))
    margin = int(np.ceil(0.5 * rmax)) + 3

    print("Centered-increment midpoint margin:", margin, flush=True)
    print("Midpoint box:", (nx - 2*margin, ny - 2*margin, nz - 2*margin), flush=True)

    rng = np.random.RandomState(seed)

    ix = rng.randint(margin, nx - margin, size=N_SAMPLE)
    iy = rng.randint(margin, ny - margin, size=N_SAMPLE)
    iz = rng.randint(margin, nz - margin, size=N_SAMPLE)

    mid = np.column_stack([ix, iy, iz]).astype(np.float32)

    print("First midpoint:", mid[0].tolist(), flush=True)

    jmag = compute_jmag(B)
    j_mid = jmag[ix, iy, iz].astype(np.float32)
    j_rep = np.repeat(j_mid, NPHI)

    print("Block-averaging B for fast B_L ...", flush=True)
    Bds = block_average_vec(B, BL_DOWNSAMPLE)
    print("Bds shape:", Bds.shape, flush=True)

    mid_ds = (mid - 0.5 * (BL_DOWNSAMPLE - 1)) / float(BL_DOWNSAMPLE)

    phis = 2.0 * np.pi * np.arange(NPHI, dtype=np.float64) / float(NPHI)

    summary = []
    all_q = []
    all_theta = []
    all_sin = []
    all_A = []
    all_dB2 = []

    for r in R_LIST:
        print("\n" + "-" * 100, flush=True)
        print("r =", int(r), flush=True)

        sigma_full = 0.5 * float(r)
        sigma_ds = sigma_full / float(BL_DOWNSAMPLE)

        print("Gaussian B_L sigma_full =", sigma_full, "sigma_ds =", sigma_ds, flush=True)

        BLds = np.empty_like(Bds, dtype=np.float32)
        for c in range(3):
            BLds[..., c] = gaussian_filter(
                Bds[..., c],
                sigma=sigma_ds,
                mode="nearest",
                truncate=3.0
            ).astype(np.float32)

        BL_mid = trilinear_vec_chunked(BLds, mid_ds).astype(np.float64)
        bhat, okB = unit_rows(BL_mid)

        e1, e2 = perp_basis(bhat)

        dirs = (
            np.cos(phis)[None, :, None] * e1[:, None, :]
            + np.sin(phis)[None, :, None] * e2[:, None, :]
        ).astype(np.float32)

        dot_test = np.sum(dirs.astype(np.float64) * bhat[:, None, :], axis=2)
        abs_dot = np.abs(dot_test[okB, :])

        mean_abs_dot = float(np.mean(abs_dot))
        max_abs_dot = float(np.max(abs_dot))

        print("mean |rhat · bLhat| =", "%.3e" % mean_abs_dot, flush=True)
        print("max  |rhat · bLhat| =", "%.3e" % max_abs_dot, flush=True)

        dirs_flat = dirs.reshape(N_SAMPLE * NPHI, 3)
        mid_flat = np.repeat(mid, NPHI, axis=0)

        xplus = mid_flat + 0.5 * float(r) * dirs_flat
        xminus = mid_flat - 0.5 * float(r) * dirs_flat

        print("Interpolating centered endpoints ...", flush=True)

        U_plus = trilinear_vec_chunked(U, xplus)
        U_minus = trilinear_vec_chunked(U, xminus)
        B_plus = trilinear_vec_chunked(B, xplus)
        B_minus = trilinear_vec_chunked(B, xminus)

        dU = U_plus - U_minus
        dB = B_plus - B_minus

        dzp = dU + dB
        dzm = dU - dB

        normp = np.linalg.norm(dzp, axis=1)
        normm = np.linalg.norm(dzm, axis=1)

        ok = (normp > 1.0e-12) & (normm > 1.0e-12)
        ok = ok & np.repeat(okB, NPHI)

        q = np.full(N_SAMPLE * NPHI, np.nan, dtype=np.float32)
        q[ok] = np.sum(dzp[ok] * dzm[ok], axis=1) / (normp[ok] * normm[ok])
        q = np.clip(q, -1.0, 1.0)

        theta_deg = np.full(N_SAMPLE * NPHI, np.nan, dtype=np.float32)
        theta_deg[ok] = np.degrees(np.arccos(np.abs(q[ok]))).astype(np.float32)

        sin_theta = np.full(N_SAMPLE * NPHI, np.nan, dtype=np.float32)
        sin_theta[ok] = np.sqrt(np.maximum(0.0, 1.0 - q[ok]*q[ok])).astype(np.float32)

        A = np.full(N_SAMPLE * NPHI, np.nan, dtype=np.float32)
        A[ok] = (normp[ok] * normm[ok]).astype(np.float32)

        dB2 = np.full(N_SAMPLE * NPHI, np.nan, dtype=np.float32)
        dB2[ok] = np.sum(dB[ok] * dB[ok], axis=1).astype(np.float32)

        mean_all = float(np.nanmean(theta_deg))
        mean_topA = mean_top_fraction(theta_deg, A, frac=0.10)
        mean_topj = mean_top_fraction(theta_deg, j_rep, frac=0.10)

        mean_sin = float(np.nanmean(sin_theta))
        A_weighted_sin = weighted_mean(sin_theta, A)
        cov_A_sin = covariance(A, sin_theta)
        mean_A = float(np.nanmean(A))
        norm_cov = cov_A_sin / mean_A if mean_A != 0 else np.nan

        n_valid = int(np.count_nonzero(ok))
        n_total = int(N_SAMPLE * NPHI)

        print("valid events =", n_valid, "/", n_total, flush=True)
        print("mean angle all       = %.3f deg" % mean_all, flush=True)
        print("mean angle top 10%% A = %.3f deg" % mean_topA, flush=True)
        print("mean angle top 10%% j = %.3f deg" % mean_topj, flush=True)
        print("<sin theta>          = %.6f" % mean_sin, flush=True)
        print("<A sin theta>/<A>    = %.6f" % A_weighted_sin, flush=True)
        print("Cov(A,sin theta)/<A> = %.6e" % norm_cov, flush=True)

        summary.append({
            "r": int(r),
            "sigma_full": float(sigma_full),
            "sigma_ds": float(sigma_ds),
            "mean_abs_rhat_dot_bhat": mean_abs_dot,
            "max_abs_rhat_dot_bhat": max_abs_dot,
            "n_valid": n_valid,
            "n_total": n_total,
            "mean_angle_all_deg": mean_all,
            "mean_angle_top10_A_deg": mean_topA,
            "mean_angle_top10_j_deg": mean_topj,
            "mean_sin_theta": mean_sin,
            "A_weighted_mean_sin_theta": A_weighted_sin,
            "cov_A_sin_theta": cov_A_sin,
            "mean_A": mean_A,
            "normalized_cov_A_sin_theta": norm_cov
        })

        all_q.append(q)
        all_theta.append(theta_deg)
        all_sin.append(sin_theta)
        all_A.append(A)
        all_dB2.append(dB2)

        del BLds, BL_mid, bhat, e1, e2, dirs, dot_test, abs_dot
        del dirs_flat, mid_flat, xplus, xminus
        del U_plus, U_minus, B_plus, B_minus
        del dU, dB, dzp, dzm, normp, normm

    print("\nSaving cube outputs ...", flush=True)

    np.savez(
        out_npz,
        stem=np.array([stem]),
        r_list=R_LIST.astype(np.float32),
        phis=phis.astype(np.float32),
        mid=mid.astype(np.float32),
        q=np.asarray(all_q, dtype=np.float32),
        theta_deg=np.asarray(all_theta, dtype=np.float32),
        sin_theta=np.asarray(all_sin, dtype=np.float32),
        A=np.asarray(all_A, dtype=np.float32),
        dB2=np.asarray(all_dB2, dtype=np.float32),
        j_mid=j_mid.astype(np.float32),
        j_rep=j_rep.astype(np.float32),
        n_sample=np.array([N_SAMPLE], dtype=np.int64),
        nphi=np.array([NPHI], dtype=np.int64),
        seed=np.array([seed], dtype=np.int64),
        bl_downsample=np.array([BL_DOWNSAMPLE], dtype=np.int64),
    )

    with open(out_json, "w") as f:
        json.dump({
            "description": "perpBL final15 320^3 cube",
            "stem": stem,
            "cube_index": cube_index,
            "increment": "centered local-BL-perpendicular",
            "sigma_rule": "sigma_B(r)=r/2",
            "U_FILE": str(u_file),
            "B_FILE": str(b_file),
            "R_LIST": R_LIST.astype(int).tolist(),
            "N_SAMPLE": N_SAMPLE,
            "NPHI": NPHI,
            "SEED": seed,
            "BL_DOWNSAMPLE": BL_DOWNSAMPLE,
            "summary": summary,
            "OUT_NPZ": str(out_npz)
        }, f, indent=2)

    print("Saved:", out_npz, flush=True)
    print("Saved:", out_json, flush=True)
    print("Done cube:", stem, flush=True)

    del U, B, jmag, j_mid, j_rep, Bds
    del all_q, all_theta, all_sin, all_A, all_dB2


# ============================================================
# MAIN
# ============================================================

stems = [line.strip() for line in STEMS_FILE.read_text().splitlines() if line.strip()]

if MAX_CUBES is not None:
    stems = stems[:MAX_CUBES]

print("=" * 100, flush=True)
print("perpBL final15 320^3 batch run", flush=True)
print("=" * 100, flush=True)
print("Number of cubes:", len(stems), flush=True)
print("N_SAMPLE:", N_SAMPLE, flush=True)
print("R_LIST:", R_LIST.astype(int).tolist(), flush=True)
print("Output directory:", OUTDIR, flush=True)
print("=" * 100, flush=True)

for i, stem in enumerate(stems, 1):
    run_one_cube(stem, i)

print("\nALL DONE.", flush=True)