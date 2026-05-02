#!/usr/bin/env python3
"""
Magnetic path-line Richardson-pair dispersion test on a 320^3 JHTDB MHD cutout.

Longer-time version.

This script:
    1. Downloads 30 magnetic-field B(x,t) cutouts from mhd1024.
    2. Uses a time stride of 16 JHTDB indices.
    3. Seeds 2000 close magnetic path-line pairs near the cube centre.
    4. Integrates

            dX/dt = B(X,t)

       using trilinear spatial interpolation and linear time interpolation.
    5. Measures pair dispersion <R^2(t)>, excess dispersion
       <R^2(t)> - <R^2(0)>, local log-log slopes, and valid-pair fraction.

Important:
    - These are magnetic path-line tracers / virtual magnetic disturbances,
      not material fluid particles.
    - The magnetic field is treated as an Alfvén-speed-like velocity.
    - The correct mhd1024 time step is hard-coded as 0.0025, based on
      pyJHTDB/dbinfo.py:
          mhd1024['dt'] = 2.5e-3
    - With TIME_STRIDE=16 and N_TIMES=30, the total time span is 1.16.

Run:
    cd /home/idies/workspace/Storage/elenceq/mhd_work/jhtdb_mhd1024
    export JHTDB_TOKEN="com.icloud.elenceq-093dd6ec"
    python -u magnetic_pathline_richardson_320_stride16.py
"""

from __future__ import annotations

import os
import gc
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# User configuration
# ============================================================

BASE = Path("/home/idies/workspace/Storage/elenceq/mhd_work/jhtdb_mhd1024")

RUN_NAME = "mag_pathline_richardson_320_nt30_stride16_npairs2000"
RUN_DIR = BASE / RUN_NAME
RAW_DIR = RUN_DIR / "raw_b_cubes"
OUT_DIR = RUN_DIR / "results"

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "mhd1024"
VARIABLE = "magneticfield"

# Correct mhd1024 physical time step per integer time index.
MHD1024_DT = 2.5e-3

# Spatial cube.
# 1-based inclusive JHTDB/giverny ranges.
# 353..672 gives length 320.
N_CUBE = 320
X_START = 353
Y_START = 353
Z_START = 353
X_END = X_START + N_CUBE - 1
Y_END = Y_START + N_CUBE - 1
Z_END = Z_START + N_CUBE - 1

# Temporal sampling.
# TIME_START + TIME_STRIDE*(N_TIMES-1) must be <= 1023 for mhd1024.
N_TIMES = 30
TIME_START = 22
TIME_STRIDE = 16
TIME_VALUES = [TIME_START + TIME_STRIDE * i for i in range(N_TIMES)]
SNAPSHOT_DT = MHD1024_DT * TIME_STRIDE

if TIME_VALUES[-1] > 1023:
    raise ValueError(
        f"Last requested time index {TIME_VALUES[-1]} exceeds mhd1024 max index 1023."
    )

# Download behavior.
DOWNLOAD_IF_MISSING = True
FORCE_REDOWNLOAD = False

# Pair seeding.
N_PAIRS = 2000
R0_GRID = 4.0
SEED = 12345

# Central seeding box in local grid coordinates.
# Local coordinates run from 0 to 319.
CENTER_BOX_LO = 120.0
CENTER_BOX_HI = 200.0

# Integration.
# With TIME_STRIDE=16, each saved snapshot interval is dt=0.04.
# Eight RK4 substeps gives h=0.005.
SUBSTEPS_PER_SNAPSHOT = 8

# Main run uses +B. Set to [+1.0, -1.0] if you later want both directions.
SIGNS = [+1.0]

# Spatial spacing for mhd1024: 2*pi/1024 = pi/512.
DX = float(np.pi / 512.0)
DY = float(np.pi / 512.0)
DZ = float(np.pi / 512.0)


# ============================================================
# Paths
# ============================================================

def cube_path(time_value: int) -> Path:
    return RAW_DIR / (
        f"b_{DATASET}_t{int(time_value):04d}_"
        f"x{X_START}-{X_END}_y{Y_START}-{Y_END}_z{Z_START}-{Z_END}.npy"
    )


def metadata_path() -> Path:
    return RAW_DIR / "cutout_metadata.json"


def result_path(sign: float) -> Path:
    s = "plusB" if sign > 0 else "minusB"
    return OUT_DIR / f"magnetic_pathline_richardson_{s}.npz"


# ============================================================
# JHTDB / giverny download
# ============================================================

def require_token() -> str:
    token = os.environ.get("JHTDB_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "JHTDB_TOKEN is not set.\n\n"
            "Run first:\n"
            "    export JHTDB_TOKEN='your-token-here'\n"
        )
    return token


def extract_array_from_cutout_dataset(ds) -> np.ndarray:
    """
    giverny getCutout returns an xarray Dataset with one data variable.

    Expected shape:
        (z, y, x, 3)

    where the final axis is (Bx, By, Bz).
    """
    if not hasattr(ds, "data_vars"):
        raise TypeError("Expected xarray Dataset from getCutout.")

    names = list(ds.data_vars.keys())
    if len(names) != 1:
        raise RuntimeError(f"Expected one data variable, got {names}")

    arr = ds[names[0]].values

    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise RuntimeError(
            f"Expected cutout array shape (z,y,x,3), got {arr.shape}"
        )

    return np.asarray(arr, dtype=np.float32)


def write_metadata() -> None:
    meta = {
        "dataset": DATASET,
        "variable": VARIABLE,
        "N_cube": N_CUBE,
        "x_start": X_START,
        "x_end": X_END,
        "y_start": Y_START,
        "y_end": Y_END,
        "z_start": Z_START,
        "z_end": Z_END,
        "time_start": TIME_START,
        "time_stride": TIME_STRIDE,
        "time_values": TIME_VALUES,
        "mhd1024_dt_per_index": MHD1024_DT,
        "snapshot_dt": SNAPSHOT_DT,
        "total_time_span": (N_TIMES - 1) * SNAPSHOT_DT,
        "dx": DX,
        "dy": DY,
        "dz": DZ,
        "N_pairs": N_PAIRS,
        "R0_grid": R0_GRID,
        "center_box_lo": CENTER_BOX_LO,
        "center_box_hi": CENTER_BOX_HI,
        "substeps_per_snapshot": SUBSTEPS_PER_SNAPSHOT,
        "note": (
            "Array shape is (z,y,x,3). Vector components are assumed to be "
            "(Bx,By,Bz). Positions in the integrator are local grid coordinates "
            "(x,y,z) in [0,N_cube). Magnetic path-line equation is dX/dt=B(X,t)."
        ),
    }

    with open(metadata_path(), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[metadata] saved: {metadata_path()}")


def download_one_cutout(time_value: int, cube) -> None:
    from giverny.turbulence_toolkit import getCutout

    out = cube_path(time_value)

    if out.exists() and not FORCE_REDOWNLOAD:
        print(f"[download] exists: {out.name}")
        return

    axes_ranges = np.array(
        [
            [X_START, X_END],
            [Y_START, Y_END],
            [Z_START, Z_END],
            [int(time_value), int(time_value)],
        ],
        dtype=np.int32,
    )

    strides = np.array([1, 1, 1, 1], dtype=np.int32)

    print(f"[download] Requesting {VARIABLE} t={time_value}, cube {N_CUBE}^3...")
    ds = getCutout(
        cube,
        VARIABLE,
        axes_ranges,
        strides,
        trace_memory=False,
        verbose=True,
    )

    arr = extract_array_from_cutout_dataset(ds)

    print(f"[download] received array shape={arr.shape}, dtype={arr.dtype}")

    expected_shape = (N_CUBE, N_CUBE, N_CUBE, 3)
    if arr.shape != expected_shape:
        raise RuntimeError(f"Unexpected cutout shape {arr.shape}; expected {expected_shape}")

    np.save(out, arr)
    print(f"[download] saved: {out}")

    del arr, ds
    gc.collect()


def download_all_cutouts() -> None:
    if not DOWNLOAD_IF_MISSING:
        print("[download] DOWNLOAD_IF_MISSING=False; skipping download stage.")
        return

    token = require_token()

    from giverny.turbulence_dataset import turb_dataset

    cube = turb_dataset(
        dataset_title=DATASET,
        output_path=str(RAW_DIR),
        auth_token=token,
    )

    write_metadata()

    for t in TIME_VALUES:
        download_one_cutout(t, cube)


def check_all_cubes_exist() -> None:
    missing = [str(cube_path(t)) for t in TIME_VALUES if not cube_path(t).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing B cube files:\n"
            + "\n".join(missing)
            + "\n\nSet DOWNLOAD_IF_MISSING=True or fix the download stage."
        )


def load_b_cube(time_value: int):
    """
    Memory-map one saved B cube. Shape is (z,y,x,3).
    """
    return np.load(cube_path(time_value), mmap_mode="r")


# ============================================================
# Pair initialization
# ============================================================

def seed_pairs() -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        positions0: shape (2*N_PAIRS, 3), local grid coordinates (x,y,z)
        pair_ids:   shape (2*N_PAIRS,), pair id for each endpoint
    """
    rng = np.random.default_rng(SEED)

    centres = rng.uniform(
        low=CENTER_BOX_LO,
        high=CENTER_BOX_HI,
        size=(N_PAIRS, 3),
    )

    dirs = rng.normal(size=(N_PAIRS, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]

    x1 = centres - 0.5 * R0_GRID * dirs
    x2 = centres + 0.5 * R0_GRID * dirs

    positions0 = np.empty((2 * N_PAIRS, 3), dtype=np.float64)
    positions0[0::2, :] = x1
    positions0[1::2, :] = x2

    pair_ids = np.repeat(np.arange(N_PAIRS), 2)

    return positions0, pair_ids


# ============================================================
# Interpolation and integration
# ============================================================

def inside_domain(pos: np.ndarray, shape_zyxv: Tuple[int, int, int, int]) -> np.ndarray:
    """
    pos shape: (N,3), local grid coordinates ordered as (x,y,z).

    For trilinear interpolation, require:
        0 <= x < nx-1
        0 <= y < ny-1
        0 <= z < nz-1
    """
    nz, ny, nx, _ = shape_zyxv

    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    return (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(z) &
        (x >= 0.0) & (x < nx - 1.0) &
        (y >= 0.0) & (y < ny - 1.0) &
        (z >= 0.0) & (z < nz - 1.0)
    )


def trilinear_b(field: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    Trilinear interpolation of B.

    field shape: (z,y,x,3)
    pos shape:   (N,3), local grid coordinates ordered as (x,y,z)

    returns:
        B shape: (N,3), components (Bx,By,Bz)
    """
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    wz = (z - z0)[:, None]

    c000 = field[z0, y0, x0, :]
    c100 = field[z0, y0, x1, :]
    c010 = field[z0, y1, x0, :]
    c110 = field[z0, y1, x1, :]
    c001 = field[z1, y0, x0, :]
    c101 = field[z1, y0, x1, :]
    c011 = field[z1, y1, x0, :]
    c111 = field[z1, y1, x1, :]

    c00 = c000 * (1.0 - wx) + c100 * wx
    c10 = c010 * (1.0 - wx) + c110 * wx
    c01 = c001 * (1.0 - wx) + c101 * wx
    c11 = c011 * (1.0 - wx) + c111 * wx

    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy

    c = c0 * (1.0 - wz) + c1 * wz

    return c.astype(np.float64, copy=False)


def velocity_grid_units(
    pos: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    theta: float,
    sign: float,
) -> np.ndarray:
    """
    Evaluate d(local grid coordinate)/dt.

    B is in physical length / physical time.
    Local grid coordinate x_grid = x_phys / dx.

    Therefore:
        dx_grid/dt = Bx / dx
        dy_grid/dt = By / dy
        dz_grid/dt = Bz / dz

    The field is linearly interpolated in time between b0 and b1.
    """
    out = np.full_like(pos, np.nan, dtype=np.float64)
    m = inside_domain(pos, b0.shape)

    if not np.any(m):
        return out

    bb0 = trilinear_b(b0, pos[m])
    bb1 = trilinear_b(b1, pos[m])
    bb = (1.0 - theta) * bb0 + theta * bb1

    out[m, 0] = sign * bb[:, 0] / DX
    out[m, 1] = sign * bb[:, 1] / DY
    out[m, 2] = sign * bb[:, 2] / DZ

    return out


def rk4_step(
    pos: np.ndarray,
    h_phys: float,
    theta0: float,
    theta_h: float,
    b0: np.ndarray,
    b1: np.ndarray,
    sign: float,
) -> np.ndarray:
    """
    One RK4 substep.

    theta0 is the start-time fraction inside the current saved-snapshot interval.
    theta_h is h_phys / SNAPSHOT_DT.
    """
    k1 = velocity_grid_units(pos, b0, b1, theta0, sign)
    k2 = velocity_grid_units(pos + 0.5 * h_phys * k1, b0, b1, theta0 + 0.5 * theta_h, sign)
    k3 = velocity_grid_units(pos + 0.5 * h_phys * k2, b0, b1, theta0 + 0.5 * theta_h, sign)
    k4 = velocity_grid_units(pos + h_phys * k3, b0, b1, theta0 + theta_h, sign)

    return pos + (h_phys / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ============================================================
# Diagnostics
# ============================================================

def compute_pair_r2(pos: np.ndarray, pair_alive: np.ndarray) -> np.ndarray:
    """
    Return R^2 for all pairs; invalid pairs get NaN.
    """
    r2 = np.full(N_PAIRS, np.nan, dtype=np.float64)

    x1 = pos[0::2, :]
    x2 = pos[1::2, :]
    d = x2 - x1

    r2_all = np.sum(d * d, axis=1)
    r2[pair_alive] = r2_all[pair_alive]

    return r2


def finite_log_slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Local derivative d log(y) / d log(x), using np.gradient on finite positive values.
    """
    slope = np.full_like(y, np.nan, dtype=np.float64)
    m = (x > 0) & np.isfinite(x) & np.isfinite(y) & (y > 0)

    if np.count_nonzero(m) >= 3:
        slope[m] = np.gradient(np.log(y[m]), np.log(x[m]))

    return slope


def summarize_r2(times: np.ndarray, r2_all: np.ndarray, n_valid: np.ndarray) -> Dict[str, np.ndarray]:
    mean_r2 = np.nanmean(r2_all, axis=1)
    median_r2 = np.nanmedian(r2_all, axis=1)
    q25_r2 = np.nanpercentile(r2_all, 25, axis=1)
    q75_r2 = np.nanpercentile(r2_all, 75, axis=1)

    initial_mean_r2 = mean_r2[0]
    excess_mean_r2 = mean_r2 - initial_mean_r2

    alpha_mean_r2 = finite_log_slope(times, mean_r2)
    alpha_excess_r2 = finite_log_slope(times, excess_mean_r2)

    return {
        "times": times,
        "mean_r2": mean_r2,
        "median_r2": median_r2,
        "q25_r2": q25_r2,
        "q75_r2": q75_r2,
        "excess_mean_r2": excess_mean_r2,
        "alpha_mean_r2": alpha_mean_r2,
        "alpha_excess_r2": alpha_excess_r2,
        "n_valid": n_valid,
        "valid_fraction": n_valid / float(N_PAIRS),
    }


def save_summary_csv(summary: Dict[str, np.ndarray], sign: float) -> None:
    s = "plusB" if sign > 0 else "minusB"
    csv_path = OUT_DIR / f"summary_{s}.csv"

    arr = np.column_stack(
        [
            summary["times"],
            summary["mean_r2"],
            summary["median_r2"],
            summary["q25_r2"],
            summary["q75_r2"],
            summary["excess_mean_r2"],
            summary["alpha_mean_r2"],
            summary["alpha_excess_r2"],
            summary["n_valid"],
            summary["valid_fraction"],
        ]
    )

    header = (
        "time,"
        "mean_R2,"
        "median_R2,"
        "q25_R2,"
        "q75_R2,"
        "excess_mean_R2,"
        "local_slope_mean_R2,"
        "local_slope_excess_mean_R2,"
        "n_valid,"
        "valid_fraction"
    )

    np.savetxt(csv_path, arr, delimiter=",", header=header, comments="")
    print(f"[output] saved CSV: {csv_path}")


def plot_mean_r2(summary: Dict[str, np.ndarray], sign: float) -> None:
    s = "plusB" if sign > 0 else "minusB"

    t = summary["times"]
    mean_r2 = summary["mean_r2"]
    q25 = summary["q25_r2"]
    q75 = summary["q75_r2"]

    fig, ax = plt.subplots(figsize=(6.6, 4.8))

    m = (t > 0) & np.isfinite(mean_r2) & (mean_r2 > 0)
    ax.loglog(t[m], mean_r2[m], marker="o", label=r"$\langle R^2(t)\rangle$")

    if np.count_nonzero(m) >= 2:
        idxs = np.where(m)[0]
        idx = idxs[max(0, min(2, len(idxs) - 1))]
        tref = t[m]
        ref = mean_r2[idx] * (tref / t[idx]) ** 3
        ax.loglog(tref, ref, linestyle="--", label=r"$t^3$ reference")

    mq = (t > 0) & np.isfinite(q25) & np.isfinite(q75) & (q25 > 0) & (q75 > 0)
    if np.any(mq):
        ax.fill_between(t[mq], q25[mq], q75[mq], alpha=0.2, label="IQR")

    ax.set_xlabel("time")
    ax.set_ylabel(r"$\langle R^2\rangle$ [grid cells$^2$]")
    ax.set_title(f"Magnetic path-line pair dispersion: {s}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    p = OUT_DIR / f"mean_R2_loglog_{s}.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"[plot] saved: {p}")


def plot_excess_r2(summary: Dict[str, np.ndarray], sign: float) -> None:
    s = "plusB" if sign > 0 else "minusB"

    t = summary["times"]
    excess = summary["excess_mean_r2"]

    fig, ax = plt.subplots(figsize=(6.6, 4.8))

    m = (t > 0) & np.isfinite(excess) & (excess > 0)
    ax.loglog(t[m], excess[m], marker="o", label=r"$\langle R^2(t)\rangle-\langle R^2(0)\rangle$")

    if np.count_nonzero(m) >= 2:
        idxs = np.where(m)[0]
        idx = idxs[max(0, min(2, len(idxs) - 1))]
        tref = t[m]
        ref = excess[idx] * (tref / t[idx]) ** 3
        ax.loglog(tref, ref, linestyle="--", label=r"$t^3$ reference")

    ax.set_xlabel("time")
    ax.set_ylabel(r"excess pair separation [grid cells$^2$]")
    ax.set_title(f"Excess magnetic path-line dispersion: {s}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    p = OUT_DIR / f"excess_R2_loglog_{s}.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"[plot] saved: {p}")


def plot_slopes(summary: Dict[str, np.ndarray], sign: float) -> None:
    s = "plusB" if sign > 0 else "minusB"

    t = summary["times"]
    a1 = summary["alpha_mean_r2"]
    a2 = summary["alpha_excess_r2"]

    fig, ax = plt.subplots(figsize=(6.6, 4.5))

    m1 = (t > 0) & np.isfinite(a1)
    m2 = (t > 0) & np.isfinite(a2)

    ax.plot(t[m1], a1[m1], marker="o", label=r"slope of $\langle R^2\rangle$")
    ax.plot(t[m2], a2[m2], marker="s", label=r"slope of excess $\langle R^2\rangle$")
    ax.axhline(3.0, linestyle="--", label=r"Richardson slope $3$")

    ax.set_xscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$d\log(\cdot)/d\log t$")
    ax.set_title(f"Local scaling slopes: {s}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    p = OUT_DIR / f"local_slopes_{s}.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"[plot] saved: {p}")


def plot_valid_fraction(summary: Dict[str, np.ndarray], sign: float) -> None:
    s = "plusB" if sign > 0 else "minusB"

    t = summary["times"]
    valid_fraction = summary["valid_fraction"]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    ax.plot(t, valid_fraction, marker="o")
    ax.set_xlabel("time")
    ax.set_ylabel("surviving pair fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Boundary survival: {s}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    p = OUT_DIR / f"valid_fraction_{s}.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"[plot] saved: {p}")


def plot_results(summary: Dict[str, np.ndarray], sign: float) -> None:
    plot_mean_r2(summary, sign)
    plot_excess_r2(summary, sign)
    plot_slopes(summary, sign)
    plot_valid_fraction(summary, sign)


# ============================================================
# Main integration
# ============================================================

def integrate_for_sign(sign: float) -> None:
    print("\n" + "=" * 72)
    print(f"[integrate] Starting magnetic path-line integration for sign={sign:+.0f}")
    print("=" * 72)

    times = np.arange(N_TIMES, dtype=np.float64) * SNAPSHOT_DT

    pos, pair_ids = seed_pairs()
    pair_alive = np.ones(N_PAIRS, dtype=bool)

    pos_hist = np.full((N_TIMES, 2 * N_PAIRS, 3), np.nan, dtype=np.float32)
    r2_all = np.full((N_TIMES, N_PAIRS), np.nan, dtype=np.float64)
    n_valid = np.zeros(N_TIMES, dtype=np.int64)

    r2_all[0, :] = compute_pair_r2(pos, pair_alive)
    n_valid[0] = int(np.count_nonzero(pair_alive))
    pos_hist[0, :, :] = pos.astype(np.float32)

    print(f"[integrate] DX,DY,DZ = {DX}, {DY}, {DZ}")
    print(f"[integrate] mhd1024 dt per index = {MHD1024_DT}")
    print(f"[integrate] TIME_STRIDE = {TIME_STRIDE}")
    print(f"[integrate] snapshot_dt = {SNAPSHOT_DT}")
    print(f"[integrate] total time span = {(N_TIMES - 1) * SNAPSHOT_DT}")
    print(f"[integrate] RK4 substeps per interval = {SUBSTEPS_PER_SNAPSHOT}")
    print(f"[integrate] RK4 h = {SNAPSHOT_DT / SUBSTEPS_PER_SNAPSHOT}")
    print(f"[integrate] initial <R^2> = {np.nanmean(r2_all[0]):.8g}")
    print(f"[integrate] initial valid pairs = {n_valid[0]} / {N_PAIRS}")

    for k in range(N_TIMES - 1):
        t0 = TIME_VALUES[k]
        t1 = TIME_VALUES[k + 1]

        print(f"\n[integrate] interval {k+1}/{N_TIMES-1}: index {t0} -> {t1}")

        b0 = load_b_cube(t0)
        b1 = load_b_cube(t1)

        h_phys = SNAPSHOT_DT / float(SUBSTEPS_PER_SNAPSHOT)
        theta_h = 1.0 / float(SUBSTEPS_PER_SNAPSHOT)

        for sub in range(SUBSTEPS_PER_SNAPSHOT):
            if not np.any(pair_alive):
                print("[integrate] all pairs have left the cube.")
                break

            theta0 = sub * theta_h

            active_tracers = np.repeat(pair_alive, 2)
            active_indices = np.where(active_tracers)[0]

            pos_active = pos[active_indices, :]

            pos_new = rk4_step(
                pos_active,
                h_phys=h_phys,
                theta0=theta0,
                theta_h=theta_h,
                b0=b0,
                b1=b1,
                sign=sign,
            )

            pos[active_indices, :] = pos_new

            tracer_inside = np.zeros(2 * N_PAIRS, dtype=bool)
            tracer_inside[active_indices] = inside_domain(pos[active_indices, :], b0.shape)

            new_pair_alive = tracer_inside[0::2] & tracer_inside[1::2]
            pair_alive &= new_pair_alive

            dead_tracers = np.repeat(~pair_alive, 2)
            pos[dead_tracers, :] = np.nan

            print(
                f"    substep {sub+1}/{SUBSTEPS_PER_SNAPSHOT}: "
                f"valid pairs = {np.count_nonzero(pair_alive)}"
            )

        r2_all[k + 1, :] = compute_pair_r2(pos, pair_alive)
        n_valid[k + 1] = int(np.count_nonzero(pair_alive))
        pos_hist[k + 1, :, :] = pos.astype(np.float32)

        if n_valid[k + 1] > 0:
            mean_r2 = np.nanmean(r2_all[k + 1])
            med_r2 = np.nanmedian(r2_all[k + 1])
            excess = mean_r2 - np.nanmean(r2_all[0])
            print(
                f"[diagnostic] index={t1}, time={times[k+1]:.6g}, "
                f"<R^2>={mean_r2:.8g}, "
                f"excess={excess:.8g}, "
                f"median R^2={med_r2:.8g}, "
                f"valid={n_valid[k+1]}/{N_PAIRS}"
            )
        else:
            print(f"[diagnostic] index={t1}, no valid pairs remain.")

        del b0, b1
        gc.collect()

    summary = summarize_r2(times, r2_all, n_valid)

    out = result_path(sign)

    np.savez_compressed(
        out,
        times=times,
        time_values=np.asarray(TIME_VALUES),
        r2_all=r2_all,
        n_valid=n_valid,
        valid_fraction=summary["valid_fraction"],
        mean_r2=summary["mean_r2"],
        median_r2=summary["median_r2"],
        q25_r2=summary["q25_r2"],
        q75_r2=summary["q75_r2"],
        excess_mean_r2=summary["excess_mean_r2"],
        alpha_mean_r2=summary["alpha_mean_r2"],
        alpha_excess_r2=summary["alpha_excess_r2"],
        pos_hist=pos_hist,
        pair_ids=pair_ids,
        sign=sign,
        N_PAIRS=N_PAIRS,
        R0_GRID=R0_GRID,
        TIME_START=TIME_START,
        TIME_STRIDE=TIME_STRIDE,
        TIME_VALUES=np.asarray(TIME_VALUES),
        MHD1024_DT=MHD1024_DT,
        SNAPSHOT_DT=SNAPSHOT_DT,
        SUBSTEPS_PER_SNAPSHOT=SUBSTEPS_PER_SNAPSHOT,
        DX=DX,
        DY=DY,
        DZ=DZ,
        X_START=X_START,
        X_END=X_END,
        Y_START=Y_START,
        Y_END=Y_END,
        Z_START=Z_START,
        Z_END=Z_END,
    )

    print(f"[output] saved NPZ: {out}")

    save_summary_csv(summary, sign)
    plot_results(summary, sign)


# ============================================================
# Extra quick diagnostics
# ============================================================

def print_b_magnitude_estimate() -> None:
    """
    Estimate magnetic displacement per saved snapshot from the first cube.
    """
    f = cube_path(TIME_VALUES[0])
    B = np.load(f, mmap_mode="r")

    Bs = B[::8, ::8, ::8, :].astype(np.float64)
    mag = np.sqrt(np.sum(Bs * Bs, axis=-1))

    mean_mag = float(np.mean(mag))
    rms_mag = float(np.sqrt(np.mean(mag ** 2)))
    max_mag = float(np.max(mag))

    mean_cells = mean_mag * SNAPSHOT_DT / DX
    rms_cells = rms_mag * SNAPSHOT_DT / DX
    max_cells = max_mag * SNAPSHOT_DT / DX

    print("\n[B diagnostic from first cube]")
    print(f"mean |B| = {mean_mag:.8g}")
    print(f"rms  |B| = {rms_mag:.8g}")
    print(f"max  |B| = {max_mag:.8g}")
    print(f"mean |B| * snapshot_dt / dx = {mean_cells:.8g} grid cells")
    print(f"rms  |B| * snapshot_dt / dx = {rms_cells:.8g} grid cells")
    print(f"max  |B| * snapshot_dt / dx = {max_cells:.8g} grid cells")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("\nMagnetic path-line Richardson test: longer stride-16 run")
    print("=" * 72)
    print(f"RUN_DIR = {RUN_DIR}")
    print(f"DATASET = {DATASET}")
    print(f"VARIABLE = {VARIABLE}")
    print(f"cube = x{X_START}-{X_END}, y{Y_START}-{Y_END}, z{Z_START}-{Z_END}")
    print(f"N_CUBE = {N_CUBE}")
    print(f"N_TIMES = {N_TIMES}")
    print(f"TIME_START = {TIME_START}")
    print(f"TIME_STRIDE = {TIME_STRIDE}")
    print(f"TIME_VALUES = {TIME_VALUES[0]} ... {TIME_VALUES[-1]}")
    print(f"MHD1024_DT = {MHD1024_DT}")
    print(f"SNAPSHOT_DT = {SNAPSHOT_DT}")
    print(f"TOTAL_TIME = {(N_TIMES - 1) * SNAPSHOT_DT}")
    print(f"N_PAIRS = {N_PAIRS}")
    print(f"R0_GRID = {R0_GRID}")
    print(f"CENTER_BOX = [{CENTER_BOX_LO}, {CENTER_BOX_HI}]")
    print(f"SUBSTEPS_PER_SNAPSHOT = {SUBSTEPS_PER_SNAPSHOT}")
    print("=" * 72)

    download_all_cutouts()
    check_all_cubes_exist()
    print_b_magnitude_estimate()

    for sign in SIGNS:
        integrate_for_sign(sign)

    print("\nDone.")
    print(f"Results are in: {OUT_DIR}")
    print("\nInspect first:")
    print(f"  {OUT_DIR / 'summary_plusB.csv'}")
    print(f"  {OUT_DIR / 'mean_R2_loglog_plusB.png'}")
    print(f"  {OUT_DIR / 'excess_R2_loglog_plusB.png'}")
    print(f"  {OUT_DIR / 'local_slopes_plusB.png'}")
    print(f"  {OUT_DIR / 'valid_fraction_plusB.png'}")


if __name__ == "__main__":
    main()