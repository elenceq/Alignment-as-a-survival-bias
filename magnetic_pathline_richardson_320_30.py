#!/usr/bin/env python3
"""
Magnetic path-line Richardson-pair dispersion test on a 320^3 JHTDB MHD cutout.

Goal:
    Download 30 magnetic-field snapshots B(x,t) on the same 320^3 cube.
    Seed 2000 close pairs near the centre of the cube.
    Integrate magnetic path lines

        dX/dt = B(X,t)

    using trilinear spatial interpolation and linear time interpolation.
    Measure <R^2(t)> and the local log-log slope.

Designed for SciServer / JHTDB / giverny.

Important:
    1. This tracks "magnetic path-line tracers", not material particles.
    2. Pair endpoints are dropped once either endpoint exits the local cutout.
    3. The time step must be in physical simulation-time units. The script tries
       to read the dataset time spacing from giverny metadata; if that fails,
       it uses USER_SNAPSHOT_DT below.
"""

from __future__ import annotations

import os
import gc
import json
import math
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# User configuration
# ============================================================

BASE = Path("/home/idies/workspace/Storage/elenceq/mhd_work/jhtdb_mhd1024")
RUN_NAME = "mag_pathline_richardson_320_nt30_npairs2000"

RUN_DIR = BASE / RUN_NAME
RAW_DIR = RUN_DIR / "raw_b_cubes"
OUT_DIR = RUN_DIR / "results"

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "mhd1024"
VARIABLE = "magneticfield"

# A central 320^3 cube in a 1024^3 domain, using giverny/JHTDB 1-based inclusive ranges.
# 353..672 has length 320.
N_CUBE = 320
X_START = 353
Y_START = 353
Z_START = 353
X_END = X_START + N_CUBE - 1
Y_END = Y_START + N_CUBE - 1
Z_END = Z_START + N_CUBE - 1

# Time indices/snapshots.
# Adjust TIME_START if you want a different temporal window.
N_TIMES = 30
TIME_START = 22
TIME_STRIDE = 1
TIME_VALUES = [TIME_START + TIME_STRIDE * i for i in range(N_TIMES)]

# If giverny metadata cannot provide physical dt, use this fallback.
# The Richardson exponent is slope-based, but trajectory integration needs dt
# in units consistent with B.
USER_SNAPSHOT_DT = 1.0

# Download behavior.
DOWNLOAD_IF_MISSING = True
FORCE_REDOWNLOAD = False

# Pair seeding.
N_PAIRS = 2000
R0_GRID = 4.0                 # initial separation in grid cells
SEED = 12345

# Seed pair centres inside this local cube range.
# This leaves a buffer from boundaries.
CENTER_BOX_LO = 120.0
CENTER_BOX_HI = 200.0

# Integration.
SUBSTEPS_PER_SNAPSHOT = 4     # RK4 substeps between saved JHTDB snapshots
SIGNS = [+1.0]                # use [+1.0, -1.0] if you want both dX/dt = +/-B

# Fallback physical spacing for a 2*pi periodic 1024^3 box.
# The script overwrites this from cutout coordinates if available.
FALLBACK_DX = 2.0 * np.pi / 1024.0
FALLBACK_DY = 2.0 * np.pi / 1024.0
FALLBACK_DZ = 2.0 * np.pi / 1024.0


# ============================================================
# Utility paths
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
            "JHTDB_TOKEN is not set.\n"
            "Set it before running, e.g.\n\n"
            "    export JHTDB_TOKEN='your-real-token-here'\n\n"
            "A 320^3 cutout is far larger than the public testing-token limit."
        )
    return token


def try_get_dataset_dt(cube) -> Optional[float]:
    """
    Try to read physical time spacing from giverny metadata.
    Returns None if unavailable.
    """
    try:
        from giverny.turbulence_gizmos.basic_gizmos import get_time_dt
        dt = float(get_time_dt(cube.metadata, DATASET, "getcutout"))
        if np.isfinite(dt) and dt > 0:
            return dt
    except Exception as exc:
        print(f"[metadata] Could not read dataset dt from giverny metadata: {exc}")
    return None


def extract_spacing_from_xarray(ds) -> Tuple[float, float, float]:
    """
    Extract dx, dy, dz from xarray coordinates if present.
    Fallback to 2*pi/1024.
    """
    dx, dy, dz = FALLBACK_DX, FALLBACK_DY, FALLBACK_DZ

    try:
        if "xcoor" in ds.coords:
            x = np.asarray(ds.coords["xcoor"].values, dtype=float)
            if len(x) >= 2:
                dx = float(x[1] - x[0])

        if "ycoor" in ds.coords:
            y = np.asarray(ds.coords["ycoor"].values, dtype=float)
            if len(y) >= 2:
                dy = float(y[1] - y[0])

        if "zcoor" in ds.coords:
            z = np.asarray(ds.coords["zcoor"].values, dtype=float)
            if len(z) >= 2:
                dz = float(z[1] - z[0])
    except Exception as exc:
        print(f"[metadata] Could not extract xarray spacing; using fallback spacing: {exc}")

    return dx, dy, dz


def extract_array_from_cutout_dataset(ds) -> np.ndarray:
    """
    giverny getCutout returns an xarray Dataset with one data variable for one time.
    Expected array shape: (z, y, x, values), values=(Bx, By, Bz).
    """
    if not hasattr(ds, "data_vars"):
        raise TypeError("Expected xarray Dataset from getCutout, but got a different object.")

    names = list(ds.data_vars.keys())
    if len(names) != 1:
        raise RuntimeError(f"Expected one data variable in cutout result, got {names}")

    arr = ds[names[0]].values

    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise RuntimeError(
            f"Expected cutout array shape (z,y,x,3), got {arr.shape}"
        )

    return np.asarray(arr, dtype=np.float32)


def download_one_cutout(time_value: int, cube, first_download: bool) -> Dict:
    """
    Download one B cutout and save it as .npy.
    Returns metadata from this cutout.
    """
    from giverny.turbulence_toolkit import getCutout

    out = cube_path(time_value)
    if out.exists() and not FORCE_REDOWNLOAD:
        print(f"[download] exists: {out.name}")
        return {}

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

    if arr.shape != (N_CUBE, N_CUBE, N_CUBE, 3):
        raise RuntimeError(
            f"Unexpected cutout shape {arr.shape}; expected "
            f"({N_CUBE},{N_CUBE},{N_CUBE},3)."
        )

    np.save(out, arr)
    print(f"[download] saved: {out}")

    meta = {}
    if first_download:
        dx, dy, dz = extract_spacing_from_xarray(ds)
        dataset_dt = try_get_dataset_dt(cube)
        if dataset_dt is None:
            snapshot_dt = USER_SNAPSHOT_DT
        else:
            snapshot_dt = dataset_dt * TIME_STRIDE

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
            "time_values": TIME_VALUES,
            "time_stride": TIME_STRIDE,
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "snapshot_dt": snapshot_dt,
            "note": (
                "Array shape is (z,y,x,3), vector components are assumed "
                "to be (Bx,By,Bz). Positions in the integrator are local "
                "grid coordinates (x,y,z) in [0,N_cube)."
            ),
        }

        with open(metadata_path(), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[metadata] dx,dy,dz = {dx}, {dy}, {dz}")
        print(f"[metadata] snapshot_dt = {snapshot_dt}")
        print(f"[metadata] saved: {metadata_path()}")

    del arr, ds
    gc.collect()
    return meta


def download_all_cutouts() -> None:
    """
    Download all missing B snapshots.
    """
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

    first_download = not metadata_path().exists()

    for t in TIME_VALUES:
        meta = download_one_cutout(t, cube, first_download=first_download)
        if meta:
            first_download = False


# ============================================================
# Metadata and loading
# ============================================================

def load_or_make_metadata() -> Dict:
    if metadata_path().exists():
        with open(metadata_path(), "r") as f:
            return json.load(f)

    print("[metadata] No metadata file found. Using fallback spacing and USER_SNAPSHOT_DT.")
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
        "time_values": TIME_VALUES,
        "time_stride": TIME_STRIDE,
        "dx": FALLBACK_DX,
        "dy": FALLBACK_DY,
        "dz": FALLBACK_DZ,
        "snapshot_dt": USER_SNAPSHOT_DT,
    }
    with open(metadata_path(), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def check_all_cubes_exist() -> None:
    missing = [str(cube_path(t)) for t in TIME_VALUES if not cube_path(t).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing B cube files:\n" + "\n".join(missing) +
            "\n\nSet DOWNLOAD_IF_MISSING=True or fix the download stage."
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
# Interpolation and RK4 integration
# ============================================================

def inside_domain(pos: np.ndarray, shape_zyxv: Tuple[int, int, int, int]) -> np.ndarray:
    """
    pos is shape (N,3) in local grid coordinates (x,y,z).
    For trilinear interpolation, require x,y,z in [0, N-1).
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
    spacing: Tuple[float, float, float],
    sign: float,
) -> np.ndarray:
    """
    Evaluate d(local grid coordinate)/dt at positions.

    B is in physical length / physical time.
    local grid coordinate x_grid = x_phys / dx.
    Therefore:

        dx_grid/dt = Bx / dx

    Linear interpolation in time between b0 and b1.
    """
    out = np.full_like(pos, np.nan, dtype=np.float64)
    m = inside_domain(pos, b0.shape)

    if not np.any(m):
        return out

    bb0 = trilinear_b(b0, pos[m])
    bb1 = trilinear_b(b1, pos[m])
    bb = (1.0 - theta) * bb0 + theta * bb1

    dx, dy, dz = spacing

    out[m, 0] = sign * bb[:, 0] / dx
    out[m, 1] = sign * bb[:, 1] / dy
    out[m, 2] = sign * bb[:, 2] / dz

    return out


def rk4_step(
    pos: np.ndarray,
    h_phys: float,
    theta0: float,
    theta_h: float,
    b0: np.ndarray,
    b1: np.ndarray,
    spacing: Tuple[float, float, float],
    sign: float,
) -> np.ndarray:
    """
    One RK4 substep.

    theta0 is the start-time fraction in the current snapshot interval.
    theta_h is the substep size in units of the snapshot interval, i.e.
    h_phys / snapshot_dt.
    """
    k1 = velocity_grid_units(pos, b0, b1, theta0, spacing, sign)
    k2 = velocity_grid_units(pos + 0.5 * h_phys * k1, b0, b1, theta0 + 0.5 * theta_h, spacing, sign)
    k3 = velocity_grid_units(pos + 0.5 * h_phys * k2, b0, b1, theta0 + 0.5 * theta_h, spacing, sign)
    k4 = velocity_grid_units(pos + h_phys * k3, b0, b1, theta0 + theta_h, spacing, sign)

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


def summarize_r2(times: np.ndarray, r2_all: np.ndarray, n_valid: np.ndarray) -> Dict[str, np.ndarray]:
    mean_r2 = np.nanmean(r2_all, axis=1)
    median_r2 = np.nanmedian(r2_all, axis=1)
    q25 = np.nanpercentile(r2_all, 25, axis=1)
    q75 = np.nanpercentile(r2_all, 75, axis=1)

    alpha = np.full_like(mean_r2, np.nan, dtype=np.float64)
    m = (times > 0) & np.isfinite(mean_r2) & (mean_r2 > 0)

    if np.count_nonzero(m) >= 3:
        alpha[m] = np.gradient(np.log(mean_r2[m]), np.log(times[m]))

    return {
        "times": times,
        "mean_r2": mean_r2,
        "median_r2": median_r2,
        "q25_r2": q25,
        "q75_r2": q75,
        "alpha": alpha,
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
            summary["alpha"],
            summary["n_valid"],
            summary["valid_fraction"],
        ]
    )

    header = "time,mean_R2,median_R2,q25_R2,q75_R2,local_slope_alpha,n_valid,valid_fraction"
    np.savetxt(csv_path, arr, delimiter=",", header=header, comments="")
    print(f"[output] saved CSV: {csv_path}")


def plot_results(summary: Dict[str, np.ndarray], sign: float) -> None:
    s = "plusB" if sign > 0 else "minusB"

    t = summary["times"]
    mean_r2 = summary["mean_r2"]
    q25 = summary["q25_r2"]
    q75 = summary["q75_r2"]
    alpha = summary["alpha"]
    valid_fraction = summary["valid_fraction"]

    # Plot <R^2(t)> with t^3 reference.
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    m = (t > 0) & np.isfinite(mean_r2) & (mean_r2 > 0)
    ax.loglog(t[m], mean_r2[m], marker="o", label=r"$\langle R^2(t)\rangle$")

    if np.count_nonzero(m) >= 2:
        idx = np.where(m)[0][1] if np.count_nonzero(m) > 1 else np.where(m)[0][0]
        tref = t[m]
        ref = mean_r2[idx] * (tref / t[idx]) ** 3
        ax.loglog(tref, ref, linestyle="--", label=r"$t^3$ reference")

    # Interquartile band, if positive.
    mq = (t > 0) & np.isfinite(q25) & np.isfinite(q75) & (q25 > 0) & (q75 > 0)
    if np.any(mq):
        ax.fill_between(t[mq], q25[mq], q75[mq], alpha=0.2, label="IQR")

    ax.set_xlabel("time")
    ax.set_ylabel(r"pair separation $\langle R^2\rangle$ [grid cells$^2$]")
    ax.set_title(f"Magnetic path-line pair dispersion: {s}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    p = OUT_DIR / f"mean_R2_loglog_{s}.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"[plot] saved: {p}")

    # Plot local slope.
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    m = (t > 0) & np.isfinite(alpha)
    ax.plot(t[m], alpha[m], marker="o")
    ax.axhline(3.0, linestyle="--", label=r"Richardson slope $3$")
    ax.set_xscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$d\log\langle R^2\rangle / d\log t$")
    ax.set_title(f"Local scaling slope: {s}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    p = OUT_DIR / f"local_slope_{s}.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"[plot] saved: {p}")

    # Plot surviving fraction.
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
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


# ============================================================
# Main integration
# ============================================================

def integrate_for_sign(sign: float, meta: Dict) -> None:
    print("\n" + "=" * 72)
    print(f"[integrate] Starting magnetic path-line integration for sign={sign:+.0f}")
    print("=" * 72)

    dx = float(meta.get("dx", FALLBACK_DX))
    dy = float(meta.get("dy", FALLBACK_DY))
    dz = float(meta.get("dz", FALLBACK_DZ))
    spacing = (dx, dy, dz)

    snapshot_dt = float(meta.get("snapshot_dt", USER_SNAPSHOT_DT))
    if not np.isfinite(snapshot_dt) or snapshot_dt <= 0:
        raise RuntimeError(f"Bad snapshot_dt={snapshot_dt}")

    times = np.arange(N_TIMES, dtype=np.float64) * snapshot_dt

    pos, pair_ids = seed_pairs()
    pair_alive = np.ones(N_PAIRS, dtype=bool)

    pos_hist = np.full((N_TIMES, 2 * N_PAIRS, 3), np.nan, dtype=np.float32)
    r2_all = np.full((N_TIMES, N_PAIRS), np.nan, dtype=np.float64)
    n_valid = np.zeros(N_TIMES, dtype=np.int64)

    # Initial diagnostics.
    r2_all[0, :] = compute_pair_r2(pos, pair_alive)
    n_valid[0] = int(np.count_nonzero(pair_alive))
    pos_hist[0, :, :] = pos.astype(np.float32)

    print(f"[integrate] dx,dy,dz = {spacing}")
    print(f"[integrate] snapshot_dt = {snapshot_dt}")
    print(f"[integrate] initial <R^2> = {np.nanmean(r2_all[0]):.6g}")
    print(f"[integrate] initial valid pairs = {n_valid[0]} / {N_PAIRS}")

    # March between snapshots.
    for k in range(N_TIMES - 1):
        t0 = TIME_VALUES[k]
        t1 = TIME_VALUES[k + 1]

        print(f"\n[integrate] interval {k+1}/{N_TIMES-1}: t={t0} -> t={t1}")

        b0 = load_b_cube(t0)
        b1 = load_b_cube(t1)

        h_phys = snapshot_dt / float(SUBSTEPS_PER_SNAPSHOT)
        theta_h = 1.0 / float(SUBSTEPS_PER_SNAPSHOT)

        for sidx in range(SUBSTEPS_PER_SNAPSHOT):
            if not np.any(pair_alive):
                print("[integrate] all pairs have left the cube.")
                break

            theta0 = sidx * theta_h

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
                spacing=spacing,
                sign=sign,
            )

            pos[active_indices, :] = pos_new

            # Drop pairs if either endpoint is outside after this substep.
            tracer_inside = np.zeros(2 * N_PAIRS, dtype=bool)
            tracer_inside[active_indices] = inside_domain(pos[active_indices, :], b0.shape)

            new_pair_alive = tracer_inside[0::2] & tracer_inside[1::2]
            pair_alive &= new_pair_alive

            # Set dead-pair positions to NaN to avoid accidental reuse.
            dead_tracers = np.repeat(~pair_alive, 2)
            pos[dead_tracers, :] = np.nan

            print(
                f"    substep {sidx+1}/{SUBSTEPS_PER_SNAPSHOT}: "
                f"valid pairs = {np.count_nonzero(pair_alive)}"
            )

        # Diagnostics at the next stored time.
        r2_all[k + 1, :] = compute_pair_r2(pos, pair_alive)
        n_valid[k + 1] = int(np.count_nonzero(pair_alive))
        pos_hist[k + 1, :, :] = pos.astype(np.float32)

        if n_valid[k + 1] > 0:
            print(
                f"[diagnostic] t_index={t1}, "
                f"<R^2>={np.nanmean(r2_all[k+1]):.6g}, "
                f"median R^2={np.nanmedian(r2_all[k+1]):.6g}, "
                f"valid={n_valid[k+1]}/{N_PAIRS}"
            )
        else:
            print(f"[diagnostic] t_index={t1}, no valid pairs remain.")

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
        alpha=summary["alpha"],
        pos_hist=pos_hist,
        pair_ids=pair_ids,
        sign=sign,
        R0_GRID=R0_GRID,
        N_PAIRS=N_PAIRS,
        SUBSTEPS_PER_SNAPSHOT=SUBSTEPS_PER_SNAPSHOT,
        dx=dx,
        dy=dy,
        dz=dz,
        snapshot_dt=snapshot_dt,
    )

    print(f"[output] saved NPZ: {out}")

    save_summary_csv(summary, sign)
    plot_results(summary, sign)


def main() -> None:
    print("\nMagnetic path-line Richardson test")
    print("=" * 72)
    print(f"RUN_DIR = {RUN_DIR}")
    print(f"DATASET = {DATASET}")
    print(f"VARIABLE = {VARIABLE}")
    print(f"cube = x{X_START}-{X_END}, y{Y_START}-{Y_END}, z{Z_START}-{Z_END}")
    print(f"N_TIMES = {N_TIMES}")
    print(f"TIME_VALUES = {TIME_VALUES[0]} ... {TIME_VALUES[-1]}")
    print(f"N_PAIRS = {N_PAIRS}")
    print(f"R0_GRID = {R0_GRID}")
    print("=" * 72)

    download_all_cutouts()
    check_all_cubes_exist()

    meta = load_or_make_metadata()
    print(f"[metadata] using metadata from: {metadata_path()}")
    print(json.dumps(meta, indent=2))

    for sign in SIGNS:
        integrate_for_sign(sign, meta)

    print("\nDone.")
    print(f"Results are in: {OUT_DIR}")


if __name__ == "__main__":
    main()