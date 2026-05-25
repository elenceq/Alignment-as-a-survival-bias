#!/usr/bin/env python3
"""
AP07_320_timewindow_source_depletion.py

Local-BL-perpendicular finite-time source-depletion test for the
20-cube 320^3 JHTDB time-window data.

This is the corrected production script. It uses Gaussian-coarse-grained
local magnetic field directions with L = r/2, chooses centered increments
in the local perpendicular plane at the starting time, and follows the same
sampled midpoint-direction pairs to the target snapshot.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates


STATE_NAMES = ["B", "HS", "HL"]
STATE_TO_INT = {"B": 0, "HS": 1, "HL": 2}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def raw_root() -> Path:
    return project_root().parent / "raw"


def default_manifest() -> Path:
    return project_root() / "00_admin" / "AP06_320_timewindow_manifest_20.csv"


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def raw_path(row: pd.Series, t: int, field: str) -> Path:
    tag = (
        f"t{int(t):04d}_"
        f"x{int(row.x0)}-{int(row.x1)}_"
        f"y{int(row.y0)}-{int(row.y1)}_"
        f"z{int(row.z0)}-{int(row.z1)}"
    )
    return raw_root() / f"mhd1024_{field}_{tag}.npy"


def load_field(row: pd.Series, t: int, field: str) -> np.ndarray:
    path = raw_path(row, t, field)
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path, mmap_mode="r")
    if tuple(arr.shape) != (320, 320, 320, 3):
        raise ValueError(f"Bad shape {arr.shape}: {path}")
    if arr.dtype != np.float32:
        raise ValueError(f"Bad dtype {arr.dtype}: {path}")
    return arr


def block_average_3d_vector(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return np.asarray(arr, dtype=np.float32)
    n = arr.shape[0]
    if arr.shape[:3] != (n, n, n) or arr.shape[3] != 3:
        raise ValueError(f"Expected vector cube, got shape {arr.shape}")
    if n % factor != 0:
        raise ValueError(f"Cube size {n} not divisible by downsample factor {factor}")
    m = n // factor
    a = np.asarray(arr, dtype=np.float32)
    return a.reshape(m, factor, m, factor, m, factor, 3).mean(axis=(1, 3, 5)).astype(np.float32)


def gaussian_bl_downsampled(b: np.ndarray, r: int, downsample: int, sigma_factor: float) -> np.ndarray:
    b_small = block_average_3d_vector(b, downsample)
    sigma_small = float(sigma_factor * r) / float(downsample)
    if sigma_small <= 0:
        raise ValueError("Gaussian sigma must be positive.")

    out = np.empty_like(b_small, dtype=np.float32)
    for c in range(3):
        out[..., c] = gaussian_filter(b_small[..., c], sigma=sigma_small, mode="nearest")
    return out


def interpolate_vectors(field: np.ndarray, points: np.ndarray, order: int = 1) -> np.ndarray:
    coords = [points[:, 0], points[:, 1], points[:, 2]]
    vals = np.empty((points.shape[0], 3), dtype=np.float64)
    for c in range(3):
        vals[:, c] = map_coordinates(
            field[..., c],
            coords,
            order=order,
            mode="nearest",
            prefilter=False,
        )
    return vals


def unit_vectors(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1)
    bad = n <= 0.0
    if np.any(bad):
        v = v.copy()
        v[bad, :] = np.array([0.0, 0.0, 1.0])
        n = np.linalg.norm(v, axis=1)
    return v / n[:, None]


def perpendicular_basis(bhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = bhat.shape[0]
    ref = np.zeros((n, 3), dtype=np.float64)
    use_z = np.abs(bhat[:, 2]) < 0.90
    ref[use_z, 2] = 1.0
    ref[~use_z, 1] = 1.0

    e1 = np.cross(bhat, ref)
    e1 = unit_vectors(e1)
    e2 = np.cross(bhat, e1)
    e2 = unit_vectors(e2)
    return e1, e2


def make_local_perp_samples(
    b_start: np.ndarray,
    r: int,
    n_samples: int,
    n_phi: int,
    rng: np.random.Generator,
    downsample: int,
    sigma_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    if r <= 0:
        raise ValueError(f"Bad r={r}")
    if n_phi < 1:
        raise ValueError("n_phi must be positive")

    n = b_start.shape[0]
    half = 0.5 * float(r)
    margin = int(math.ceil(half)) + 3
    if 2 * margin >= n:
        raise ValueError(f"r={r} too large for cube size {n}")

    n_mid = int(math.ceil(n_samples / n_phi))
    centers = rng.uniform(float(margin), float(n - 1 - margin), size=(n_mid, 3))

    bl_small = gaussian_bl_downsampled(
        b=b_start,
        r=r,
        downsample=downsample,
        sigma_factor=sigma_factor,
    )

    small_n = bl_small.shape[0]
    coords_small = (centers - 0.5 * (downsample - 1)) / float(downsample)
    coords_small = np.clip(coords_small, 0.0, float(small_n - 1))
    bl_at_centers = interpolate_vectors(bl_small, coords_small, order=1)
    bhat = unit_vectors(bl_at_centers)

    e1, e2 = perpendicular_basis(bhat)

    phis = 2.0 * np.pi * np.arange(n_phi, dtype=np.float64) / float(n_phi)
    dirs = []
    cents = []
    for phi in phis:
        d = math.cos(float(phi)) * e1 + math.sin(float(phi)) * e2
        dirs.append(d)
        cents.append(centers)

    centers_all = np.vstack(cents)
    dirs_all = np.vstack(dirs)

    if len(centers_all) > n_samples:
        keep = rng.choice(len(centers_all), size=n_samples, replace=False)
        centers_all = centers_all[keep]
        dirs_all = dirs_all[keep]

    dirs_all = unit_vectors(dirs_all)
    return centers_all.astype(np.float64), dirs_all.astype(np.float64)


def compute_increments_for_pairs(
    u: np.ndarray,
    b: np.ndarray,
    centers: np.ndarray,
    directions: np.ndarray,
    r: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half_vec = 0.5 * float(r) * directions
    p1 = centers + half_vec
    p2 = centers - half_vec

    u1 = interpolate_vectors(u, p1, order=1)
    u2 = interpolate_vectors(u, p2, order=1)
    b1 = interpolate_vectors(b, p1, order=1)
    b2 = interpolate_vectors(b, p2, order=1)

    du = u1 - u2
    db = b1 - b2

    dzp = du + db
    dzm = du - db

    plus_sq = np.sum(dzp * dzp, axis=1).astype(np.float64)
    minus_sq = np.sum(dzm * dzm, axis=1).astype(np.float64)

    plus_norm = np.sqrt(plus_sq)
    minus_norm = np.sqrt(minus_sq)
    pm = plus_norm * minus_norm

    c = np.zeros_like(pm, dtype=np.float64)
    valid = pm > 0.0
    c[valid] = np.sum(dzp[valid] * dzm[valid], axis=1) / pm[valid]
    c = np.clip(c, -1.0, 1.0)
    theta = np.arccos(np.abs(c))

    return plus_sq, minus_sq, pm, theta


def assign_states(pm: np.ndarray, theta: np.ndarray, sector_mode: str) -> np.ndarray:
    states = np.zeros(len(pm), dtype=np.int8)
    high_cut = float(np.quantile(pm, 0.90))
    high = pm >= high_cut

    if sector_mode == "mean_theta":
        theta_cut = float(np.mean(theta))
        states[high & (theta <= theta_cut)] = STATE_TO_INT["HS"]
        states[high & (theta > theta_cut)] = STATE_TO_INT["HL"]
        return states

    if sector_mode == "terciles":
        theta_high = theta[high]
        if len(theta_high) == 0:
            return states
        lo = float(np.quantile(theta_high, 1.0 / 3.0))
        hi = float(np.quantile(theta_high, 2.0 / 3.0))
        states[high & (theta <= lo)] = STATE_TO_INT["HS"]
        states[high & (theta >= hi)] = STATE_TO_INT["HL"]
        return states

    raise ValueError(f"Unknown sector_mode={sector_mode}")


def matrix_counts(from_states: np.ndarray, to_states: np.ndarray) -> np.ndarray:
    code = from_states.astype(np.int64) * 3 + to_states.astype(np.int64)
    counts = np.bincount(code, minlength=9).astype(np.float64)
    return counts.reshape(3, 3)


def sector_sums(
    states: np.ndarray,
    plus_sq: np.ndarray,
    minus_sq: np.ndarray,
    pm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.bincount(states, minlength=3).astype(np.float64)
    s_plus = np.bincount(states, weights=plus_sq, minlength=3).astype(np.float64)
    s_minus = np.bincount(states, weights=minus_sq, minlength=3).astype(np.float64)
    s_pm = np.bincount(states, weights=pm, minlength=3).astype(np.float64)
    return counts, s_plus, s_minus, s_pm


def process_cube(
    row: pd.Series,
    r_list: list[int],
    lags: list[int],
    n_samples: int,
    n_phi: int,
    sector_mode: str,
    seed: int,
    downsample: int,
    sigma_factor: float,
) -> tuple[list[dict], list[dict]]:
    label = str(row.label)
    times = [int(row.t0), int(row.t1), int(row.t2), int(row.t3), int(row.t4)]

    transition_rows: list[dict] = []
    sector_rows: list[dict] = []

    print(f"\n[cube] {label} times={times}", flush=True)

    field_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def get_fields(t: int) -> tuple[np.ndarray, np.ndarray]:
        if t not in field_cache:
            field_cache[t] = (load_field(row, t, "velocity"), load_field(row, t, "magneticfield"))
        return field_cache[t]

    for r in r_list:
        print(f"  r={r}", flush=True)

        for start_index in range(len(times) - 1):
            t_start = times[start_index]
            u_start, b_start = get_fields(t_start)

            rng = np.random.default_rng(seed + 1000003 * start_index + 1009 * int(r))
            centers, directions = make_local_perp_samples(
                b_start=b_start,
                r=int(r),
                n_samples=int(n_samples),
                n_phi=int(n_phi),
                rng=rng,
                downsample=int(downsample),
                sigma_factor=float(sigma_factor),
            )

            plus0, minus0, pm0, theta0 = compute_increments_for_pairs(
                u=u_start,
                b=b_start,
                centers=centers,
                directions=directions,
                r=int(r),
            )
            states0 = assign_states(pm=pm0, theta=theta0, sector_mode=sector_mode)

            for lag in lags:
                target_index = start_index + int(lag)
                if target_index >= len(times):
                    continue

                t_target = times[target_index]
                u_target, b_target = get_fields(t_target)
                plus1, minus1, pm1, theta1 = compute_increments_for_pairs(
                    u=u_target,
                    b=b_target,
                    centers=centers,
                    directions=directions,
                    r=int(r),
                )
                states1 = assign_states(pm=pm1, theta=theta1, sector_mode=sector_mode)

                mat = matrix_counts(states0, states1)
                for i, from_name in enumerate(STATE_NAMES):
                    for j, to_name in enumerate(STATE_NAMES):
                        transition_rows.append(
                            {
                                "label": label,
                                "r": int(r),
                                "lag": int(lag),
                                "start_index": int(start_index),
                                "from_state": from_name,
                                "to_state": to_name,
                                "count": float(mat[i, j]),
                            }
                        )

                counts, s_plus, s_minus, s_pm = sector_sums(
                    states=states0,
                    plus_sq=plus0,
                    minus_sq=minus0,
                    pm=pm0,
                )
                for i, state_name in enumerate(STATE_NAMES):
                    sector_rows.append(
                        {
                            "label": label,
                            "r": int(r),
                            "lag": int(lag),
                            "start_index": int(start_index),
                            "state": state_name,
                            "count": float(counts[i]),
                            "sum_S2p": float(s_plus[i]),
                            "sum_S2m": float(s_minus[i]),
                            "sum_S2pm": float(s_pm[i]),
                        }
                    )

    return transition_rows, sector_rows


def aggregate_counts(transitions: pd.DataFrame, sectors: pd.DataFrame, by_cube: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys_t = ["r", "lag", "from_state", "to_state"]
    keys_s = ["r", "lag", "state"]
    if by_cube:
        keys_t = ["label"] + keys_t
        keys_s = ["label"] + keys_s

    t = transitions.groupby(keys_t, as_index=False)["count"].sum()
    s = sectors.groupby(keys_s, as_index=False)[["count", "sum_S2p", "sum_S2m", "sum_S2pm"]].sum()
    return t, s


def compute_balance(transitions: pd.DataFrame, by_cube: bool) -> pd.DataFrame:
    group_keys = ["r", "lag"]
    if by_cube:
        group_keys = ["label"] + group_keys

    rows = []
    for group_vals, g in transitions.groupby(group_keys):
        if by_cube:
            label, r, lag = group_vals
        else:
            r, lag = group_vals
            label = "ENSEMBLE"

        mat = pd.DataFrame(0.0, index=STATE_NAMES, columns=STATE_NAMES)
        for _, x in g.iterrows():
            mat.loc[x["from_state"], x["to_state"]] += float(x["count"])

        total = float(mat.to_numpy().sum())
        if total <= 0:
            continue

        row_totals = mat.sum(axis=1)
        for state in STATE_NAMES:
            row_total = float(row_totals.loc[state])
            if row_total > 0:
                N_i = row_total / total
                P_self = float(mat.loc[state, state] / row_total)
                D_i = 1.0 - P_self
            else:
                N_i = np.nan
                P_self = np.nan
                D_i = np.nan

            incoming = 0.0
            for other in STATE_NAMES:
                if other != state:
                    incoming += float(mat.loc[other, state])

            gamma_plus = incoming / total
            gamma_minus = N_i * D_i if np.isfinite(N_i) and np.isfinite(D_i) else np.nan
            gamma_plus_over_D = gamma_plus / D_i if np.isfinite(D_i) and D_i > 0 else np.nan

            rows.append(
                {
                    "label": label,
                    "r": int(r),
                    "lag": int(lag),
                    "state": state,
                    "N_i": N_i,
                    "P_self": P_self,
                    "D_i": D_i,
                    "gamma_plus": gamma_plus,
                    "gamma_minus": gamma_minus,
                    "gamma_plus_over_D": gamma_plus_over_D,
                    "gamma_plus_minus_gamma_minus": gamma_plus - gamma_minus if np.isfinite(gamma_minus) else np.nan,
                }
            )

    return pd.DataFrame(rows).sort_values(["label", "lag", "r", "state"]).reset_index(drop=True)


def compute_sector_means(sectors: pd.DataFrame, by_cube: bool) -> pd.DataFrame:
    keys = ["r", "lag", "state"]
    if by_cube:
        keys = ["label"] + keys

    s = sectors.groupby(keys, as_index=False)[["count", "sum_S2p", "sum_S2m", "sum_S2pm"]].sum()

    rows = []
    for _, x in s.iterrows():
        count = float(x["count"])
        if count > 0:
            A_p = float(x["sum_S2p"] / count)
            A_m = float(x["sum_S2m"] / count)
            A_pm = float(x["sum_S2pm"] / count)
        else:
            A_p = np.nan
            A_m = np.nan
            A_pm = np.nan

        rows.append(
            {
                "label": x["label"] if by_cube else "ENSEMBLE",
                "r": int(x["r"]),
                "lag": int(x["lag"]),
                "state": x["state"],
                "count": count,
                "A_p": A_p,
                "A_m": A_m,
                "A_pm": A_pm,
            }
        )

    return pd.DataFrame(rows).sort_values(["label", "lag", "r", "state"]).reset_index(drop=True)


def reconstruct(balance: pd.DataFrame, sector_means: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for label in sorted(set(balance["label"])):
        b_label = balance[balance["label"] == label]
        s_label = sector_means[sector_means["label"] == label]

        for (r, lag), gb in b_label.groupby(["r", "lag"]):
            gs = s_label[(s_label["r"] == r) & (s_label["lag"] == lag)]
            if len(gs) == 0:
                continue

            for alpha, col in [("+", "A_p"), ("-", "A_m"), ("+-", "A_pm")]:
                direct = 0.0
                sd = 0.0
                sum_sd_weights = 0.0

                for state in STATE_NAMES:
                    brow = gb[gb["state"] == state]
                    srow = gs[gs["state"] == state]
                    if len(brow) != 1 or len(srow) != 1:
                        continue

                    brow = brow.iloc[0]
                    srow = srow.iloc[0]
                    A_i = float(srow[col])
                    N_i = float(brow["N_i"])
                    w_sd = float(brow["gamma_plus_over_D"])

                    if np.isfinite(A_i) and np.isfinite(N_i):
                        direct += N_i * A_i
                    if np.isfinite(A_i) and np.isfinite(w_sd):
                        sd += w_sd * A_i
                        sum_sd_weights += w_sd

                rows.append(
                    {
                        "label": label,
                        "r": int(r),
                        "lag": int(lag),
                        "alpha": alpha,
                        "S2_direct": direct,
                        "S2_sd": sd,
                        "sum_sd_weights": sum_sd_weights,
                        "sqrt_direct": math.sqrt(direct) if direct > 0 else np.nan,
                        "sqrt_sd": math.sqrt(sd) if sd > 0 else np.nan,
                    }
                )

    return pd.DataFrame(rows).sort_values(["label", "lag", "alpha", "r"]).reset_index(drop=True)


def fit_h(x: np.ndarray, y: np.ndarray, fit_min: int, fit_max: int) -> float:
    mask = (x >= fit_min) & (x <= fit_max) & np.isfinite(y) & (y > 0)
    if np.sum(mask) < 3:
        return np.nan
    slope, _ = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(slope)


def mean_sem_by_cube(cube_recon: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (r, lag, alpha), g in cube_recon.groupby(["r", "lag", "alpha"]):
        yd = g["sqrt_direct"].to_numpy(dtype=float)
        ys = g["sqrt_sd"].to_numpy(dtype=float)
        yd = yd[np.isfinite(yd)]
        ys = ys[np.isfinite(ys)]

        rows.append(
            {
                "r": int(r),
                "lag": int(lag),
                "alpha": alpha,
                "direct_mean": float(np.mean(yd)) if len(yd) else np.nan,
                "direct_sem": float(np.std(yd, ddof=1) / math.sqrt(len(yd))) if len(yd) > 1 else np.nan,
                "sd_mean": float(np.mean(ys)) if len(ys) else np.nan,
                "sd_sem": float(np.std(ys, ddof=1) / math.sqrt(len(ys))) if len(ys) > 1 else np.nan,
                "n_cubes_direct": int(len(yd)),
                "n_cubes_sd": int(len(ys)),
            }
        )

    return pd.DataFrame(rows).sort_values(["lag", "alpha", "r"]).reset_index(drop=True)


def make_slope_table(ensemble_recon: pd.DataFrame, cube_mean: pd.DataFrame, fit_min: int, fit_max: int) -> pd.DataFrame:
    rows = []
    for lag in sorted(ensemble_recon["lag"].unique()):
        for alpha in ["+", "-", "+-"]:
            ge = ensemble_recon[(ensemble_recon["label"] == "ENSEMBLE") & (ensemble_recon["lag"] == lag) & (ensemble_recon["alpha"] == alpha)]
            gm = cube_mean[(cube_mean["lag"] == lag) & (cube_mean["alpha"] == alpha)]
            rows.append(
                {
                    "lag": int(lag),
                    "alpha": alpha,
                    "h_ensemble_direct": fit_h(ge["r"].to_numpy(float), ge["sqrt_direct"].to_numpy(float), fit_min, fit_max),
                    "h_ensemble_sd": fit_h(ge["r"].to_numpy(float), ge["sqrt_sd"].to_numpy(float), fit_min, fit_max),
                    "h_cube_mean_direct": fit_h(gm["r"].to_numpy(float), gm["direct_mean"].to_numpy(float), fit_min, fit_max),
                    "h_cube_mean_sd": fit_h(gm["r"].to_numpy(float), gm["sd_mean"].to_numpy(float), fit_min, fit_max),
                    "fit_min": int(fit_min),
                    "fit_max": int(fit_max),
                }
            )
    return pd.DataFrame(rows)


def anchored_line(x: np.ndarray, y: np.ndarray, fit_min: int, fit_max: int, h: float) -> tuple[np.ndarray, np.ndarray]:
    mask = (x >= fit_min) & (x <= fit_max) & np.isfinite(y) & (y > 0)
    xx = np.asarray(x[mask], dtype=float)
    yy = np.asarray(y[mask], dtype=float)
    if len(xx) < 2:
        raise ValueError("Not enough valid points to anchor reference line.")
    order = np.argsort(xx)
    xx = xx[order]
    yy = yy[order]
    rr = np.array([fit_min, fit_max], dtype=float)
    r0 = math.sqrt(float(fit_min * fit_max))
    y0 = float(np.exp(np.interp(np.log(r0), np.log(xx), np.log(yy))))
    return rr, y0 * (rr / r0) ** h


def plot_s2(cube_mean: pd.DataFrame, out_pdf: Path, lag: int, fit_min: int, fit_max: int) -> dict:
    panels = [
        ("+", r"$\left(S_2^+\right)^{1/2}$"),
        ("-", r"$\left(S_2^-\right)^{1/2}$"),
        ("+-", r"$\left(S_2^{+-}\right)^{1/2}$"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
    slopes: dict[str, float] = {}

    for ax, (alpha, ylabel) in zip(axes, panels):
        g = cube_mean[(cube_mean["lag"] == lag) & (cube_mean["alpha"] == alpha)].sort_values("r")
        if len(g) == 0:
            raise RuntimeError(f"No data for lag={lag}, alpha={alpha}")

        r = g["r"].to_numpy(dtype=float)
        yd = g["direct_mean"].to_numpy(dtype=float)
        sd = g["direct_sem"].to_numpy(dtype=float)
        ys = g["sd_mean"].to_numpy(dtype=float)
        ss = g["sd_sem"].to_numpy(dtype=float)

        h_direct = fit_h(r, yd, fit_min, fit_max)
        h_sd = fit_h(r, ys, fit_min, fit_max)
        slopes[f"{alpha}_direct"] = h_direct
        slopes[f"{alpha}_sd"] = h_sd

        ax.set_xscale("log")
        ax.set_yscale("log")

        direct_line, = ax.plot(r, yd, "o-", linewidth=1.4, markersize=4.0, label=rf"direct, $h={h_direct:.3f}$")
        ax.errorbar(r, yd, yerr=sd, fmt="none", ecolor="black", elinewidth=0.9, capsize=3, capthick=0.9)

        sd_line, = ax.plot(r, ys, "s-", linewidth=1.2, markersize=3.8, label=rf"source--depletion, $h={h_sd:.3f}$")
        ax.errorbar(r, ys, yerr=ss, fmt="none", ecolor="black", elinewidth=0.9, capsize=3, capthick=0.9)

        rr14, yy14 = anchored_line(r, yd, fit_min, fit_max, 1.0 / 4.0)
        rr13, yy13 = anchored_line(r, yd, fit_min, fit_max, 1.0 / 3.0)
        h14_line, = ax.plot(rr14, yy14, "--", linewidth=1.1, label=r"$h=1/4$")
        h13_line, = ax.plot(rr13, yy13, ":", linewidth=1.3, label=r"$h=1/3$")

        ax.set_xlabel(r"$r$")
        ax.set_ylabel(ylabel)
        ax.set_xticks([32, 64, 128, 192])
        ax.set_xticklabels(["32", "64", "128", "192"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(direction="in", which="both")
        ax.legend(handles=[direct_line, sd_line, h14_line, h13_line], frameon=False, fontsize=8, loc="upper left", handlelength=2.2)

    fig.tight_layout(w_pad=1.4)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return slopes


def plot_balance(balance: pd.DataFrame, out_pdf: Path, lag: int) -> None:
    b = balance[(balance["label"] == "ENSEMBLE") & (balance["lag"] == lag)]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))

    for ax, state in zip(axes, STATE_NAMES):
        g = b[b["state"] == state].sort_values("r")
        ax.plot(g["r"], g["N_i"], "o-", linewidth=1.2, markersize=4, label=r"$N_i$")
        ax.plot(g["r"], g["gamma_plus_over_D"], "s-", linewidth=1.1, markersize=3.5, label=r"$\gamma_i^+/D_i$")
        ax.set_title(state)
        ax.set_xlabel(r"$r$")
        ax.set_ylabel("fraction")
        ax.legend(frameon=False, fontsize=8)
        ax.tick_params(direction="in", which="both")

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(default_manifest()))
    parser.add_argument("--run-label", default="AP07_320x20_localPerp_meanTheta_n30000")
    parser.add_argument("--r-list", default="32,48,64,80,96,128,160,192")
    parser.add_argument("--lags", default="1,2,4")
    parser.add_argument("--fit-min", type=int, default=32)
    parser.add_argument("--fit-max", type=int, default=160)
    parser.add_argument("--primary-lag", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=30000)
    parser.add_argument("--n-phi", type=int, default=8)
    parser.add_argument("--sector-mode", choices=["mean_theta", "terciles"], default="mean_theta")
    parser.add_argument("--max-cubes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--bl-sigma-factor", type=float, default=0.5)
    parser.add_argument("--bl-downsample", type=int, default=4)
    args = parser.parse_args()

    root = project_root()
    manifest = Path(args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(manifest)

    r_list = parse_int_list(args.r_list)
    lags = parse_int_list(args.lags)

    outdir = root / "04_outputs" / "AP07_320_timewindow_source_depletion" / args.run_label
    plotdir = root / "05_figures" / "AP07_320_timewindow_source_depletion"
    outdir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest)
    if args.max_cubes is not None:
        df = df.head(int(args.max_cubes)).copy()

    print(f"project_root:     {root}")
    print(f"manifest:         {manifest}")
    print(f"run_label:        {args.run_label}")
    print(f"n_cubes:          {len(df)}")
    print(f"r_list:           {r_list}")
    print(f"lags:             {lags}")
    print(f"fit range:        {args.fit_min}..{args.fit_max}")
    print(f"n_samples:        {args.n_samples}")
    print(f"n_phi:            {args.n_phi}")
    print(f"sector_mode:      {args.sector_mode}")
    print(f"BL Gaussian L:    {args.bl_sigma_factor} * r")
    print(f"BL downsample:    {args.bl_downsample}")
    print("sampling:         local-BL-perpendicular at start time; same midpoint-direction pairs at target time")

    all_transition_rows: list[dict] = []
    all_sector_rows: list[dict] = []

    for i, row in df.iterrows():
        t_rows, s_rows = process_cube(
            row=row,
            r_list=r_list,
            lags=lags,
            n_samples=int(args.n_samples),
            n_phi=int(args.n_phi),
            sector_mode=args.sector_mode,
            seed=int(args.seed + 7919 * i),
            downsample=int(args.bl_downsample),
            sigma_factor=float(args.bl_sigma_factor),
        )
        all_transition_rows.extend(t_rows)
        all_sector_rows.extend(s_rows)

    transitions_raw = pd.DataFrame(all_transition_rows)
    sectors_raw = pd.DataFrame(all_sector_rows)

    transitions_ens, sectors_ens = aggregate_counts(transitions_raw, sectors_raw, by_cube=False)
    transitions_cube, sectors_cube = aggregate_counts(transitions_raw, sectors_raw, by_cube=True)

    balance_ens = compute_balance(transitions_ens, by_cube=False)
    balance_cube = compute_balance(transitions_cube, by_cube=True)

    sector_means_ens = compute_sector_means(sectors_ens, by_cube=False)
    sector_means_cube = compute_sector_means(sectors_cube, by_cube=True)

    recon_ens = reconstruct(balance_ens, sector_means_ens)
    recon_cube = reconstruct(balance_cube, sector_means_cube)
    cube_mean = mean_sem_by_cube(recon_cube)
    slopes = make_slope_table(recon_ens, cube_mean, fit_min=args.fit_min, fit_max=args.fit_max)

    transitions_raw.to_csv(outdir / "AP07_transition_counts_raw.csv", index=False)
    sectors_raw.to_csv(outdir / "AP07_sector_sums_raw.csv", index=False)
    balance_ens.to_csv(outdir / "AP07_balance_ensemble.csv", index=False)
    balance_cube.to_csv(outdir / "AP07_balance_by_cube.csv", index=False)
    sector_means_ens.to_csv(outdir / "AP07_sector_means_ensemble.csv", index=False)
    sector_means_cube.to_csv(outdir / "AP07_sector_means_by_cube.csv", index=False)
    recon_ens.to_csv(outdir / "AP07_reconstruction_ensemble.csv", index=False)
    recon_cube.to_csv(outdir / "AP07_reconstruction_by_cube.csv", index=False)
    cube_mean.to_csv(outdir / "AP07_reconstruction_cube_mean_sem.csv", index=False)
    slopes.to_csv(outdir / "AP07_fit_slopes.csv", index=False)

    s2_pdf = plotdir / f"{args.run_label}_S2_lag{args.primary_lag}.pdf"
    balance_pdf = plotdir / f"{args.run_label}_balance_lag{args.primary_lag}.pdf"
    primary_slopes = plot_s2(cube_mean, s2_pdf, lag=args.primary_lag, fit_min=args.fit_min, fit_max=args.fit_max)
    plot_balance(balance_ens, balance_pdf, lag=args.primary_lag)

    metadata = {
        "manifest": str(manifest),
        "run_label": args.run_label,
        "n_cubes": int(len(df)),
        "r_list": r_list,
        "lags": lags,
        "fit_min": int(args.fit_min),
        "fit_max": int(args.fit_max),
        "primary_lag": int(args.primary_lag),
        "n_samples_per_cube_per_r_start": int(args.n_samples),
        "n_phi": int(args.n_phi),
        "sector_mode": args.sector_mode,
        "local_perpendicular": True,
        "BL_filter": "Gaussian coarse-grained magnetic field",
        "BL_sigma_L": "L = bl_sigma_factor * r",
        "bl_sigma_factor": float(args.bl_sigma_factor),
        "bl_downsample": int(args.bl_downsample),
        "transition_geometry": "directions perpendicular to BL at start time; same midpoint-direction pairs evaluated at target time",
        "primary_slopes": primary_slopes,
        "outputs": {
            "outdir": str(outdir),
            "s2_pdf": str(s2_pdf),
            "balance_pdf": str(balance_pdf),
        },
    }
    with open(outdir / "AP07_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nWrote:")
    print(outdir)
    print(s2_pdf)
    print(balance_pdf)

    print("\nFit slopes:")
    print(slopes.to_string(index=False))


if __name__ == "__main__":
    main()