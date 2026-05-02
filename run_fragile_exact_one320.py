from pathlib import Path
import re, json, time, argparse
import numpy as np
from scipy.ndimage import convolve1d
from scipy.optimize import minimize_scalar

# ============================================================
# Exact compact-kernel replication for one pressure-complete 320^3 cube
#
# Geometric part follows saved 448^3 JSON convention:
#   kernel: separable compact-support bump
#   phi(s)=exp(-1/(1-s^2)), |s|<1
#   ell = support radius in grid cells
#   margin = ell + 1
#
# G_l^± = (z_l^± - z_l^∓).grad z_l^± - grad Pi_l + N_l^±
# Pi = p + |b|^2/2
# N_l^± = -div tau_l^± + nu Laplacian(z_l^±) + F_l
# tau_l^± = filter(z^∓ z^±) - z_l^∓ z_l^±
# C_l^± = max singular value of centered-difference Jacobian of G_l^±
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--stem", default="t0022_x321-640_y641-960_z641-960")
parser.add_argument("--nbase", type=int, default=100000)
parser.add_argument("--skip-stochastic", action="store_true")
parser.add_argument("--skip-geometric", action="store_true")
args = parser.parse_args()

BASE = Path("/home/idies/workspace/Storage/elenceq/mhd_work/jhtdb_mhd1024")
RAW = BASE / "raw"
OUT = BASE / "processed" / "fragile_alignment_exact_320"
OUT.mkdir(parents=True, exist_ok=True)

stem = args.stem
u_path = RAW / f"mhd1024_velocity_{stem}.npy"
b_path = RAW / f"mhd1024_magneticfield_{stem}.npy"
p_path = RAW / f"mhd1024_pressure_{stem}.npy"

print("stem:", stem, flush=True)
print("u:", u_path, u_path.exists(), flush=True)
print("b:", b_path, b_path.exists(), flush=True)
print("p:", p_path, p_path.exists(), flush=True)

m = re.match(r"t(\d+)_x(\d+)-(\d+)_y(\d+)-(\d+)_z(\d+)-(\d+)", stem)
if not m:
    raise ValueError(f"Could not parse stem: {stem}")
tidx, x0, x1, y0, y1, z0, z1 = map(int, m.groups())

dx = 2*np.pi/1024.0
nu = 1.1e-4
eta = 1.1e-4
f0 = 0.25
kf = 2.0

u = np.load(u_path, mmap_mode="r")
b = np.load(b_path, mmap_mode="r")
p = np.load(p_path, mmap_mode="r")
N = u.shape[0]

print("shape:", u.shape, b.shape, p.shape, flush=True)
if u.shape != (N, N, N, 3) or b.shape != (N, N, N, 3) or p.shape != (N, N, N):
    raise ValueError("Unexpected shapes")

# Load full cube into RAM for geometric part.
# This is intentional: exact compact-kernel stress calculation repeatedly accesses fields.
u_full = np.asarray(u, dtype=np.float32)
b_full = np.asarray(b, dtype=np.float32)
p_full = np.asarray(p, dtype=np.float32)

zp = u_full + b_full
zm = u_full - b_full
Pi = p_full + 0.5*np.sum(b_full*b_full, axis=-1)

def bump_kernel(radius):
    r = int(radius)
    k = np.arange(-r, r+1, dtype=np.float64)
    s = k / float(r)
    w = np.zeros_like(s)
    mask = np.abs(s) < 1.0
    w[mask] = np.exp(-1.0/(1.0 - s[mask]**2))
    w /= w.sum()
    return w.astype(np.float32)

def filt_scalar(a, ell):
    w = bump_kernel(ell)
    out = np.asarray(a, dtype=np.float32)
    for ax in range(3):
        out = convolve1d(out, w, axis=ax, mode="nearest")
    return out.astype(np.float32, copy=False)

def filt_vec(v, ell):
    out = np.empty_like(v, dtype=np.float32)
    for c in range(3):
        out[..., c] = filt_scalar(v[..., c], ell)
    return out

def grad_scalar(f):
    gx, gy, gz = np.gradient(f, dx, dx, dx, edge_order=2)
    return np.stack([gx, gy, gz], axis=-1).astype(np.float32)

def lap_vec(v):
    out = np.empty_like(v, dtype=np.float32)
    for c in range(3):
        f = v[..., c]
        gxx = np.gradient(np.gradient(f, dx, axis=0, edge_order=2), dx, axis=0, edge_order=2)
        gyy = np.gradient(np.gradient(f, dx, axis=1, edge_order=2), dx, axis=1, edge_order=2)
        gzz = np.gradient(np.gradient(f, dx, axis=2, edge_order=2), dx, axis=2, edge_order=2)
        out[..., c] = gxx + gyy + gzz
    return out.astype(np.float32)

def advective(zadv_l, ztar_l):
    out = np.empty_like(ztar_l, dtype=np.float32)
    a0, a1, a2 = zadv_l[...,0], zadv_l[...,1], zadv_l[...,2]
    for c in range(3):
        gx, gy, gz = np.gradient(ztar_l[..., c], dx, dx, dx, edge_order=2)
        out[..., c] = a0*gx + a1*gy + a2*gz
    return out.astype(np.float32)

def forcing_field_filtered(ell):
    xs = (np.arange(x0, x1+1, dtype=np.float32) - 1.0) * dx
    ys = (np.arange(y0, y1+1, dtype=np.float32) - 1.0) * dx
    zs = (np.arange(z0, z1+1, dtype=np.float32) - 1.0) * dx

    sx = np.sin(kf*xs)[:,None,None]
    cx = np.cos(kf*xs)[:,None,None]
    sy = np.sin(kf*ys)[None,:,None]
    cy = np.cos(kf*ys)[None,:,None]
    cz = np.cos(kf*zs)[None,None,:]

    F = np.zeros((N, N, N, 3), dtype=np.float32)
    F[...,0] = f0 * sx * cy * cz
    F[...,1] = -f0 * cx * sy * cz
    # F[...,2] = 0
    return filt_vec(F, ell)

def div_tau(advector, target, adv_l, target_l, ell):
    """
    tau_i_j = filter(advector_j * target_i) - adv_l_j * target_l_i
    div_tau_i = sum_j d_j tau_i_j
    """
    div = np.zeros_like(target_l, dtype=np.float32)

    for i in range(3):
        acc = np.zeros(target_l.shape[:-1], dtype=np.float32)
        for j in range(3):
            prod = np.asarray(advector[..., j] * target[..., i], dtype=np.float32)
            tau = filt_scalar(prod, ell) - adv_l[..., j] * target_l[..., i]
            d_tau = np.gradient(tau, dx, axis=j, edge_order=2)
            acc += d_tau.astype(np.float32)
            del prod, tau, d_tau
        div[..., i] = acc
        del acc

    return div

def max_spectral_norm_jacobian(G, margin, chunk_x=4):
    """
    Exact max singular value of 3x3 Jacobian of vector field G over valid interior.
    Uses chunked SVD to avoid forming all matrices at once.
    """
    derivs = []
    for i in range(3):
        for ax in range(3):
            derivs.append(np.gradient(G[..., i], dx, axis=ax, edge_order=2).astype(np.float32))

    sl_y = slice(margin, N-margin)
    sl_z = slice(margin, N-margin)
    max_sv = 0.0

    for xa in range(margin, N-margin, chunk_x):
        xb = min(N-margin, xa+chunk_x)
        sl_x = slice(xa, xb)
        shape = derivs[0][sl_x, sl_y, sl_z].shape
        npts = int(np.prod(shape))

        M = np.empty((npts, 3, 3), dtype=np.float32)
        idx = 0
        for i in range(3):
            for ax in range(3):
                M[:, i, ax] = derivs[idx][sl_x, sl_y, sl_z].reshape(-1)
                idx += 1

        svals = np.linalg.svd(M, compute_uv=False)
        local = float(np.max(svals[:, 0]))
        if local > max_sv:
            max_sv = local
        del M, svals

    del derivs
    return float(max_sv)

def compute_C_for_sign(ell, sign, zp_l, zm_l, Pi_l, F_l):
    gradPi = grad_scalar(Pi_l)

    if sign == "+":
        ztar, zadv = zp, zm
        ztar_l, zadv_l = zp_l, zm_l
        transport = zp_l - zm_l
    else:
        ztar, zadv = zm, zp
        ztar_l, zadv_l = zm_l, zp_l
        transport = zm_l - zp_l

    print(f"    sign {sign}: advective term", flush=True)
    adv = advective(transport, ztar_l)

    print(f"    sign {sign}: stress divergence", flush=True)
    divt = div_tau(zadv, ztar, zadv_l, ztar_l, ell)

    print(f"    sign {sign}: laplacian + forcing + G", flush=True)
    Nterm = -divt + nu*lap_vec(ztar_l) + F_l
    G = adv - gradPi + Nterm

    margin = ell + 1
    print(f"    sign {sign}: max singular value, margin={margin}", flush=True)
    C = max_spectral_norm_jacobian(G, margin=margin)

    del gradPi, adv, divt, Nterm, G
    return C

# ------------------------------------------------------------
# Geometric C_l exact compact-kernel test
# ------------------------------------------------------------
if not args.skip_geometric:
    ell_list = [32, 40, 48, 64, 80, 96, 128]
    rows = []

    print("\n=== GEOMETRIC EXACT COMPACT-KERNEL TEST ===", flush=True)

    for ell in ell_list:
        t0 = time.time()
        margin = ell + 1
        nvalid = N - 2*margin
        if nvalid <= 0:
            print(f"Skipping ell={ell}: no valid interior", flush=True)
            continue

        print(f"\nell={ell}, margin={margin}, valid interior={nvalid}^3", flush=True)

        print("  filtering z+ z- Pi F", flush=True)
        zp_l = filt_vec(zp, ell)
        zm_l = filt_vec(zm, ell)
        Pi_l = filt_scalar(Pi, ell)
        F_l = forcing_field_filtered(ell)

        C_plus = compute_C_for_sign(ell, "+", zp_l, zm_l, Pi_l, F_l)
        C_minus = compute_C_for_sign(ell, "-", zp_l, zm_l, Pi_l, F_l)

        row = {
            "ell_grid": int(ell),
            "ell_physical": float(ell*dx),
            "kernel": "separable_smooth_compact_bump",
            "support_radius_grid": int(ell),
            "margin_used": int(margin),
            "valid_interior": int(nvalid),
            "C_plus": float(C_plus),
            "C_minus": float(C_minus),
            "C_avg": float(0.5*(C_plus + C_minus)),
            "seconds": float(time.time() - t0),
        }
        rows.append(row)

        print("  RESULT ell={ell}: C+={cp:.6g}, C-={cm:.6g}, Cavg={ca:.6g}, seconds={sec:.1f}".format(
            ell=ell, cp=C_plus, cm=C_minus, ca=row["C_avg"], sec=row["seconds"]
        ), flush=True)

        del zp_l, zm_l, Pi_l, F_l

    geo = {
        "cube": {
            "stem": stem,
            "time_index": tidx,
            "x_range_1based": [x0, x1],
            "y_range_1based": [y0, y1],
            "z_range_1based": [z0, z1],
            "grid_size": int(N),
            "dx": float(dx),
        },
        "physics": {
            "z_definition": "z^± = u ± b",
            "pressure_used": "Pi = p + |b|^2 / 2",
            "nu": nu,
            "eta": eta,
            "forcing": {
                "type": "Taylor-Green",
                "kf": kf,
                "f0": f0,
                "Fx": "f0*sin(kf*x)*cos(kf*y)*cos(kf*z)",
                "Fy": "-f0*cos(kf*x)*sin(kf*y)*cos(kf*z)",
                "Fz": "0",
            },
        },
        "kernel": {
            "type": "smooth compact-support separable bump",
            "phi(s)": "exp(-1/(1-s^2)) for |s|<1, else 0",
            "ell_meaning": "support radius in grid cells, per coordinate",
        },
        "discrete_definition": {
            "tau": "tau_l^± = filter(z^∓ z^±) - z_l^∓ z_l^±",
            "N": "N_l^± = -div tau_l^± + nu Laplacian(z_l^±) + F_l",
            "G": "G_l^± = (z_l^± - z_l^∓)·grad z_l^± - grad Pi_l + N_l^±",
            "C": "max singular value of centered-difference Jacobian of G_l^± over valid interior",
        },
        "rows": rows,
    }

    geo_path = OUT / f"Cl_exact_compact_320_{stem}.json"
    with open(geo_path, "w") as f:
        json.dump(geo, f, indent=2)

    print("\nSaved geometric JSON:", geo_path, flush=True)

# ------------------------------------------------------------
# Stochastic strong-event/null test
# ------------------------------------------------------------
if not args.skip_stochastic:
    print("\n=== STOCHASTIC ANGLE-AMPLITUDE TEST ===", flush=True)

    # use memmaps to avoid duplicate RAM
    u_mm = np.load(u_path, mmap_mode="r")
    b_mm = np.load(b_path, mmap_mode="r")

    r_list = np.array([32, 40, 48, 64, 80, 96, 128], dtype=int)
    n_base = int(args.nbase)
    n_phi = 8
    rng = np.random.RandomState(1234)
    q_edges = [0.00, 0.50, 0.80, 0.95, 1.00]
    bin_labels = [f"{int(100*q_edges[i])}-{int(100*q_edges[i+1])}%" for i in range(len(q_edges)-1)]

    def interp_xy_vec(arr, xs, ys, zs):
        x0i = np.floor(xs).astype(np.int64)
        y0i = np.floor(ys).astype(np.int64)
        x1i = x0i + 1
        y1i = y0i + 1
        wx = (xs - x0i).astype(np.float32)
        wy = (ys - y0i).astype(np.float32)

        v00 = arr[x0i, y0i, zs, :]
        v10 = arr[x1i, y0i, zs, :]
        v01 = arr[x0i, y1i, zs, :]
        v11 = arr[x1i, y1i, zs, :]

        out = ((1-wx)*(1-wy))[:,None]*v00 + (wx*(1-wy))[:,None]*v10 \
            + ((1-wx)*wy)[:,None]*v01 + (wx*wy)[:,None]*v11
        return out.astype(np.float32, copy=False)

    def rho_unsigned_norm_log(theta, a):
        th = np.clip(theta, 1e-10, 0.5*np.pi - 1e-10)
        if a < 1e-10:
            return np.log(np.sin(th))
        Z = np.sinh(a)/a
        return np.log(np.sin(th)) + np.log(np.cosh(a*np.cos(th))) - np.log(Z)

    def fit_a(theta, max_n=60000):
        theta = np.asarray(theta, dtype=np.float64)
        if theta.size > max_n:
            idx = rng.choice(theta.size, size=max_n, replace=False)
            theta = theta[idx]
        def nll(a):
            return -float(np.sum(rho_unsigned_norm_log(theta, a)))
        res = minimize_scalar(nll, bounds=(0.0, 12.0), method="bounded",
                              options={"xatol": 1e-3, "maxiter": 80})
        return float(res.x)

    rows = []

    for r in r_list:
        t0 = time.time()
        print(f"\nr={r}", flush=True)

        margin = r + 3
        xs0 = rng.uniform(margin, N-margin-2, size=n_base).astype(np.float32)
        ys0 = rng.uniform(margin, N-margin-2, size=n_base).astype(np.float32)
        zs0 = rng.randint(0, N, size=n_base).astype(np.int64)

        theta_parts = []
        A_parts = []

        for phi in np.linspace(0, 2*np.pi, n_phi, endpoint=False):
            dxr = r*np.cos(phi)
            dyr = r*np.sin(phi)

            u0s = interp_xy_vec(u_mm, xs0, ys0, zs0)
            b0s = interp_xy_vec(b_mm, xs0, ys0, zs0)
            u1s = interp_xy_vec(u_mm, xs0+dxr, ys0+dyr, zs0)
            b1s = interp_xy_vec(b_mm, xs0+dxr, ys0+dyr, zs0)

            dzp = (u1s + b1s) - (u0s + b0s)
            dzm = (u1s - b1s) - (u0s - b0s)

            np_norm = np.linalg.norm(dzp, axis=1)
            nm_norm = np.linalg.norm(dzm, axis=1)
            A = np_norm * nm_norm
            good = A > 1e-14

            cosang = np.sum(dzp[good]*dzm[good], axis=1) / A[good]
            cosang = np.clip(cosang, -1.0, 1.0)
            theta = np.arccos(np.abs(cosang)).astype(np.float32)

            theta_parts.append(theta)
            A_parts.append(A[good].astype(np.float32))

        theta = np.concatenate(theta_parts)
        A = np.concatenate(A_parts)
        A_null = rng.permutation(A)
        sinth = np.sin(theta)

        theta_unw = float(np.mean(theta))
        theta_w = float(np.sum(A*theta)/np.sum(A))
        theta_null = float(np.sum(A_null*theta)/np.sum(A_null))

        cov_real = float(np.mean((A-A.mean())*(sinth-sinth.mean())))
        cov_null = float(np.mean((A_null-A_null.mean())*(sinth-sinth.mean())))

        q = np.quantile(A, q_edges)
        theta_bins = []
        theta_bins_null = []
        a_bins = []
        a_bins_null = []

        for k in range(len(q_edges)-1):
            lo, hi = q[k], q[k+1]
            if k == len(q_edges)-2:
                sel = (A >= lo) & (A <= hi)
                seln = (A_null >= lo) & (A_null <= hi)
            else:
                sel = (A >= lo) & (A < hi)
                seln = (A_null >= lo) & (A_null < hi)

            th_real = theta[sel]
            th_null = theta[seln]

            theta_bins.append(float(np.degrees(np.mean(th_real))))
            theta_bins_null.append(float(np.degrees(np.mean(th_null))))
            a_bins.append(fit_a(th_real))
            a_bins_null.append(fit_a(th_null))

        row = {
            "r": int(r),
            "n_events": int(theta.size),
            "theta_unweighted_deg": float(np.degrees(theta_unw)),
            "theta_A_weighted_deg": float(np.degrees(theta_w)),
            "theta_shuffled_null_deg": float(np.degrees(theta_null)),
            "cov_A_sintheta": cov_real,
            "cov_A_sintheta_null": cov_null,
            "bin_labels": bin_labels,
            "theta_bins_deg": theta_bins,
            "theta_bins_null_deg": theta_bins_null,
            "a_bins": a_bins,
            "a_bins_null": a_bins_null,
            "seconds": float(time.time() - t0),
        }
        rows.append(row)

        print("  theta_unw={:.3f}, theta_A={:.3f}, theta_null={:.3f}, cov={:.3e}, nullcov={:.3e}, seconds={:.1f}".format(
            row["theta_unweighted_deg"], row["theta_A_weighted_deg"],
            row["theta_shuffled_null_deg"], row["cov_A_sintheta"],
            row["cov_A_sintheta_null"], row["seconds"]
        ), flush=True)

    stoch = {
        "cube": {
            "stem": stem,
            "time_index": tidx,
            "grid_size": int(N),
            "x_range_1based": [x0, x1],
            "y_range_1based": [y0, y1],
            "z_range_1based": [z0, z1],
        },
        "method": {
            "r_list": r_list.tolist(),
            "n_base": n_base,
            "n_phi": n_phi,
            "directions": "8 xy-plane directions, bilinear interpolation in x,y",
            "A": "A_r = |delta z+| |delta z-|",
            "theta": "folded unsigned angle arccos(|cos theta|)",
            "null": "A_r randomly permuted relative to theta_r",
            "bins": q_edges,
        },
        "rows": rows,
    }

    stoch_path = OUT / f"stochastic_compact_320_{stem}.json"
    with open(stoch_path, "w") as f:
        json.dump(stoch, f, indent=2)

    print("\nSaved stochastic JSON:", stoch_path, flush=True)

print("\nDONE", flush=True)
