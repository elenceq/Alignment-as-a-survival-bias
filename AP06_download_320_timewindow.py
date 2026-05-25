#!/usr/bin/env python3
"""
AP06_download_320_timewindow.py

Download missing 320^3 time-window cutouts for the AmplitudeProject
source-depletion ensemble test.

Input:
  00_admin/AP06_missing_timewindow_files_20.csv

Output:
  raw files are written to the existing parent raw/ directory, using names:
    mhd1024_velocity_tXXXX_xA-B_yC-D_zE-F.npy
    mhd1024_magneticfield_tXXXX_xA-B_yC-D_zE-F.npy

Rules:
  - No filesystem discovery.
  - Download only rows listed in the missing manifest.
  - Skip existing valid files.
  - Write through .tmp.npy first, then atomically rename.
  - Verify final shape is (320, 320, 320, 3).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SHAPE = (320, 320, 320, 3)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def raw_dir() -> Path:
    return project_root().parent / "raw"


def default_manifest() -> Path:
    return project_root() / "00_admin" / "AP06_missing_timewindow_files_20.csv"


def output_dir() -> Path:
    return project_root() / "04_outputs" / "AP06_download_320_timewindow"


def load_giverny(dataset_name: str, auth_token: str):
    from pathlib import Path
    import inspect
    from giverny.turbulence_dataset import turb_dataset
    from giverny.turbulence_toolkit import getCutout

    if not auth_token:
        raise ValueError("Missing JHTDB/Giverny auth token.")

    cache_dir = output_dir() / "giverny_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    sig = inspect.signature(turb_dataset)
    params = sig.parameters

    kwargs = {}
    if "dataset_title" in params:
        kwargs["dataset_title"] = dataset_name
    elif "dataset_name" in params:
        kwargs["dataset_name"] = dataset_name

    if "output_path" in params:
        kwargs["output_path"] = str(cache_dir)

    if "auth_token" in params:
        kwargs["auth_token"] = auth_token
    elif "token" in params:
        kwargs["token"] = auth_token

    try:
        if kwargs:
            cube = turb_dataset(**kwargs)
        else:
            cube = turb_dataset(dataset_name, str(cache_dir), auth_token)
    except TypeError:
        try:
            cube = turb_dataset(dataset_name, str(cache_dir), auth_token)
        except TypeError:
            try:
                cube = turb_dataset(dataset_name, auth_token)
            except TypeError:
                cube = turb_dataset(dataset_name)

    for attr in ["auth_token", "authToken", "token"]:
        try:
            setattr(cube, attr, auth_token)
        except Exception:
            pass

    for method in ["set_auth_token", "setAuthToken", "set_token", "add_token"]:
        if hasattr(cube, method):
            try:
                getattr(cube, method)(auth_token)
            except TypeError:
                pass

    if hasattr(cube, "initialize"):
        try:
            cube.initialize()
        except TypeError:
            try:
                cube.initialize(dataset_name)
            except TypeError:
                try:
                    cube.initialize(auth_token)
                except TypeError:
                    pass

    return cube, getCutout

def dataset_to_array(obj) -> np.ndarray:
    if hasattr(obj, "data_vars"):
        names = list(obj.data_vars)
        if len(names) == 0:
            raise TypeError("xarray Dataset has no data variables.")
        arr = obj[names[0]].values
        dims = tuple(obj[names[0]].dims)

        arr = np.asarray(arr)
        arr = np.squeeze(arr)

        if arr.ndim != 4:
            raise ValueError(f"After squeeze, expected 4D array, got shape {arr.shape}, dims={dims}")

        comp_axes = [i for i, s in enumerate(arr.shape) if s == 3]
        if len(comp_axes) != 1:
            raise ValueError(f"Cannot identify vector-component axis in shape {arr.shape}, dims={dims}")

        comp_axis = comp_axes[0]
        arr = np.moveaxis(arr, comp_axis, -1)

        spatial = arr.shape[:3]
        if sorted(spatial) != [320, 320, 320]:
            raise ValueError(f"Unexpected spatial shape after component move: {arr.shape}, dims={dims}")

        return np.asarray(arr, dtype=np.float32)

    if hasattr(obj, "values"):
        arr = np.asarray(obj.values)
    else:
        arr = np.asarray(obj)

    arr = np.squeeze(arr)

    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array, got shape {arr.shape}")

    comp_axes = [i for i, s in enumerate(arr.shape) if s == 3]
    if len(comp_axes) != 1:
        raise ValueError(f"Cannot identify vector-component axis in shape {arr.shape}")

    arr = np.moveaxis(arr, comp_axes[0], -1)

    return np.asarray(arr, dtype=np.float32)


def call_getcutout(cube, getCutout, variable: str, t: int, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int):
    xyzt_axes_ranges = np.array(
        [
            [int(x0), int(x1)],
            [int(y0), int(y1)],
            [int(z0), int(z1)],
            [int(t), int(t)],
        ],
        dtype=np.int32,
    )
    xyzt_strides = np.array([1, 1, 1, 1], dtype=np.int32)

    calls = [
        lambda: getCutout(cube, variable, xyzt_axes_ranges, xyzt_strides, False, False),
        lambda: getCutout(cube, variable, xyzt_axes_ranges, xyzt_strides, False),
        lambda: getCutout(cube, variable, xyzt_axes_ranges, xyzt_strides),
    ]

    last_error = None
    for fn in calls:
        try:
            return fn()
        except TypeError as exc:
            last_error = exc

    raise last_error

def expected_path(row, variable: str) -> Path:
    if variable == "velocity":
        return Path(row["velocity"])
    if variable == "magneticfield":
        return Path(row["magnetic"])
    raise ValueError(variable)


def valid_npy(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return tuple(arr.shape) == EXPECTED_SHAPE
    except Exception:
        return False


def download_one(cube, getCutout, row, variable: str, force: bool = False) -> dict:
    target = expected_path(row, variable)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and valid_npy(target) and not force:
        return {
            "label": row["label"],
            "t": int(row["t"]),
            "variable": variable,
            "path": str(target),
            "status": "skipped_exists",
            "seconds": 0.0,
            "shape": str(EXPECTED_SHAPE),
        }

    tmp = target.with_suffix(".tmp.npy")
    if tmp.exists():
        tmp.unlink()

    t0 = time.time()

    print(
        f"[download] {row['label']} t={int(row['t']):04d} {variable} "
        f"x{int(row['x0'])}-{int(row['x1'])} "
        f"y{int(row['y0'])}-{int(row['y1'])} "
        f"z{int(row['z0'])}-{int(row['z1'])}",
        flush=True,
    )

    obj = call_getcutout(
        cube=cube,
        getCutout=getCutout,
        variable=variable,
        t=int(row["t"]),
        x0=int(row["x0"]),
        x1=int(row["x1"]),
        y0=int(row["y0"]),
        y1=int(row["y1"]),
        z0=int(row["z0"]),
        z1=int(row["z1"]),
    )

    arr = dataset_to_array(obj)

    if tuple(arr.shape) != EXPECTED_SHAPE:
        raise ValueError(f"{variable} returned shape {arr.shape}, expected {EXPECTED_SHAPE}")

    np.save(tmp, arr)

    if not valid_npy(tmp):
        raise RuntimeError(f"Temporary file failed validation: {tmp}")

    os.replace(tmp, target)

    seconds = time.time() - t0

    return {
        "label": row["label"],
        "t": int(row["t"]),
        "variable": variable,
        "path": str(target),
        "status": "downloaded",
        "seconds": seconds,
        "shape": str(arr.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(default_manifest()))
    parser.add_argument("--dataset", default="mhd1024")
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--max-snapshots", type=int, default=None)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-name", default="AP06_download_progress.csv")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    outdir = output_dir()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest)

    required = {
        "label", "t", "x0", "x1", "y0", "y1", "z0", "z1",
        "velocity", "magnetic", "complete",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    work = df[~df["complete"].astype(bool)].copy()
    work = work.iloc[int(args.start_row):].reset_index(drop=True)

    if args.max_snapshots is not None:
        work = work.head(int(args.max_snapshots)).copy()

    if len(work) == 0:
        print("No incomplete snapshots to download.")
        return

    print(f"Project root: {project_root()}")
    print(f"Raw dir:      {raw_dir()}")
    print(f"Manifest:     {manifest}")
    print(f"Rows:         {len(work)}")
    print(f"Dataset:      {args.dataset}")
    print(flush=True)

    auth_token = args.auth_token or os.environ.get("JHTDB_TOKEN") or os.environ.get("JHTDB_AUTH_TOKEN")
    cube, getCutout = load_giverny(args.dataset, auth_token)

    records = []
    progress_csv = outdir / args.progress_name

    for i, row in work.iterrows():
        print(f"\n[row {i + 1}/{len(work)}] {row['label']} t={int(row['t']):04d}", flush=True)

        for variable in ["velocity", "magneticfield"]:
            try:
                rec = download_one(
                    cube=cube,
                    getCutout=getCutout,
                    row=row,
                    variable=variable,
                    force=args.force,
                )
                records.append(rec)
                print(f"  {variable}: {rec['status']} {rec['seconds']:.1f}s", flush=True)
            except Exception as exc:
                rec = {
                    "label": row["label"],
                    "t": int(row["t"]),
                    "variable": variable,
                    "path": str(expected_path(row, variable)),
                    "status": "FAILED",
                    "seconds": np.nan,
                    "shape": "",
                    "error": repr(exc),
                }
                records.append(rec)
                pd.DataFrame(records).to_csv(progress_csv, index=False)
                print(f"  {variable}: FAILED {exc!r}", flush=True)
                raise

        pd.DataFrame(records).to_csv(progress_csv, index=False)

    metadata = {
        "manifest": str(manifest),
        "dataset": args.dataset,
        "max_snapshots": args.max_snapshots,
        "start_row": args.start_row,
        "force": args.force,
        "progress_csv": str(progress_csv),
        "n_records": len(records),
    }

    with open(outdir / "AP06_download_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone.")
    print(f"Progress CSV: {progress_csv}")


if __name__ == "__main__":
    main()