#!/usr/bin/env python3
from pathlib import Path
import zipfile
import hashlib
import datetime
import os

# ============================================================
# make_github_update_zip_v1.py
#
# Purpose:
#   Build a clean ZIP containing the updated scripts, figure PDFs/PNGs,
#   LaTeX tables, metadata/config files, and small processed summaries
#   needed to update the GitHub repository.
#
# It deliberately excludes raw fields and large per-cube NPZ files.
# ============================================================

BASE = Path.cwd()
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR = BASE / "github_update_zips"
OUTDIR.mkdir(parents=True, exist_ok=True)

ZIP_PATH = OUTDIR / f"mhd_dynamic_alignment_github_update_{STAMP}.zip"

# Hard exclusions: never include raw DNS fields or huge binary outputs.
EXCLUDE_PARTS = {
    "raw",
    "__pycache__",
    ".git",
    ".ipynb_checkpoints",
}

EXCLUDE_SUFFIXES = {
    ".npy",
    ".h5",
    ".hdf5",
    ".mat",
}

# Large per-cube NPZ files are reproducible and usually too large for GitHub.
EXCLUDE_NAME_PATTERNS = [
    "processed/perpBL_v1/per_cube/*.npz",
]

# Include patterns. These are intentionally broad enough to catch the
# renamed/fixed plotting scripts but narrow enough to avoid raw data.
INCLUDE_PATTERNS = [
    # Root-level production / plotting / analysis scripts
    "*.py",

    # Configs and cube lists
    "processed/perpBL_v1/config/*.txt",
    "processed/perpBL_v1/config/*.json",

    # Final figures used in the paper and supplement
    "processed/perpBL_v1/figures/*.pdf",
    "processed/perpBL_v1/figures/*.png",

    # Final tables and compact numerical table files
    "processed/perpBL_v1/tables/*.tex",
    "processed/perpBL_v1/tables/*.txt",
    "processed/perpBL_v1/tables/*.npz",
    "processed/perpBL_v1/tables/*.json",

    # Ensemble-level compact summaries
    "processed/perpBL_v1/ensemble/*.npz",
    "processed/perpBL_v1/ensemble/*.json",
    "processed/perpBL_v1/ensemble/*.txt",
    "processed/perpBL_v1/ensemble/*.tex",

    # Small per-cube JSON summaries only, not the large NPZ arrays
    "processed/perpBL_v1/per_cube/*_summary.json",

    # Existing manuscript figure files if stored at root
    "*.pdf",
    "*.png",

    # Optional docs if present
    "README*",
    "LICENSE*",
    "requirements*.txt",
    "environment*.yml",
    "environment*.yaml",
]

def rel(path: Path) -> str:
    return path.relative_to(BASE).as_posix()

def is_excluded(path: Path) -> bool:
    r = rel(path)

    for part in path.parts:
        if part in EXCLUDE_PARTS:
            return True

    if path.suffix in EXCLUDE_SUFFIXES:
        return True

    # Exclude large per-cube NPZ arrays.
    for pat in EXCLUDE_NAME_PATTERNS:
        if path.match(pat):
            return True

    # Exclude the zip output directory itself.
    if r.startswith("github_update_zips/"):
        return True

    return False

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Collect files.
files = set()
for pat in INCLUDE_PATTERNS:
    for p in BASE.glob(pat):
        if p.is_file() and not is_excluded(p):
            files.add(p)

files = sorted(files, key=lambda p: rel(p))

# Write manifest text.
manifest_lines = []
manifest_lines.append("# GitHub update ZIP manifest")
manifest_lines.append(f"# Created: {datetime.datetime.now().isoformat(timespec='seconds')}")
manifest_lines.append(f"# Base: {BASE}")
manifest_lines.append("")
manifest_lines.append("# Included files:")
manifest_lines.append("")

total_bytes = 0
for p in files:
    size = p.stat().st_size
    total_bytes += size
    manifest_lines.append(f"{sha256(p)}  {size:12d}  {rel(p)}")

manifest_lines.append("")
manifest_lines.append(f"# File count: {len(files)}")
manifest_lines.append(f"# Total included size: {total_bytes / (1024**2):.2f} MB")
manifest_text = "\n".join(manifest_lines) + "\n"

manifest_path = OUTDIR / f"MANIFEST_{STAMP}.txt"
manifest_path.write_text(manifest_text)

# Create zip.
with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in files:
        z.write(p, arcname=rel(p))
    z.write(manifest_path, arcname="MANIFEST.txt")

print("=" * 100)
print("GitHub update ZIP created")
print("=" * 100)
print(f"ZIP:       {ZIP_PATH}")
print(f"MANIFEST:  {manifest_path}")
print(f"Files:     {len(files)}")
print(f"Size:      {ZIP_PATH.stat().st_size / (1024**2):.2f} MB")
print("=" * 100)
print("Included top-level summary:")
for p in files[:20]:
    print("  ", rel(p))
if len(files) > 20:
    print(f"  ... and {len(files)-20} more")
print("=" * 100)
print("Excluded by design: raw fields, .npy/.mat/.h5 files, .git, __pycache__, and large per-cube NPZ arrays.")
