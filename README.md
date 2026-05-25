# Dynamic Alignment as a Statistical Survival Effect

This repository contains analysis scripts, processed diagnostic summaries, and plotting material for the paper.

The raw JHTDB and Wind data are not redistributed here. The repository stores scripts, manifests, processed CSV summaries, and figure-generation files needed to reproduce the reported diagnostics from the downloaded data.

## Main retention-theory files

- `AP06_download_320_timewindow.py`: downloads the twenty 320^3 five-snapshot JHTDB time windows used for the finite-time retention test.
- `AP06_320_timewindow_manifest_20.csv`: manifest of the twenty cubes and their five time snapshots.
- `AP07_320_timewindow_source_depletion.py`: local-B_L-perpendicular twenty-cube retention calculation.
- `AP07_make_fig_source_depletion.py`: plot-only script for the source-depletion reconstruction figure.
- `AP07_reconstruction_cube_mean_sem.csv`: processed cube-level mean/SEM data used for the source-depletion reconstruction figure.
- `AP07_fit_slopes.csv` or `AP07_fit_slopes_clean_plot.csv`: fitted slopes for the direct and retention-balance reconstructions.
- `AP07_320x20_localPerp_meanTheta_n30000_S2_lag1.pdf`: final source-depletion reconstruction figure.

## Method note

The finite-time retention calculation uses centered Elsasser increments whose separation directions are chosen perpendicular to the local Gaussian-filtered magnetic field B_L. The Gaussian coarse-graining scale is L = r/2, and eight directions are sampled in the local perpendicular plane.

## Data note

Raw simulation cutouts must be obtained separately from the Johns Hopkins Turbulence Database. NASA Wind data products are obtained from CDAWeb. Processed summaries and plotting inputs are included here for reproducibility of the reported figures.
