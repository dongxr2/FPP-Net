# Packaging validation — 2026-11-08

The portable repository was executed on the local Python3.11 environment, independently of the historical research-module import chain.

Completed checks:

1. Rebuilt all18KAIST and56CEITEC processed records from the original local TDMS/MAT files. All74raw-file hashes and74derived feature-array hashes matched the frozen release.
2. Verified all counts, canonical record order, train/validation/test record separation and CEITEC load separation. Each eligible window is tested exactly once across outer folds.
3. Loaded all18released FPP-Net models on CPU. Per-seed target probabilities matched the original saved predictions within the declared1e-4tolerance. Final ensemble results: KAIST57.375%window/75%record; CEITEC96.466767%window/100%record.
4. Refit the spectral LR baseline from the packaged features. Results: KAIST59.597222%window/63.888889%record; CEITEC86.168981%window/94.444444%record.
5. Re-ran all four acquisition-group/split analyses from the packaged45-record LR features.
6. Re-ran frequency-position analysis using only the matching outer fold's frozen models and test windows. Half-norm positions26/27; profile correlations0.409137/0.712639.
7. Executed one-epoch KAIST fold0/seed20261014 training smoke test, including both source validation runs and the refit branch. These development weights are excluded from the repository/release and are not paper results.
8. Passed frontend common-gain invariance, architecture/parameter/backpropagation and nonfinite-input unit checks.

The other classical baseline commands have been matched to their source configurations, but full retraining of every comparator and seed was not repeated for packaging. The download commands use official endpoints and checksums; multi-gigabyte files already available locally were not unnecessarily downloaded again. CI is supplied but has not yet run on GitHub, since no repository was published in this session.
