# Derived datasets and original measurements

This project constructs **evaluation tasks** from public measurements. KAIST-XM and CEITEC-XL are not newly acquired datasets.

| Property | KAIST-XM | CEITEC-XL |
|---|---|---|
| Source | Mendeley rgn5brrgrn v5 | Zenodo 13889418 v1.0 |
| Signal | one vibration channel | three SubSys1 currents |
| Sampling rate | 25,600 Hz | 10,000 Hz |
| Window | 5120 samples / 0.2 s | 1710 samples / 0.171 s |
| Records | 18: 3 healthy, 15 faulty | 56: 8 healthy, 48 faulty |
| Windows | 10,800: 1800 healthy, 9000 faulty | 6717: 960 healthy, 5757 faulty |
| Domain | motor rating | load group |
| Input features | `(B,2,1024)` | `(B,6,1024)` |
| Label | 0 healthy, 1 stator fault | 0 healthy, 1 interturn fault |

KAIST: retain acquisition group A (January 25, 2022, ai3), count each healthy acquisition only once, and merge interturn/intercoil faults into the positive class. The 3 kW subset contains no intercoil records. This is binary fault detection, not three-class fault-type classification.

CEITEC: retain a nonoverlapping window only if **every** speed sample is 500–4000 rpm. Most records retain120windows; 5 Nm/1turn retains121; 35 Nm/5turn and35 Nm/6turn retain118 each. This is a consequence of speed-boundary screening, not missing labels. The 160-window cap does not bind any record. No health reference from the target is used.

## Release assets

- `kaist-xm-processed-v1.zip`: 18 per-record NPZ files.
- `ceitec-xl-processed-v1.zip`: 56 per-record NPZ files.
- `kaist-audit-features-v1.zip`: 45-record, 128-dimensional LR features for the additional group/split analyses.

Each asset has paths rooted at `data/processed/`; extract into the repository root. Each primary record NPZ contains:

| Key | Meaning |
|---|---|
| `features` | float32 relative log-power + residual, already scaled/clipped by the frontend |
| `label` | binary label per window |
| `record` | original audit record ID, not remapped |
| `domain` | frozen motor/load-group ID |
| `start`, `end` | zero-based original sample interval, end-exclusive |

Do **not** feed these spectral tensors through `spectral_features` again. They are direct inputs to `FPPNet.forward`. Original waveforms use `spectral_features(raw, sampling_rate)` first. Classification must not consume record/domain IDs, file names, speed, severity, or load metadata.

Manifests are ordered to preserve the exact window indexing used by `configs/splits.json`. They include raw-file SHA256 and feature-array SHA256. `python run.py check` verifies hashes, counts and partitions.

## Rebuild from original public sources

```bash
python run.py download --dataset kaist --raw-dir data/raw
python run.py download --dataset ceitec --raw-dir data/raw
python run.py prepare --dataset all --raw-dir data/raw
python run.py check
```

The downloader runs sequentially and checks official hashes and sizes. KAIST is distributed as three large ZIP archives containing multiple modalities; downloading them is much larger than the selected vibration subset. Only vibration TDMS files are extracted. CEITEC comprises56MAT files. If a host changes its endpoint, download from the official landing page and place extracted files anywhere beneath `data/raw/`; preparation finds files by exact basename and verifies their hashes.

- [KAIST source](https://data.mendeley.com/datasets/rgn5brrgrn/5)
- [CEITEC source](https://zenodo.org/records/13889418)

For KAIST, native floating-point signal precision is preserved through centering. For CEITEC, raw current windows are converted to float32 before the frontend to match the archived experiment. FFT/interpolation, 50Hz Gaussian smoothing, floors, clipping and channel concatenation are defined in `fpp_net.py`.

Both sources use CC BY4.0. See [NOTICE.md](../NOTICE.md) for authors, DOIs and modifications. Release archives carry the same attribution notice; the MIT software license does not apply to these measurements.
