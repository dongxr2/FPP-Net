# Attribution and scope of licenses

The MIT license applies to the original software in this repository. It does **not** relicense the third-party experimental measurements or their derived feature arrays. Dependencies are installed separately and retain their own licenses.

## KAIST source data — CC BY 4.0

Wonho Jung, SungHyun Yun, Yoon-Seop Lim, Sungjin Cheong, and Yong-Hwa Park. *Vibration and Current Dataset of Three-Phase Permanent Magnet Synchronous Motors with Stator Faults*, Mendeley Data, version 5 (2022). DOI: [10.17632/rgn5brrgrn.5](https://doi.org/10.17632/rgn5brrgrn.5).

Associated data article: Jung et al., *Data in Brief* 47 (2023), 108952, DOI: [10.1016/j.dib.2023.108952](https://doi.org/10.1016/j.dib.2023.108952).

Changes in this project: duplicate healthy measurements counted once; selection of 18 January ai3 vibration records; nonoverlapping 0.2-second windows; binary health labels; frozen motor-disjoint evaluation; relative log-power and local residual features. A separate 45-record feature asset supports acquisition-group checks. Neither raw waveforms nor fault labels are repaired or relabeled. The selected subset is configuration-consistent, not independently certified free of every possible confound.

## CEITEC source data — CC BY 4.0

Matus Kozovsky. *Measurement of interturn short-circuits emulation on dual three-phase PMS motor*, Zenodo, version 1.0 (2024). DOI: [10.5281/zenodo.13889418](https://doi.org/10.5281/zenodo.13889418).

Changes in this project: three SubSys1 currents selected; complete 1710-sample windows retained only when every speed sample is within 500–4000 rpm; float32 raw-current conversion matching the experiment; binary health labels; frozen load-disjoint evaluation; relative log-power and local residual features. No individual current samples are replaced or repaired.

The original sources list [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) (checked 2026-11-08). Attribute the original creators, link the license, and identify these modifications when sharing the derived data. No endorsement by KAIST, CEITEC, the original creators, or their institutions is implied. Source rights metadata are recorded in `data/sources/licenses.json`.

The draft manuscript and data are not covered by the software MIT grant. No manuscript DOI or publication status is asserted here.
