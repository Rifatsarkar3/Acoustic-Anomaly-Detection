# Lightweight CPU-bound CNN for Acoustic Anomaly Detection under Severe Noise

Source code and manuscript files for a paper submitted to **Applied
Acoustics** (Elsevier):

> **Lightweight CPU-bound convolutional neural networks for acoustic anomaly
> detection of industrial machinery under severe noise: spectral
> representation, class imbalance, and hardware-decoupled deployment**
>
> Sakar Mohammad Raziul Hasan Rifat, Akter Labani, Tao Zhang\*, Saleh Mahamat
> Aboubakar Ousmane, Boudjelkha Mohammed Djamel Eddine
> Department of Mechanical Engineering, Huai'an University, Huai'an, Jiangsu,
> China
> \*Corresponding author: taozhang2021@hau.edu.cn

## Summary

A systematic experimental study of a lightweight 2D-CNN (3B-FC128, ~2–3.3M
parameters) for supervised acoustic anomaly detection on the [MIMII
dataset](https://zenodo.org/record/3384388) at the extreme −6 dB SNR
operating point, where the published unsupervised baseline attains AUC ≈
0.64.

- Lightweight 2D-CNN detects machine faults on MIMII at −6 dB far above
  autoencoder baselines.
- Isolates a class-imbalance decision-collapse failure mode in the most
  imbalanced category (pneumatic valves, 7.7:1) and shows it is fixed only by
  the *joint* application of log-Mel spectral input and class-weighted loss —
  cepstral (MFCC) features remain collapsed even when re-weighted.
- Ten-seed protocol with bootstrap confidence intervals, partial AUC, and
  calibrated decision thresholds, since single-run results for the most
  imbalanced category are seed-unstable.
- Under one identical protocol, matches or beats a 2×-larger spectral–temporal
  fusion baseline (STgram-MFN) on mean AUC while being the only compared model
  that never collapses to chance on any seed.
- Benchmarks deployment on a commodity six-core CPU sharing a fully-loaded
  GPU visual-inspection workload: ~500 FPS of acoustic inference on two CPU
  threads, zero VRAM use, ≤6% throughput degradation under concurrent GPU
  load.

10-seed mean AUC (proposed log-Mel + class-weighted configuration):

| Machine | AUC (mean ± SD) |
|---|---|
| Fan | 0.941 ± 0.009 |
| Pump | 0.945 ± 0.011 |
| Slider | 0.988 ± 0.006 |
| Valve | 0.962 ± 0.026 |

## Repository contents

```
main.tex                        LaTeX source (elsarticle, Applied Acoustics review format)
references.bib                  Bibliography (37 entries)
figures/                        All figures cited in main.tex, plus the graphical abstract
graphical_abstract_standalone.png   Graphical abstract (separate copy, for journal upload)
highlights.txt                  Article highlights (3–5 bullets, journal requirement)
cover_letter.md                 Cover letter to the editor
main_reference_copy.pdf         Compiled PDF, for reference (not a build artifact to edit)
SUBMISSION_MANIFEST.md          Maps each file to its Editorial Manager upload slot
```

## Building the PDF

Requires a LaTeX distribution with the `elsarticle` class (TeX Live, MiKTeX,
or [tectonic](https://tectonic-typesetting.github.io/)).

```bash
tectonic main.tex
# or
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Data availability

The MIMII dataset is publicly available (Purohit et al., DCASE 2019). Trained
model weights and per-seed experiment logs will be added to this repository
upon acceptance.

## Citation

Citation details will be added upon publication.
