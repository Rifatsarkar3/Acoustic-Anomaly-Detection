# Acoustic Anomaly Detection — Project Guide

## Workspace Layout

```
root/
├── audio_loader.py          # Core dataset class — MIMII_AcousticDataset
├── cpu_model.py             # Core model — AcousticAnomalyDetector (2D-CNN)
├── train_audio.py           # Training entry point
├── requirements.txt
├── CLAUDE.md                # This file
│
├── 01_posweight_retrain.py  # Valve pos-weight fix
├── 02_multiseed_validation.py
├── 03_thread_sweep.py
├── 04_latency_distribution.py
├── 05_architecture_ablation.py
├── 06_representation_ablation.py
├── 07_snr_sweep.py
├── 08_cpu_vs_gpu_inference.py
├── 09_split_sensitivity.py
├── 10_lomo_cv.py            # LOMO-CV — machine-unit generalization test
├── 11_autoencoder_baseline.py  # Conv AE unsupervised baseline
├── 12_logmel_multiseed.py   # 10-seed validation of PROPOSED config (Log-Mel + pos_weight)
├── 13_spectrogram_figure.py # Log-Mel exemplar figure (no training)
├── 14_diagrams.py           # Pipeline + architecture diagrams (no training)
├── 15_supervised_baselines_multiseed.py  # MobileNetV3 + STgram-MFN-Lite, 10-seed, fair protocol
├── 16_model_complexity.py   # consistent params/latency/FPS benchmark (3 models)
├── 17_lomo_logmel.py        # LOMO under PROPOSED config (Log-Mel + pos_weight, valve included)
│
├── results/                 # All .txt results, .png plots, final .json outputs
├── paper/                   # Submission package: main.tex (elsarticle review format),
│   ├── figures/             #   references.bib (33 verified), highlights.txt,
│   └── archive/             #   cover_letter.md, main.pdf; archive/ = superseded drafts
├── paper_backup/            # Snapshot of the validated manuscript (kept in sync)
├── v2_scripts/              # DROPPED CNN-Conformer+ArcFace pivot — future work only, do NOT submit
├── model_weights/           # Trained .pth files (validated/ and posweight/)
├── legacy/                  # Old task*.py scripts — do not run, keep for reference
├── utils/                   # Utility scripts (text_combiner, md_to_docx, etc.)
│
├── MIMII_Dataset_-6dB/      # Primary dataset (used for all paper results)
├── MIMII_Dataset_0dB/
└── MIMII_Dataset_6dB/
```

All experiment scripts **must be run from the project root** — they import
`audio_loader` and `cpu_model` as local modules.

## Architecture (from cpu_model.py)

- Block 1: Conv2d(1,16,3,pad=1) → BN → ReLU → MaxPool(2,2)
- Block 2: Conv2d(16,32,3,pad=1) → BN → ReLU → MaxPool(2,2)
- Block 3: Conv2d(32,64,3,pad=1) → BN → ReLU → MaxPool(2,2)
- Classifier: Flatten → Linear(16000,128) → ReLU → Dropout(0.3) → Linear(128,1)
- Input: (1, 40, 400) — MFCC features, 40 coefficients, 400 time frames
- Total params: 2,071,777

## Confirmed Paper Metrics (the floor — never go below)

| Machine | Acc    | Prec   | Recall | F1     | AUC    |
|---------|--------|--------|--------|--------|--------|
| Fan     | 85.86% | 0.7599 | 0.7333 | 0.7464 | 0.9011 |
| Pump    | 95.72% | 0.8909 | 0.6203 | 0.7313 | 0.9445 |
| Slider  | 95.12% | 0.9839 | 0.7625 | 0.8592 | 0.9730 |
| Valve   | 92.33% | 0.6120 | 0.5200 | 0.6952 | 0.9386 |

External reference: Purohit et al. (MIMII/DCASE 2019) AUC ≈ 0.64 at −6 dB.

**Provenance caveat (2026-06-11):** the Valve row above is the seed-42
3B-FC128 row of `results/architecture_ablation.txt`, which used **MFCC**
(not Log-Mel) + pos_weight. Valve is seed-unstable under MFCC (10-seed AUC
0.616 ± 0.112); the authoritative valve evidence is the 10-seed Log-Mel run
in `results/logmel_multiseed_results.txt` (script 12). The manuscript reports
multi-seed statistics, with single-seed values labeled as such.

## Validated Multi-Seed Metrics (proposed config — new AUC floor, 2026-06-11)

10 seeds, Log-Mel 64 + pos_weight + 3B-FC128, −6 dB, 80/20 split
(`results/logmel_multiseed_results.txt`):

| Machine | AUC (mean ± SD)  | pAUC (FPR≤0.1) | F1@0.5 | F1@calibrated τ |
|---------|------------------|----------------|--------|-----------------|
| Fan     | 0.9408 ± 0.0094 | 0.809          | 0.758  | 0.791           |
| Pump    | 0.9452 ± 0.0108 | 0.879          | 0.696  | 0.760           |
| Slider  | 0.9883 ± 0.0055 | 0.970          | 0.911  | 0.925           |
| Valve   | 0.9624 ± 0.0260 | 0.866          | 0.681  | 0.731           |

Valve never collapses under Log-Mel (per-seed AUC min 0.9115); under MFCC it
collapsed on 6/10 seeds. Log-Mel beats MFCC mean AUC on **every** machine.

## Target Journal (decided 2026-06-11)

**Applied Acoustics (Elsevier).** elsarticle class, abstract ≤250 words,
3–5 highlights ≤85 chars, Vancouver numbered refs, CRediT + declarations.
Submission files in `paper/`: main.tex, references.bib, highlights.txt,
cover_letter.md, figures/. Compile with tectonic
(`%LOCALAPPDATA%\tectonic-bin\tectonic.exe main.tex` from `paper/`).

## Current Status (2026-06-15)

Submitting the **validated 2D-CNN** (3B-FC128, Log-Mel + pos_weight) to Applied
Acoustics. The CNN-Conformer + ArcFace "v2" pivot was **dropped** (archived in
`paper/archive/`; `v2_scripts/` = future work only). Integrated into
`paper/main.tex`: a fair multi-seed supervised baseline comparison (MobileNetV3,
STgram-MFN-Lite — scripts 15/16) and a Log-Mel LOMO redo (script 17, valve
included). Manuscript compiles clean and is internally consistent; abstract 232
words; remaining open item is author-side (none code-blocking).

## Non-Negotiable Rules

**Rule 1 — Novelty is deployment + phenomenon + rigor, not architecture.**
The defensible contributions are (a) the CPU-acoustic / GPU-visual co-residency
deployment benchmark, (b) the valve class-imbalance decision-collapse phenomenon
and its joint Log-Mel + pos_weight remedy, and (c) multi-seed + honest LOMO rigor.
A vanilla 2D-CNN (or CNN+attention) is not itself a novelty claim in 2026. Do not
remove or water down a contribution without an explicit user decision.

**Rule 2 — Every number must belong to the model that produced it.** No metric
may be attributed to a model/feature/loss/seed-count/split that did not generate
it; abstract, tables, and body must stay internally consistent; single-seed
values are always labeled as such. (Lesson from the half-pivoted draft that
reported the CNN's 10-seed AUCs as the Conformer's.)

**Rule 3 — Adopt changes only on confirmed, multi-seed improvement.** Nothing
enters the *proposed* model until it beats the validated AUC floor across the full
10-seed protocol. Single-seed wins do not count (valve especially). New models
enter as validated baselines/ablations, never as silent replacements.

**Rule 4 — Never degrade the validated metrics (the floor).** 10-seed mean AUC
floor — never go below: Fan 0.9408 · Pump 0.9452 · Slider 0.9883 · Valve 0.9624
(see Validated Multi-Seed Metrics). Any change that worsens a reported metric is a
regression.

**Rule 5 — Smoke test before any full/multi-seed run.** A single smoke run
(seed 42, ≤10 epochs, 1 model, 1 machine/fold) before any multi-hour run; proceed
only if it passes. Long runs must checkpoint/resume.

**Rule 6 — Fair comparison.** Any baseline uses the identical proposed-config
protocol (same input, loss, seeds, split, calibration, metrics); only the
architecture differs. No rigor asymmetry (e.g., 10-seed proposed vs 1-seed
baseline).

**Rule 7 — Benchmark externally and engage the literature.** External reference:
Purohit et al. (MIMII/DCASE 2019) AUC ≈ 0.64 at −6 dB — a result is "better" only
if it beats that, not just a prior run. For Applied Acoustics, engage recent
(2024–26) and in-journal work (e.g., Wißbrock 2024) and scope every claim to its
evidence. LOMO scope = machine-unit generalization (units of the same model), not
cross-machine-type.
