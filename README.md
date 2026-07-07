# Acoustic Anomaly Detection

Lightweight 2D-CNN for supervised acoustic anomaly detection on the MIMII
dataset at -6 dB SNR, submitted to *Applied Acoustics* (Elsevier).

- Ten-seed protocol (bootstrap CIs, pAUC, calibrated thresholds, significance tests)
- Fixes a valve class-imbalance decision collapse via joint log-Mel input + class-weighted loss
- Matches a 2x-larger fusion baseline (STgram-MFN) on AUC while being the only model that never collapses on any seed
- CPU-only inference at ~500 FPS on 2 threads, zero VRAM, <=6% latency loss under concurrent GPU load

Full details are in the abstract and body of [`main.tex`](main.tex).

## Repository layout

```
main.tex, references.bib, highlights.txt,     # Applied Acoustics submission package
cover_letter_applied_acoustics.docx,          # (elsarticle format)
figures/, main_reference_copy.pdf

code/                        # source that produced the paper's results
├── audio_loader.py          # MIMII_AcousticDataset
├── cpu_model.py              # AcousticAnomalyDetector (2D-CNN, 3B-FC128)
├── train_audio.py            # training entry point
├── requirements.txt
├── 01_posweight_retrain.py   # valve pos-weight fix
├── 02_multiseed_validation.py
├── 03_thread_sweep.py
├── 04_latency_distribution.py
├── 05_architecture_ablation.py
├── 06_representation_ablation.py
├── 07_snr_sweep.py
├── 08_cpu_vs_gpu_inference.py
├── 09_split_sensitivity.py
├── 10_lomo_cv.py             # leave-one-machine-out CV
├── 11_autoencoder_baseline.py
├── 12_logmel_multiseed.py    # 10-seed validation of the proposed config
├── 13_spectrogram_figure.py
├── 14_diagrams.py
├── 15_supervised_baselines_multiseed.py  # MobileNetV3 + STgram-MFN-Lite, fair protocol
├── 16_model_complexity.py
├── 17_lomo_logmel.py         # LOMO under proposed config (Log-Mel + pos_weight)
└── utils/                    # Directory_map, md_to_docx, tex_to_docx, text_combiner, setup_audio_data

results/                      # .txt / .json / .png outputs referenced by the paper
CLAUDE.md                     # project guide: architecture, validated metrics, decision rules
```

Experiment scripts must be run from the `code/` directory — they import
`audio_loader` and `cpu_model` as local modules.

## Architecture

- Block 1: Conv2d(1,16,3,pad=1) -> BN -> ReLU -> MaxPool(2,2)
- Block 2: Conv2d(16,32,3,pad=1) -> BN -> ReLU -> MaxPool(2,2)
- Block 3: Conv2d(32,64,3,pad=1) -> BN -> ReLU -> MaxPool(2,2)
- Classifier: Flatten -> Linear(16000,128) -> ReLU -> Dropout(0.3) -> Linear(128,1)
- Input: (1, 40, 400) log-Mel/MFCC features
- Total params: 2,071,777

## Validated multi-seed metrics (proposed config)

10 seeds, Log-Mel 64 + pos_weight + 3B-FC128, -6 dB, 80/20 split
(`results/logmel_multiseed_results.txt`):

| Machine | AUC (mean +/- SD) | pAUC (FPR<=0.1) | F1@0.5 | F1@calibrated tau |
|---------|--------------------|------------------|--------|---------------------|
| Fan     | 0.9408 +/- 0.0094 | 0.809 | 0.758 | 0.791 |
| Pump    | 0.9452 +/- 0.0108 | 0.879 | 0.696 | 0.760 |
| Slider  | 0.9883 +/- 0.0055 | 0.970 | 0.911 | 0.925 |
| Valve   | 0.9624 +/- 0.0260 | 0.866 | 0.681 | 0.731 |

External reference: Purohit et al. (MIMII/DCASE 2019) AUC ~ 0.64 at -6 dB.

## Data and weights (not included in this repo)

- **Dataset**: MIMII (`-6dB`, `0dB`, `6dB` splits), publicly available from the
  [MIMII Dataset project](https://zenodo.org/record/3384388). Place under
  `MIMII_Dataset_-6dB/` etc. at the same level as `code/` before running.
- **Trained weights**: not versioned here (binary, regenerable). Reproduce via
  `code/train_audio.py`; multi-seed sweeps via `code/12_logmel_multiseed.py`.

## Setup

```bash
cd code
pip install -r requirements.txt
python train_audio.py
```
