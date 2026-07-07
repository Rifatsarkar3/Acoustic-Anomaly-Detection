"""
EXPERIMENT 11 — Convolutional Autoencoder Unsupervised Baseline
===============================================================
Purpose: Establish a learned unsupervised baseline comparable to
Purohit et al. (MIMII/DCASE 2019, AUC ~ 0.64 at −6 dB).

Method:
  - Train a Conv autoencoder on NORMAL samples only
  - Anomaly score = MSE reconstruction error on the test set
  - Higher MSE → more anomalous (unsupervised, no labels used in training)
  - Report AUC and compare to: (a) Purohit et al. 0.64, (b) our supervised CNN

Architecture (mirrors the depth of our supervised 2D-CNN):
  Encoder: Conv(1→16) → Conv(16→32) → Conv(32→64)
  Decoder: ConvTranspose(64→32) → ConvTranspose(32→16) → ConvTranspose(16→1)

Input: (1, 40, 400) MFCC tensors — same as supervised model

Rule 4 — smoke test first:
  Run with --smoke (fan only, 5 epochs) before full run.

Usage:
    python 11_autoencoder_baseline.py          # full run (all 4 machines)
    python 11_autoencoder_baseline.py --smoke  # smoke test
"""

import os, time, json, argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
from audio_loader import MIMII_AcousticDataset

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DATA_DIR   = r"E:\Yolo-Thermal\Acoustic Anomaly Detection\MIMII_Dataset_-6dB"
OUTPUT_DIR = r"E:\Yolo-Thermal\Acoustic Anomaly Detection\results"
MACHINES   = ["fan", "pump", "slider", "valve"]
BATCH_SIZE = 32
EPOCHS     = 20          # AEs need more epochs than supervised
LR         = 1e-3
SEED       = 42
# ──────────────────────────────────────────────────────────────────────────────


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16,  1, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(machine_type: str, device, epochs: int):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    full_dataset = MIMII_AcousticDataset(data_dir=DATA_DIR, machine_type=machine_type)

    # Separate normal (label=0) and all samples for scoring
    normal_indices = [i for i, s in enumerate(full_dataset.samples) if s['label'] == 0.0]
    all_indices    = list(range(len(full_dataset)))

    # Hold out 20% of normal for validation (monitor over-fitting)
    n_normal = len(normal_indices)
    np.random.shuffle(normal_indices)
    val_cut   = int(0.2 * n_normal)
    val_norm  = normal_indices[:val_cut]
    train_norm = normal_indices[val_cut:]

    train_loader = DataLoader(Subset(full_dataset, train_norm),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model     = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for mfccs, _ in train_loader:
            mfccs = mfccs.to(device)
            # Per-sample z-score normalisation — stabilises AE reconstruction loss
            mu  = mfccs.mean(dim=(1, 2, 3), keepdim=True)
            std = mfccs.std(dim=(1, 2, 3), keepdim=True) + 1e-8
            x   = (mfccs - mu) / std
            recon = model(x)
            recon = recon[:, :, :x.shape[2], :x.shape[3]]
            loss  = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * mfccs.size(0)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:>3}/{epochs}  train_loss={epoch_loss/len(train_norm):.6f}")

    return model, full_dataset, all_indices


def score_and_evaluate(model, full_dataset, all_indices, device):
    model.eval()
    criterion = nn.MSELoss(reduction='none')

    scores, labels = [], []
    loader = DataLoader(Subset(full_dataset, all_indices),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    with torch.no_grad():
        for mfccs, lbls in loader:
            mfccs = mfccs.to(device)
            mu  = mfccs.mean(dim=(1, 2, 3), keepdim=True)
            std = mfccs.std(dim=(1, 2, 3), keepdim=True) + 1e-8
            x   = (mfccs - mu) / std
            recon = model(x)
            recon = recon[:, :, :x.shape[2], :x.shape[3]]
            mse = criterion(recon, x).view(mfccs.size(0), -1).mean(dim=1)
            scores.extend(mse.cpu().numpy())
            labels.extend(lbls.numpy().flatten())

    scores = np.array(scores)
    labels = np.array(labels)
    auc = roc_auc_score(labels, scores)
    return auc, scores, labels


def main(smoke: bool = False):
    device   = torch.device('cpu')
    epochs   = 5 if smoke else EPOCHS
    machines = MACHINES[:1] if smoke else MACHINES

    if smoke:
        print("*** SMOKE TEST MODE — fan only, 5 epochs ***")

    results = []
    t_total = time.time()

    for machine in machines:
        print(f"\n{'='*60}")
        print(f"  AUTOENCODER BASELINE — {machine.upper()}")
        print(f"{'='*60}")
        t0 = time.time()

        model, full_dataset, all_indices = train_autoencoder(machine, device, epochs)
        auc, scores, labels = score_and_evaluate(model, full_dataset, all_indices, device)

        elapsed = (time.time() - t0) / 60
        n_normal = int((labels == 0).sum())
        n_abnormal = int((labels == 1).sum())

        print(f"  AUC={auc:.4f}  "
              f"(normal={n_normal}, abnormal={n_abnormal})  [{elapsed:.1f} min]")
        print(f"  Reference -- Purohit et al. DCASE 2019: AUC ~0.640")

        results.append({
            "machine": machine, "auc": float(auc),
            "n_normal": n_normal, "n_abnormal": n_abnormal,
        })

    total_time = (time.time() - t_total) / 3600

    print(f"\n{'='*60}")
    print("AUTOENCODER BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Machine':<10} {'AE AUC':>10}  {'vs Purohit 0.640':>18}")
    print("-" * 45)
    for r in results:
        delta = r['auc'] - 0.640
        sign  = "+" if delta >= 0 else ""
        print(f"{r['machine']:<10} {r['auc']:>10.4f}  {sign}{delta:>+.4f}")

    print(f"\nTotal runtime: {total_time:.2f} hours")

    if not smoke:
        txt_path  = os.path.join(OUTPUT_DIR, "autoencoder_baseline_results.txt")
        json_path = os.path.join(OUTPUT_DIR, "autoencoder_baseline_results.json")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("CONV AUTOENCODER BASELINE — MIMII -6dB\n")
            f.write(f"Epochs={EPOCHS}  Batch={BATCH_SIZE}  LR={LR}  Seed={SEED}\n")
            f.write("Score: per-sample MSE reconstruction error (higher = more anomalous)\n")
            f.write("Reference: Purohit et al. DCASE 2019 AUC ~ 0.640\n\n")
            f.write(f"{'Machine':<10} {'AE AUC':>10} {'vs Purohit':>12} {'N Normal':>10} {'N Abnormal':>12}\n")
            f.write("-" * 58 + "\n")
            for r in results:
                delta = r['auc'] - 0.640
                f.write(f"{r['machine']:<10} {r['auc']:>10.4f} {delta:>+12.4f} "
                        f"{r['n_normal']:>10} {r['n_abnormal']:>12}\n")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\n[Saved] {txt_path}")
        print(f"[Saved] {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: fan only, 5 epochs")
    args = parser.parse_args()
    main(smoke=args.smoke)
