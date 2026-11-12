"""
REVIEW SCRIPT 2 — Multi-Seed Validation + Bootstrap 95% CI
============================================================
Runs full training + evaluation across 10 seeds.
Reports: mean ± std, min, max, 95% bootstrap CI for AUC/F1/Recall.

NOTE: Run this overnight — ~6-8 hours total on Ryzen 5 7500F.

Usage:
    python 02_multiseed_validation.py

Outputs:
    - multiseed_results.txt   → full per-seed breakdown + summary
"""

import os, time, torch, torch.nn as nn, torch.optim as optim, numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from scipy.stats import bootstrap as scipy_bootstrap
from audio_loader import MIMII_AcousticDataset
from cpu_model import AcousticAnomalyDetector

# ── CONFIG ────────────────────────────────────────────────────
DATA_DIR   = r"E:\Yolo-Thermal\Acoustic Anomaly Detection\MIMII_Dataset_-6dB"
MACHINES   = ["fan", "pump", "slider", "valve"]
SEEDS      = [42, 0, 123, 456, 789, 1, 2024, 999, 7, 314]
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-4
# Use pos_weight from Script 1 results (apply fix from day 1)
CLASS_RATIOS = {"fan": 2.8, "pump": 8.2, "slider": 3.6, "valve": 7.7}
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
# ──────────────────────────────────────────────────────────────

def train_eval(machine, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cpu")

    full_ds  = MIMII_AcousticDataset(data_dir=DATA_DIR, machine_type=machine)
    train_sz = int(0.8 * len(full_ds))
    test_sz  = len(full_ds) - train_sz
    gen      = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(full_ds, [train_sz, test_sz], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = AcousticAnomalyDetector().to(device)
    pw        = torch.tensor([CLASS_RATIOS[machine]])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    for _ in range(EPOCHS):
        model.train()
        for mfccs, labels in train_loader:
            optimizer.zero_grad()
            criterion(model(mfccs), labels).backward()
            optimizer.step()

    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for mfccs, labels in test_loader:
            probs = torch.sigmoid(model(mfccs)).squeeze().numpy()
            true  = labels.squeeze().numpy()
            if probs.ndim == 0:
                probs = np.array([probs]); true = np.array([true])
            all_probs.extend(probs.tolist())
            all_labels.extend(true.tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(float)

    return dict(
        auc    = roc_auc_score(y_true, y_prob),
        f1     = f1_score(y_true, y_pred, zero_division=0),
        recall = recall_score(y_true, y_pred, zero_division=0),
        acc    = (y_pred == y_true).mean() * 100,
    )


def bootstrap_ci(values, stat_fn=np.mean, n_resamples=1000, ci=0.95):
    """95% bootstrap CI for a list of scalar values."""
    rng    = np.random.default_rng(42)
    data   = np.array(values)
    result = scipy_bootstrap(
        (data,), statistic=stat_fn,
        n_resamples=n_resamples, confidence_level=ci,
        random_state=rng, method='percentile'
    )
    return result.confidence_interval.low, result.confidence_interval.high


if __name__ == "__main__":
    print("=" * 65)
    print("REVIEW SCRIPT 2 — Multi-Seed Validation (10 seeds)")
    print(f"Seeds: {SEEDS}")
    print("=" * 65)

    all_results = {m: [] for m in MACHINES}
    start = time.time()

    for machine in MACHINES:
        print(f"\n[{machine.upper()}]")
        for seed in SEEDS:
            t0 = time.time()
            r  = train_eval(machine, seed)
            all_results[machine].append(r)
            print(f"  Seed {seed:>4}: AUC={r['auc']:.4f}  F1={r['f1']:.4f}  "
                  f"Recall={r['recall']:.4f}  Acc={r['acc']:.2f}%  "
                  f"[{time.time()-t0:.0f}s]")

    total_hrs = (time.time() - start) / 3600
    print(f"\nTotal runtime: {total_hrs:.2f} hours")

    # ── Build summary ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY — Mean ± Std  |  Min / Max  |  95% CI")
    print("=" * 65)

    out_path = os.path.join(OUT_DIR, "multiseed_results.txt")
    with open(out_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"MULTI-SEED VALIDATION ({len(SEEDS)} seeds, pos_weight applied)\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write("=" * 70 + "\n\n")

        for machine in MACHINES:
            rows    = all_results[machine]
            for metric in ["auc", "f1", "recall", "acc"]:
                vals   = [r[metric] for r in rows]
                mean_v = np.mean(vals)
                std_v  = np.std(vals)
                min_v  = np.min(vals)
                max_v  = np.max(vals)
                ci_lo, ci_hi = bootstrap_ci(vals)

                line = (f"  {metric.upper():<8}: {mean_v:.4f} ± {std_v:.4f}  "
                        f"[{min_v:.4f} – {max_v:.4f}]  "
                        f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
                print(f"{machine.upper()} {line}")
                f.write(f"{machine.upper()} {line}\n")

            # Per-seed detail
            f.write(f"\n  {machine.upper()} per-seed breakdown:\n")
            f.write(f"  {'Seed':>6} {'AUC':>8} {'F1':>8} {'Recall':>8} {'Acc%':>8}\n")
            for i, seed in enumerate(SEEDS):
                r = all_results[machine][i]
                f.write(f"  {seed:>6} {r['auc']:>8.4f} {r['f1']:>8.4f} "
                        f"{r['recall']:>8.4f} {r['acc']:>8.2f}\n")
            f.write("\n")

    print(f"\n[SAVED] {out_path}")
