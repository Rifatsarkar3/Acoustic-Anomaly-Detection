"""
REVIEW SCRIPT 1 — Weighted Loss Retraining (pos_weight fix)
============================================================
Fixes the class imbalance problem that caused Valve to collapse
(TP=0, Recall=0, AUC=0.5616).

Runs TWO passes per machine:
  Pass A: Original (no pos_weight) — reproduces your existing results
  Pass B: Weighted (pos_weight = class ratio) — the fix

Outputs:
  - posweight_results.txt  → full before/after comparison table
  - model_weights/posweight/  → saved .pth for weighted models

Usage:
    python 01_posweight_retrain.py

Runtime: ~2-3 hours on Ryzen 5 7500F
"""

import os, time, torch, torch.nn as nn, torch.optim as optim, numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from audio_loader import MIMII_AcousticDataset
from cpu_model import AcousticAnomalyDetector

# ── CONFIG ────────────────────────────────────────────────────
DATA_DIR   = r"E:\Yolo-Thermal\Acoustic Anomaly Detection\MIMII_Dataset_-6dB"
MACHINES   = ["fan", "pump", "slider", "valve"]
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-4
SEED       = 42
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))

# Class ratios from your class_balance.txt
CLASS_RATIOS = {"fan": 2.8, "pump": 8.2, "slider": 3.6, "valve": 7.7}
# ──────────────────────────────────────────────────────────────

def run_experiment(machine, use_pos_weight, label):
    device = torch.device("cpu")

    full_ds   = MIMII_AcousticDataset(data_dir=DATA_DIR, machine_type=machine)
    train_sz  = int(0.8 * len(full_ds))
    test_sz   = len(full_ds) - train_sz
    gen       = torch.Generator().manual_seed(SEED)
    train_ds, test_ds = random_split(full_ds, [train_sz, test_sz], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = AcousticAnomalyDetector().to(device)
    pw        = torch.tensor([CLASS_RATIOS[machine]]) if use_pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # Training
    for epoch in range(EPOCHS):
        model.train()
        for mfccs, labels in train_loader:
            optimizer.zero_grad()
            criterion(model(mfccs), labels).backward()
            optimizer.step()
        ep_num = epoch + 1
        if ep_num % 5 == 0:
            print(f"  [{label}] {machine.upper()} epoch {ep_num}/{EPOCHS}")

    # Evaluation
    model.eval()
    all_labels, all_probs = [], []
    t_loss, total = 0.0, 0
    with torch.no_grad():
        for mfccs, labels in test_loader:
            out    = model(mfccs)
            t_loss += criterion(out, labels).item() * mfccs.size(0)
            probs  = torch.sigmoid(out).squeeze().numpy()
            true   = labels.squeeze().numpy()
            if probs.ndim == 0:
                probs = np.array([probs]); true = np.array([true])
            all_probs.extend(probs.tolist())
            all_labels.extend(true.tolist())
            total += labels.size(0)

    y_true  = np.array(all_labels)
    y_prob  = np.array(all_probs)
    y_pred  = (y_prob >= 0.5).astype(float)

    acc  = (y_pred == y_true).mean() * 100
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    loss = t_loss / total

    # Save weighted model weights
    if use_pos_weight:
        wdir = os.path.join(OUT_DIR, "model_weights", "posweight")
        os.makedirs(wdir, exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join(wdir, f"cpu_audio_{machine}_posweight.pth"))

    return dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc, loss=loss)


if __name__ == "__main__":
    print("=" * 65)
    print("REVIEW SCRIPT 1 — pos_weight Weighted Loss Retraining")
    print("=" * 65)

    results = {}
    for machine in MACHINES:
        print(f"\n[{machine.upper()}]")
        t0 = time.time()
        results[machine] = {
            "before": run_experiment(machine, use_pos_weight=False, label="Before"),
            "after" : run_experiment(machine, use_pos_weight=True,  label="After"),
        }
        print(f"  Done in {(time.time()-t0)/60:.1f} min")

    # ── Print summary ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("BEFORE vs AFTER pos_weight")
    print("=" * 65)
    metrics = ["acc", "prec", "rec", "f1", "auc"]
    header  = f"{'Machine':<10} {'Cond':<8} {'Acc%':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'AUC':>7}"
    print(header)
    print("-" * 65)
    for m in MACHINES:
        for cond in ["before", "after"]:
            r = results[m][cond]
            tag = "BEFORE" if cond == "before" else "AFTER "
            print(f"{m:<10} {tag} {r['acc']:>7.2f} {r['prec']:>7.4f} "
                  f"{r['rec']:>7.4f} {r['f1']:>7.4f} {r['auc']:>7.4f}")
        print()

    # ── Save to file ──────────────────────────────────────────
    out = os.path.join(OUT_DIR, "posweight_results.txt")
    with open(out, "w") as f:
        f.write("=" * 65 + "\n")
        f.write("pos_weight WEIGHTED LOSS — Before vs After\n")
        f.write(f"Machines: {MACHINES} | Seed={SEED} | Epochs={EPOCHS}\n")
        f.write(f"Class ratios used: {CLASS_RATIOS}\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"{'Machine':<10} {'Condition':<8} {'Acc%':>7} "
                f"{'Prec':>7} {'Recall':>7} {'F1':>7} {'AUC':>7} {'Loss':>8}\n")
        f.write("-" * 65 + "\n")
        for m in MACHINES:
            for cond in ["before", "after"]:
                r   = results[m][cond]
                tag = "BEFORE" if cond == "before" else "AFTER"
                f.write(f"{m:<10} {tag:<8} {r['acc']:>7.2f} "
                        f"{r['prec']:>7.4f} {r['rec']:>7.4f} "
                        f"{r['f1']:>7.4f} {r['auc']:>7.4f} {r['loss']:>8.4f}\n")
            f.write("\n")

    print(f"\n[SAVED] {out}")
