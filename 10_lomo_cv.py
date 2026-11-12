"""
EXPERIMENT 10 — Leave-One-Machine-ID-Out Cross-Validation (LOMO-CV)
====================================================================
Purpose: Quantify whether the random-split AUC results in the paper are
inflated by machine-ID leakage. Tests generalization across physical
machine units (not across machine types).

Protocol:
  - 4 machine IDs per type: id_00, id_02, id_04, id_06
  - 4-fold LOMO: each fold trains on 3 IDs, tests on the 1 held-out ID
  - No recording from the held-out machine appears in training
  - Report mean ± SD AUC across 4 folds; also report per-fold values

Scope: fan, pump, slider  (valve skipped in headline claim — see paper §3.1.1)
Dataset: MIMII_Dataset_-6dB

Rule 4 — smoke test first:
  Run with SMOKE_TEST = True (1 machine, 1 fold, 5 epochs) before full run.

Usage:
    python 10_lomo_cv.py               # full run
    python 10_lomo_cv.py --smoke       # smoke test (1 fold, fan only)
"""

import os, sys, time, json, argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from scipy.stats import bootstrap as scipy_bootstrap
from audio_loader import MIMII_AcousticDataset
from cpu_model import AcousticAnomalyDetector

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DATA_DIR   = r"E:\Yolo-Thermal\Acoustic Anomaly Detection\MIMII_Dataset_-6dB"
OUTPUT_DIR = r"E:\Yolo-Thermal\Acoustic Anomaly Detection\results"
MACHINES   = ["fan", "pump", "slider"]   # valve reported separately
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-4
SEED       = 42
# ──────────────────────────────────────────────────────────────────────────────


def get_machine_ids(data_dir: str, machine_type: str) -> list[str]:
    path = os.path.join(data_dir, machine_type)
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def train_one_fold(train_indices, test_indices, full_dataset, device, epochs):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_ds = Subset(full_dataset, train_indices)
    test_ds  = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = AcousticAnomalyDetector().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(epochs):
        model.train()
        for mfccs, labels in train_loader:
            mfccs, labels = mfccs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(mfccs), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for mfccs, labels in test_loader:
            mfccs = mfccs.to(device)
            probs = torch.sigmoid(model(mfccs)).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().flatten())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc  = roc_auc_score(all_labels, all_probs)
    preds = (all_probs >= 0.5).astype(int)
    f1   = f1_score(all_labels, preds, zero_division=0)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec  = recall_score(all_labels, preds, zero_division=0)
    acc  = (preds == all_labels).mean() * 100

    return {"auc": auc, "f1": f1, "precision": prec, "recall": rec, "acc": acc,
            "n_train": len(train_indices), "n_test": len(test_indices)}


def bootstrap_ci(values, n_resamples=9999, confidence=0.95):
    data = (np.array(values),)
    res  = scipy_bootstrap(data, np.mean, n_resamples=n_resamples,
                           confidence_level=confidence, random_state=SEED,
                           method='percentile')
    return res.confidence_interval.low, res.confidence_interval.high


def run_lomo(machine_type: str, device, epochs: int, smoke: bool = False):
    print(f"\n{'='*60}")
    print(f"  LOMO-CV — {machine_type.upper()}")
    print(f"{'='*60}")

    full_dataset = MIMII_AcousticDataset(data_dir=DATA_DIR, machine_type=machine_type)
    machine_ids  = get_machine_ids(DATA_DIR, machine_type)
    print(f"  Machine IDs found: {machine_ids}")

    # Build per-ID index maps from samples list
    id_to_indices: dict[str, list[int]] = {mid: [] for mid in machine_ids}
    for idx, sample in enumerate(full_dataset.samples):
        mid = sample['machine_id']
        if mid in id_to_indices:
            id_to_indices[mid].append(idx)

    folds = machine_ids[:1] if smoke else machine_ids   # smoke: 1 fold only

    fold_results = []
    for held_out_id in folds:
        t0 = time.time()
        test_indices  = id_to_indices[held_out_id]
        train_indices = [i for mid, idxs in id_to_indices.items()
                         if mid != held_out_id for i in idxs]

        print(f"\n  Fold held-out={held_out_id} | "
              f"train={len(train_indices)} | test={len(test_indices)}")

        result = train_one_fold(train_indices, test_indices, full_dataset, device, epochs)
        result['held_out_id'] = held_out_id
        fold_results.append(result)

        print(f"    AUC={result['auc']:.4f}  F1={result['f1']:.4f}  "
              f"Acc={result['acc']:.2f}%  [{(time.time()-t0)/60:.1f} min]")

    aucs = [r['auc'] for r in fold_results]
    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)

    if len(aucs) >= 2:
        ci_lo, ci_hi = bootstrap_ci(aucs)
    else:
        ci_lo = ci_hi = mean_auc

    print(f"\n  LOMO AUC: {mean_auc:.4f} ± {std_auc:.4f}  "
          f"(95% CI [{ci_lo:.4f}, {ci_hi:.4f}])")

    return {"machine": machine_type, "fold_results": fold_results,
            "mean_auc": mean_auc, "std_auc": std_auc,
            "ci_lo": ci_lo, "ci_hi": ci_hi}


def main(smoke: bool = False):
    device  = torch.device('cpu')
    epochs  = 3 if smoke else EPOCHS
    machines = MACHINES[:1] if smoke else MACHINES

    if smoke:
        print("*** SMOKE TEST MODE — 1 machine, 1 fold, 3 epochs ***")

    all_results = []
    t_total = time.time()

    for machine in machines:
        res = run_lomo(machine, device, epochs, smoke=smoke)
        all_results.append(res)

    total_time = (time.time() - t_total) / 3600

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("LOMO-CV SUMMARY")
    print(f"{'='*60}")
    print(f"{'Machine':<10} {'LOMO AUC':>10} {'±SD':>8} {'95% CI':>20}")
    print("-" * 55)
    for r in all_results:
        ci = f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
        print(f"{r['machine']:<10} {r['mean_auc']:>10.4f} {r['std_auc']:>8.4f} {ci:>20}")

    print(f"\nTotal runtime: {total_time:.2f} hours")

    # ── Save results ───────────────────────────────────────────────────────
    if not smoke:
        tag = "lomo_cv_results"
        txt_path  = os.path.join(OUTPUT_DIR, f"{tag}.txt")
        json_path = os.path.join(OUTPUT_DIR, f"{tag}.json")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("LOMO-CV RESULTS — MIMII -6dB\n")
            f.write(f"Epochs={EPOCHS}  Batch={BATCH_SIZE}  LR={LR}  Seed={SEED}\n")
            f.write("Machines: fan, pump, slider  (valve excluded — unstable under random split)\n\n")
            f.write(f"{'Machine':<10} {'LOMO AUC':>10} {'±SD':>8} {'95% CI':>22}\n")
            f.write("-" * 55 + "\n")
            for r in all_results:
                ci = f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
                f.write(f"{r['machine']:<10} {r['mean_auc']:>10.4f} "
                        f"{r['std_auc']:>8.4f} {ci:>22}\n")
            f.write("\nPer-fold breakdown:\n")
            for r in all_results:
                f.write(f"\n  {r['machine'].upper()}\n")
                for fold in r['fold_results']:
                    f.write(f"    held-out={fold['held_out_id']}  "
                            f"AUC={fold['auc']:.4f}  F1={fold['f1']:.4f}  "
                            f"Acc={fold['acc']:.2f}%  "
                            f"n_train={fold['n_train']}  n_test={fold['n_test']}\n")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n[Saved] {txt_path}")
        print(f"[Saved] {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 1 machine, 1 fold, 3 epochs")
    args = parser.parse_args()
    main(smoke=args.smoke)
