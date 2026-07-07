"""
SCRIPT 17 - Leave-One-Machine-ID-Out CV under the PROPOSED config
=================================================================
Redoes LOMO-CV with the proposed configuration (log-Mel 64 + pos_weight +
3B-FC128) instead of the earlier MFCC/unweighted version, so the
generalization analysis is consistent with the proposed model. Because
log-Mel makes the valve stable in-domain, the valve is now INCLUDED
(all four machine types, 4 folds each = held-out id_00/02/04/06).

Mirrors script 12's feature pipeline and model exactly; only the split
changes (train on three machine IDs, test on the held-out ID).

Usage:
    python 17_lomo_logmel.py                 # full run
    python 17_lomo_logmel.py --machines fan --smoke   # smoke: 1 machine, 1 fold, 2 epochs
"""
import os, time, json, argparse
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import librosa
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from scipy.stats import bootstrap as scipy_bootstrap

ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "MIMII_Dataset_-6dB")
OUT_DIR  = os.path.join(ROOT, "results")
MACHINES_ALL = ["fan", "pump", "slider", "valve"]
BATCH_SIZE, LR, N_MELS, MAX_LEN, EPOCHS, SEED = 32, 1e-4, 64, 400, 10, 42
CLASS_RATIOS = {"fan": 2.8, "pump": 8.2, "slider": 3.6, "valve": 7.7}


def build_cnn(n_freq_bins):
    flat = 64 * (n_freq_bins // 8) * (MAX_LEN // 8)
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Flatten(), nn.Linear(flat, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))


def extract_features(machine):
    """Log-Mel features (as script 12) plus a per-sample machine-id array."""
    mdir = os.path.join(DATA_DIR, machine)
    files, ids = [], []
    for mid in sorted(os.listdir(mdir)):
        id_path = os.path.join(mdir, mid)
        if not os.path.isdir(id_path):
            continue
        for cond, lbl in [("normal", 0.0), ("abnormal", 1.0)]:
            cp = os.path.join(id_path, cond)
            if not os.path.exists(cp):
                continue
            for f in os.listdir(cp):
                if f.endswith(".wav"):
                    files.append((os.path.join(cp, f), lbl)); ids.append(mid)
    n = len(files)
    X = np.zeros((n, 1, N_MELS, MAX_LEN), dtype=np.float32)
    y = np.zeros((n, 1), dtype=np.float32)
    for i, (path, lbl) in enumerate(files):
        sig, sr = librosa.load(path, sr=16000)
        mel = librosa.power_to_db(
            librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=N_MELS), ref=np.max)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        mel = mel[:, :MAX_LEN] if mel.shape[1] > MAX_LEN else \
            np.pad(mel, ((0, 0), (0, MAX_LEN - mel.shape[1])), mode="constant")
        X[i, 0] = mel; y[i, 0] = lbl
    return torch.from_numpy(X), torch.from_numpy(y), np.array(ids)


def train_eval_fold(X, y, train_mask, test_mask, machine):
    torch.manual_seed(SEED); np.random.seed(SEED)
    tr = TensorDataset(X[train_mask], y[train_mask])
    te = TensorDataset(X[test_mask], y[test_mask])
    tl = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    el = DataLoader(te, batch_size=256, shuffle=False, num_workers=0)
    model = build_cnn(N_MELS)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CLASS_RATIOS[machine]]))
    opt = optim.AdamW(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in tl:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in el:
            probs.extend(torch.sigmoid(model(xb)).squeeze(1).numpy().tolist())
            labels.extend(yb.squeeze(1).numpy().tolist())
    probs, labels = np.array(probs), np.array(labels)
    preds = (probs >= 0.5).astype(float)
    return dict(auc=roc_auc_score(labels, probs),
                f1=f1_score(labels, preds, zero_division=0),
                recall=recall_score(labels, preds, zero_division=0),
                n_train=int(train_mask.sum()), n_test=int(test_mask.sum()))


def bootstrap_ci(vals):
    if len(vals) < 2:
        return float("nan"), float("nan")
    res = scipy_bootstrap((np.array(vals),), np.mean, n_resamples=9999,
                          confidence_level=0.95, random_state=SEED, method="percentile")
    return res.confidence_interval.low, res.confidence_interval.high


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--machines", nargs="+", default=MACHINES_ALL)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    all_results = []
    for machine in args.machines:
        print(f"\n{'='*60}\n  LOMO (log-Mel + pos_weight) — {machine.upper()}\n{'='*60}")
        X, y, ids = extract_features(machine)
        uids = sorted(set(ids.tolist()))
        folds = uids[:1] if args.smoke else uids
        fold_results = []
        for held in folds:
            t0 = time.time()
            test_mask = ids == held
            train_mask = ~test_mask
            r = train_eval_fold(X, y, train_mask, test_mask, machine)
            r["held_out_id"] = held
            fold_results.append(r)
            print(f"  held-out={held}  AUC={r['auc']:.4f}  F1={r['f1']:.4f}  "
                  f"n_train={r['n_train']} n_test={r['n_test']}  [{time.time()-t0:.0f}s]")
        aucs = [r["auc"] for r in fold_results]
        lo, hi = bootstrap_ci(aucs)
        rec = dict(machine=machine, fold_results=fold_results,
                   mean_auc=float(np.mean(aucs)), std_auc=float(np.std(aucs)),
                   ci_lo=lo, ci_hi=hi)
        all_results.append(rec)
        print(f"  LOMO AUC: {rec['mean_auc']:.4f} +/- {rec['std_auc']:.4f}  "
              f"95% CI [{lo:.4f}, {hi:.4f}]")
        del X, y

    print(f"\n{'='*60}\nLOMO (log-Mel + pos_weight) SUMMARY\n{'='*60}")
    for r in all_results:
        print(f"{r['machine']:<8} {r['mean_auc']:.4f} +/- {r['std_auc']:.4f}  "
              f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")

    if not args.smoke:
        with open(os.path.join(OUT_DIR, "lomo_logmel_results.txt"), "w", encoding="utf-8") as f:
            f.write("LOMO-CV RESULTS (PROPOSED config: log-Mel 64 + pos_weight + 3B-FC128) — MIMII -6dB\n")
            f.write(f"Epochs={EPOCHS} Batch={BATCH_SIZE} LR={LR} Seed={SEED}; all four machines (valve included)\n\n")
            f.write(f"{'Machine':<8} {'LOMO AUC':>10} {'±SD':>8} {'95% CI':>22}\n" + "-"*52 + "\n")
            for r in all_results:
                f.write(f"{r['machine']:<8} {r['mean_auc']:>10.4f} {r['std_auc']:>8.4f} "
                        f"{'['+format(r['ci_lo'],'.4f')+', '+format(r['ci_hi'],'.4f')+']':>22}\n")
            f.write("\nPer-fold:\n")
            for r in all_results:
                f.write(f"\n  {r['machine'].upper()}\n")
                for fold in r['fold_results']:
                    f.write(f"    held-out={fold['held_out_id']}  AUC={fold['auc']:.4f}  "
                            f"F1={fold['f1']:.4f}  n_train={fold['n_train']} n_test={fold['n_test']}\n")
        with open(os.path.join(OUT_DIR, "lomo_logmel_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        print("\n[SAVED] results/lomo_logmel_results.txt / .json")
