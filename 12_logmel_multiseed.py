"""
SCRIPT 12 - Multi-Seed Validation of the PROPOSED configuration
================================================================
Log-Mel (64 bands) + pos_weight + 3B-FC128, 10 seeds, all machines.

Motivation: the existing multiseed_results.txt used MFCC features, while
the manuscript's proposed configuration is Log-Mel + pos_weight. This
script produces the authoritative multi-seed evidence for the proposed
configuration, including bootstrap 95% CIs, pAUC (max_fpr=0.1, DCASE-
comparable), and train-set-calibrated decision thresholds.

Features are precomputed once per machine and held in RAM, so the 10
seeds reuse the same feature tensors (large speedup vs. re-extracting
per epoch).

File enumeration order matches audio_loader.MIMII_AcousticDataset
(id_XX -> normal, abnormal), so random_split with the same generator
seed reproduces the exact train/test partitions of prior experiments.

Usage:
    python 12_logmel_multiseed.py                      # full run
    python 12_logmel_multiseed.py --machines valve --seeds 42 --epochs 2 --tag smoke
"""

import os, time, json, argparse
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import librosa
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             roc_auc_score, precision_recall_curve)
from scipy.stats import bootstrap as scipy_bootstrap

ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "MIMII_Dataset_-6dB")
OUT_DIR  = os.path.join(ROOT, "results")
MACHINES_ALL = ["fan", "pump", "slider", "valve"]
SEEDS_ALL    = [42, 0, 123, 456, 789, 1, 2024, 999, 7, 314]
BATCH_SIZE = 32
LR         = 1e-4
N_MELS     = 64
MAX_LEN    = 400
CLASS_RATIOS = {"fan": 2.8, "pump": 8.2, "slider": 3.6, "valve": 7.7}


def build_cnn(n_freq_bins):
    flat = 64 * (n_freq_bins // 8) * (MAX_LEN // 8)
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2))
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(flat,128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128,1))
        def forward(self, x): return self.classifier(self.features(x))
    return CNN()


def list_files(machine):
    """Same enumeration order as audio_loader.MIMII_AcousticDataset."""
    mdir = os.path.join(DATA_DIR, machine)
    files = []
    for mid in os.listdir(mdir):
        id_path = os.path.join(mdir, mid)
        if not os.path.isdir(id_path):
            continue
        for cond, lbl in [("normal", 0.0), ("abnormal", 1.0)]:
            cp = os.path.join(id_path, cond)
            if not os.path.exists(cp):
                continue
            for f in os.listdir(cp):
                if f.endswith(".wav"):
                    files.append((os.path.join(cp, f), lbl))
    return files


def extract_features(machine):
    files = list_files(machine)
    n = len(files)
    X = np.zeros((n, 1, N_MELS, MAX_LEN), dtype=np.float32)
    y = np.zeros((n, 1), dtype=np.float32)
    t0 = time.time()
    for i, (path, lbl) in enumerate(files):
        sig, sr = librosa.load(path, sr=16000)
        mel = librosa.power_to_db(
            librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=N_MELS), ref=np.max)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        if mel.shape[1] > MAX_LEN:
            mel = mel[:, :MAX_LEN]
        else:
            mel = np.pad(mel, ((0,0),(0, MAX_LEN-mel.shape[1])), mode="constant")
        X[i, 0] = mel
        y[i, 0] = lbl
        if (i+1) % 1000 == 0:
            print(f"    extracted {i+1}/{n}  [{time.time()-t0:.0f}s]")
    print(f"    feature extraction done: {n} files in {time.time()-t0:.0f}s")
    return torch.from_numpy(X), torch.from_numpy(y)


def predict(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for feat, lbl in loader:
            p = torch.sigmoid(model(feat)).squeeze(1).numpy()
            probs.extend(p.tolist())
            labels.extend(lbl.squeeze(1).numpy().tolist())
    return np.array(labels), np.array(probs)


def train_eval(X, y, machine, seed, epochs):
    torch.manual_seed(seed); np.random.seed(seed)
    ds = TensorDataset(X, y)
    train_sz = int(0.8 * len(ds))
    gen = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(ds, [train_sz, len(ds)-train_sz], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    eval_train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

    model = build_cnn(N_MELS)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CLASS_RATIOS[machine]]))
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    for _ in range(epochs):
        model.train()
        for feat, lbl in train_loader:
            optimizer.zero_grad()
            criterion(model(feat), lbl).backward()
            optimizer.step()

    # Train-set threshold calibration (max F1 on PR curve)
    yt_train, p_train = predict(model, eval_train_loader)
    prec, rec, thr = precision_recall_curve(yt_train, p_train)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    best = np.argmax(f1s[:-1])
    tau = float(thr[best])

    y_true, y_prob = predict(model, test_loader)
    y05  = (y_prob >= 0.5).astype(float)
    ytau = (y_prob >= tau).astype(float)

    return dict(
        seed      = seed,
        auc       = roc_auc_score(y_true, y_prob),
        pauc      = roc_auc_score(y_true, y_prob, max_fpr=0.1),
        acc       = float((y05 == y_true).mean() * 100),
        precision = precision_score(y_true, y05, zero_division=0),
        recall    = recall_score(y_true, y05, zero_division=0),
        f1        = f1_score(y_true, y05, zero_division=0),
        tau       = tau,
        acc_cal       = float((ytau == y_true).mean() * 100),
        precision_cal = precision_score(y_true, ytau, zero_division=0),
        recall_cal    = recall_score(y_true, ytau, zero_division=0),
        f1_cal        = f1_score(y_true, ytau, zero_division=0),
    )


def bootstrap_ci(values, n_resamples=1000, ci=0.95):
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(42)
    res = scipy_bootstrap((np.array(values),), statistic=np.mean,
                          n_resamples=n_resamples, confidence_level=ci,
                          random_state=rng, method="percentile")
    return res.confidence_interval.low, res.confidence_interval.high


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--machines", nargs="+", default=MACHINES_ALL)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS_ALL)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    ckpt_path = os.path.join(OUT_DIR, f"logmel_multiseed_checkpoint{suffix}.json")
    out_path  = os.path.join(OUT_DIR, f"logmel_multiseed_results{suffix}.txt")

    ckpt = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        print(f"[RESUME] loaded checkpoint with {sum(len(v) for v in ckpt.values())} runs")

    print("=" * 70)
    print("SCRIPT 12 - Multi-seed validation, PROPOSED config")
    print(f"Log-Mel({N_MELS}) + pos_weight + 3B-FC128 | epochs={args.epochs}")
    print(f"Machines: {args.machines} | Seeds: {args.seeds}")
    print("=" * 70)

    t_start = time.time()
    for machine in args.machines:
        done = {r["seed"] for r in ckpt.get(machine, [])}
        todo = [s for s in args.seeds if s not in done]
        if not todo:
            print(f"\n[{machine.upper()}] all seeds done, skipping")
            continue
        print(f"\n[{machine.upper()}] extracting Log-Mel features...")
        X, y = extract_features(machine)
        for seed in todo:
            t0 = time.time()
            r = train_eval(X, y, machine, seed, args.epochs)
            ckpt.setdefault(machine, []).append(r)
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f, indent=1)
            print(f"  seed {seed:>4}: AUC={r['auc']:.4f}  pAUC={r['pauc']:.4f}  "
                  f"F1={r['f1']:.4f}  F1cal={r['f1_cal']:.4f}  "
                  f"Rec={r['recall']:.4f}  Acc={r['acc']:.2f}%  "
                  f"tau={r['tau']:.3f}  [{time.time()-t0:.0f}s]")
        del X, y

    print(f"\nTotal runtime: {(time.time()-t_start)/3600:.2f} h")

    # ---- summary ---------------------------------------------------
    metrics = ["auc", "pauc", "f1", "f1_cal", "recall", "recall_cal",
               "precision", "precision_cal", "acc", "acc_cal"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("MULTI-SEED VALIDATION - PROPOSED CONFIG\n")
        f.write(f"Log-Mel ({N_MELS} bands) + pos_weight + 3B-FC128 | "
                f"epochs={args.epochs} | batch={BATCH_SIZE} | lr={LR}\n")
        f.write(f"Seeds: {args.seeds}\n")
        f.write("Threshold calibration: max-F1 on training-set PR curve\n")
        f.write("pAUC: partial AUC, max_fpr=0.1 (DCASE-comparable)\n")
        f.write("=" * 70 + "\n\n")
        for machine in args.machines:
            rows = ckpt.get(machine, [])
            if not rows:
                continue
            for metric in metrics:
                vals = [r[metric] for r in rows]
                lo, hi = bootstrap_ci(vals)
                line = (f"{machine.upper():<7} {metric.upper():<14}: "
                        f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}  "
                        f"[{np.min(vals):.4f} - {np.max(vals):.4f}]  "
                        f"95% CI [{lo:.4f}, {hi:.4f}]")
                print(line)
                f.write(line + "\n")
            f.write(f"\n  {machine.upper()} per-seed breakdown:\n")
            f.write(f"  {'Seed':>6} {'AUC':>8} {'pAUC':>8} {'F1':>8} {'F1cal':>8} "
                    f"{'Recall':>8} {'Prec':>8} {'Acc%':>8} {'tau':>7}\n")
            for r in sorted(rows, key=lambda r: args.seeds.index(r["seed"])
                            if r["seed"] in args.seeds else 99):
                f.write(f"  {r['seed']:>6} {r['auc']:>8.4f} {r['pauc']:>8.4f} "
                        f"{r['f1']:>8.4f} {r['f1_cal']:>8.4f} {r['recall']:>8.4f} "
                        f"{r['precision']:>8.4f} {r['acc']:>8.2f} {r['tau']:>7.3f}\n")
            f.write("\n")
    print(f"\n[SAVED] {out_path}")
    print(f"[SAVED] {ckpt_path}")
