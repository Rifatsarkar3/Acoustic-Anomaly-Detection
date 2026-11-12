"""
SCRIPT 15 - Multi-Seed Supervised BASELINES (fair comparison)
=============================================================
Runs the MobileNetV3 and STgram-MFN(-Lite) baselines under the EXACT
protocol of the proposed configuration (script 12): Log-Mel (64) input,
class-weighted BCE (pos_weight), AdamW lr=1e-4, batch 32, 80/20 split
redrawn per seed with the same generator, train-set F1 threshold
calibration, and the same 10 seeds. Only the network architecture differs,
so the resulting AUCs are an apples-to-apples architecture comparison
against the proposed 3B-FC128 (Table tab:multiseed in the manuscript).

This addresses the reviewer-expected "modern supervised baseline" gap
without changing the proposed model. ArcFace is intentionally NOT used:
the baselines use the same class-weighted BCE as the proposed model.

Usage:
    python 15_supervised_baselines_multiseed.py --model mobilenet
    python 15_supervised_baselines_multiseed.py --model stgram
    # smoke test (Rule 4):
    python 15_supervised_baselines_multiseed.py --model mobilenet \
        --machines fan --seeds 42 --epochs 2 --tag smoke
"""

import os, sys, time, json, argparse
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
sys.path.insert(0, os.path.join(ROOT, "v2_scripts"))
from models import MobileNetV3Audio, STgramMFN_Lite  # noqa: E402

MACHINES_ALL = ["fan", "pump", "slider", "valve"]
SEEDS_ALL    = [42, 0, 123, 456, 789, 1, 2024, 999, 7, 314]
BATCH_SIZE = 32
LR         = 1e-4
N_MELS     = 64
MAX_LEN    = 400
CLASS_RATIOS = {"fan": 2.8, "pump": 8.2, "slider": 3.6, "valve": 7.7}


def build_model(name):
    if name == "mobilenet":
        return MobileNetV3Audio(num_classes=1)
    elif name == "stgram":
        return STgramMFN_Lite(num_classes=1)
    raise ValueError(name)


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
            mel = np.pad(mel, ((0, 0), (0, MAX_LEN-mel.shape[1])), mode="constant")
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
            out = model(feat)
            p = torch.sigmoid(out).squeeze(1).numpy()
            probs.extend(p.tolist())
            labels.extend(lbl.squeeze(1).numpy().tolist())
    return np.array(labels), np.array(probs)


def train_eval(X, y, machine, seed, epochs, model_name):
    torch.manual_seed(seed); np.random.seed(seed)
    ds = TensorDataset(X, y)
    train_sz = int(0.8 * len(ds))
    gen = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(ds, [train_sz, len(ds)-train_sz], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    eval_train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

    model = build_model(model_name)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CLASS_RATIOS[machine]]))
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    for _ in range(epochs):
        model.train()
        for feat, lbl in train_loader:
            optimizer.zero_grad()
            criterion(model(feat), lbl).backward()
            optimizer.step()

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
    ap.add_argument("--model", required=True, choices=["mobilenet", "stgram"])
    ap.add_argument("--machines", nargs="+", default=MACHINES_ALL)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS_ALL)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    base = f"baselines_multiseed_{args.model}{suffix}"
    ckpt_path = os.path.join(OUT_DIR, f"{base}_checkpoint.json")
    out_path  = os.path.join(OUT_DIR, f"{base}_results.txt")

    ckpt = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        print(f"[RESUME] loaded checkpoint with {sum(len(v) for v in ckpt.values())} runs")

    print("=" * 70)
    print(f"SCRIPT 15 - Multi-seed baseline: {args.model.upper()}")
    print(f"Log-Mel({N_MELS}) + pos_weight + BCE | epochs={args.epochs}")
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
            r = train_eval(X, y, machine, seed, args.epochs, args.model)
            ckpt.setdefault(machine, []).append(r)
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f, indent=1)
            print(f"  seed {seed:>4}: AUC={r['auc']:.4f}  pAUC={r['pauc']:.4f}  "
                  f"F1={r['f1']:.4f}  Rec={r['recall']:.4f}  Acc={r['acc']:.2f}%  "
                  f"[{time.time()-t0:.0f}s]")
        del X, y

    print(f"\nTotal runtime: {(time.time()-t_start)/3600:.2f} h")

    metrics = ["auc", "pauc", "f1", "f1_cal", "recall", "recall_cal",
               "precision", "precision_cal", "acc", "acc_cal"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"MULTI-SEED SUPERVISED BASELINE - {args.model.upper()}\n")
        f.write(f"Log-Mel ({N_MELS} bands) + pos_weight + BCE | "
                f"epochs={args.epochs} | batch={BATCH_SIZE} | lr={LR}\n")
        f.write(f"Seeds: {args.seeds}\n")
        f.write("Same protocol as proposed config (script 12); architecture only differs.\n")
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
            f.write(f"  {'Seed':>6} {'AUC':>8} {'pAUC':>8} {'F1':>8} "
                    f"{'Recall':>8} {'Acc%':>8}\n")
            for r in sorted(rows, key=lambda r: args.seeds.index(r["seed"])
                            if r["seed"] in args.seeds else 99):
                f.write(f"  {r['seed']:>6} {r['auc']:>8.4f} {r['pauc']:>8.4f} "
                        f"{r['f1']:>8.4f} {r['recall']:>8.4f} {r['acc']:>8.2f}\n")
            f.write("\n")
    print(f"\n[SAVED] {out_path}")
    print(f"[SAVED] {ckpt_path}")
