"""
SCRIPT 19 - Revision experiments (ten seeds, GPU-trained, protocol of script 12)
================================================================================
Every experiment uses the training/evaluation protocol of script 12 (AdamW lr 1e-4, batch 32,
10 epochs, BCE with pos_weight = normal:abnormal ratio, train-set max-F1 threshold calibration,
seeds {42, 0, 123, 456, 789, 1, 2024, 999, 7, 314}, random_split with the seed's generator).
Models are built on the CPU right after torch.manual_seed(seed) and then moved to the GPU, so the
initial weights and the train/test partition are identical to a CPU run of script 12; the batch order
matches only in the first epoch (on the GPU, dropout draws from the CUDA generator instead of the CPU
one), and the arithmetic differs (cuDNN vs. oneDNN). Runs are deterministic on a given GPU.

Experiments (--exp):
  equiv      proposed config (log-Mel + W, 3B-FC128) and MFCC + W re-trained on the GPU, to check
             that the device does not change the ten-seed statistics of Table 5 (CPU)
  factorial  {MFCC, log-Mel} x unweighted, 3B-FC128 (the weighted cells are the 'equiv' runs)
  arch       2B-FC64, 2B-FC128, 3B-FC64, 3B-FC256, 4B-FC128 (log-Mel + W)
  snr        0 dB and +6 dB (log-Mel + W, 3B-FC128)
  split      70/30 and 85/15 (log-Mel + W, 3B-FC128)
  valcal     the 80/20 test set of each seed is kept; the training part is split 70/10 (train/val);
             30 epochs with per-epoch train loss and validation AUC; at epoch 10 the threshold is
             calibrated on the training set and, separately, on the validation set
             (models: 3B-FC128, MobileNetV3, STgram-MFN-Lite)
  enroll     leave-one-machine-ID-out with k% of the held-out unit added to training
             (k = 0, 10, 25; normal-only or labelled), seeds 42, 0, 123
  equiv_id   the log-Mel + W runs of 'equiv' repeated to record the AUC per machine unit (identical models)
  enroll10   the same for all ten seeds, plus false-alarm rate and recall on the held-out unit at 0.5, at the
             training-calibrated threshold, and at a unit-specific threshold set to 5% FPR on half of the unit's
             unseen normal clips (cross-fitted; np.quantile rule); seeds 42, 0, 123 reproduce 'enroll' exactly
  enroll10tau  the k = 0 and normal-only 10% conditions of enroll10 again (identical models), with a tie-aware unit
             threshold (at most 5% of the calibration normals flagged; collapsed units with constant scores get a
             threshold above them instead of one that flags every clip)
Features are read from the cache written by utils/cache_features.py.

Usage: python 19_revision_experiments.py --exp arch [--machines valve --seeds 42 --tag smoke]
Output: results/revision_<exp>[_tag].json (one record per run, resumable)
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, json, sys, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from sklearn.metrics import (f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "v2_scripts"))
from models import MobileNetV3Audio, STgramMFN_Lite  # noqa: E402

FEAT = os.environ.get("MIMII_FEATURES", r"E:\MIMII\features")
OUT = os.path.join(ROOT, "results")
MACHINES = ["fan", "pump", "slider", "valve"]
SEEDS = [42, 0, 123, 456, 789, 1, 2024, 999, 7, 314]
CLASS_RATIOS = {"fan": 2.8, "pump": 8.2, "slider": 3.6, "valve": 7.7}
BATCH, LR, MAX_LEN = 32, 1e-4, 400
DEV = torch.device("cuda")
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
torch.use_deterministic_algorithms(True, warn_only=True)


def build_variant(n_blocks, fc, n_bins):
    """Script 05's builder generalised to the input height; (3, 128) reproduces script 12's model."""
    ch = [1] + [16 * 2 ** i for i in range(n_blocks)]
    layers = []
    for i in range(n_blocks):
        layers += [nn.Conv2d(ch[i], ch[i + 1], 3, padding=1), nn.BatchNorm2d(ch[i + 1]), nn.ReLU(), nn.MaxPool2d(2, 2)]
    flat = ch[-1] * (n_bins // 2 ** n_blocks) * (MAX_LEN // 2 ** n_blocks)
    return nn.Sequential(*layers, nn.Flatten(), nn.Linear(flat, fc), nn.ReLU(), nn.Dropout(0.3), nn.Linear(fc, 1))


def build(model, n_bins):
    if model == "mobilenet":
        return MobileNetV3Audio(num_classes=1)
    if model == "stgram":
        return STgramMFN_Lite(num_classes=1)
    nb, fc = model.split("B-FC")                         # e.g. "3B-FC128"
    return build_variant(int(nb), int(fc), n_bins)


def load(machine, snr="m6", rep="logmel"):
    X = torch.from_numpy(np.load(os.path.join(FEAT, snr, f"{machine}_{rep}.npy")))
    meta = np.genfromtxt(os.path.join(FEAT, snr, f"{machine}_meta.csv"), delimiter=",", names=True,
                         dtype=None, encoding="utf-8")
    y = torch.from_numpy(meta["label"].astype(np.float32).reshape(-1, 1))
    return X, y, meta["machine_id"].astype(str)


def predict(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs.append(torch.sigmoid(model(xb.to(DEV))).squeeze(1).cpu().numpy())
            labels.append(yb.squeeze(1).numpy())
    return np.concatenate(labels), np.concatenate(probs)


def max_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.argmax(f1s[:-1])])


def test_metrics(y, p, tau, prefix=""):
    y05, yt = (p >= 0.5).astype(float), (p >= tau).astype(float)
    return {prefix + "auc": roc_auc_score(y, p), prefix + "pauc": roc_auc_score(y, p, max_fpr=0.1),
            prefix + "acc": float((y05 == y).mean() * 100), prefix + "precision": precision_score(y, y05, zero_division=0),
            prefix + "recall": recall_score(y, y05, zero_division=0), prefix + "f1": f1_score(y, y05, zero_division=0),
            prefix + "tau": tau, prefix + "f1_cal": f1_score(y, yt, zero_division=0),
            prefix + "recall_cal": recall_score(y, yt, zero_division=0),
            prefix + "precision_cal": precision_score(y, yt, zero_division=0)}


def train(X, y, train_idx, machine, seed, model_name, weighted, epochs, n_bins, monitor=None, snapshot=None):
    """Script 12's training loop on the GPU. `monitor` = {name: loader} evaluated after every epoch;
    `snapshot(epoch, model)` is called after every epoch (used for the epoch-10 evaluation)."""
    torch.manual_seed(seed); np.random.seed(seed)
    ds = TensorDataset(X, y)
    model = build(model_name, n_bins).to(DEV)
    loader = DataLoader(Subset(ds, train_idx), batch_size=BATCH, shuffle=True, num_workers=0)
    pw = torch.tensor([CLASS_RATIOS[machine]], device=DEV) if weighted else None
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = optim.AdamW(model.parameters(), lr=LR)
    curve = []
    for ep in range(1, epochs + 1):
        model.train(); tot, nb = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        rec = {"epoch": ep, "train_loss": tot / nb}
        for name, ld in (monitor or {}).items():
            rec[name + "_auc"] = roc_auc_score(*predict(model, ld))
        curve.append(rec)
        if snapshot:
            snapshot(ep, model)
    return model, curve


def split_8020(n, seed, frac=0.8):
    gen = torch.Generator().manual_seed(seed)
    tr, te = random_split(range(n), [int(frac * n), n - int(frac * n)], generator=gen)
    return list(tr.indices), list(te.indices)


def standard_run(X, y, machine, seed, model, weighted, n_bins, frac=0.8, epochs=10, ids=None):
    tr, te = split_8020(len(y), seed, frac)
    ds = TensorDataset(X, y)
    m, _ = train(X, y, tr, machine, seed, model, weighted, epochs, n_bins)
    tau = max_f1_threshold(*predict(m, DataLoader(Subset(ds, tr), batch_size=256)))
    yt, pt = predict(m, DataLoader(Subset(ds, te), batch_size=256))
    out = test_metrics(yt, pt, tau)
    if seed == 42 and ids is None:                       # kept for the illustrative confusion-matrix/ROC figures
        out.update(y_true=yt.astype(int).tolist(), y_prob=np.round(pt, 6).tolist())
    if ids is not None:                                  # per-unit AUC (DCASE convention), for the STgram-MFN comparison
        u = ids[te]
        per = {k: roc_auc_score(yt[u == k], pt[u == k]) for k in sorted(set(u)) if len(set(yt[u == k])) == 2}
        out.update(auc_per_id=per, auc_id_mean=float(np.mean(list(per.values()))))
    return out


def run_valcal(X, y, machine, seed, model, n_bins, epochs=30, snap=10):
    tr80, te = split_8020(len(y), seed)
    g = torch.Generator().manual_seed(seed + 1000)
    perm = torch.randperm(len(tr80), generator=g).tolist()
    n_val = round(len(y) * 0.1)
    val = [tr80[i] for i in perm[:n_val]]; tr = [tr80[i] for i in perm[n_val:]]
    ds = TensorDataset(X, y)
    L = lambda idx: DataLoader(Subset(ds, idx), batch_size=256)
    out = {}

    def snapshot(ep, m):
        if ep == snap:
            yv, pv = predict(m, L(val)); yt, pt = predict(m, L(te))
            out.update(test_metrics(yt, pt, max_f1_threshold(*predict(m, L(tr))), "e10_trcal_"))
            out.update({k: v for k, v in test_metrics(yt, pt, max_f1_threshold(yv, pv), "e10_valcal_").items()
                        if k.startswith("e10_valcal_") and ("tau" in k or "cal" in k)})
        out.setdefault("test_auc_by_epoch", []).append(roc_auc_score(*predict(m, L(te))))

    _, curve = train(X, y, tr, machine, seed, model, True, epochs, n_bins, monitor={"val": L(val)}, snapshot=snapshot)
    out["curve"] = curve
    best = max(curve, key=lambda r: r["val_auc"])["epoch"]
    out["best_val_epoch"] = best
    out["test_auc_at_best_val_epoch"] = out["test_auc_by_epoch"][best - 1]
    out.update(n_train=len(tr), n_val=len(val), n_test=len(te))
    return out


def fpr_threshold(neg, target):
    """Lowest threshold that flags at most `target` of the calibration normals with `score >= threshold`; tied scores
    stay below it, so a unit with constant scores gets a threshold above them (nothing flagged)."""
    s = np.sort(neg)[::-1]
    k = int(np.floor(target * len(s)))
    if k == 0:
        return float(np.nextafter(s[0], np.inf))
    t = s[k - 1]
    return float(np.nextafter(t, np.inf)) if (s >= t).sum() > k else float(t)


def unit_thresholds(m, ds, train_idx, test, yt, pt, seed, target_fpr=0.05, rule="quantile"):
    """False-alarm rate (FPR) and recall (TPR) on the held-out unit's test clips at three thresholds: 0.5, the
    training-calibrated max-F1 threshold, and a unit-specific threshold set to `target_fpr` on half of the unit's
    unseen normal clips and evaluated on the other half (two-fold cross-fitting, averaged)."""
    rate = lambda s, t: float((s >= t).mean())
    neg, pos = pt[yt == 0], pt[yt == 1]
    tau_tr = max_f1_threshold(*predict(m, DataLoader(Subset(ds, train_idx), batch_size=256)))
    out = {"tau_train": tau_tr, "fpr_05": rate(neg, 0.5), "tpr_05": rate(pos, 0.5),
           "fpr_train": rate(neg, tau_tr), "tpr_train": rate(pos, tau_tr)}
    perm = np.random.default_rng(seed + 7).permutation(len(neg))
    halves = (perm[: len(neg) // 2], perm[len(neg) // 2:])
    folds = []
    for cal, ev in (halves, halves[::-1]):
        t = fpr_threshold(neg[cal], target_fpr) if rule == "conservative" else float(np.quantile(neg[cal], 1 - target_fpr))
        folds.append((t, rate(neg[ev], t), rate(pos, t)))
    out.update(tau_unit=float(np.mean([f[0] for f in folds])), fpr_unit=float(np.mean([f[1] for f in folds])),
               tpr_unit=float(np.mean([f[2] for f in folds])), target_fpr=target_fpr, rule=rule)
    return out


def run_enroll(X, y, ids, machine, seed, held, frac, mode, n_bins, thresholds=False, rule="quantile"):
    rng = np.random.default_rng(seed)
    held_idx = np.where(ids == held)[0]
    other = np.where(ids != held)[0]
    lab = y[held_idx, 0].numpy()
    pools = [held_idx[lab == 0]] + ([held_idx[lab == 1]] if mode == "labelled" else [])  # stratified by class
    enrol = np.concatenate([rng.choice(p, size=int(round(frac * len(p))), replace=False) for p in pools])
    k = len(enrol)
    test = np.setdiff1d(held_idx, enrol)
    ds = TensorDataset(X, y)
    train_idx = np.concatenate([other, enrol]).tolist()
    m, _ = train(X, y, train_idx, machine, seed, "3B-FC128", True, 10, n_bins)
    yt, pt = predict(m, DataLoader(Subset(ds, test.tolist()), batch_size=256))
    out = {"auc": roc_auc_score(yt, pt), "n_enrol": int(k), "n_test": int(len(test)), "n_test_abnormal": int(yt.sum())}
    if thresholds:                                       # evaluation only; training is unchanged
        out.update(unit_thresholds(m, ds, train_idx, test, yt, pt, seed, rule=rule))
    return out


def jobs(exp):
    if exp == "equiv_id":                                 # = equiv log-Mel + W, re-run to record per-unit AUC
        return [dict(model="3B-FC128", rep="logmel", weighted=True)]
    if exp == "equiv":
        return [dict(model="3B-FC128", rep="logmel", weighted=True), dict(model="3B-FC128", rep="mfcc", weighted=True)]
    if exp == "factorial":                                # weighted cells = the 'equiv' runs
        return [dict(model="3B-FC128", rep=r, weighted=False) for r in ("mfcc", "logmel")]
    if exp == "arch":
        return [dict(model=m, rep="logmel", weighted=True) for m in ("2B-FC64", "2B-FC128", "3B-FC64", "3B-FC256", "4B-FC128")]
    if exp == "snr":
        return [dict(model="3B-FC128", rep="logmel", weighted=True, snr=s) for s in ("0", "6")]
    if exp == "split":
        return [dict(model="3B-FC128", rep="logmel", weighted=True, frac=f) for f in (0.7, 0.85)]
    if exp == "valcal":
        return [dict(model=m, rep="logmel") for m in ("3B-FC128", "mobilenet", "stgram")]
    if exp == "enroll10tau":                              # the two commissioning conditions, tie-aware unit threshold
        return [dict(frac=0.0, mode="labelled"), dict(frac=0.1, mode="normal")]
    if exp in ("enroll", "enroll10"):
        return [dict(frac=f, mode=md) for f, md in ((0.0, "labelled"), (0.1, "normal"), (0.25, "normal"),
                                                     (0.1, "labelled"), (0.25, "labelled"))]
    raise ValueError(exp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--machines", nargs="+", default=MACHINES)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    seeds = a.seeds or ([42, 0, 123] if a.exp == "enroll" else SEEDS)
    path = os.path.join(OUT, f"revision_{a.exp}{'_' + a.tag if a.tag else ''}.json")
    done = json.load(open(path)) if os.path.exists(path) else []
    key = lambda r: json.dumps({k: r[k] for k in sorted(r) if k in ("cfg", "machine", "seed", "held")}, sort_keys=True)
    seen = {key(r) for r in done}
    print(f"{a.exp}: {len(done)} runs already in {os.path.basename(path)} | torch {torch.__version__} "
          f"| {torch.cuda.get_device_name(0)}", flush=True)
    for cfg in jobs(a.exp):
        for machine in a.machines:
            snr, rep = {"0": "0", "6": "p6"}.get(cfg.get("snr"), "m6"), cfg.get("rep", "logmel")
            X, y, ids = load(machine, snr, rep)
            n_bins = X.shape[2]
            for seed in seeds:
                for held in (sorted(set(ids)) if a.exp.startswith("enroll") else [None]):
                    rec = {"cfg": cfg, "machine": machine, "seed": seed}
                    if held:
                        rec["held"] = held
                    if key(rec) in seen:
                        continue
                    t0 = time.time()
                    if a.exp == "valcal":
                        r = run_valcal(X, y, machine, seed, cfg["model"], n_bins)
                    elif a.exp.startswith("enroll"):
                        r = run_enroll(X, y, ids, machine, seed, held, cfg["frac"], cfg["mode"], n_bins,
                                       thresholds=a.exp in ("enroll10", "enroll10tau"),
                                       rule="conservative" if a.exp == "enroll10tau" else "quantile")
                    else:
                        r = standard_run(X, y, machine, seed, cfg["model"], cfg["weighted"], n_bins, cfg.get("frac", 0.8),
                                         ids=ids if a.exp == "equiv_id" else None)
                    rec.update(r, seconds=round(time.time() - t0, 1))
                    done.append(rec); seen.add(key(rec))
                    json.dump(done, open(path, "w"), indent=0)
                    auc = r.get("auc", r.get("e10_trcal_auc"))
                    print(f"  {json.dumps(cfg)} {machine:<6} seed {seed:>4}{' ' + held if held else ''}: "
                          f"AUC {auc:.4f} [{rec['seconds']}s]", flush=True)
    print("DONE", a.exp, flush=True)
