"""
SCRIPT 21 - Figures for the revision experiments (script 19) and the deployment benchmark (script 18)
=====================================================================================================
All figures are drawn at the printed width of the manuscript (13.86 cm) so that their 7-8 pt text
prints at that size. Reads only result files; no training.

Usage: python 21_revision_figures.py [factorial arch snr convergence deployment]
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
W = 13.86 / 2.54
MACH = ["fan", "pump", "slider", "valve"]
COL = {"fan": "#2E86AB", "pump": "#E63946", "slider": "#2D6A4F", "valve": "#6A0572"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False})


def load(name):
    return json.load(open(os.path.join(RES, name)))


def save(fig, name):
    p = os.path.join(RES, name)
    fig.savefig(p, dpi=600, facecolor="white"); plt.close(fig); print("[SAVED]", p)


def runs(exp, machine=None, **cfg):
    return [r for r in load(f"revision_{exp}.json") if (machine is None or r["machine"] == machine)
            and all(r["cfg"].get(k) == v for k, v in cfg.items())]


FACT = [("mfcc", False, "MFCC, unweighted"), ("mfcc", True, "MFCC, weighted"),
        ("logmel", False, "Log-Mel, unweighted"), ("logmel", True, "Log-Mel, weighted")]


def factorial_runs(machine, rep, weighted):
    return runs("equiv" if weighted else "factorial", machine, rep=rep, weighted=weighted)


def seed42(machine, rep, weighted):
    """Seed-42 test predictions (utils/factorial_seed42_predictions.py re-runs them deterministically)."""
    return next(r for r in load("revision_factorial_seed42_predictions.json")
                if r["machine"] == machine and r["rep"] == rep and r["weighted"] == weighted)


def fig_factorial():
    # (a) valve confusion matrices at the 0.5 threshold, seed 42, for the four configurations
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.75), gridspec_kw=dict(left=0.13, right=0.985, bottom=0.24, top=0.84, wspace=0.12))
    for ax, (rep, w, title) in zip(axes, FACT):
        r = seed42("valve", rep, w)
        cm = confusion_matrix(r["y_true"], (np.array(r["y_prob"]) >= 0.5).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7.5,
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontweight="bold")
        ax.set_xticks([0, 1], ["Normal", "Abnormal"])
        ax.set_yticks([0, 1], ["Normal", "Abnormal"] if ax is axes[0] else ["", ""])
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=7.5); ax.set_xlabel("Predicted")
        if ax is axes[0]:
            ax.set_ylabel("True")
        for s in ax.spines.values():
            s.set_visible(False)
    save(fig, "fig_factorial_confusion.png")
    # (b) ROC curves, seed 42, one panel per machine, four configurations each
    fig, axes = plt.subplots(1, 4, figsize=(W, 2.1), sharey=True,
                             gridspec_kw=dict(left=0.09, right=0.99, bottom=0.4, top=0.9, wspace=0.12))
    styles = {("mfcc", False): ("#999999", "--"), ("mfcc", True): ("#999999", "-"),
              ("logmel", False): ("#1A56A8", "--"), ("logmel", True): ("#1A56A8", "-")}
    for ax, m in zip(axes, MACH):
        for rep, w, title in FACT:
            r = seed42(m, rep, w)
            fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
            c, ls = styles[(rep, w)]
            ax.plot(fpr, tpr, color=c, ls=ls, lw=1.1, label=title)
        ax.plot([0, 1], [0, 1], color="black", lw=0.5, ls=":")
        ax.set_title(m.capitalize()); ax.set_xlabel("False-positive rate")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
    axes[0].set_ylabel("True-positive rate")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.54, 0.0))
    save(fig, "fig_factorial_roc.png")


ARCH = ["2B-FC64", "2B-FC128", "3B-FC64", "3B-FC128", "3B-FC256", "4B-FC128"]


def arch_auc(variant, m):
    if variant == "3B-FC128":
        rs = runs("equiv", m, rep="logmel", weighted=True)
    else:
        rs = runs("arch", m, model=variant)
    return np.array([r["auc"] for r in rs])


def fig_arch():
    fig, ax = plt.subplots(figsize=(W, 2.1), gridspec_kw=dict(left=0.08, right=0.99, bottom=0.2, top=0.86))
    x = np.arange(len(ARCH)); bw = 0.2
    for i, m in enumerate(MACH):
        mu = [arch_auc(v, m).mean() for v in ARCH]; sd = [arch_auc(v, m).std() for v in ARCH]
        ax.bar(x + (i - 1.5) * bw, mu, bw, yerr=sd, color=COL[m], label=m.capitalize(), error_kw=dict(lw=0.7, capsize=1.5))
    ax.set_xticks(x, ARCH); ax.set_ylabel("AUC (ten-seed mean ± SD)"); ax.set_ylim(0.5, 1.0)
    ax.axvspan(2.5, 3.5, color="#FFF2CC", zorder=0)
    ax.legend(ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.16))
    save(fig, "fig_arch_multiseed.png")


def fig_snr():
    snrs = [("−6", None), ("0", "0"), ("+6", "6")]
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.1), gridspec_kw=dict(left=0.14, right=0.98, bottom=0.18, top=0.95))
    for m in MACH:
        mu, sd = [], []
        for _, s in snrs:
            rs = runs("equiv", m, rep="logmel", weighted=True) if s is None else runs("snr", m, snr=s)
            a = np.array([r["auc"] for r in rs]); mu.append(a.mean()); sd.append(a.std())
        ax.errorbar(range(3), mu, yerr=sd, marker="o", ms=3, lw=1.1, capsize=2, color=COL[m], label=m.capitalize())
    ax.axhline(0.64, color="black", ls="--", lw=0.8)
    ax.text(0.05, 0.63, "Published MIMII autoencoder at −6 dB", fontsize=6.5, va="top", ha="left")
    ax.set_xticks(range(3), [s + " dB" for s, _ in snrs]); ax.set_xlabel("SNR"); ax.set_ylabel("AUC (mean ± SD)")
    ax.set_ylim(0.55, 1.01); ax.legend(frameon=False, loc="center right", ncol=2, bbox_to_anchor=(1.0, 0.45))
    save(fig, "fig_snr_multiseed.png")


def fig_convergence():
    names = {"3B-FC128": "Proposed 3B-FC128", "mobilenet": "MobileNetV3", "stgram": "STgram-MFN-Lite"}
    cols = {"3B-FC128": "#1A56A8", "mobilenet": "#E69F00", "stgram": "#009E73"}
    fig, axes = plt.subplots(1, 4, figsize=(W, 2.0), sharey=True,
                             gridspec_kw=dict(left=0.08, right=0.99, bottom=0.33, top=0.88, wspace=0.12))
    for ax, m in zip(axes, MACH):
        for model in names:
            rs = runs("valcal", m, model=model)
            if not rs:
                continue
            c = np.array([[e["val_auc"] for e in r["curve"]] for r in rs])
            ep = np.arange(1, c.shape[1] + 1)
            ax.plot(ep, c.mean(0), color=cols[model], lw=1.1, label=names[model])
            ax.fill_between(ep, c.mean(0) - c.std(0), c.mean(0) + c.std(0), color=cols[model], alpha=0.15, lw=0)
        ax.axvline(10, color="black", ls=":", lw=0.8)
        ax.set_title(m.capitalize()); ax.set_xlabel("Epoch"); ax.set_ylim(0.5, 1.0)
    axes[0].set_ylabel("Validation AUC")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.53, 0.0))
    save(fig, "fig_convergence.png")


THREADS = [1, 2, 3, 4, 6]
VAR = {"logmel": "Log-Mel (proposed)", "mfcc": "MFCC"}


def fig_deployment():
    """Thread sweep (throughput and degradation under GPU load) and latency histograms from script 18."""
    d = load("deployment_benchmark.json"); R, raw = d["results"], d["raw_latencies_ms"]
    tp = lambda v, t, cond: np.array([R[f"A|{v}|t{t}|rep{r}"][cond]["clips_per_s"] for r in range(3)])
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw=dict(left=0.1, right=0.99, bottom=0.36, top=0.96, wspace=0.35))
    for v, c in (("logmel", "#1A56A8"), ("mfcc", "#999999")):
        iso = np.array([tp(v, t, "isolated") for t in THREADS]); con = np.array([tp(v, t, "concurrent") for t in THREADS])
        axes[0].plot(THREADS, iso.mean(1), "o-", color=c, ms=3, lw=1.1, label=f"{VAR[v]}, GPU idle")
        axes[0].plot(THREADS, con.mean(1), "s--", color=c, ms=3, lw=1.1, label=f"{VAR[v]}, GPU loaded")
        deg = np.maximum(0, (iso - con) / iso * 100); m = deg.mean(1)
        axes[1].errorbar(THREADS, m, yerr=[m - deg.min(1), deg.max(1) - m], fmt="o-", color=c, ms=3, lw=1.1, capsize=2,
                         label=VAR[v])
    for ax in axes:
        ax.set_xlabel("CPU threads"); ax.set_xticks(THREADS)
    # one legend below both panels: in-panel legends covered data points (colours also identify the right panel)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.54, 0.0))
    axes[0].set_ylabel("Throughput (clips/s)"); axes[1].set_ylabel("Degradation (%)")
    save(fig, "fig_thread_sweep.png")
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.9), gridspec_kw=dict(left=0.1, right=0.98, bottom=0.23, top=0.88, wspace=0.3))
    for ax, b, title in ((axes[0], 1, "Single clip (batch 1)"), (axes[1], 16, "Batch of 16 clips")):
        for cond, c, lab in (("isolated", "#1A56A8", "GPU idle"), ("concurrent", "#E63946", "GPU loaded")):
            x = np.array(raw[f"B|logmel|b{b}|{cond}"]); hi = np.percentile(x, 99.5)
            ax.hist(x[x <= hi], bins=50, color=c, alpha=0.6, label=lab)
        ax.set_title(title); ax.set_xlabel("Latency (ms)"); ax.set_ylabel("Count"); ax.legend(frameon=False)
    save(fig, "fig_latency.png")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["factorial", "arch", "snr", "convergence"] + (
        ["deployment"] if os.path.exists(os.path.join(RES, "deployment_benchmark.json")) else [])
    for t in todo:
        globals()["fig_" + t]()
