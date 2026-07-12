"""
SCRIPT 14 - Pipeline + architecture diagrams (Figures 1 and 2)
===============================================================
Pure-matplotlib schematic diagrams; no training, no dataset access.

Outputs:
    results/fig_pipeline.png      - dual-stream CPU/GPU deployment pipeline
    results/fig_architecture.png  - 3B-FC128 layer diagram
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(ROOT, "results")


def box(ax, x, y, w, h, text, fc="#E8F0FE", ec="#1A56A8", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, color="#333333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, color=color, lw=1.4))


def pipeline_figure():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.2); ax.axis("off")

    # acoustic branch (CPU)
    ax.add_patch(FancyBboxPatch((0.2, 2.9), 11.6, 2.1,
                 boxstyle="round,pad=0.03", fc="#F3F9F1", ec="#3B7A3B",
                 lw=1.6, alpha=0.85))
    ax.text(0.45, 4.75, "Acoustic branch - CPU only (AMD Ryzen 5 7500F, 2 threads, 0 MB VRAM)",
            fontsize=10, fontweight="bold", color="#2D5E2D")
    box(ax, 0.5, 3.2, 1.7, 1.0, "Microphone\n16 kHz WAV\n(10 s clips)", fc="#FFFFFF", ec="#3B7A3B")
    box(ax, 2.7, 3.2, 2.0, 1.0, "Log-Mel\nspectrogram\n64 bands x 400 frames", fc="#FFFFFF", ec="#3B7A3B")
    box(ax, 5.2, 3.2, 2.2, 1.0, "3B-FC128 CNN\n(3 conv blocks +\nFC-128 classifier)", fc="#D9EAD3", ec="#3B7A3B", bold=True)
    box(ax, 7.9, 3.2, 1.8, 1.0, "Sigmoid\np(abnormal)", fc="#FFFFFF", ec="#3B7A3B")
    box(ax, 10.1, 3.2, 1.5, 1.0, "Anomaly\nalert", fc="#FCE5CD", ec="#B45309")
    arrow(ax, 2.2, 3.7, 2.7, 3.7); arrow(ax, 4.7, 3.7, 5.2, 3.7)
    arrow(ax, 7.4, 3.7, 7.9, 3.7); arrow(ax, 9.7, 3.7, 10.1, 3.7)

    # visual branch (GPU)
    ax.add_patch(FancyBboxPatch((0.2, 0.3), 11.6, 2.1,
                 boxstyle="round,pad=0.03", fc="#FDF2F2", ec="#A83232",
                 lw=1.6, alpha=0.85))
    ax.text(0.45, 2.15, "Visual branch - GPU (NVIDIA RTX 5070, full VRAM budget retained)",
            fontsize=10, fontweight="bold", color="#7A2424")
    box(ax, 0.5, 0.6, 1.7, 1.0, "Camera /\nthermal\nimaging", fc="#FFFFFF", ec="#A83232")
    box(ax, 2.7, 0.6, 2.0, 1.0, "Pre-\nprocessing", fc="#FFFFFF", ec="#A83232")
    box(ax, 5.2, 0.6, 2.2, 1.0, "Visual inference\n(e.g. ResNet-50 /\nYOLO-class models)", fc="#F4CCCC", ec="#A83232", bold=True)
    box(ax, 7.9, 0.6, 1.8, 1.0, "Defect\ndetections", fc="#FFFFFF", ec="#A83232")
    box(ax, 10.1, 0.6, 1.5, 1.0, "Quality\ngate", fc="#FCE5CD", ec="#B45309")
    arrow(ax, 2.2, 1.1, 2.7, 1.1); arrow(ax, 4.7, 1.1, 5.2, 1.1)
    arrow(ax, 7.4, 1.1, 7.9, 1.1); arrow(ax, 9.7, 1.1, 10.1, 1.1)

    ax.text(6.0, 2.62, "Hardware-decoupled: no shared VRAM, ≤ 6% acoustic throughput "
            "degradation under full GPU load (MFCC variant, 2-thread operating point)",
            ha="center", fontsize=9.5,
            style="italic", color="#333333")
    plt.tight_layout()
    p = os.path.join(OUT, "fig_pipeline.png")
    plt.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[SAVED] {p}")


def architecture_figure():
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.8); ax.axis("off")

    layers = [
        ("Input\n1 x 64 x 400", "#FFFFFF", "#555555"),
        ("Conv 3x3, 16\nBN + ReLU\nMaxPool 2x2\n16 x 32 x 200", "#E8F0FE", "#1A56A8"),
        ("Conv 3x3, 32\nBN + ReLU\nMaxPool 2x2\n32 x 16 x 100", "#D2E3FC", "#1A56A8"),
        ("Conv 3x3, 64\nBN + ReLU\nMaxPool 2x2\n64 x 8 x 50", "#AECBFA", "#1A56A8"),
        ("Flatten\n25,600", "#FFF8E1", "#B45309"),
        ("FC 128\nReLU\nDropout 0.3", "#FFE0B2", "#B45309"),
        ("FC 1\nlogit", "#FFCC80", "#B45309"),
        ("Sigmoid\np(abnormal)", "#F4CCCC", "#A83232"),
    ]
    x = 0.3
    w, gap = 1.35, 0.32
    for i, (txt, fc, ec) in enumerate(layers):
        box(ax, x, 1.0, w, 1.8, txt, fc=fc, ec=ec, fs=8.5,
            bold=(i in (1, 2, 3, 5)))
        if i < len(layers) - 1:
            arrow(ax, x + w, 1.9, x + w + gap, 1.9)
        x += w + gap
    ax.text(6.5, 0.45, "3B-FC128: three convolutional blocks (16-32-64 channels) + 128-unit classifier. "
            "MFCC input (1 x 40 x 400) yields 2.07 M parameters; Log-Mel input (1 x 64 x 400) yields 3.30 M.",
            ha="center", fontsize=9, style="italic")
    ax.text(6.5, 3.45, "3B-FC128 architecture", ha="center", fontsize=12,
            fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, "fig_architecture.png")
    plt.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[SAVED] {p}")


if __name__ == "__main__":
    pipeline_figure()
    architecture_figure()
