"""
SCRIPT 14 - Pipeline + architecture diagrams
=============================================
Pure-matplotlib schematic diagrams; no training, no dataset access.

Both figures are drawn at their printed size (13.86 cm = 5.46 in, the MDPI text
width) so that the font sizes set here are the sizes that appear on paper
(8-9 pt); earlier versions were drawn on 12-14 in canvases and shrank to ~4 pt.

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
WIDTH_IN = 13.86 / 2.54          # printed width in the manuscript
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def box(ax, x, y, w, h, text, fc="#E8F0FE", ec="#1A56A8", fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, x1, y1, x2, y2, color="#333333", **kw):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=9, color=color, lw=1.0, **kw))


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=600, facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {p}")


def pipeline_figure():
    # axes units = inches, so box sizes and font sizes share one scale
    W, H = WIDTH_IN, 2.45
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    bw, bh, gap, x0 = 0.88, 0.6, 0.16, 0.17
    xs = [x0 + i * (bw + gap) for i in range(5)]
    lane_h = 0.98
    lanes = [
        (1.35, "#F3F9F1", "#3B7A3B", "#2D5E2D", "#D9EAD3",
         "Acoustic branch: CPU only (AMD Ryzen 5 7500F, 2 threads, no VRAM)",
         ["Microphone\n16 kHz,\n10 s clips", "Log-Mel\nspectrogram\n64 × 400", "3B-FC128\nCNN",
          "Sigmoid\np(abnormal)", "Anomaly\nalert"]),
        (0.06, "#FDF2F2", "#A83232", "#7A2424", "#F4CCCC",
         "Visual branch: GPU (NVIDIA RTX 5070, full VRAM budget)",
         ["Camera /\nthermal\nimaging", "Pre-\nprocessing", "Visual\ninference\n(ResNet-50)",
          "Defect\ndetections", "Quality\ngate"]),
    ]
    for y, fc, ec, tc, hc, title, labels in lanes:
        ax.add_patch(FancyBboxPatch((0.06, y), W - 0.12, lane_h, boxstyle="round,pad=0.01,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=1.1))
        ax.text(0.15, y + lane_h - 0.15, title, fontsize=8, fontweight="bold", color=tc, va="center")
        for i, (x, lab) in enumerate(zip(xs, labels)):
            box(ax, x, y + 0.09, bw, bh, lab, fc="#FCE5CD" if i == 4 else (hc if i == 2 else "#FFFFFF"),
                ec="#B45309" if i == 4 else ec, fs=7.5, bold=(i == 2))
            if i < 4:
                arrow(ax, x + bw, y + 0.09 + bh / 2, x + bw + gap, y + 0.09 + bh / 2)
    ax.text(W / 2, 1.195, "No shared GPU memory: the acoustic model never uses the GPU",
            ha="center", va="center", fontsize=7.5, style="italic", color="#333333")
    save(fig, "fig_pipeline.png")


def architecture_figure():
    W, H = WIDTH_IN, 2.45
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    x0, gap = 0.3, 0.19
    bw, bh = (W - x0 - 0.05 - 3 * gap) / 4, 0.82
    xs = [x0 + i * (bw + gap) for i in range(4)]
    y1, y2 = 1.42, 0.27
    rows = [
        (y1, "Feature extractor", [
            ("Input\nlog-Mel\n1 × 64 × 400", "#FFFFFF", "#555555"),
            ("Block 1\nConv 3×3, 16 ch\nBN + ReLU\nMaxPool 2×2\n→ 16 × 32 × 200", "#E8F0FE", "#1A56A8"),
            ("Block 2\nConv 3×3, 32 ch\nBN + ReLU\nMaxPool 2×2\n→ 32 × 16 × 100", "#D2E3FC", "#1A56A8"),
            ("Block 3\nConv 3×3, 64 ch\nBN + ReLU\nMaxPool 2×2\n→ 64 × 8 × 50", "#AECBFA", "#1A56A8")]),
        (y2, "Classifier", [
            ("Flatten\n25,600 features", "#FFF8E1", "#B45309"),
            ("Fully connected,\n128 units\nReLU, dropout 0.3", "#FFE0B2", "#B45309"),
            ("Fully connected,\n1 unit (logit)", "#FFCC80", "#B45309"),
            ("Sigmoid\np(abnormal)", "#F4CCCC", "#A83232")]),
    ]
    for y, title, cells in rows:                     # row labels run vertically in the left margin
        ax.text(0.13, y + bh / 2, title, rotation=90, ha="center", va="center", fontsize=8, fontweight="bold")
        for i, (x, (txt, fc, ec)) in enumerate(zip(xs, cells)):
            box(ax, x, y, bw, bh, txt, fc=fc, ec=ec, fs=7.5)
            if i < 3:
                arrow(ax, x + bw, y + bh / 2, x + bw + gap, y + bh / 2)
    # Block 3 -> Flatten: straight segments through the empty band between the rows
    yc = (y1 + y2 + bh) / 2
    xb3, xfl = xs[3] + bw / 2, xs[0] + bw / 2
    ax.plot([xb3, xb3, xfl], [y1, yc, yc], color="#333333", lw=1.0, solid_capstyle="butt")
    arrow(ax, xfl, yc, xfl, y2 + bh, shrinkA=0)
    ax.text(W / 2, 0.1, "Parameters: 2.07 M with MFCC input (1 × 40 × 400); "
            "3.30 M with log-Mel input (1 × 64 × 400)", ha="center", fontsize=7.5, style="italic")
    save(fig, "fig_architecture.png")


if __name__ == "__main__":
    pipeline_figure()
    architecture_figure()
