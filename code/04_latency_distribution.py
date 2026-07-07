"""
REVIEW SCRIPT 4 — Latency Distribution: P50 / P95 / P99
=========================================================
Collects per-sample latency distribution over 1000 inference cycles.
Reports mean, std, P50, P95, P99.
Also measures throughput stability over time (jitter analysis).

Outputs:
  - latency_distribution.txt
  - latency_distribution_plot.png

Usage:
    python 04_latency_distribution.py

Runtime: ~5 minutes
"""

import os, time, torch, torch.multiprocessing as mp, numpy as np
import torchvision.models as models
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cpu_model import AcousticAnomalyDetector

# ── CONFIG ────────────────────────────────────────────────────
NUM_CYCLES  = 1000        # More cycles = tighter distribution
BATCH_SIZE  = 16
WARMUP      = 50
N_THREADS   = 2           # Proposed configuration
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
# ──────────────────────────────────────────────────────────────

def collect_latencies(n_threads, label, with_gpu=False):
    torch.set_num_threads(n_threads)
    device = torch.device("cpu")
    model  = AcousticAnomalyDetector().to(device)
    model.eval()
    dummy  = torch.randn(BATCH_SIZE, 1, 40, 400)

    # Warmup
    for _ in range(WARMUP):
        with torch.no_grad(): _ = model(dummy)

    latencies_ms = []
    for _ in range(NUM_CYCLES):
        t0 = time.perf_counter()
        with torch.no_grad(): _ = model(dummy)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    lats = np.array(latencies_ms)
    # Per-sample latency = batch latency / batch_size
    per_sample = lats / BATCH_SIZE

    stats = dict(
        label    = label,
        batch_mean_ms = np.mean(lats),
        batch_std_ms  = np.std(lats),
        batch_p50_ms  = np.percentile(lats, 50),
        batch_p95_ms  = np.percentile(lats, 95),
        batch_p99_ms  = np.percentile(lats, 99),
        batch_max_ms  = np.max(lats),
        sample_mean_ms = np.mean(per_sample),
        sample_p95_ms  = np.percentile(per_sample, 95),
        sample_p99_ms  = np.percentile(per_sample, 99),
        fps            = (NUM_CYCLES * BATCH_SIZE) / (np.sum(lats) / 1000),
        raw_lats       = lats,
    )
    return stats


def gpu_stress():
    if not torch.cuda.is_available(): return
    device = torch.device("cuda")
    model  = models.resnet50(pretrained=False).to(device)
    model.eval()
    dummy  = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)
    while True:
        with torch.no_grad(): _ = model(dummy)


if __name__ == "__main__":
    mp.freeze_support()
    print("=" * 65)
    print("REVIEW SCRIPT 4 — Latency Distribution Analysis")
    print("=" * 65)

    # Phase 1: Isolated
    print("\nPhase 1: Isolated (GPU idle)...")
    iso = collect_latencies(N_THREADS, "Isolated", with_gpu=False)
    print(f"  Mean={iso['batch_mean_ms']:.2f}ms  P95={iso['batch_p95_ms']:.2f}ms  "
          f"P99={iso['batch_p99_ms']:.2f}ms  FPS={iso['fps']:.2f}")

    # Phase 2: Concurrent
    print("\nPhase 2: Concurrent (GPU at 100%)...")
    manager  = mp.Manager()
    gpu_proc = mp.Process(target=gpu_stress)
    gpu_proc.start()
    time.sleep(2)
    con = collect_latencies(N_THREADS, "Concurrent", with_gpu=True)
    gpu_proc.terminate(); gpu_proc.join()
    print(f"  Mean={con['batch_mean_ms']:.2f}ms  P95={con['batch_p95_ms']:.2f}ms  "
          f"P99={con['batch_p99_ms']:.2f}ms  FPS={con['fps']:.2f}")

    # ── Save text ─────────────────────────────────────────────
    out_txt = os.path.join(OUT_DIR, "latency_distribution.txt")
    with open(out_txt, "w") as f:
        f.write("=" * 65 + "\n")
        f.write("LATENCY DISTRIBUTION — AMD Ryzen 5 7500F | 2 threads\n")
        f.write(f"Batch size={BATCH_SIZE}, Cycles={NUM_CYCLES}, Warmup={WARMUP}\n")
        f.write("=" * 65 + "\n\n")
        for stats in [iso, con]:
            f.write(f"[{stats['label']}]\n")
            f.write(f"  Batch latency  — Mean: {stats['batch_mean_ms']:.2f} ms  "
                    f"Std: {stats['batch_std_ms']:.2f} ms\n")
            f.write(f"                   P50:  {stats['batch_p50_ms']:.2f} ms  "
                    f"P95: {stats['batch_p95_ms']:.2f} ms  "
                    f"P99: {stats['batch_p99_ms']:.2f} ms  "
                    f"Max: {stats['batch_max_ms']:.2f} ms\n")
            f.write(f"  Per-sample     — Mean: {stats['sample_mean_ms']:.3f} ms  "
                    f"P95: {stats['sample_p95_ms']:.3f} ms  "
                    f"P99: {stats['sample_p99_ms']:.3f} ms\n")
            f.write(f"  Throughput     — {stats['fps']:.2f} FPS\n")
            # Real-time margin using P99
            clip_duration_ms = 10_000  # 10 second MIMII clip
            margin = clip_duration_ms / stats['sample_p99_ms']
            f.write(f"  Real-time margin (P99): {margin:.0f}x "
                    f"(P99={stats['sample_p99_ms']:.2f}ms vs 10s clip)\n\n")

    # ── Plot ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    for idx, (stats, color) in enumerate([(iso, '#2E86AB'), (con, '#E63946')]):
        ax = fig.add_subplot(gs[0, idx])
        lats = stats['raw_lats']
        ax.hist(lats, bins=60, color=color, alpha=0.75, edgecolor='white',
                linewidth=0.5)
        for pct, val, ls in [(50, stats['batch_p50_ms'], '--'),
                              (95, stats['batch_p95_ms'], '-.'),
                              (99, stats['batch_p99_ms'], ':')]:
            ax.axvline(val, color='#333333', linewidth=1.5, linestyle=ls,
                       label=f'P{pct}={val:.1f}ms')
        ax.axvline(stats['batch_mean_ms'], color='black', linewidth=2,
                   label=f"Mean={stats['batch_mean_ms']:.1f}ms")
        ax.set_xlabel('Batch Latency (ms)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f"{stats['label']}\n({stats['fps']:.1f} FPS)",
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(
        'Inference Latency Distribution — 2-Thread CPU-Bound 2D-CNN\n'
        'AMD Ryzen 5 7500F | Batch=16 | 1000 cycles',
        fontsize=11, fontweight='bold'
    )
    out_plot = os.path.join(OUT_DIR, "latency_distribution_plot.png")
    plt.savefig(out_plot, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n[SAVED] {out_txt}")
    print(f"[SAVED] {out_plot}")
